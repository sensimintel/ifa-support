# -*- coding: utf-8 -*-
"""
Depth Anything 3 Web 服务：一个页面、一个 8060 端口，左右两栏对比。
- 左栏：官方 Gradio UI（点云 / 网格 / 3D 量距等），通过 gr.mount_gradio_app 挂在同一 FastAPI 的 /gradio。
- 右栏：自研扩展面板（/panel），可在网页上调参对比，支持三种产物：
    · 深度图      —— 彩色深度图（越亮=越近）
    · 点云+相机    —— DA3 导出 scene.glb（点云 + 相机线框），可 3D 转视角
    · 网格 mesh   —— 由深度反投影自建三角网格 GLB，可 3D 转视角
  可调参数：process_res、conf_thresh_percentile、num_max_points、show_cameras。
- 顶层 / 是左右分栏首页，用两个同源 iframe 分别嵌入 /gradio 与 /panel。
- 关键约束：5090 GPU 与产线服务共享，官方 Gradio 与右栏共用同一个模型单例（进程内只加载一份
  权重），并用 GPU 锁串行化推理，避免加载两份权重撑爆显存。
- 绑定 0.0.0.0，局域网内可直接用 http://<5090局域网IP>:8060 访问。
"""
import base64
import io
import json
import os
import shutil
import socket
import struct
import sys
import threading
import time
import uuid
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
import trimesh
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from PIL import Image, ImageOps

# 把 DA3 源码目录加入 import 路径（本服务独立于 DA3 仓，只引用其 src）
DA3_ROOT = Path("/home/odyss/Depth-Anything-3")
sys.path.append(str(DA3_ROOT / "src"))
from depth_anything_3.api import DepthAnything3  # noqa: E402
from depth_anything_3.app.gradio_app import DepthAnything3App  # noqa: E402

MODEL_DIR = str(DA3_ROOT / "models" / "DA3NESTED-GIANT-LARGE-1.1")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROCESS_RES = 504  # DA3 默认处理分辨率

# GLB / mesh 产物落盘目录（每次推理一个子目录，超量自动清理）
GLB_DIR = Path("/home/odyss/da3-web/glb_out")
GLB_DIR.mkdir(parents=True, exist_ok=True)
GLB_KEEP = 24  # 最多保留最近多少次产物

app = FastAPI(title="DA3 Depth Web")
_model = None
_model_lock = threading.Lock()   # 保护模型单例的加载
_gpu_lock = threading.Lock()     # 串行化所有 GPU 推理（右栏 + 官方 Gradio 共用同一模型）


# ══════════════════════════════════════════════════════════════════════
# 电子秤实时重量模块（零依赖：socket 手写 Modbus TCP，后台线程统一轮询并缓存）
#   秤A  SJ101CX @ 192.168.0.80  寄存器0    字序HH-LL  分度0.1  站号1
#   秤B  Y31X04  @ 192.168.0.90  寄存器450  字序LL-HH  分度0.1  站号1
# ══════════════════════════════════════════════════════════════════════
SCALES = [
    {"id": "A", "name": "秤A · SJ101CX", "host": "192.168.0.80", "port": 502,
     "unit": 1, "addr": 0, "word_order": "HH-LL", "division": 0.1},
    {"id": "B", "name": "秤B · Y31X04", "host": "192.168.0.90", "port": 502,
     "unit": 1, "addr": 450, "word_order": "LL-HH", "division": 0.1},
]
SCALE_POLL_INTERVAL = 0.4   # 后台轮询间隔（秒）
SCALE_TIMEOUT = 1.2         # 单次 Modbus 读超时（秒）
_scale_latest = {s["id"]: {"ok": False, "weight": None, "raw": None} for s in SCALES}
_scale_lock = threading.Lock()


def _recv_exact(sock, n):
    """从 socket 精确读取 n 个字节。"""
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("连接被对端关闭")
        buf += chunk
    return buf


def _read_scale_int32(host, port, unit, addr, word_order, timeout):
    """手写 Modbus TCP FC3：读 2 个保持寄存器并解码为有符号 32 位整数。"""
    req = struct.pack(">HHHBBHH", 1, 0, 6, unit, 0x03, addr, 2)
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        sock.sendall(req)
        head = _recv_exact(sock, 9)          # MBAP(7)+功能码(1)+字节数(1)
        if head[7] & 0x80:
            raise IOError(f"Modbus 异常响应，功能码 0x{head[7]:02X}")
        data = _recv_exact(sock, head[8])
    regs = struct.unpack(">" + "H" * (head[8] // 2), data)
    low, high = (regs[0], regs[1]) if word_order.upper() == "LL-HH" else (regs[1], regs[0])
    raw_u = ((high << 16) | low) & 0xFFFFFFFF
    return struct.unpack(">i", struct.pack(">I", raw_u))[0]


def _scale_poller():
    """后台线程：周期性读两个秤，写入缓存。"""
    while True:
        for s in SCALES:
            try:
                raw = _read_scale_int32(s["host"], s["port"], s["unit"],
                                        s["addr"], s["word_order"], SCALE_TIMEOUT)
                with _scale_lock:
                    _scale_latest[s["id"]] = {
                        "ok": True, "weight": round(raw * s["division"], 1), "raw": raw}
            except Exception:  # 读失败：标记离线，保留上次读数
                with _scale_lock:
                    prev = _scale_latest[s["id"]]
                    _scale_latest[s["id"]] = {
                        "ok": False, "weight": prev.get("weight"), "raw": prev.get("raw")}
        time.sleep(SCALE_POLL_INTERVAL)


# 启动后台轮询线程（daemon：随主进程退出）
threading.Thread(target=_scale_poller, daemon=True).start()


def get_model():
    """懒加载并缓存 DA3 模型单例（fp32 权重；forward 内部用 autocast bf16 计算）。

    右栏推理与官方 Gradio 都复用这同一个实例，进程内只占一份权重（约 6.5GB）。
    """
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                print(f"[da3-web] 正在从 {MODEL_DIR} 加载模型到 {DEVICE} ...", flush=True)
                t0 = time.time()
                _model = DepthAnything3.from_pretrained(MODEL_DIR).to(DEVICE).eval()
                print(f"[da3-web] 模型加载完成，耗时 {time.time() - t0:.1f}s", flush=True)
    return _model


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    """把单通道深度图上色为 BGR 彩色图。用 2%-98% 分位裁剪去离群，越亮=越近。"""
    d = depth.astype(np.float32)
    valid = np.isfinite(d)
    if not valid.any():
        return np.zeros((*d.shape, 3), dtype=np.uint8)
    lo, hi = np.percentile(d[valid], [2, 98])
    dn = np.clip((d - lo) / (hi - lo + 1e-8), 0, 1)
    dn = 1.0 - dn  # 反转：深度小(近)→高值→亮，符合直觉
    u8 = (dn * 255).astype(np.uint8)
    color = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)  # BGR
    color[~valid] = 0
    return color


def to_data_uri_bgr(bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", bgr)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def to_data_uri_rgb(rgb: np.ndarray) -> str:
    return to_data_uri_bgr(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def _rgb_uint8(pred, arr_fallback):
    """从 prediction 取与深度同尺寸的 RGB（uint8）；缺省回退原图。"""
    if getattr(pred, "processed_images", None) is not None:
        rgb = np.asarray(pred.processed_images)[0]
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb * (255 if rgb.max() <= 1.0 else 1), 0, 255).astype(np.uint8)
        return rgb
    return arr_fallback


def build_mesh_glb(pred, out_path, conf_thresh_percentile=40.0, edge_ratio=0.06):
    """由单视图 depth + 内参反投影出结构化点，构建带顶点色的三角网格并导出为 GLB。

    - 用 intrinsics 把每个像素反投影到相机坐标；剔除天空 / 低置信 / 深度不连续（避免边缘拉丝）。
    - 顶点色取 processed_images。model-viewer 可直接 3D 转视角查看。
    """
    depth = np.asarray(pred.depth)[0].astype(np.float32)  # H,W
    H, W = depth.shape
    K = np.asarray(pred.intrinsics)[0].astype(np.float32)  # 3x3
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    rgb = _rgb_uint8(pred, None)
    if rgb is None or rgb.shape[:2] != depth.shape:
        rgb = np.full((H, W, 3), 180, np.uint8)

    valid = np.isfinite(depth) & (depth > 0)
    if getattr(pred, "sky", None) is not None:
        valid &= ~np.asarray(pred.sky)[0].astype(bool)
    if getattr(pred, "conf", None) is not None:
        conf = np.asarray(pred.conf)[0].astype(np.float32)
        finite = np.isfinite(conf)
        if finite.any():
            thr = np.percentile(conf[finite], conf_thresh_percentile)
            valid &= conf >= thr

    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    z = depth
    x = (us - cx) / fx * z
    y = (vs - cy) / fy * z
    # gltf 约定 +Y 向上、相机看 -Z：翻转 y/z 让默认视角更自然（用户仍可自由转）
    verts = np.stack([x, -y, -z], axis=-1).reshape(-1, 3).astype(np.float32)
    cols = rgb.reshape(-1, 3)

    idx = np.arange(H * W).reshape(H, W)
    tl, tr = idx[:-1, :-1], idx[:-1, 1:]
    bl, br = idx[1:, :-1], idx[1:, 1:]
    quad_valid = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]
    d4 = np.stack([depth[:-1, :-1], depth[:-1, 1:], depth[1:, :-1], depth[1:, 1:]])
    dmax, dmin, dmed = d4.max(0), d4.min(0), np.median(d4, 0)
    cont = (dmax - dmin) <= (edge_ratio * np.maximum(dmed, 1e-6))  # 深度不连续处不连面
    keep = quad_valid & cont
    tl, tr, bl, br = tl[keep], tr[keep], bl[keep], br[keep]
    faces = np.concatenate(
        [np.stack([tl, bl, tr], -1), np.stack([tr, bl, br], -1)], axis=0)
    if len(faces) == 0:
        raise RuntimeError("有效网格面为 0（可能置信阈值过高或深度不连续过多）")
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=cols, process=False)
    mesh.export(out_path)
    return len(verts), len(faces)


def _prune_glb():
    """只保留最近 GLB_KEEP 个产物子目录，清理旧的。"""
    try:
        dirs = sorted([p for p in GLB_DIR.iterdir() if p.is_dir()],
                      key=lambda p: p.stat().st_mtime, reverse=True)
        for p in dirs[GLB_KEEP:]:
            shutil.rmtree(p, ignore_errors=True)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════
# 顶层分栏首页：左 iframe=官方 Gradio(/gradio)，右 iframe=扩展面板(/panel)
# ══════════════════════════════════════════════════════════════════════
SPLIT_PAGE = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DA3 · 官方 Gradio ＋ 扩展面板</title>
<style>
 *{box-sizing:border-box}
 html,body{margin:0;height:100%;font-family:system-ui,-apple-system,'Segoe UI',sans-serif;background:#0d0d0f}
 .top{height:48px;display:flex;align-items:center;gap:14px;padding:0 18px;color:#fff;background:#1c1c1e;font-size:14px;flex-wrap:wrap}
 .top b{font-size:15px}
 .top .tag{font-size:12px;color:#9a9aa0}
 .wrap{display:flex;height:calc(100% - 48px)}
 .pane{flex:1 1 50%;min-width:0;display:flex;flex-direction:column;border-right:1px solid #2c2c2e}
 .pane:last-child{border-right:0}
 .pane .bar{height:34px;display:flex;align-items:center;padding:0 14px;color:#e5e5ea;background:#141416;font-size:13px;font-weight:600;border-bottom:1px solid #2c2c2e}
 .pane .bar .dot{width:8px;height:8px;border-radius:50%;margin-right:8px}
 .pane .bar a{margin-left:auto;font-size:12px;font-weight:500;color:#0a84ff;text-decoration:none}
 iframe{flex:1;width:100%;border:0;background:#fff}
 @media(max-width:900px){.wrap{flex-direction:column;height:auto}.pane{height:90vh;border-right:0;border-bottom:1px solid #2c2c2e}}
</style></head><body>
 <div class="top"><b>Depth Anything 3</b>
  <span class="tag">左：官方 Gradio（点云 / 网格 / 3D 量距）　·　右：扩展面板（深度图 · 点云 · 网格 · 电子秤，可调参转视角）　·　同一 8060 端口</span></div>
 <div class="wrap">
  <div class="pane">
   <div class="bar"><span class="dot" style="background:#34c759"></span>官方 Gradio UI
    <a href="/gradio" target="_blank">单独打开 ↗</a></div>
   <iframe src="/gradio" title="官方 Gradio"></iframe>
  </div>
  <div class="pane">
   <div class="bar"><span class="dot" style="background:#0a84ff"></span>扩展面板（自研）
    <a href="/panel" target="_blank">单独打开 ↗</a></div>
   <iframe src="/panel" title="扩展面板"></iframe>
  </div>
 </div>
</body></html>"""


# ══════════════════════════════════════════════════════════════════════
# 扩展面板：调参 + 三种产物（深度图 / 点云GLB / 网格GLB），前端 fetch + model-viewer
# ══════════════════════════════════════════════════════════════════════
PANEL_PAGE = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DA3 扩展面板 · 深度/点云/网格</title>
<script type="module" src="https://unpkg.com/@google/model-viewer@3.5.0/dist/model-viewer.min.js"></script>
<style>
 body{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;max-width:1180px;margin:24px auto;padding:0 16px;color:#1c1c1e;background:#f5f5f7}
 h1{font-size:21px;margin:.2em 0} .sub{color:#6b6b70;font-size:13px;margin-bottom:18px}
 .nav{display:flex;gap:14px;margin-bottom:16px;font-size:14px;align-items:center}
 .nav a{padding:6px 14px;border-radius:980px;background:#fff;box-shadow:0 1px 3px rgba(0,0,0,.08);color:#0071e3;text-decoration:none}
 .nav a.active{background:#0071e3;color:#fff}
 .nav .home{margin-left:auto;box-shadow:none;background:transparent;color:#6b6b70}
 .card{background:#fff;border-radius:14px;padding:18px 20px;box-shadow:0 1px 4px rgba(0,0,0,.08);margin-bottom:18px}
 label{font-size:13px;color:#3a3a3c;display:block;margin:0 0 4px}
 .row{display:flex;flex-wrap:wrap;gap:18px;align-items:flex-end}
 .fld{flex:1 1 180px;min-width:150px}
 select,input[type=file]{width:100%;font-size:14px}
 select{padding:8px;border:1px solid #d0d0d5;border-radius:8px;background:#fff}
 input[type=range]{width:100%}
 .rngval{font-variant-numeric:tabular-nums;color:#0071e3;font-weight:600}
 .glbopts{border-top:1px dashed #e0e0e5;margin-top:14px;padding-top:14px}
 button{background:#0071e3;color:#fff;border:0;border-radius:980px;padding:10px 26px;font-size:15px;cursor:pointer;margin-top:6px}
 button:disabled{opacity:.5;cursor:default}
 .hint{font-size:12px;color:#8e8e93;margin-top:6px}
 .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
 .grid figure{margin:0} .grid img{width:100%;border-radius:10px;display:block;background:#000}
 figcaption{font-size:13px;color:#6b6b70;margin-top:6px;text-align:center}
 model-viewer{width:100%;height:520px;background:#111319;border-radius:12px;--poster-color:transparent}
 .meta{font-size:13px;color:#6b6b70;margin-top:10px}
 a.dl{color:#0071e3;text-decoration:none}
 .err{color:#c1121f}
 @media(max-width:720px){.grid{grid-template-columns:1fr}}
</style></head><body>
<div class="nav"><a class="active" href="/panel">深度 / 点云 / 网格</a><a href="/weight">电子秤实时重量</a><a class="home" href="/" target="_top">↗ 对比首页</a></div>
<h1>Depth Anything 3 · 扩展面板</h1>
<div class="sub">上传一张图，选产物类型 + 调参，右侧点云 / 网格可用鼠标 3D 转视角。模型：DA3NESTED-GIANT-LARGE-1.1</div>

<div class="card">
 <div class="row">
  <div class="fld"><label>图片</label><input type="file" id="file" accept="image/*"></div>
  <div class="fld"><label>产物类型 export_format</label>
   <select id="fmt">
    <option value="depth">深度图（彩色）</option>
    <option value="glb">点云 + 相机（GLB · 可转视角）</option>
    <option value="mesh">网格 mesh（GLB · 可转视角）</option>
   </select></div>
  <div class="fld"><label>处理分辨率 process_res <span class="rngval" id="prv">504</span></label>
   <input type="range" id="pr" min="196" max="896" step="28" value="504"></div>
 </div>
 <div class="glbopts" id="glbopts">
  <div class="row">
   <div class="fld"><label>置信度裁剪分位 conf_thresh_percentile <span class="rngval" id="ctv">40</span>%</label>
    <input type="range" id="ct" min="0" max="90" step="5" value="40"></div>
   <div class="fld" id="nmpwrap"><label>点云最大点数 num_max_points <span class="rngval" id="nmv">0.8</span>M</label>
    <input type="range" id="nmp" min="0.1" max="2" step="0.1" value="0.8"></div>
   <div class="fld" id="camwrap"><label>相机线框 show_cameras</label>
    <select id="cam"><option value="1">显示</option><option value="0">隐藏</option></select></div>
  </div>
  <div class="hint" id="glbhint">点云用 DA3 官方 scene.glb 导出；网格由深度反投影自建三角面（conf 分位越高越干净，num_max_points 仅点云生效）。</div>
 </div>
 <button id="go">生成</button>
 <span class="hint" id="tip"></span>
</div>

<div class="card" id="out" style="display:none"></div>

<script>
const $=id=>document.getElementById(id);
$('pr').oninput=()=>$('prv').textContent=$('pr').value;
$('ct').oninput=()=>$('ctv').textContent=$('ct').value;
$('nmp').oninput=()=>$('nmv').textContent=(+$('nmp').value).toFixed(1);
function syncOpts(){const f=$('fmt').value;
 $('glbopts').style.display=(f==='depth')?'none':'block';
 $('nmpwrap').style.display=(f==='glb')?'block':'none';
 $('camwrap').style.display=(f==='glb')?'block':'none';}
$('fmt').onchange=syncOpts;syncOpts();

$('go').onclick=async()=>{
 const f=$('file').files[0];
 if(!f){$('tip').textContent='请先选择图片';return;}
 const fd=new FormData();
 fd.append('file',f);
 fd.append('export_format',$('fmt').value);
 fd.append('process_res',$('pr').value);
 fd.append('conf_thresh_percentile',$('ct').value);
 fd.append('num_max_points',Math.round(+$('nmp').value*1e6));
 fd.append('show_cameras',$('cam').value);
 $('go').disabled=true;$('go').textContent='推理中…';$('tip').textContent='';
 const out=$('out');out.style.display='block';out.innerHTML='<div class="meta">⏳ GPU 推理中，请稍候…</div>';
 try{
  const r=await fetch('/api/infer',{method:'POST',body:fd});
  const j=await r.json();
  if(!r.ok||j.error){out.innerHTML='<div class="err">出错：'+(j.error||('HTTP '+r.status))+'</div>';}
  else if(j.mode==='depth'){
   out.innerHTML=`<div class="grid">
     <figure><img src="${j.input_uri}"><figcaption>输入图（已按 EXIF 转正）</figcaption></figure>
     <figure><img src="${j.depth_uri}"><figcaption>深度图（越亮=越近）</figcaption></figure></div>
    <div class="meta">推理耗时 ${j.dt.toFixed(2)}s ｜ 深度范围 ${j.dmin.toFixed(3)} ~ ${j.dmax.toFixed(3)} ｜ 分辨率 ${j.shape[0]}×${j.shape[1]}</div>`;
  }else{
   out.innerHTML=`<model-viewer src="${j.glb_url}" camera-controls touch-action="pan-y" auto-rotate
      camera-orbit="0deg 80deg 30%" field-of-view="28deg" min-camera-orbit="auto auto 3%"
      interaction-prompt="none" shadow-intensity="0.3" exposure="1.35"></model-viewer>
    <div class="meta">${j.label} ｜ 推理耗时 ${j.dt.toFixed(2)}s ｜ ${j.stat} ｜
      <a class="dl" href="${j.glb_url}" download>下载 GLB ↓</a> ｜ 用鼠标拖拽转视角、滚轮缩放</div>`;
  }
 }catch(e){out.innerHTML='<div class="err">请求失败：'+e+'</div>';}
 $('go').disabled=false;$('go').textContent='生成';
};
</script>
</body></html>"""


@app.get("/", response_class=HTMLResponse)
def home():
    """左右分栏对比首页。"""
    return SPLIT_PAGE


@app.get("/panel", response_class=HTMLResponse)
def panel():
    """扩展面板：调参 + 深度图/点云/网格。"""
    return PANEL_PAGE


@app.post("/api/infer")
async def api_infer(
    file: UploadFile = File(...),
    export_format: str = Form("depth"),          # depth | glb | mesh
    process_res: int = Form(504),
    conf_thresh_percentile: float = Form(40.0),
    num_max_points: int = Form(800000),
    show_cameras: str = Form("1"),
):
    """统一推理入口：按 export_format 返回深度图 data-uri 或 GLB 下载地址。"""
    raw = await file.read()
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    except Exception as e:
        return JSONResponse({"error": f"读取图片失败：{e}"}, status_code=400)
    arr = np.array(img)
    res = int(max(140, min(896, process_res)))
    show_cam = str(show_cameras) in ("1", "true", "True", "on")

    model = get_model()
    try:
        with _gpu_lock:  # 与官方 Gradio 共用同一模型，串行化 GPU 推理
            t0 = time.time()
            if export_format in ("glb", "mesh"):
                token = uuid.uuid4().hex
                outdir = GLB_DIR / token
                outdir.mkdir(parents=True, exist_ok=True)
                if export_format == "glb":
                    with torch.no_grad():
                        pred = model.inference(
                            [arr], process_res=res, export_dir=str(outdir),
                            export_format="glb",
                            conf_thresh_percentile=float(conf_thresh_percentile),
                            num_max_points=int(num_max_points), show_cameras=show_cam)
                    dt = time.time() - t0
                    glb = outdir / "scene.glb"
                    sz = glb.stat().st_size / 1024 if glb.exists() else 0
                    depth = np.asarray(pred.depth)[0]
                    _prune_glb()
                    return JSONResponse({
                        "mode": "glb", "glb_url": f"/glb/{token}/scene.glb", "dt": dt,
                        "label": "点云 + 相机线框", "stat": f"GLB {sz:.0f}KB",
                        "shape": [int(depth.shape[1]), int(depth.shape[0])]})
                else:  # mesh：先出 prediction（含 intrinsics），再自建网格
                    with torch.no_grad():
                        pred = model.inference([arr], process_res=res, export_format="mini_npz")
                    glb = outdir / "scene.glb"
                    nv, nf = build_mesh_glb(
                        pred, str(glb), conf_thresh_percentile=float(conf_thresh_percentile))
                    dt = time.time() - t0
                    sz = glb.stat().st_size / 1024 if glb.exists() else 0
                    _prune_glb()
                    return JSONResponse({
                        "mode": "mesh", "glb_url": f"/glb/{token}/scene.glb", "dt": dt,
                        "label": "三角网格 mesh",
                        "stat": f"顶点 {nv:,} · 面 {nf:,} · GLB {sz:.0f}KB"})
            else:  # depth
                with torch.no_grad():
                    pred = model.inference([arr], process_res=res, export_format="mini_npz")
                dt = time.time() - t0
                depth = np.asarray(pred.depth)[0]
                base_rgb = _rgb_uint8(pred, arr)
                return JSONResponse({
                    "mode": "depth",
                    "input_uri": to_data_uri_rgb(base_rgb),
                    "depth_uri": to_data_uri_bgr(colorize_depth(depth)),
                    "dt": dt,
                    "dmin": float(np.nanmin(depth)), "dmax": float(np.nanmax(depth)),
                    "shape": [int(depth.shape[1]), int(depth.shape[0])]})
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return JSONResponse(
            {"error": "GPU 显存不足（5090 与产线共享）。请调低 process_res 或稍后重试。"},
            status_code=507)
    except Exception as e:
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)


@app.get("/glb/{token}/{name}")
def serve_glb(token: str, name: str):
    """按 token 提供生成的 GLB（校验为 32 位 hex，仅允许 scene.glb，防目录穿越）。"""
    if len(token) != 32 or any(c not in "0123456789abcdef" for c in token) or name != "scene.glb":
        return JSONResponse({"error": "非法路径"}, status_code=400)
    p = GLB_DIR / token / "scene.glb"
    if not p.exists():
        return JSONResponse({"error": "产物已过期或不存在"}, status_code=404)
    return FileResponse(str(p), media_type="model/gltf-binary", filename="scene.glb")


# ── 电子秤：JSON 接口 + 实时看板页 ─────────────────────────────────────
@app.get("/api/weights")
def api_weights():
    """返回两个秤的最新缓存读数（后台线程每 0.4s 刷新）。"""
    with _scale_lock:
        scales = [{"id": s["id"], "name": s["name"], "host": s["host"],
                   "ok": _scale_latest[s["id"]]["ok"],
                   "weight": _scale_latest[s["id"]]["weight"],
                   "raw": _scale_latest[s["id"]]["raw"]} for s in SCALES]
    return JSONResponse({"scales": scales, "ts": time.time()})


WEIGHT_PAGE = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>电子秤实时重量</title>
<style>
 body{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;max-width:1100px;margin:32px auto;padding:0 16px;color:#1c1c1e;background:#f5f5f7}
 h1{font-size:22px} .sub{color:#6b6b70;font-size:14px;margin-bottom:24px}
 .nav{display:flex;gap:18px;margin-bottom:18px;font-size:14px;align-items:center}
 .nav a{padding:6px 14px;border-radius:980px;background:#fff;box-shadow:0 1px 3px rgba(0,0,0,.08);color:#0071e3;text-decoration:none}
 .nav a.active{background:#0071e3;color:#fff}
 .nav .home{margin-left:auto;box-shadow:none;background:transparent;color:#6b6b70}
 .grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}
 .card{background:#fff;border-radius:14px;padding:24px;box-shadow:0 1px 4px rgba(0,0,0,.08)}
 .head{display:flex;align-items:center;justify-content:space-between;margin-bottom:14px}
 .name{font-size:16px;font-weight:600;color:#1c1c1e}
 .dot{width:10px;height:10px;border-radius:50%;background:#ff3b30}
 .dot.on{background:#34c759}
 .weight{font-variant-numeric:tabular-nums;font-weight:700;font-size:64px;line-height:1;letter-spacing:-1px}
 .unit{font-size:22px;color:#8e8e93;margin-left:6px;font-weight:500}
 .meta{margin-top:12px;font-size:12px;color:#8e8e93;display:flex;justify-content:space-between}
 canvas{margin-top:14px;width:100%;height:56px;display:block}
 .off .weight{color:#c7c7cc}
 @media(max-width:720px){.grid{grid-template-columns:1fr}}
</style></head><body>
<div class="nav"><a href="/panel">深度 / 点云 / 网格</a><a class="active" href="/weight">电子秤实时重量</a><a class="home" href="/" target="_top">↗ 对比首页</a></div>
<h1>电子秤 · 实时重量</h1>
<div class="sub">数据每 0.4s 由服务端轮询 · 页面每 0.5s 刷新</div>
<div class="grid" id="grid"></div>
<script>
const HIST={};
function card(s){return `<div class="card" id="c_${s.id}">
 <div class="head"><span class="name">${s.name}</span><span class="dot" id="d_${s.id}"></span></div>
 <div><span class="weight" id="w_${s.id}">--</span><span class="unit">g</span></div>
 <canvas id="cv_${s.id}"></canvas>
 <div class="meta"><span id="ip_${s.id}"></span><span id="t_${s.id}"></span></div></div>`;}
function spark(id){const cv=document.getElementById('cv_'+id);if(!cv)return;
 const dpr=devicePixelRatio||1,w=cv.clientWidth,h=cv.clientHeight;cv.width=w*dpr;cv.height=h*dpr;
 const g=cv.getContext('2d');g.scale(dpr,dpr);g.clearRect(0,0,w,h);
 const d=HIST[id]||[];if(d.length<2)return;const mn=Math.min(...d),mx=Math.max(...d),r=(mx-mn)||1;
 g.beginPath();d.forEach((v,i)=>{const x=i/(d.length-1)*w,y=h-6-((v-mn)/r)*(h-12);i?g.lineTo(x,y):g.moveTo(x,y);});
 g.strokeStyle='#0071e3';g.lineWidth=2;g.stroke();}
async function tick(){try{
 const j=await (await fetch('/api/weights',{cache:'no-store'})).json();
 j.scales.forEach(s=>{const w=document.getElementById('w_'+s.id),d=document.getElementById('d_'+s.id),
  c=document.getElementById('c_'+s.id),t=document.getElementById('t_'+s.id),ip=document.getElementById('ip_'+s.id);
  if(s.weight!=null){w.textContent=s.weight.toFixed(1);(HIST[s.id]=HIST[s.id]||[]).push(s.weight);
   if(HIST[s.id].length>80)HIST[s.id].shift();spark(s.id);}
  d.className='dot'+(s.ok?' on':'');c.className='card'+(s.ok?'':' off');
  ip.textContent=s.host+'  raw='+(s.raw==null?'-':s.raw);
  t.textContent=s.ok?new Date().toLocaleTimeString('zh-CN'):'离线';});
}catch(e){}}
fetch('/api/weights').then(r=>r.json()).then(j=>{
 document.getElementById('grid').innerHTML=j.scales.map(card).join('');tick();setInterval(tick,500);});
</script></body></html>"""


@app.get("/weight", response_class=HTMLResponse)
def weight_page():
    return WEIGHT_PAGE


# ══════════════════════════════════════════════════════════════════════
# 兼容 shim：DA3 的 Gradio UI 按旧版 gradio 写，本机是 gradio 6.1.0，
# 丢弃已移除的装饰性 kwargs（如 Gallery 的 show_download_button），避免改上游源码。
# ══════════════════════════════════════════════════════════════════════
_GRADIO6_REMOVED_KWARGS = {"show_download_button", "show_share_button"}


def _shim_gradio_component(cls):
    _orig_init = cls.__init__

    def __init__(self, *args, **kwargs):
        for _k in _GRADIO6_REMOVED_KWARGS:
            kwargs.pop(_k, None)
        return _orig_init(self, *args, **kwargs)

    cls.__init__ = __init__


for _cls in (gr.Gallery, gr.Image, gr.Video, gr.Model3D):
    _shim_gradio_component(_cls)


# ══════════════════════════════════════════════════════════════════════
# 让官方 Gradio 复用同一个模型单例并串行化 GPU（避免进程内两份权重撑爆显存）
# ══════════════════════════════════════════════════════════════════════
import depth_anything_3.app.modules.model_inference as _mi_mod  # noqa: E402


def _shared_initialize_model(self, device="cuda"):
    """官方 UI 的模型初始化改为复用本服务的共享单例。"""
    self.model = get_model()


_orig_run_inference = _mi_mod.ModelInference.run_inference


def _locked_run_inference(self, *args, **kwargs):
    """官方 UI 的推理也走同一把 GPU 锁，与右栏串行，避免并发撞显存。"""
    with _gpu_lock:
        return _orig_run_inference(self, *args, **kwargs)


_mi_mod.ModelInference.initialize_model = _shared_initialize_model
_mi_mod.ModelInference.run_inference = _locked_run_inference


# ══════════════════════════════════════════════════════════════════════
# 挂载官方 Gradio UI 到同一 FastAPI（同端口，路径 /gradio）
# ══════════════════════════════════════════════════════════════════════
os.environ.setdefault("DA3_MODEL_DIR", MODEL_DIR)
_GRADIO_WORKSPACE = str(Path("/home/odyss/da3-web/workspace/gradio"))
_GRADIO_GALLERY = str(Path("/home/odyss/da3-web/workspace/gallery"))
Path(_GRADIO_WORKSPACE).mkdir(parents=True, exist_ok=True)
Path(_GRADIO_GALLERY).mkdir(parents=True, exist_ok=True)

_da3_gradio_app = DepthAnything3App(
    model_dir=MODEL_DIR, workspace_dir=_GRADIO_WORKSPACE, gallery_dir=_GRADIO_GALLERY)
_gradio_blocks = _da3_gradio_app.create_app()
_gradio_blocks.queue(max_size=20)
app = gr.mount_gradio_app(app, _gradio_blocks, path="/gradio")
