# -*- coding: utf-8 -*-
"""
Depth Anything 3 极简 Web 服务：浏览器上传一张图 → 返回彩色深度图。
- 启动时把 GIANT-LARGE 模型常驻加载到 GPU，之后每次请求只做推理。
- 纯服务端渲染，零前端依赖：GET / 出上传页，POST /infer 出结果页（内嵌 base64）。
- 绑定 0.0.0.0，局域网内可直接用 http://<5090局域网IP>:8060 访问。
"""
import base64
import io
import json
import socket
import struct
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image, ImageOps

# 把 DA3 源码目录加入 import 路径（本服务独立于 DA3 仓，只引用其 src）
DA3_ROOT = Path("/home/odyss/Depth-Anything-3")
sys.path.append(str(DA3_ROOT / "src"))
from depth_anything_3.api import DepthAnything3  # noqa: E402

MODEL_DIR = str(DA3_ROOT / "models" / "DA3NESTED-GIANT-LARGE-1.1")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROCESS_RES = 504  # DA3 默认处理分辨率

app = FastAPI(title="DA3 Depth Web")
_model = None


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
    """懒加载并缓存模型；首次调用会把权重搬到 GPU（约需十几到几十秒）。"""
    global _model
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


PAGE = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Depth Anything 3 · 深度图</title>
<style>
 body{{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;max-width:1100px;margin:32px auto;padding:0 16px;color:#1c1c1e;background:#f5f5f7}}
 h1{{font-size:22px}} .sub{{color:#6b6b70;font-size:14px;margin-bottom:24px}}
 .card{{background:#fff;border-radius:14px;padding:20px;box-shadow:0 1px 4px rgba(0,0,0,.08);margin-bottom:20px}}
 input[type=file]{{margin:8px 0}}
 button{{background:#0071e3;color:#fff;border:0;border-radius:980px;padding:10px 22px;font-size:15px;cursor:pointer}}
 button:disabled{{opacity:.5}}
 .grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
 .grid figure{{margin:0}} .grid img{{width:100%;border-radius:10px;display:block}}
 figcaption{{font-size:13px;color:#6b6b70;margin-top:6px;text-align:center}}
 .meta{{font-size:13px;color:#6b6b70;margin-top:8px}}
 a{{color:#0071e3;text-decoration:none}}
 @media(max-width:720px){{.grid{{grid-template-columns:1fr}}}}
 .nav{{display:flex;gap:18px;margin-bottom:18px;font-size:14px}}
 .nav a{{padding:6px 14px;border-radius:980px;background:#fff;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
 .nav a.active{{background:#0071e3;color:#fff}}
</style></head><body>
<div class="nav"><a class="active" href="/">深度图</a><a href="/weight">电子秤实时重量</a></div>
<h1>Depth Anything 3 · 单图深度估计</h1>
<div class="sub">上传一张图片，返回彩色深度图（越亮 = 越近）。模型：DA3NESTED-GIANT-LARGE-1.1</div>
<div class="card">
 <form action="/infer" method="post" enctype="multipart/form-data" onsubmit="this.querySelector('button').disabled=true;this.querySelector('button').innerText='推理中…约几秒';">
  <input type="file" name="file" accept="image/*" required><br>
  <button type="submit">生成深度图</button>
 </form>
</div>
{result}
</body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE.format(result="")


@app.post("/infer", response_class=HTMLResponse)
async def infer(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img).convert("RGB")  # 修正手机拍照方向
    except Exception as e:
        return PAGE.format(result=f'<div class="card">读取图片失败：{e}</div>')

    arr = np.array(img)
    model = get_model()
    t0 = time.time()
    with torch.no_grad():
        pred = model.inference([arr], process_res=PROCESS_RES, export_format="mini_npz")
    dt = time.time() - t0

    depth = np.asarray(pred.depth)[0]  # (N,H,W) 取第一帧
    # 用推理内部处理过的图做对比（与深度图同尺寸）；缺省则用原图
    if getattr(pred, "processed_images", None) is not None:
        base_rgb = np.asarray(pred.processed_images)[0]
        if base_rgb.dtype != np.uint8:
            base_rgb = np.clip(base_rgb * (255 if base_rgb.max() <= 1.0 else 1), 0, 255).astype(np.uint8)
    else:
        base_rgb = arr

    depth_color = colorize_depth(depth)
    dmin, dmax = float(np.nanmin(depth)), float(np.nanmax(depth))
    result = f"""<div class="card">
 <div class="grid">
  <figure><img src="{to_data_uri_rgb(base_rgb)}"><figcaption>输入图</figcaption></figure>
  <figure><img src="{to_data_uri_bgr(depth_color)}"><figcaption>深度图（越亮=越近）</figcaption></figure>
 </div>
 <div class="meta">推理耗时 {dt:.2f}s ｜ 深度范围 {dmin:.3f} ~ {dmax:.3f} ｜ 分辨率 {depth.shape[1]}×{depth.shape[0]} ｜ <a href="/">← 再传一张</a></div>
</div>"""
    return PAGE.format(result=result)


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
 .nav{display:flex;gap:18px;margin-bottom:18px;font-size:14px}
 .nav a{padding:6px 14px;border-radius:980px;background:#fff;box-shadow:0 1px 3px rgba(0,0,0,.08);color:#0071e3;text-decoration:none}
 .nav a.active{background:#0071e3;color:#fff}
 .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:20px}
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
<div class="nav"><a href="/">深度图</a><a class="active" href="/weight">电子秤实时重量</a></div>
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
