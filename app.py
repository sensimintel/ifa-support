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
import re
import shutil
import socket
import struct
import sys
import threading
import time
import urllib.request
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

from frame_relay import router as frame_router, set_processor  # noqa: E402

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


# ══════════════════════════════════════════════════════════════════════
# LocateAnything 检测：调 5090 本地 LocateAnything 网关（localhost:8000，OpenAI 兼容），
# 在帧上定位 food / drink，并把检测框内点云的 3D 包围盒画成彩色粗线框叠到点云 GLB 上。
#   · 输入务必用 DA3 处理后的 processed_images（与深度同一张图），检测框归一化坐标零错位。
#   · 坐标契约：响应 content 里 `<ref>label</ref><box><x1><y1><x2><y2></box>`，0-999 归一化，
#     未命中 `<box>None</box>`；多类以 `</c>` 分隔（逗号分隔在真实第一视角帧上会崩，勿用）。
# ══════════════════════════════════════════════════════════════════════
LOCATE_ENDPOINT = "http://127.0.0.1:8000/v1/chat/completions"
LOCATE_MODEL = "nvidia/LocateAnything-3B"
LOCATE_TARGETS = "food</c>drink"          # 两类语义：食物 / 饮品
LOCATE_TIMEOUT = 20.0                     # 单次检测超时（秒）；已挪到 GPU 锁外，不阻塞产线
LOCATE_COLORS = {"food": (222, 52, 52),   # food = 红
                 "drink": (46, 120, 235)}  # drink = 蓝
_LOCATE_RE = re.compile(r"<ref>(.*?)</ref>\s*<box>(.*?)</box>", re.S)
_LOCATE_INT_RE = re.compile(r"<(-?\d+)>")


def _locate_food_drink(rgb: np.ndarray):
    """调本地 LocateAnything 检测 food/drink。

    入参 rgb 为 uint8 RGB（应传 DA3 的 processed_images，与深度图同尺寸同视图，检测框零错位）。
    返回 [(label, nx1, ny1, nx2, ny2), ...]，坐标为相对图像宽高的归一化 0-1（左上、右下）。
    任何失败（网络/解析）都吞掉并返回 []，保证点云仍能出（只是没框）。"""
    try:
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            return []
        b64 = base64.b64encode(buf.tobytes()).decode()
        payload = {
            "model": LOCATE_MODEL,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text",
                 "text": f"Locate all the instances that matches the following description: {LOCATE_TARGETS}."},
            ]}],
            "max_tokens": 512, "temperature": 0.0,
        }
        req = urllib.request.Request(
            LOCATE_ENDPOINT, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=LOCATE_TIMEOUT) as r:
            data = json.loads(r.read().decode())
        content = data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[da3-web] LocateAnything 调用失败：{type(e).__name__}: {e}", flush=True)
        return []
    dets = []
    for label, box in _LOCATE_RE.findall(content):
        label = label.strip().lower()
        if "none" in box.lower():
            continue
        ints = [int(x) for x in _LOCATE_INT_RE.findall(box)]
        # 每 4 个整数一个框，兼容单/多实例
        for k in range(0, len(ints) - 3, 4):
            x1, y1, x2, y2 = ints[k:k + 4]
            nx1, ny1 = min(x1, x2) / 1000.0, min(y1, y2) / 1000.0
            nx2, ny2 = max(x1, x2) / 1000.0, max(y1, y2) / 1000.0
            if nx2 - nx1 < 0.01 or ny2 - ny1 < 0.01:  # 退化框丢弃
                continue
            dets.append((label, nx1, ny1, nx2, ny2))
    return dets


def _box_wireframe_meshes(lo, hi, color_rgb, radius):
    """用细圆柱拼一个 3D 轴对齐包围盒（AABB）的 12 条棱，返回 mesh 列表。

    比 1px 线段醒目（model-viewer 里线宽固定很细），演示时框更清晰。"""
    x0, y0, z0 = float(lo[0]), float(lo[1]), float(lo[2])
    x1, y1, z1 = float(hi[0]), float(hi[1]), float(hi[2])
    c = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], dtype=np.float64)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    col = np.array([color_rgb[0], color_rgb[1], color_rgb[2], 255], dtype=np.uint8)
    meshes = []
    for a, b in edges:
        p0, p1 = c[a], c[b]
        if np.linalg.norm(p1 - p0) < 1e-9:
            continue
        try:
            cyl = trimesh.creation.cylinder(radius=radius, segment=[p0, p1], sections=6)
        except Exception:
            continue
        cyl.visual.vertex_colors = np.tile(col, (len(cyl.vertices), 1))
        meshes.append(cyl)
    return meshes


def build_pointcloud_boxes_glb(pred, detections, out_path, conf_thresh_percentile=40.0,
                               num_max_points=800000, show_cameras=True):
    """自建单视图点云并叠加 food/drink 3D 检测框，导出 GLB。

    坐标系用「相机坐标系」（相机=原点/拍摄位置、光轴固定，仅 flip Y/Z 到 glTF 约定），
    不套 DA3 逐帧 pose、不做中位数居中——相机不动则坐标系帧间固定，点云不漂。
    点云与框来自同一套反投影，框严格与点云对齐。返回命中的 label 列表。"""
    from depth_anything_3.utils.export import glb as _glb  # 复用官方对齐/相机/降采样

    depth = np.asarray(pred.depth).astype(np.float32)       # (N,H,W)
    K = np.asarray(pred.intrinsics).astype(np.float64)      # (N,3,3)
    N, H, W = depth.shape
    ext = pred.extrinsics
    if ext is None:  # 单视图缺外参：相机置于原点（identity）
        ext = np.tile(np.eye(4, dtype=np.float64), (N, 1, 1))
    else:
        ext = np.asarray(ext).astype(np.float64)
    rgb = _rgb_uint8(pred, None)
    if rgb is None or rgb.shape[:2] != (H, W):
        rgb = np.full((H, W, 3), 180, np.uint8)

    i = 0  # DA3 单图推理，只处理第 0 帧
    d = depth[i]
    valid = np.isfinite(d) & (d > 0)
    if getattr(pred, "sky", None) is not None:
        valid &= ~np.asarray(pred.sky)[i].astype(bool)
    conf = pred.conf
    if conf is not None:
        conf = np.asarray(conf).astype(np.float32)
        conf_thr = _glb.get_conf_thresh(pred, None, 1.05,
                                        conf_thresh_percentile, 90.0)
        valid &= conf[i] >= conf_thr

    # 反投影到「相机坐标系」（相机=原点、光轴固定），不套 DA3 pose、不减中位数中心——
    # 坐标系帧间固定：相机物理不动 → 点云只随真实场景深度变，切断两个漂移源：
    #   (a) 中位数居中把局部变化放大成整帧平移；(c) 单图 pose 估计逐帧漂。
    # glTF 约定翻转 Y/Z（相机看 -Z、Y 向上）。残留 (b) 深度尺度呼吸后续再治。
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    pix = np.stack([us, vs, np.ones_like(us)], -1).reshape(-1, 3).astype(np.float64)
    K_inv = np.linalg.inv(K[i])
    rays = K_inv @ pix.T                                # (3, H*W)
    Xc = (rays * d.reshape(-1)[None, :]).T              # (H*W, 3) 相机坐标（原点=相机光心）
    Xa = Xc.copy()
    Xa[:, 1] *= -1.0                                    # flip Y（图像 y 向下 → glTF Y 向上）
    Xa[:, 2] *= -1.0                                    # flip Z（相机光轴 +Z → glTF 相机看 -Z）
    Xa_grid = Xa.reshape(H, W, 3)
    vmask = valid.reshape(-1)

    scene = trimesh.Scene()
    if scene.metadata is None:
        scene.metadata = {}
    # 相机线框的对齐矩阵：仅 flip Y/Z（相机固定在原点），与点云同坐标系
    A_cam = np.eye(4); A_cam[1, 1] = -1.0; A_cam[2, 2] = -1.0
    scene.metadata["hf_alignment"] = A_cam

    # 点云（降采样后加入场景）
    pc_pts = Xa[vmask].astype(np.float32)
    pc_cols = rgb.reshape(-1, 3)[vmask].astype(np.uint8)
    pc_pts, pc_cols = _glb._filter_and_downsample(pc_pts, pc_cols, int(num_max_points))
    # 裁离群点：个别深度估计异常的远点会把点云包围盒撑爆，导致 model-viewer 取景距离算成
    # 负/极小值、画面全黑（尤其自适应/近距视角）。按到中位数中心的距离去掉最远 ~1.5%，让包围盒紧致。
    if pc_pts.shape[0] > 200:
        c = np.median(pc_pts, axis=0)
        dist = np.linalg.norm(pc_pts - c, axis=1)
        keep = dist <= np.percentile(dist, 98.5)
        pc_pts, pc_cols = pc_pts[keep], pc_cols[keep]
    if pc_pts.shape[0] > 0:
        scene.add_geometry(trimesh.points.PointCloud(vertices=pc_pts, colors=pc_cols))
        # model-viewer 的 load 事件与 getDimensions/取景依赖场景里存在三角面 mesh——纯点云(POINTS)
        # + 相机线框(LINES) 无 mesh 时不触发 load、getDimensions 返回 0 → 画面全黑（无检测框那几帧）。
        # 加一个 1mm 极小三角面作 mesh 锚（几乎不可见），确保触发 load；包围盒仍含点云范围，取景不受影响。
        _anchor = trimesh.Trimesh(
            vertices=np.array([[0, 0, 0], [1e-3, 0, 0], [0, 1e-3, 0]], dtype=np.float32),
            faces=np.array([[0, 1, 2]]), process=False)
        scene.add_geometry(_anchor)

    scene_scale = _glb._estimate_scene_scale(pc_pts, fallback=1.0)
    radius = max(scene_scale * 0.004, 1e-4)            # 框线粗细随场景尺度

    # 逐检测框：取框内有效像素的对齐点，算 3D AABB（2%/98% 分位抗离群），画粗线框
    hit_labels = []
    for (label, nx1, ny1, nx2, ny2) in detections:
        u1, u2 = int(nx1 * W), int(np.ceil(nx2 * W))
        v1, v2 = int(ny1 * H), int(np.ceil(ny2 * H))
        u1, u2 = max(0, u1), min(W, u2)
        v1, v2 = max(0, v1), min(H, v2)
        if u2 <= u1 or v2 <= v1:
            continue
        sub_valid = valid[v1:v2, u1:u2].reshape(-1)
        sub_pts = Xa_grid[v1:v2, u1:u2].reshape(-1, 3)[sub_valid]
        if sub_pts.shape[0] < 20:        # 框内有效深度太少，无法定位 3D 盒
            continue
        lo = np.percentile(sub_pts, 2, axis=0)
        hi = np.percentile(sub_pts, 98, axis=0)
        hi = np.where(hi - lo < 1e-3, lo + 1e-3, hi)   # 防退化成面/线
        color = LOCATE_COLORS.get(label, (255, 200, 0))
        for m in _box_wireframe_meshes(lo, hi, color, radius):
            scene.add_geometry(m)
        hit_labels.append(label)

    # 相机线框（相机固定在原点：用 identity pose，配合 metadata 的 A_cam 只做 flip）
    if show_cameras:
        try:
            _ext_id = np.tile(np.eye(4, dtype=np.float64), (N, 1, 1))
            _glb._add_cameras_to_scene(
                scene=scene, K=K, ext_w2c=_ext_id,
                image_sizes=[(H, W)] * N, scale=scene_scale * 0.03)
        except Exception:
            pass

    scene.export(out_path)
    return hit_labels


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
# 顶层分栏首页：左 iframe=扩展面板(/panel，设备帧+DA3)，右 iframe=电子秤(/weight)
# ══════════════════════════════════════════════════════════════════════
SPLIT_PAGE = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DA3 · 设备实时帧 ＋ 电子秤</title>
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
  <span class="tag">左：设备实时帧 + DA3 产物（深度图 · 点云 · 网格，可调参转视角）　·　右：电子秤实时重量　·　同一 8060 端口</span></div>
 <div class="wrap">
  <div class="pane">
   <div class="bar"><span class="dot" style="background:#0a84ff"></span>设备实时帧 · DA3 产物
    <a href="/panel" target="_blank">单独打开 ↗</a></div>
   <iframe src="/panel" title="设备实时帧 + DA3 产物"></iframe>
  </div>
  <div class="pane">
   <div class="bar"><span class="dot" style="background:#34c759"></span>电子秤实时重量
    <a href="/weight" target="_blank">单独打开 ↗</a></div>
   <iframe src="/weight" title="电子秤实时重量"></iframe>
  </div>
 </div>
</body></html>"""


# ══════════════════════════════════════════════════════════════════════
# 扩展面板：调参 + 三种产物（深度图 / 点云GLB / 网格GLB），前端 fetch + model-viewer
# ══════════════════════════════════════════════════════════════════════
PANEL_PAGE = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DA3 扩展面板 · 设备实时帧</title>
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
 select{width:100%;font-size:14px;padding:8px;border:1px solid #d0d0d5;border-radius:8px;background:#fff}
 input[type=range]{width:100%}
 .rngval{font-variant-numeric:tabular-nums;color:#0071e3;font-weight:600}
 .glbopts{border-top:1px dashed #e0e0e5;margin-top:14px;padding-top:14px}
 .hint{font-size:12px;color:#8e8e93;margin-top:6px}
 .status{font-size:13px;color:#3a3a3c;margin-top:14px;padding-top:12px;border-top:1px solid #f0f0f2}
 .status b{font-variant-numeric:tabular-nums}
 .status .err{color:#c1121f}
 .status .dim{color:#8e8e93}
 .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
 .grid figure{margin:0}
 .box{width:100%;height:460px;background:#0b0d10;border-radius:12px;overflow:hidden;position:relative;display:flex;align-items:center;justify-content:center}
 .box img{max-width:100%;max-height:100%;display:block}
 .box model-viewer{width:100%;height:100%;--poster-color:transparent}
 .wait{color:#7a828c;font-size:14px}
 figcaption{font-size:13px;color:#6b6b70;margin-top:8px;text-align:center}
 figcaption .m{color:#8e8e93}
 @media(max-width:720px){.grid{grid-template-columns:1fr}}
</style></head><body>
<div class="nav"><a class="active" href="/panel">深度 / 点云 / 网格</a><a href="/weight">电子秤实时重量</a><a class="home" href="/" target="_top">↗ 对比首页</a></div>
<h1>Depth Anything 3 · 扩展面板</h1>
<div class="sub">实时展示设备帧：左 = 接收到的帧，右 = DA3 产物（按下方控件实时生成）。改产物类型 / 分辨率 / 调参，右框即时重算。模型：DA3NESTED-GIANT-LARGE-1.1</div>

<div class="card">
 <div class="row">
  <div class="fld"><label>产物类型 export_format</label>
   <select id="fmt">
    <option value="depth">深度图（彩色）</option>
    <option value="glb">点云 + food/drink 框（GLB · 可转视角）</option>
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
  <div class="hint">点云由深度反投影自建，并叠加 LocateAnything 检测的 food（红框）/drink（蓝框）3D 包围盒；网格由深度反投影自建三角面（conf 分位越高越干净，num_max_points 仅点云生效）。分辨率越高越细、越吃显存，OOM 时会提示调低。</div>
 </div>
 <div class="status" id="status">等待设备帧… 请让手机 App（或模拟脚本）向 /api/frame 发帧。</div>
</div>

<div class="card" id="camcard" style="display:none">
 <div class="row" style="align-items:center;margin-bottom:2px;gap:22px">
  <div class="fld" style="flex:0 0 auto"><label style="display:inline;margin:0"><input type="checkbox" id="crot"> 自动旋转（默认关）</label></div>
  <div class="fld" style="flex:0 0 auto"><label style="display:inline;margin:0"><input type="checkbox" id="cadapt"> 跨场景自适应取景（保留角度/FOV，中心·距离自动跟随点云）</label></div>
 </div>
 <div class="row">
  <div class="fld"><label>方位角 θ <span class="rngval" id="cthv">-17</span>°</label>
   <input type="range" id="cth" min="-180" max="180" step="1" value="-17"></div>
  <div class="fld"><label>俯仰角 φ <span class="rngval" id="cphv">40</span>°</label>
   <input type="range" id="cph" min="0" max="180" step="1" value="40"></div>
  <div class="fld"><label>距离 radius <span class="rngval" id="crdv">1.5</span>m</label>
   <input type="range" id="crd" min="0.3" max="10" step="0.1" value="1.5"></div>
  <div class="fld"><label>视场 FOV <span class="rngval" id="cfvv">45.5</span>°</label>
   <input type="range" id="cfv" min="10" max="60" step="0.5" value="45.5"></div>
 </div>
 <div class="hint">拖动右侧 3D 视图或调滑块调整视角；下方是当前相机的<b>实测量化参数</b>（绝对值，可直接作为默认视角）。调到满意后点「复制」把参数发给我。</div>
 <div class="status" style="border-top:1px solid #f0f0f2;margin-top:8px">当前相机：<code id="camnow" style="font-size:12px;word-break:break-all;color:#0071e3">—（先切到点云/网格产物并拖一下视图）</code></div>
 <button id="camcopy" style="margin-top:10px;padding:7px 16px;border:0;border-radius:8px;background:#0071e3;color:#fff;font-size:13px;cursor:pointer">复制当前视角参数</button>
</div>

<div class="grid">
 <figure>
  <div class="box"><img id="raw" style="display:none"><span class="wait" id="rawwait">等待设备帧…</span></div>
  <figcaption>接收到的设备帧 <span class="m" id="rawmeta"></span></figcaption>
 </figure>
 <figure>
  <div class="box">
   <img id="prodimg" style="display:none">
   <model-viewer id="mv" style="display:none" camera-controls touch-action="pan-y"
     camera-orbit="-17deg 40deg 1.5m" field-of-view="45.5deg" camera-target="-0.389m -0.623m 1.582m"
     min-camera-orbit="-Infinity 0deg 1%" max-camera-orbit="Infinity 180deg 2000%"
     min-field-of-view="10deg" max-field-of-view="60deg"
     interaction-prompt="none" shadow-intensity="0.3" exposure="1.35"></model-viewer>
   <span class="wait" id="prodwait">等待产物…</span>
  </div>
  <figcaption>DA3 产物 <span class="m" id="prodmeta"></span></figcaption>
 </figure>
</div>

<script>
const $=id=>document.getElementById(id);
$('pr').oninput=()=>$('prv').textContent=$('pr').value;
$('ct').oninput=()=>$('ctv').textContent=$('ct').value;
$('nmp').oninput=()=>$('nmv').textContent=(+$('nmp').value).toFixed(1);

function syncOpts(){const f=$('fmt').value;
 $('glbopts').style.display=(f==='depth')?'none':'block';
 $('nmpwrap').style.display=(f==='glb')?'block':'none';
 $('camwrap').style.display=(f==='glb')?'block':'none';
 $('camcard').style.display=(f==='depth')?'none':'block';}  // 相机视角调节仅对 model-viewer 产物

// ── 3D 视角调节（仅点云/网格）：滑块/鼠标调视角，锁定绝对视角，每帧 reload 强制拉回，杜绝漂移 ──
// 漂移根因：camera-orbit 是相对 camera-target(模型中心) 的球坐标；每帧新点云中心/尺度不同，
// model-viewer 载入新模型会 auto-frame 重新对准中心，故角度数值不变、看向的点变了 → 画面飘。
// 治法：① 锁定绝对 camera-target + camera-orbit(米) + fov；② 只在「用户手动交互」时更新锁定值
//（忽略 auto-frame 触发的 camera-change，避免把漂移固化）；③ 每帧 load 后强制 apply 锁定视角。
const mv=$('mv');
const camState={theta:-17,phi:40,radius:1.5,fov:45.5};  // 默认视角（角度/FOV=用户调定；radius 改绝对米，% 在 model-viewer 对点云换算不稳）
let locked=null, lastCam='';   // locked: {orbit,target,fov} 绝对值字符串；null=尚未锁定（用初始默认）
let interacting=false, interactTimer;   // 「用户正在调」标志：交互期间新帧 load 不拉回，避免打断
function markInteract(){interacting=true;clearTimeout(interactTimer);interactTimer=setTimeout(()=>{interacting=false;},600);}
let adaptive=false;   // 取景模式：false=固定视角（锁定绝对值），true=自适应（角度固定，中心/距离每帧自动）

function readNow(){  // 读当前实测相机（角度→deg，距离/中心→米，绝对值）
  const o=mv.getCameraOrbit(), t=mv.getCameraTarget(), f=mv.getFieldOfView();
  return {th:o.theta*180/Math.PI, ph:o.phi*180/Math.PI, r:o.radius,
          tx:t.x, ty:t.y, tz:t.z, fov:f};
}
function lockFromNow(){  // 把「当前实测视角」固化为锁定视角，并刷新只读框
  try{
    const n=readNow();
    // radius 用「实测绝对米」锁定（getCameraOrbit 在米模式下可靠，鼠标缩放也能锁）；
    // 异常(<=0)时回落滑块值。绝对米绕开了 model-viewer 对点云的 % 距离换算 bug。
    const rm=(n.r>0 ? n.r : camState.radius);
    locked={orbit:n.th.toFixed(2)+'deg '+n.ph.toFixed(2)+'deg '+rm.toFixed(3)+'m',
            target:n.tx.toFixed(4)+'m '+n.ty.toFixed(4)+'m '+n.tz.toFixed(4)+'m',
            fov:n.fov.toFixed(2)+'deg'};
    lastCam='camera-orbit="'+n.th.toFixed(1)+'deg '+n.ph.toFixed(1)+'deg '+rm.toFixed(2)+'m" '
           +'field-of-view="'+n.fov.toFixed(1)+'deg" '
           +'camera-target="'+n.tx.toFixed(3)+'m '+n.ty.toFixed(3)+'m '+n.tz.toFixed(3)+'m"';
    $('camnow').textContent=lastCam;
  }catch(e){}
}
let lastAdaptive=null;   // 上一次算出的自适应视角{orbit,target,fov}，供换 src 前预设 + dims 未就绪时沿用
function applyLocked(){  // 强制把相机拉回锁定视角（压制 auto-frame）
  if(!locked)return;
  mv.cameraTarget=locked.target; mv.cameraOrbit=locked.orbit; mv.fieldOfView=locked.fov;
  if(mv.jumpCameraToGoal)mv.jumpCameraToGoal();
}
function applyAdaptive(){  // 自适应：保留角度 θ/φ 与 FOV；每帧按 getDimensions 算距离、对准 getBoundingBoxCenter
  // 绝对米（getDimensions/getBoundingBoxCenter 可靠）；不用 %/auto（对点云会算负距离/黑）。
  const d=mv.getDimensions(), c=mv.getBoundingBoxCenter();
  const maxDim=Math.max(d.x, d.y, d.z);
  if(maxDim>0.05){  // 包围盒就绪：按新点云尺度/中心算并更新缓存；未就绪则沿用上次缓存（不抖不黑）
    const ADAPT_FILL=0.8;  // 取景充满度：1.0=恰好框住 maxDim；<1 拉近放大
    const dist=(maxDim / 2) / Math.tan(camState.fov * Math.PI / 360) * ADAPT_FILL;
    lastAdaptive={orbit:camState.theta+'deg '+camState.phi+'deg '+dist.toFixed(3)+'m',
                  target:c.x+'m '+c.y+'m '+c.z+'m', fov:camState.fov+'deg'};
  }
  if(!lastAdaptive)return;
  mv.cameraTarget=lastAdaptive.target; mv.cameraOrbit=lastAdaptive.orbit; mv.fieldOfView=lastAdaptive.fov;
  if(mv.jumpCameraToGoal)mv.jumpCameraToGoal();
  lastCam='camera-orbit="'+lastAdaptive.orbit+'" field-of-view="'+lastAdaptive.fov+'" '
         +'camera-target="'+lastAdaptive.target+'"  （自适应：每帧跟随点云中心/尺度）';
  $('camnow').textContent=lastCam;
}
function preApplyView(){  // 换新点云「之前」先把镜头摆到当前锁定/自适应视角（设成属性），
  // 使新模型加载的第一帧就在正确视角，消除「先闪默认视角再跳回」的抖动。
  const v = adaptive ? lastAdaptive : locked;
  if(!v)return;
  mv.setAttribute('camera-target', v.target);
  mv.setAttribute('camera-orbit', v.orbit);
  mv.setAttribute('field-of-view', v.fov);
}
function applyFromSliders(){  // 滑块档位 → 设相机
  if(adaptive){ applyAdaptive(); return; }   // 自适应：中心/距离自动，θ/φ/FOV 生效
  mv.cameraOrbit=camState.theta+'deg '+camState.phi+'deg '+camState.radius+'m';
  mv.fieldOfView=camState.fov+'deg';
  if(mv.jumpCameraToGoal)mv.jumpCameraToGoal();
  setTimeout(lockFromNow, 50); // 固定模式：读回米值锁定
}
[['cth','theta'],['cph','phi'],['crd','radius'],['cfv','fov']].forEach(([id,key])=>{
  $(id).addEventListener('input',()=>{markInteract();camState[key]=+$(id).value;$(id+'v').textContent=$(id).value;applyFromSliders();});
});
$('crot').addEventListener('change',()=>{$('crot').checked?mv.setAttribute('auto-rotate',''):mv.removeAttribute('auto-rotate');});
$('cadapt').addEventListener('change',()=>{  // 切换 固定视角 ↔ 自适应取景
  adaptive=$('cadapt').checked;
  $('crd').disabled=adaptive;                 // 自适应下距离自动，radius 滑块停用
  $('crdv').style.opacity=adaptive?0.4:1;
  markInteract();
  adaptive?applyAdaptive():applyFromSliders();
});
// 只在「用户手动拖动」时更新锁定视角；auto-frame/程序化触发的 camera-change 一律忽略（否则会追着漂移跑）
let camDebounce;
mv.addEventListener('camera-change',(e)=>{
  if(!(e.detail && e.detail.source==='user-interaction'))return;
  markInteract();
  clearTimeout(camDebounce); camDebounce=setTimeout(lockFromNow, 120);
});
// 每帧点云载入后：用户正在调则不打断；自适应每帧自动框住点云；固定模式强制拉回锁定视角
mv.addEventListener('load',()=>{
  if(interacting)return;
  if(adaptive){
    applyAdaptive();                         // 立即定位（dims 未就绪则用缓存，避免抖/黑）
    setTimeout(()=>{ if(!interacting&&adaptive)applyAdaptive(); }, 80);  // 待 dims 就绪后精调到新帧
    return;
  }
  locked ? applyLocked() : lockFromNow();
});
$('camcopy').addEventListener('click',()=>{
  if(!lastCam)return;
  if(navigator.clipboard)navigator.clipboard.writeText(lastCam).catch(()=>{});
  const b=$('camcopy');b.textContent='已复制 ✓';setTimeout(()=>b.textContent='复制当前视角参数',1200);
});

function currentConfig(){return {
  export_format:$('fmt').value,
  process_res:+$('pr').value,
  conf_thresh_percentile:+$('ct').value,
  num_max_points:Math.round(+$('nmp').value*1e6),
  show_cameras:$('cam').value
};}

let pushTimer=null;
function pushConfig(){
  clearTimeout(pushTimer);
  pushTimer=setTimeout(()=>{
    fetch('/api/frame/config',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify(currentConfig())}).catch(()=>{});
  },300);
}
// 控件任意改动 → 同步显隐 + 推配置（debounce）
['fmt','pr','ct','nmp','cam'].forEach(id=>{
  $(id).addEventListener('change',()=>{syncOpts();pushConfig();});
  $(id).addEventListener('input',pushConfig);
});
syncOpts();pushConfig();  // 首次把面板初值下发后端

let lastSeq=-1,lastProdKey='';
async function tick(){
 try{
  const s=await(await fetch('/api/frame/status',{cache:'no-store'})).json();
  // 左框：接收帧
  if(s.has_frame && s.seq!==lastSeq){lastSeq=s.seq;
    $('raw').src='/api/frame/latest?t='+s.seq;$('raw').style.display='block';$('rawwait').style.display='none';}
  $('rawmeta').textContent = s.has_frame ? ('帧 '+s.seq+(s.interval?(' · 间隔 '+s.interval.toFixed(1)+'s'):'')) : '';

  // 右框：DA3 产物（图片=深度图；模型=GLB）
  const prodKey=(s.product_kind||'')+':'+(s.product_url||'')+':'+s.product_seq;
  if(s.product_kind && prodKey!==lastProdKey){
    if(s.product_kind==='image'){
      lastProdKey=prodKey;
      $('prodimg').src='/api/frame/latest-product?t='+s.product_seq;
      $('prodimg').style.display='block';$('mv').style.display='none';$('prodwait').style.display='none';
    }else if(s.product_kind==='model' && s.product_url){
      // 点云 GLB 较大、加载/解析慢；上一个没加载完(mv.loaded=false)就别换 src，
      // 否则高帧率(如 1fps)下每帧新 GLB 不断打断加载→loaded 永远 false→右框黑。
      // 跳过时不更新 lastProdKey，下个轮询周期(500ms)再重试，届时多半已加载完。
      if(mv.loaded || !mv.getAttribute('src')){
        lastProdKey=prodKey;
        preApplyView();   // 换模型前先摆好视角，让新点云加载首帧就在锁定视角，消除抖动
        mv.src=s.product_url;
        mv.style.display='block';$('prodimg').style.display='none';$('prodwait').style.display='none';
      }
    }
  }
  // 产物元信息 + 处理状态
  let pm='';
  if(s.product_error){pm='<span class="err">'+s.product_error+'</span>';}
  else if(s.product_meta){const m=s.product_meta;
    pm=(m.label||'')+(m.dt?(' · '+m.dt.toFixed(2)+'s'):'')+(m.stat?(' · '+m.stat):'')
      +(m.shape?(' · '+m.shape[0]+'×'+m.shape[1]):'');}
  if(s.has_frame && s.product_seq<s.seq && !s.product_error) pm+=' <span class="dim">（处理中…）</span>';
  $('prodmeta').innerHTML=pm;

  // 顶部状态行
  if(!s.processor){$('status').innerHTML='<span class="err">未接入 DA3 模型（纯中继）。</span>';}
  else if(!s.has_frame){$('status').textContent='等待设备帧… 请让手机 App（或模拟脚本）向 /api/frame 发帧。';}
  else{$('status').innerHTML='接收帧 <b>'+s.seq+'</b>'+(s.interval?(' · 到达间隔 <b>'+s.interval.toFixed(1)+'s</b>'):'')
     +' · 产物帧 <b>'+s.product_seq+'</b>'+(s.product_error?' · <span class="err">'+s.product_error+'</span>':'');}
 }catch(e){/* 单次轮询失败忽略，下个周期重试 */}
}
setInterval(tick,500);tick();
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
        t0 = time.time()
        # GPU 推理放锁内串行化；点云构建 / LocateAnything 检测（CPU+网络）挪到锁外，短占锁。
        with _gpu_lock:  # 与官方 Gradio 共用同一模型，串行化 GPU 推理
            with torch.no_grad():
                pred = model.inference([arr], process_res=res, export_format="mini_npz")

        if export_format in ("glb", "mesh"):
            token = uuid.uuid4().hex
            outdir = GLB_DIR / token
            outdir.mkdir(parents=True, exist_ok=True)
            glb = outdir / "scene.glb"
            if export_format == "glb":
                # 点云叠 food/drink 检测框：LocateAnything 用原图检测（高分辨率、召回更好）。
                # DA3 预处理是等比缩放(upper_bound_resize)无裁剪，原图归一化坐标≈深度图，
                # 直接乘深度宽高映射，偏差 <2%(patch 对齐)可忽略。
                detections = _locate_food_drink(arr)
                labels = build_pointcloud_boxes_glb(
                    pred, detections, str(glb),
                    conf_thresh_percentile=float(conf_thresh_percentile),
                    num_max_points=int(num_max_points), show_cameras=show_cam)
                dt = time.time() - t0
                sz = glb.stat().st_size / 1024 if glb.exists() else 0
                depth = np.asarray(pred.depth)[0]
                n_food, n_drink = labels.count("food"), labels.count("drink")
                _prune_glb()
                return JSONResponse({
                    "mode": "glb", "glb_url": f"/glb/{token}/scene.glb", "dt": dt,
                    "label": f"点云 + food/drink 框（food×{n_food} · drink×{n_drink}）",
                    "stat": f"GLB {sz:.0f}KB",
                    "shape": [int(depth.shape[1]), int(depth.shape[0])]})
            else:  # mesh：由 prediction（含 intrinsics）自建网格
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


# ── 设备实时帧中继：接收 mobile 直发的帧 → 缓存展示 + DA3 深度处理 ──────────
def _da3_frame_processor(raw: bytes, config: dict) -> dict:
    """把收到的一帧按「面板控件配置」跑 DA3，产出：
      - export_format=depth → 彩色深度图（图片类产物，返回 JPEG 字节）
      - export_format=glb   → 点云 + food/drink 3D 检测框 + 相机线框 scene.glb（返回 url）
      - export_format=mesh  → 深度反投影自建网格 GLB（模型类产物，返回 url）
    与官方 Gradio 共用同一模型、同一把 GPU 锁串行化，避免撞显存。首帧到达才懒加载模型。
    glb 分支额外调 5090 本地 LocateAnything 检测 food/drink，框与点云同坐标系严格对齐。"""
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    except Exception as e:
        raise RuntimeError(f"读取图片失败：{e}")
    arr = np.array(img)

    fmt = str(config.get("export_format", "depth"))
    res = int(max(140, min(896, int(float(config.get("process_res", PROCESS_RES))))))
    conf = float(config.get("conf_thresh_percentile", 40.0))
    nmp = int(float(config.get("num_max_points", 800000)))
    show_cam = str(config.get("show_cameras", "1")) in ("1", "true", "True", "on", "显示")

    model = get_model()
    try:
        t0 = time.time()
        # GPU 推理放锁内串行化；点云构建 / LocateAnything 检测（CPU+网络）挪到锁外，
        # 避免检测网络往返长时间占着 GPU 锁、阻塞产线与其他产物。
        with _gpu_lock:  # 与官方 Gradio 共用同一模型，串行化 GPU 推理
            with torch.no_grad():
                pred = model.inference([arr], process_res=res, export_format="mini_npz")

        if fmt == "glb":
            token = uuid.uuid4().hex
            outdir = GLB_DIR / token
            outdir.mkdir(parents=True, exist_ok=True)
            glb = outdir / "scene.glb"
            # LocateAnything 用原图检测（高分辨率、召回更好）。DA3 预处理等比缩放无裁剪，
            # 原图归一化坐标≈深度图，直接乘深度宽高映射，偏差 <2%(patch 对齐)可忽略。
            detections = _locate_food_drink(arr)
            labels = build_pointcloud_boxes_glb(
                pred, detections, str(glb), conf_thresh_percentile=conf,
                num_max_points=nmp, show_cameras=show_cam)
            sz = glb.stat().st_size / 1024 if glb.exists() else 0
            _prune_glb()
            n_food, n_drink = labels.count("food"), labels.count("drink")
            return {"kind": "model", "url": f"/glb/{token}/scene.glb",
                    "meta": {"label": f"点云 + food/drink 框（food×{n_food} · drink×{n_drink}）",
                             "dt": time.time() - t0, "stat": f"GLB {sz:.0f}KB · res {res}"}}
        elif fmt == "mesh":
            token = uuid.uuid4().hex
            outdir = GLB_DIR / token
            outdir.mkdir(parents=True, exist_ok=True)
            glb = outdir / "scene.glb"
            nv, nf = build_mesh_glb(pred, str(glb), conf_thresh_percentile=conf)
            sz = glb.stat().st_size / 1024 if glb.exists() else 0
            _prune_glb()
            return {"kind": "model", "url": f"/glb/{token}/scene.glb",
                    "meta": {"label": "三角网格 mesh", "dt": time.time() - t0,
                             "stat": f"顶点 {nv:,} · 面 {nf:,} · GLB {sz:.0f}KB · res {res}"}}
        else:  # depth
            depth = np.asarray(pred.depth)[0]
            ok, buf = cv2.imencode(".jpg", colorize_depth(depth))
            if not ok:
                raise RuntimeError("深度图编码失败")
            return {"kind": "image", "bytes": buf.tobytes(), "content_type": "image/jpeg",
                    "meta": {"label": "彩色深度图（越亮=越近）", "dt": time.time() - t0,
                             "dmin": float(np.nanmin(depth)), "dmax": float(np.nanmax(depth)),
                             "shape": [int(depth.shape[1]), int(depth.shape[0])], "res": res}}
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise RuntimeError("GPU 显存不足（5090 与产线共享），请调低处理分辨率后重试")


app.include_router(frame_router)
set_processor(_da3_frame_processor)


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
