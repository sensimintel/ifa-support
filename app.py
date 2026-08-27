# -*- coding: utf-8 -*-
"""
Depth Anything 3 Web 服务：一个页面、一个 8060 端口，左右两栏对比。
- 右栏：自研扩展面板（/panel），可在网页上调参对比，支持三种产物：
    · 深度图      —— 彩色深度图（越亮=越近）
    · 点云+相机    —— DA3 导出 scene.glb（点云 + 相机线框），可 3D 转视角
    · 网格 mesh   —— 由深度反投影自建三角网格 GLB，可 3D 转视角
  可调参数：process_res、conf_thresh_percentile、num_max_points、show_cameras。
- 顶层 / 直接跳 /experience（主链路展示页）。
- /experience 是 IFA 浅体验区展示页：品牌化全屏实时识别 UI（实时点云背景 + 待机/识别态 + 流水视图）。
- 关键约束：5090 GPU 与产线服务共享，进程内只加载一份模型权重，并用 GPU 锁串行化推理。
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
import urllib.error
import urllib.request
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
import trimesh
from fastapi import Body, FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               RedirectResponse, Response, StreamingResponse)
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps


import devpc  # noqa: E402
import foodref  # noqa: E402
import recog_direct  # noqa: E402
import recog_prompt  # noqa: E402
import recog_log  # noqa: E402
import recog_match  # noqa: E402
import recog_sse  # noqa: E402
from frame_relay import (  # noqa: E402
    UNKNOWN_DEVICE, get_latest_frame, get_latest_frame_seq,
    get_selected_device, router as frame_router, set_pc_want_provider)


# GLB / mesh 产物落盘目录（每次推理一个子目录，超量自动清理）
GLB_DIR = Path("/home/odyss/da3-web/glb_out")
GLB_DIR.mkdir(parents=True, exist_ok=True)
# 最多保留最近多少次产物。三路产物（DA3/SAM3映射/SAM3高亮）高频推流时合计可达 3~6 个/秒，
# 客户端从拿到 url 到拉完 GLB 需数秒（远端浏览器还受 Wi-Fi 带宽限制），额度太小会让 token
# 在被拉取前就被清掉 → 前端 404、点云格子全部冻住（2026-08-13 真实发生）。150 个 ≈ 25~50s
# 寿命、磁盘峰值 ~350MB，5090 余量充足
GLB_KEEP = 150

app = FastAPI(title="DA3 Depth Web")
# 局域网演示服务：放开跨域，供 superadmin(18091) 等同网页面直调秤接口
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])
# 静态资源（/experience 用的品牌字体等）：仓内 static/ 目录随代码一起部署
STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


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
# 检测标签配色：food 红 / drink 蓝（SAM3 mask 染色、3D 框与 2D 框绘制共用）。
# ══════════════════════════════════════════════════════════════════════
LABEL_COLORS = {"food": (222, 52, 52),   # food = 红
                "drink": (46, 120, 235)}  # drink = 蓝


# ══════════════════════════════════════════════════════════════════════
# SAM3（6000pro / GCP g4-01 的 8001，自定义 REST、无鉴权；经隧道映射到 5090 本地端口）
#   · POST /v1/segment 单图分割：{image_b64, text} → instances[{obj_id,score,box_xywh_px,mask_rle}]
#     可选 {debug:true, topk:N} → 附带 debug{presence_logit/score, topk[联合分/条件原始分]}（调优页用）
#   · POST /v1/track   短视频跟踪：{frames_b64[], text, prompt_frame_index} → frames{帧idx:[objs]}
#   · mask 是 COCO 压缩 RLE，本文件自带解码（5090 无 pycocotools，不额外装依赖）
# ══════════════════════════════════════════════════════════════════════
SAM3_ENDPOINT = os.environ.get("SAM3_ENDPOINT", "http://127.0.0.1:8012").rstrip("/")
SAM3_TIMEOUT = float(os.environ.get("SAM3_TIMEOUT", "60"))
SAM3_TEXT_DEFAULT = os.environ.get("SAM3_TEXT", "food")
SAM3_TRACK_FRAMES = 5          # 跟踪用最近几帧

_recent_frames = []            # 最近若干帧 RGB（供 SAM3 track）
_recent_lock = threading.Lock()


def _push_recent_frame(rgb):
    """processor 每处理一帧就存一份，供 SAM3 track 取"过去 N 张图"。"""
    with _recent_lock:
        _recent_frames.append(rgb.copy())
        keep = SAM3_TRACK_FRAMES + 3
        if len(_recent_frames) > keep:
            del _recent_frames[:len(_recent_frames) - keep]


def _get_recent_frames(n):
    with _recent_lock:
        return list(_recent_frames[-n:])


def _rle_decode(size, counts):
    """COCO 压缩 RLE → (H,W) uint8 0/1（run-length 列优先）。"""
    h, w = int(size[0]), int(size[1])
    s = counts if isinstance(counts, str) else counts.decode()
    cnts, p = [], 0
    while p < len(s):
        x, k, more = 0, 0, 1
        while more:
            c = ord(s[p]) - 48
            x |= (c & 0x1f) << (5 * k)
            more = c & 0x20
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= (-1) << (5 * k)
        if len(cnts) > 2:
            x += cnts[-2]
        cnts.append(x)
    mask = np.zeros(h * w, np.uint8)
    idx, val = 0, 0
    for c in cnts:
        c = max(0, int(c))
        if idx >= h * w:
            break
        mask[idx:idx + c] = val
        idx += c
        val ^= 1
    return mask.reshape((h, w), order="F")


def _b64_jpg(rgb):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf.tobytes()).decode() if ok else None


def _sam3_post(path, payload):
    try:
        req = urllib.request.Request(SAM3_ENDPOINT + path, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=SAM3_TIMEOUT) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        print(f"[da3-web] SAM3 {path} 失败：{type(e).__name__}: {e}", flush=True)
        return None


def _sam3_segment_debug(rgb, text, topk=10, alpha=1.0, det_thresh=0.0):
    """单图分割（带 presence / top-K query 原始分）→ (instances, debug)；失败返回 ([], None)。
    debug 结构见 sam3_server：presence_logit/presence_score/num_queries/topk[{joint/cond}]。
    alpha/det_thresh 为「换阈值口径」旋钮：检测保留卡在 presence^α×cond > det_thresh 上。"""
    b64 = _b64_jpg(rgb)
    if not b64:
        return [], None
    r = _sam3_post("/v1/segment", {"image_b64": b64, "text": text,
                                   "debug": True, "topk": int(topk),
                                   "alpha": float(alpha), "det_thresh": float(det_thresh)})
    return ((r or {}).get("instances", []) or []), (r or {}).get("debug")


def _sam3_track(frames_rgb, text, prompt_frame_index=0):
    """短视频跟踪（按时间顺序的多帧）→ {帧idx(str): [objs]}；失败返回 {}。"""
    fb = [x for x in (_b64_jpg(f) for f in frames_rgb) if x]
    if not fb:
        return {}
    r = _sam3_post("/v1/track", {"frames_b64": fb, "text": text,
                                 "prompt_frame_index": int(prompt_frame_index)})
    return (r or {}).get("frames", {}) or {}


# ── SAM3 流式长记忆客户端（server 端滚动窗口 + 稳定公共 obj_id）───────────────
# 每个词一个常驻 session：每步只传当前 1 帧，窗口与身份注册表养在 server；
# obj_id 跨请求稳定（同一物体持续在场一直同 id）。窗口长度=显存/算力旋钮。
SAM3_STREAM_WINDOW = int(os.environ.get("SAM3_STREAM_WINDOW", "5"))
_sam3_stream_sessions = {}      # 词 -> session_id
_sam3_stream_lock = threading.Lock()


def _sam3_stream_start(word):
    r = _sam3_post("/v1/stream/start", {"text": word, "window": SAM3_STREAM_WINDOW})
    return (r or {}).get("session_id")


# ── 生产 SAM3 打分口径（控制面写、生产流式每帧读，实时生效）───────────────────
# keep = presence^α × cond > thresh；alpha=1 且 thresh=0 时完全等价模型默认行为。
# 之后新增的 SAM3 生产参数都收口到这份配置里。
_sam3_score_cfg = {"alpha": 1.0, "thresh": 0.0}
_sam3_score_lock = threading.Lock()
# 口径配置落盘持久化（gitignored 本地态）：重启/部署不丢，启动时回读
_SCORE_CFG_PATH = Path(__file__).resolve().parent / "sam3_score_cfg.json"
try:
    _sam3_score_cfg.update(json.loads(_SCORE_CFG_PATH.read_text()))
    print(f"[da3-web] 已回读生产 SAM3 口径：{_sam3_score_cfg}", flush=True)
except Exception:
    pass


def _get_score_cfg():
    with _sam3_score_lock:
        cfg = dict(_sam3_score_cfg)
    if not cfg.get("words"):   # 未配置时用默认词表（SAM3_CLOUD_TARGETS 在下文定义，运行期取）
        cfg["words"] = [{"word": q, "label": l} for (q, l) in SAM3_CLOUD_TARGETS]
    return cfg


def _sam3_stream_frame(word, rgb):
    """流式步进一帧。session 不存在/过期自动重建一次。
    每步随请求带上生产口径配置（presence α / 阈值，控制面可实时改）+ debug 捕获，
    返回 (instances, global_index, impl, debug)；流式不可用（老 server 等）返回 (None,)*4。"""
    b64 = _b64_jpg(rgb)
    if not b64:
        return None, None, None, None
    cfg = _get_score_cfg()
    with _sam3_stream_lock:
        sid = _sam3_stream_sessions.get(word)
    for _retry in range(2):
        if not sid:
            sid = _sam3_stream_start(word)
            if not sid:
                return None, None, None, None
            with _sam3_stream_lock:
                _sam3_stream_sessions[word] = sid
        r = _sam3_post("/v1/stream/frame", {"session_id": sid, "image_b64": b64,
                                            "debug": True, "topk": 10,
                                            "alpha": cfg["alpha"], "det_thresh": cfg["thresh"]})
        if r is not None:
            return (r.get("instances") or [], r.get("global_index"), r.get("impl"),
                    r.get("debug"))
        sid = None                    # 失败（session 被回收/服务重启）：重建一次再试
    return None, None, None, None


_SAM3_COLORS = [(222, 52, 52), (46, 120, 235), (26, 158, 95), (240, 150, 20),
                (160, 80, 220), (20, 180, 200), (230, 90, 160)]


def _draw_instances(rgb, instances, color=None, label_prefix="", alpha=0.45):
    """把 SAM3 实例画到图上：mask 半透明填色 + 轮廓 + box + [词名] #obj_id/score。返回新图(RGB)。
    color 指定则该组统一用它(多词时每个词一色)；否则按 obj_id 分配。多词可基于上次返回图叠加。"""
    out = rgb.copy()
    H, W = out.shape[:2]
    for i, ins in enumerate(instances or []):
        col = color if color is not None else _SAM3_COLORS[int(ins.get("obj_id", i)) % len(_SAM3_COLORS)]
        rle = ins.get("mask_rle") or {}
        if rle.get("counts") and rle.get("size"):
            try:
                m = _rle_decode(rle["size"], rle["counts"])
                if m.shape != (H, W):
                    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                sel = m.astype(bool)
                if sel.any():
                    ov = out.copy()
                    ov[sel] = col
                    out = cv2.addWeighted(ov, alpha, out, 1 - alpha, 0)
                    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(out, cs, -1, col, 2)
            except Exception as e:
                print(f"[da3-web] mask 解码失败：{type(e).__name__}: {e}", flush=True)
        box = ins.get("box_xywh_px")
        if box and len(box) == 4:
            x, y, bw, bh = [int(round(float(v))) for v in box]
            cv2.rectangle(out, (x, y), (x + bw, y + bh), col, 2)
            lbl = (label_prefix + " " if label_prefix else "") + "#%s %.2f" % (
                ins.get("obj_id", i), float(ins.get("score", 0)))
            cv2.putText(out, lbl, (x, max(14, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)
    return out


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
                               num_max_points=800000, show_cameras=True, mask_overlays=None,
                               hl_cfg=None, outlier_mad=12.0, bake_conf_alpha=False):
    """自建单视图点云并叠加 food/drink 3D 检测框，导出 GLB。

    坐标系用「相机坐标系」（相机=原点/拍摄位置、光轴固定，仅 flip Y/Z 到 glTF 约定），
    不套 DA3 逐帧 pose、不做中位数居中——相机不动则坐标系帧间固定，点云不漂。
    点云与框来自同一套反投影，框严格与点云对齐。返回命中的 label 列表。

    mask_overlays：SAM3 mask 映射，[(label, color_rgb, mask_bool(H,W)), ...]——mask 命中的点
    染成该词颜色，并按 mask 命中点算 3D AABB 画线框。同一函数/同一 pred 下点云几何与
    detections 链路完全一致，保证第二/三图点云长一个样。

    hl_cfg 非空=第四图「高亮点云」模式：mask 染色不走 0.5 混合，而是把 _apply_hl_styles
    的样式（染色/纯色/提亮/描边 + 背景压暗）直接写进顶点色，且不画 mask 的 AABB 线框
    （无框）——点云几何/降采样/裁剪与②③完全同链路，仅顶点颜色不同。"""
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
    if hl_cfg is not None:
        # 高亮模式：先按配置调底色（色相/饱和/明度，中性时跳过），再把样式化染色/纯色/
        # 提亮/描边 + 背景压暗直接写进顶点色（后续不画框）
        _gs = float(hl_cfg.get("sat", 1.0)); _gv = float(hl_cfg.get("val", 1.0))
        _gh = float(hl_cfg.get("hue", 0.0))
        if abs(_gs - 1.0) > 1e-3 or abs(_gv - 1.0) > 1e-3 or abs(_gh) > 0.5:
            _hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
            _hsv[..., 0] = np.mod(_hsv[..., 0] + _gh / 2.0, 180.0)  # OpenCV H 范围 [0,180)
            _hsv[..., 1] *= _gs
            _hsv[..., 2] *= _gv
            _h = np.mod(_hsv[..., 0], 180.0)
            _sv = np.clip(_hsv[..., 1:], 0, 255)
            _hsv = np.concatenate([_h[..., None], _sv], -1)
            rgb = cv2.cvtColor(_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        if mask_overlays:
            rgb, _ = _apply_hl_styles(rgb, mask_overlays, hl_cfg)
    elif mask_overlays:
        # SAM3 mask 命中的点染成该词颜色（半透明混合），把分割结果"染"进点云本体
        rgb = rgb.copy()
        for (_label, _col, _mk) in mask_overlays:
            rgb[_mk] = (rgb[_mk].astype(np.float32) * 0.5
                        + np.array(_col, np.float32) * 0.5).astype(np.uint8)

    i = 0  # DA3 单图推理，只处理第 0 帧
    d = depth[i]
    valid = np.isfinite(d) & (d > 0)
    if getattr(pred, "sky", None) is not None:
        valid &= ~np.asarray(pred.sky)[i].astype(bool)
    conf = pred.conf
    if conf is not None:
        conf = np.asarray(conf).astype(np.float32)
        # 绝对下限传 0（原为 1.05）：get_conf_thresh 内部 min(max(conf_thresh, p_low), p90)，
        # 写死 1.05 会把阈值钉在 1.05 上限、让分位滑块失效——远景像素 conf 天然 <1.05 被整片砍掉。
        # 传 0 后阈值 = min(p_low, p90)，分位滑块=0 即保留全部点（远景回来）。
        conf_thr = _glb.get_conf_thresh(pred, None, 0.0,
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

    # 点云（降采样后加入场景）。颜色统一带 alpha 通道：默认 255；bake_conf_alpha 时
    # 把逐点置信度（5/95 分位稳健归一）烘进 alpha [40,255]，供前端做「置信度→点大小/
    # 透明度」联动（_filter_and_downsample 纯花式索引，RGBA 原样透传）
    pc_pts = Xa[vmask].astype(np.float32)
    _rgb_flat = rgb.reshape(-1, 3)[vmask].astype(np.uint8)
    if bake_conf_alpha and conf is not None:
        _cf = conf[i].reshape(-1)[vmask].astype(np.float32)
        _lo = float(np.percentile(_cf, 5.0)); _hi = float(np.percentile(_cf, 95.0))
        _t = np.clip((_cf - _lo) / max(_hi - _lo, 1e-6), 0.0, 1.0)
        _alpha = (40.0 + _t * 215.0).astype(np.uint8)
    else:
        _alpha = np.full(_rgb_flat.shape[0], 255, np.uint8)
    pc_cols = np.concatenate([_rgb_flat, _alpha[:, None]], axis=1)
    pc_pts, pc_cols = _glb._filter_and_downsample(pc_pts, pc_cols, int(num_max_points))
    # 裁离群点：个别深度估计异常的远点会把点云包围盒撑爆，导致 model-viewer 取景距离算成
    # 负/极小值、画面全黑（尤其自适应/近距视角）。
    # 只挡「深度爆炸」的极端离群点，用深度(Z)方向的 MAD 稳健界（倍数可调）；不按到中心的
    # 欧氏距离切 98.5 分位——那会把连续的远景层当离群整片误删（远景消失+取景抖动的第二把刀）。
    if pc_pts.shape[0] > 200:
        zc = -pc_pts[:, 2]                                   # 相机看 -Z，深度值 = -z（正）
        med = float(np.median(zc))
        mad = float(np.median(np.abs(zc - med))) + 1e-6
        keep = zc <= med + float(outlier_mad) * mad          # 远端只砍极端离群，保留连续远景
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
        # 深度前景提取：2D 框边缘会漏进大量远处背景点（天花板/墙），若直接对全部框内点算
        # AABB，深度跨度被背景撑爆 → 立体框巨大且中心被背景带偏。food/drink 是画面近处的
        # 连续一簇，只保留近侧主簇（剔除远背景）再算盒子，框才贴合目标本身。
        _zc = -sub_pts[:, 2]                                   # 相机看 -Z，深度值 = -z（正）
        _z0 = float(np.percentile(_zc, 10))                   # 近侧参考深度（避开个别过近噪点）
        _fg = _zc <= max(_z0 * 1.7, _z0 + 0.08)               # 保留近侧主簇：同物体深度跨度有限
        if int(_fg.sum()) >= 20:                              # 前景点够则用之；太少退回全体避免空框
            sub_pts = sub_pts[_fg]
        lo = np.percentile(sub_pts, 2, axis=0)
        hi = np.percentile(sub_pts, 98, axis=0)
        hi = np.where(hi - lo < 1e-3, lo + 1e-3, hi)   # 防退化成面/线
        color = LABEL_COLORS.get(label, (255, 200, 0))
        for m in _box_wireframe_meshes(lo, hi, color, radius):
            scene.add_geometry(m)
        hit_labels.append(label)

    # 逐 SAM3 mask：mask 命中的有效点就是目标本体（无需前景启发式），2%/98% 分位 AABB 画线框；
    # 高亮模式（hl_cfg 非空）不画任何框
    for (label, col, mk) in ([] if hl_cfg is not None else (mask_overlays or [])):
        sel = mk & valid
        sub_pts = Xa_grid[sel]
        if sub_pts.shape[0] < 20:
            continue
        lo = np.percentile(sub_pts, 2, axis=0)
        hi = np.percentile(sub_pts, 98, axis=0)
        hi = np.where(hi - lo < 1e-3, lo + 1e-3, hi)
        for m in _box_wireframe_meshes(lo, hi, tuple(col), radius):
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


# GLB 产物 POSITION 量化回退开关（KHR_mesh_quantization）：默认开——点坐标用 uint16
# 归一化存储（12B/点 → 6B/点，产物省约 1/3 传输量），node 挂 scale/translation 还原。
# 前端 model-viewer 若出现异常（点云消失/取景错乱/尺度不对等），在 .env 置 GLB_QUANTIZE=0
# 重启即回退 float32 存储，无需改代码。
GLB_QUANTIZE = os.environ.get("GLB_QUANTIZE", "1") not in ("0", "false", "False")


def _write_pointcloud_glb(out_path, pc_pts, pc_cols, cam_wires=None, quantize=False):
    """手写二进制 glTF(GLB) 导出器：替代 trimesh Scene 组装 + export（两项合计 ~43ms/份，
    是构建耗时的最大单项；手写序列化只做必要的字节拼接，毫秒级）。

    产物结构与 trimesh 版对齐：
      · 点云：1 个 POINTS primitive（POSITION VEC3 + COLOR_0 uint8 VEC4 normalized）
      · 1mm 锚三角面：model-viewer 的 load 事件/取景依赖场景里存在三角面 mesh，必须保留
      · 相机线框：LINES primitive（逐顶点颜色），cam_wires=[(segs(M,2,3), rgb), ...]，
        None/空则不写
    quantize=True 时点云 POSITION 用 uint16 归一化存储（KHR_mesh_quantization），
    node 挂 scale/translation 还原坐标（量化误差 ≤ 包围盒边长/65535/2，场景尺度下亚毫米）；
    锚三角/相机线框保持 float32（体量可忽略，不值得引入额外还原节点）。

    GLB 布局：12B header + JSON chunk（4 对齐，空格补齐）+ BIN chunk（4 对齐，零补齐）。"""
    def _pad4(b, fill=b"\x00"):
        return b + fill * (-len(b) % 4)

    bin_parts, views, accessors = [], [], []
    _off = [0]

    def _add_view(data):
        # 追加一个 4 对齐的 bufferView，返回其下标（对齐保证后续 accessor 偏移合法）
        padded = _pad4(data)
        bin_parts.append(padded)
        views.append({"buffer": 0, "byteOffset": _off[0], "byteLength": len(data)})
        _off[0] += len(padded)
        return len(views) - 1

    meshes, nodes = [], []
    n = int(pc_pts.shape[0])
    node_pc = None
    if n > 0:
        pts = np.ascontiguousarray(pc_pts, dtype=np.float32)
        lo = pts.min(axis=0); hi = pts.max(axis=0)
        node_pc = {"mesh": 0}
        if quantize:
            # uint16 归一化量化：存 [0,65535]，accessor.normalized 解码回 [0,1]，再由
            # node 的 scale=包围盒边长 / translation=包围盒角点 还原到原坐标
            span = np.maximum(hi - lo, np.float32(1e-6))
            q = np.clip(np.rint((pts - lo) / span * 65535.0),
                        0, 65535).astype(np.uint16)
            vi = _add_view(q.tobytes())
            accessors.append({"bufferView": vi, "componentType": 5123, "count": n,
                              "type": "VEC3", "normalized": True,
                              # 规范要求 min/max 写存储原值（归一化前的整数）
                              "min": [int(x) for x in q.min(axis=0)],
                              "max": [int(x) for x in q.max(axis=0)]})
            node_pc["scale"] = [float(x) for x in span]
            node_pc["translation"] = [float(x) for x in lo]
        else:
            vi = _add_view(pts.tobytes())
            accessors.append({"bufferView": vi, "componentType": 5126, "count": n,
                              "type": "VEC3",
                              "min": [float(x) for x in lo],
                              "max": [float(x) for x in hi]})
        vi = _add_view(np.ascontiguousarray(pc_cols, dtype=np.uint8).tobytes())
        accessors.append({"bufferView": vi, "componentType": 5121, "count": n,
                          "type": "VEC4", "normalized": True})
        meshes.append({"primitives": [{"attributes": {"POSITION": 0, "COLOR_0": 1},
                                       "mode": 0}]})           # mode 0 = POINTS
        nodes.append(node_pc)
        # 1mm 三角面 mesh 锚：确保 model-viewer 触发 load（理由同单目链路）。
        # 独立 node、不挂量化还原变换——锚三角始终 float32
        anchor = np.array([[0, 0, 0], [1e-3, 0, 0], [0, 1e-3, 0]], dtype=np.float32)
        vi = _add_view(anchor.tobytes())
        accessors.append({"bufferView": vi, "componentType": 5126, "count": 3,
                          "type": "VEC3", "min": [0.0, 0.0, 0.0],
                          "max": [1e-3, 1e-3, 0.0]})
        meshes.append({"primitives": [{"attributes": {"POSITION": len(accessors) - 1},
                                       "mode": 4}]})           # mode 4 = TRIANGLES
        nodes.append({"mesh": len(meshes) - 1})
    if cam_wires:
        # 相机线框合并成一个 LINES primitive（非索引、顶点两两成段），逐顶点烘相机颜色
        seg_pts = np.concatenate([np.asarray(s, np.float32).reshape(-1, 3)
                                  for (s, _c) in cam_wires])
        seg_cols = np.concatenate(
            [np.tile(np.array([c[0], c[1], c[2], 255], np.uint8),
                     (np.asarray(s).reshape(-1, 3).shape[0], 1))
             for (s, c) in cam_wires])
        vi = _add_view(np.ascontiguousarray(seg_pts).tobytes())
        accessors.append({"bufferView": vi, "componentType": 5126,
                          "count": int(seg_pts.shape[0]), "type": "VEC3",
                          "min": [float(x) for x in seg_pts.min(axis=0)],
                          "max": [float(x) for x in seg_pts.max(axis=0)]})
        pos_ai = len(accessors) - 1
        vi = _add_view(np.ascontiguousarray(seg_cols).tobytes())
        accessors.append({"bufferView": vi, "componentType": 5121,
                          "count": int(seg_cols.shape[0]), "type": "VEC4",
                          "normalized": True})
        meshes.append({"primitives": [{"attributes": {"POSITION": pos_ai,
                                                      "COLOR_0": len(accessors) - 1},
                                       "mode": 1}]})           # mode 1 = LINES
        nodes.append({"mesh": len(meshes) - 1})

    bin_chunk = b"".join(bin_parts)
    j = {"asset": {"version": "2.0", "generator": "da3-web"},
         "scene": 0, "scenes": [{"nodes": list(range(len(nodes)))}],
         "nodes": nodes, "meshes": meshes,
         "bufferViews": views, "accessors": accessors,
         "buffers": [{"byteLength": len(bin_chunk)}]}
    if quantize and node_pc is not None and "scale" in node_pc:
        j["extensionsUsed"] = ["KHR_mesh_quantization"]
        j["extensionsRequired"] = ["KHR_mesh_quantization"]
    jb = _pad4(json.dumps(j, separators=(",", ":")).encode("utf-8"), b" ")
    bin_padded = _pad4(bin_chunk)
    total = 12 + 8 + len(jb) + 8 + len(bin_padded)
    with open(out_path, "wb") as f:
        f.write(b"glTF" + struct.pack("<II", 2, total)
                + struct.pack("<I", len(jb)) + b"JSON" + jb
                + struct.pack("<I", len(bin_padded)) + b"BIN\x00" + bin_padded)


def _hex_rgb(s, fallback=(255, 159, 10)):
    """'#rrggbb' → (r,g,b)；解析失败回退默认橙色。"""
    try:
        s = str(s).lstrip("#")
        return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
    except Exception:
        return fallback


def _apply_hl_styles(rgb, overlays, cfg):
    """第四图高亮：按配置把 SAM3 mask 命中的像素做染色/纯色/提亮/描边，背景可整体压暗。
    只改像素颜色、不画任何框/文字。返回 (处理后的 rgb, 高亮布尔 mask(H,W))。"""
    rgb = rgb.copy()
    style = str(cfg.get("style", "tint"))
    strength = min(100.0, max(0.0, float(cfg.get("strength", 65)))) / 100.0
    dim = min(90.0, max(0.0, float(cfg.get("dim", 40)))) / 100.0
    custom = (_hex_rgb(cfg.get("color", "#ff9f0a"))
              if str(cfg.get("color_mode", "auto")) == "custom" else None)
    hmask = np.zeros(rgb.shape[:2], bool)
    for (_l, _c, mk) in overlays:
        hmask |= mk
    if dim > 0 and hmask.any():
        # 背景压暗：非目标点整体调暗，聚光灯式突出目标（无目标时不压，避免整图变黑）
        bg = ~hmask
        rgb[bg] = (rgb[bg].astype(np.float32) * (1.0 - dim)).astype(np.uint8)
    for (_l, col0, mk) in overlays:
        col = np.array(custom if custom is not None else col0, np.float32)
        if style == "solid":
            rgb[mk] = col.astype(np.uint8)
        elif style == "glow":
            # 提亮：保留物体自身纹理，只抬饱和度/明度（自发光感）；颜色选择不生效
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[..., 1][mk] *= (1.0 + 0.6 * strength)
            hsv[..., 2][mk] *= (1.0 + 0.9 * strength)
            rgb = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
        elif style == "outline":
            # 描边：只染 mask 边缘一圈（腐蚀求内边界），内部保持原色
            u8 = mk.astype(np.uint8)
            edge = (u8 - cv2.erode(u8, np.ones((5, 5), np.uint8))).astype(bool)
            rgb[edge] = col.astype(np.uint8)
        else:  # tint 半透明染色：原色与高亮色按强度混合
            rgb[mk] = (rgb[mk].astype(np.float32) * (1.0 - strength)
                       + col * strength).astype(np.uint8)
    return rgb, hmask


def _render_pointcloud_image(pred, detections=None, conf_thresh_percentile=40.0,
                             view_tilt=10.0, view_zoom=1.25, splat=2, out_size=760,
                             mask_overlays=None, eye_lift=0.0, eye_back=0.0,
                             color_grade=None, hl_cfg=None, aspect=None):
    """方案A·服务端渲染：5090 就地把点云用 torch(GPU) 投影 + 画家算法 z-buffer + splat 渲成 2D 图，
    跳过 GLB 序列化与前端 model-viewer 全量加载。叠 food/drink 框。返回 RGB uint8；点云为空返回 None。

    视角=复刻②③前端 model-viewer 的「调优视角」：相机对准光轴上的点云中心(0,0,cz)、
    距离=cz×view_zoom（1.25≈前端 VIEW_K）、绕点云中心俯视 view_tilt 度、FOV=真实相机内参。
    所有量都按场景深度 cz 等比缩放（场景相对）——单目逐帧深度的尺度抖动被归一化，
    帧间视角不跳；同时保证本渲染与②③ GLB 的取景一致（除染色/高亮外画面一致）。

    mask_overlays：SAM3 mask 映射，[(label, color_rgb, mask_bool(H0,W0)), ...]——mask 命中的点
    在投影前染成该词颜色，并按 mask 命中点算 3D AABB 画框+词名（与 detections 的 2D 框链路互不影响）。
    eye_lift/eye_back：在调优视角基础上的附加抬升/后撤，单位=场景深度比例（×cz），默认 0。"""
    import math
    depth = np.asarray(pred.depth)[0].astype(np.float32)
    H0, W0 = depth.shape
    K = np.asarray(pred.intrinsics)[0].astype(np.float32)
    rgb = _rgb_uint8(pred, None)
    if rgb is None or rgb.shape[:2] != (H0, W0):
        rgb = np.full((H0, W0, 3), 180, np.uint8)
    valid = np.isfinite(depth) & (depth > 0)
    if getattr(pred, "sky", None) is not None:
        valid &= ~np.asarray(pred.sky)[0].astype(bool)
    if pred.conf is not None:
        from depth_anything_3.utils.export import glb as _glb
        conf = np.asarray(pred.conf).astype(np.float32)
        thr = _glb.get_conf_thresh(pred, None, 0.0, conf_thresh_percentile, 90.0)
        valid &= conf[0] >= thr
    if color_grade is not None:
        # 点色调色 (饱和系数, 明度系数)：对齐 model-viewer 光照/色调映射的暗调点云观感；
        # 只作用于点颜色，框/文字等 overlay 仍用原亮色
        _gs, _gv = color_grade
        _hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        _hsv[..., 1] *= _gs
        _hsv[..., 2] *= _gv
        rgb = cv2.cvtColor(np.clip(_hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
    hl_flat = None
    if mask_overlays and hl_cfg is not None:
        # 第四图高亮模式：样式化染色/提亮/描边 + 背景压暗，后续不画框；记录高亮点位掩码供二次 splat
        rgb, _hm = _apply_hl_styles(rgb, mask_overlays, hl_cfg)
        hl_flat = _hm.reshape(-1)
    elif mask_overlays:
        # SAM3 mask 命中的像素在投影前染成该词颜色（半透明混合），把分割结果"染"进点云本体
        rgb = rgb.copy()
        for (_label, col, mk) in mask_overlays:
            rgb[mk] = (rgb[mk].astype(np.float32) * 0.5
                       + np.array(col, np.float32) * 0.5).astype(np.uint8)
    # 反投影到相机坐标系（z>0 前方、y 向下）
    us, vs = np.meshgrid(np.arange(W0), np.arange(H0))
    pix = np.stack([us, vs, np.ones_like(us)], -1).reshape(-1, 3).astype(np.float32)
    Xc_all = ((np.linalg.inv(K) @ pix.T) * depth.reshape(-1)[None, :]).T   # (HW,3)
    m = valid.reshape(-1)
    Xc = Xc_all[m]
    cols = rgb.reshape(-1, 3)[m]
    if hl_flat is not None:
        hl_flat = hl_flat[m]
    if Xc.shape[0] < 50:
        return None
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        P = torch.from_numpy(np.ascontiguousarray(Xc)).to(dev).float()
        C = torch.from_numpy(np.ascontiguousarray(cols)).to(dev)
        # 调优视角（同②③前端）：对准光轴上的点云中心、按场景深度定距、绕中心俯视 tilt。
        # cz 必须用整体 min/max 包围盒中心（同 model-viewer getBoundingBoxCenter 语义）——
        # 远端天花板/远景会把包围盒撑大、相机随之拉远，③ 的"拉远小景"观感正来自这里；
        # 用分位数抗离群会把远点滤掉、相机贴近变满屏（已踩过）。一切偏移×cz → 尺度抖动天然抵消。
        tilt = math.radians(view_tilt)
        zs = Xc[:, 2]
        cz = max(0.2, (float(zs.min()) + float(zs.max())) / 2.0)
        fwd = torch.tensor([0.0, math.sin(tilt), math.cos(tilt)], device=dev)   # y 向下为正 → 视线向下俯视
        target = torch.tensor([0.0, 0.0, cz], device=dev)
        eye = target - fwd * (cz * float(view_zoom)) \
            + torch.tensor([0.0, -float(eye_lift) * cz, -float(eye_back) * cz], device=dev)
        right = torch.tensor([1.0, 0.0, 0.0], device=dev)
        up = torch.cross(right, fwd)                             # 相机"上"(世界 -y 方向)
        Rm = torch.stack([right, up, fwd], 0)                    # 行基：世界 → 相机(x右 y上 z前)

        def project(pts):
            Pv = (pts - eye) @ Rm.T
            z = Pv[:, 2].clamp(min=1e-3)
            u = f * Pv[:, 0] / z + Wout / 2.0
            v = -f * Pv[:, 1] / z + Hout / 2.0                    # y上 → 图像 v 下
            return u, v, Pv[:, 2]

        Hout = int(out_size)
        # aspect：输出画幅宽高比。缺省=照片比例；全屏展示传 16/9（同 model-viewer 全屏画布，
        # 垂直 FOV 不变、横向多出黑边——前端 cover 铺满时才不会因裁切放大而丢失取景）
        Wout = max(1, int(round(out_size * (float(aspect) if aspect else W0 / H0))))
        real_fov = 2 * math.atan(H0 / (2 * K[1, 1]))             # 真实相机垂直 fov
        fov = min(math.radians(130.0), real_fov)                 # FOV=真实内参（同前端）；远近由 view_zoom 距离控制
        f = (Hout / 2.0) / math.tan(fov / 2.0)
        u, v, zc = project(P)
        inb = (zc > 1e-3) & (u >= 0) & (u < Wout) & (v >= 0) & (v < Hout)
        u = u[inb].long(); v = v[inb].long(); zc = zc[inb]; Cin = C[inb]
        order = torch.argsort(zc, descending=True)               # 远→近：近点后写覆盖远点(z-buffer)
        u = u[order]; v = v[order]; Cin = Cin[order]
        img = torch.zeros(Hout, Wout, 3, dtype=torch.uint8, device=dev)
        for dy in range(splat):                                  # splat：每点画 splat×splat 像素，稠密些
            for dx in range(splat):
                vv = (v + dy).clamp(0, Hout - 1)
                uu = (u + dx).clamp(0, Wout - 1)
                img[vv, uu] = Cin
        if hl_flat is not None:
            # 高亮点二次 splat：按面板"点大小"倍数居中放大、后画覆盖 → 目标点比背景点更大更醒目
            Hf = torch.from_numpy(np.ascontiguousarray(hl_flat)).to(dev)[inb][order]
            k = max(1, int(round(splat * float(hl_cfg.get("point_scale", 2.0)))))
            if k > splat and bool(Hf.any()):
                uh, vh, Ch = u[Hf], v[Hf], Cin[Hf]
                off = (k - splat) // 2
                for dy in range(k):
                    for dx in range(k):
                        vv = (vh + dy - off).clamp(0, Hout - 1)
                        uu = (uh + dx - off).clamp(0, Wout - 1)
                        img[vv, uu] = Ch
        out = img.cpu().numpy()
    # 叠 food/drink 框：框内近侧主簇的 3D AABB → 8 角投影 → 画线
    if detections:
        for (label, nx1, ny1, nx2, ny2) in detections:
            u1, u2 = int(nx1 * W0), int(np.ceil(nx2 * W0))
            v1, v2 = int(ny1 * H0), int(np.ceil(ny2 * H0))
            u1, u2 = max(0, u1), min(W0, u2); v1, v2 = max(0, v1), min(H0, v2)
            if u2 <= u1 or v2 <= v1:
                continue
            sub_m = valid[v1:v2, u1:u2].reshape(-1)
            sub = Xc_all.reshape(H0, W0, 3)[v1:v2, u1:u2].reshape(-1, 3)[sub_m]
            if sub.shape[0] < 20:
                continue
            zc2 = sub[:, 2]
            z0 = np.percentile(zc2, 10)
            fg = zc2 <= max(z0 * 1.7, z0 + 0.08)                 # 近侧前景，剔除框内远背景
            if int(fg.sum()) >= 20:
                sub = sub[fg]
            lo = np.percentile(sub, 2, axis=0); hi = np.percentile(sub, 98, axis=0)
            corners = np.array([[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]], [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
                                [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]], [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]]], np.float32)
            with torch.no_grad():
                cu, cv, cz3 = project(torch.from_numpy(corners).to(dev).float())
            cu = cu.cpu().numpy(); cv = cv.cpu().numpy(); cz3 = cz3.cpu().numpy()
            color = (222, 52, 52) if label == "food" else (46, 120, 235)
            edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
            # 模拟②③ GLB 粗管线框的观感：线宽=管半径的透视投影（近粗远细），管半径随框体量走
            tube_r = max(0.006, float(np.linalg.norm(hi - lo)) * 0.018)
            for a, b in edges:
                zm = max(1e-3, (cz3[a] + cz3[b]) / 2.0)
                w = int(np.clip(f * 2.0 * tube_r / zm, 2, 14))
                cv2.line(out, (int(cu[a]), int(cv[a])), (int(cu[b]), int(cv[b])), color, w, cv2.LINE_AA)
    # 叠 SAM3 mask 映射框：mask 命中的有效点直接就是目标本体（无需近侧前景启发式），
    # 2%/98% 分位算 3D AABB → 8 角投影 → 画线 + 词名。高亮模式（第四图）不画框。
    if mask_overlays and hl_cfg is None:
        for (label, col, mk) in mask_overlays:
            sel = (mk & valid).reshape(-1)
            sub = Xc_all[sel]
            if sub.shape[0] < 20:
                continue
            lo = np.percentile(sub, 2, axis=0); hi = np.percentile(sub, 98, axis=0)
            corners = np.array([[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]], [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
                                [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]], [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]]], np.float32)
            with torch.no_grad():
                cu, cv, _cz = project(torch.from_numpy(corners).to(dev).float())
            cu = cu.cpu().numpy(); cv = cv.cpu().numpy()
            edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
            for a, b in edges:
                cv2.line(out, (int(cu[a]), int(cv[a])), (int(cu[b]), int(cv[b])), tuple(col), 2, cv2.LINE_AA)
            cv2.putText(out, str(label), (max(0, int(cu.min())), max(14, int(cv.min()) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, tuple(col), 2, cv2.LINE_AA)
    return out


# ══════════════════════════════════════════════════════════════════════
# 第三图：DA3 → SAM3 → 点云映射。processor 每帧尝试触发（后台单任务、忙时跳过本帧，不阻塞产线）：
# 取「当前帧 + 过去 N 帧」跑 SAM3 /v1/track 序列跟踪（与第二张图同语义：food红、bottle/glass→drink蓝），
# 最后一帧(=当前帧)的 mask 解码后映射到同一帧的 DA3 点云——mask 点染色 + 3D AABB 框，
# 用与 cloudimg 相同的固定相机服务端渲染成图，前端 /panel 第三框轮询展示。
# ══════════════════════════════════════════════════════════════════════
# SAM3 流式只追液体（bottle/glass → drink 蓝）。
# 流式配置（窗口/关代周期等旋钮）不动，只改目标词。
SAM3_CLOUD_TARGETS = [
    ("bottle", "drink"),     # 瓶装 → 蓝(液体)
    ("glass", "drink"),      # 杯装 → 蓝(液体)
]

# ── 第四图：SAM3 高亮点云（无框）。与第三图共用同一轮 SAM3 mask 结果，仅呈现方式不同：
# 不画 AABB 框，改为点云本体高亮。样式/强度/背景压暗/颜色由 /api/sam3hl/config 实时可调
# （只影响本图）。产物形态跟随③：fmt=glb → 与②③同一条 GLB 构建链路直渲（model-viewer
# 同管线，高亮写进顶点色；point_scale/splat 仅图模式生效，GLB 模式的点渲染由前端
# pt_size/pt_round/pt_atten 穿透 model-viewer 内部场景实时调节）；
# fmt=cloudimg → 服务端渲染 JPEG（点大小/视角等「点云整体样式」仅此路径生效）。──
_HL_STYLE_CN = {"tint": "染色", "solid": "纯色", "glow": "提亮", "outline": "描边"}
# 数值字段统一钳制表：{字段: (下限, 上限, 类型)}，高亮组 + 点云整体样式组共用一个配置
_HL_NUM_FIELDS = {
    "strength": (0, 100, int), "point_scale": (1.0, 5.0, float), "dim": (0, 90, int),
    # ── 点云整体样式（同样只作用于第四图）──
    "splat": (1, 4, int),            # 所有点的 splat 基数（高亮点在其上再乘 point_scale）
    "view_tilt": (0.0, 45.0, float), # 绕点云中心的俯视角（°，同②③调优视角）
    "view_zoom": (0.5, 2.5, float),  # 相机距离系数（×场景深度；1.25≈前端 VIEW_K）
    "eye_lift": (0.0, 1.0, float),   # 附加抬升（×场景深度，场景相对不跳帧）
    "eye_back": (0.0, 1.0, float),   # 附加后撤（×场景深度）
    "out_size": (320, 1200, int),    # 输出图高度（px）
    "sat": (0.0, 2.5, float),        # 点云底色饱和度系数（高亮色不受影响）
    "val": (0.2, 2.0, float),        # 点云底色明度系数
    "conf": (0, 90, int),            # 置信度裁剪分位（独立于产物参数，仅第四图）
    # ── 烘焙类扩展（写进 GLB，下一轮生效）──
    "hue": (-180.0, 180.0, float),   # 底色色相偏移（烘焙进顶点色，只动底色不动高亮色）
    "outlier_mad": (2.0, 20.0, float),  # 离群点裁剪强度：深度 MAD 倍数（越小裁得越狠；12=原写死值）
    # ── 点渲染（前端实时生效：/experience 穿透 model-viewer 内部 three.js 场景改
    # PointsMaterial/着色器，不进 GLB、不需重建；服务端仅存值做持久化/多端互通）──
    "pt_size": (0.5, 12.0, float),   # 点大小（px；近大远小开启时映射为世界尺寸）
    "pt_shape": (0, 2, int),         # 0=方点（three 默认） 1=圆点（裁圆） 2=柔边点（高斯衰减）
    "pt_atten": (0, 1, int),         # 1=近大远小（sizeAttenuation） 0=固定像素大小
    "pt_opacity": (0.05, 1.0, float),  # 整体不透明度
    "pt_blend": (0, 1, int),         # 1=发光叠加（additive blending） 0=正常
    "pt_density": (5, 100, int),     # 点密度 %（drawRange 抽稀显示）
    # 整体色彩（片元 uniform 实时调色，作用于最终画面含高亮色；区别于写进 GLB
    # 顶点色、只调底色不动高亮色的 sat/val/hue）
    "pt_hue": (-180.0, 180.0, float),  # 色相偏移（°，绕灰轴旋转）
    "pt_sat": (0.0, 2.5, float),       # 饱和度系数（0=灰度）
    "pt_val": (0.2, 2.5, float),       # 明度系数
    "pt_contrast": (0.5, 2.0, float),  # 对比度
    "pt_exposure": (0.3, 3.0, float),  # 曝光（model-viewer 原生属性）
    "pt_invert": (0, 1, int),          # 1=反色
    "pt_colormode": (0, 3, int),       # 着色模式：0原色 1双色调 2按深度 3按高度
    "pt_ramp_near": (0.2, 5.0, float), # 色带/深度雾 近端（m）
    "pt_ramp_far": (0.3, 6.0, float),  # 色带/深度雾 远端（m）
    "pt_fog": (0.0, 1.0, float),       # 深度雾强度（远点向背景色淡出）
    # 空间裁剪（实时，shader discard）
    "pt_clip_near": (0.0, 6.0, float), # 深度裁剪近端（m）
    "pt_clip_far": (0.3, 8.0, float),  # 深度裁剪远端（m；8=不裁）
    "pt_clip_ylo": (0.0, 1.0, float),  # 高度裁剪下限（点云高度归一化 0~1）
    "pt_clip_yhi": (0.0, 1.0, float),  # 高度裁剪上限
    # 相机与动效（实时）
    "pt_rotate": (0, 1, int),          # 1=自动旋转（model-viewer auto-rotate）
    "pt_rotate_speed": (2.0, 60.0, float),  # 旋转速度（°/s）
    "pt_fov_off": (-20.0, 20.0, float),     # FOV 偏移（°，叠加在真实相机 FOV 上）
    "pt_pulse": (0, 2, int),           # 呼吸脉冲：0关 1明度脉冲 2点大小脉冲
    "pt_pulse_speed": (0.2, 3.0, float),    # 脉冲/闪烁频率（Hz）
    "pt_sparkle": (0.0, 1.0, float),   # 星光闪烁强度（逐点错相位）
    # 置信度联动（读顶点色 alpha 通道里烘焙的置信度；任一 >0 时服务端才开始烘焙，
    # 首次开启需等下一轮 GLB，之后实时调节；仅③④，单目点云无此数据）
    "pt_conf_size": (0.0, 1.0, float),   # 置信度→点大小（低置信点变小）
    "pt_conf_alpha": (0.0, 1.0, float),  # 置信度→透明度（低置信点变透明）
}
# 十六进制颜色类配置字段（与 color 同样按 #rrggbb 校验）：双色调深/浅色、背景色
_HL_HEX_FIELDS = ("color", "pt_duo_a", "pt_duo_b", "pt_bg")
_sam3hl_lock = threading.Lock()
# 默认=②③前端 model-viewer 调优视角的等价参数（俯视10°、距离1.25×场景深度、真实FOV）：
# 除高亮/染色外与②③ GLB 取景一致；场景相对定距把单目逐帧深度的尺度抖动归一化，视角不跳。
# splat=1 保留点间缝隙的点状质感（拉远后点自然离散）。
_sam3hl_cfg = {"style": "solid", "strength": 65, "point_scale": 2.0,
               "dim": 52, "color_mode": "custom", "color": "#fffdf7",
               "splat": 1, "view_tilt": 0.0, "view_zoom": 1.40,
               "eye_lift": 0.0, "eye_back": 0.05, "out_size": 760,
               "sat": 0.0, "val": 1.35, "conf": 18,
               "hue": 0.0, "outlier_mad": 10.0,
               # 黑白银盐颗粒：极细柔边点、满密度、轻叠光，暗部仍保留噪点轮廓
               "pt_size": 0.65, "pt_shape": 2, "pt_atten": 0,
               "pt_opacity": 0.82, "pt_blend": 1, "pt_density": 100,
               "pt_hue": 0.0, "pt_sat": 0.0, "pt_val": 1.18,
               "pt_contrast": 1.35, "pt_exposure": 1.65, "pt_invert": 0,
               "pt_colormode": 0, "pt_duo_a": "#141450", "pt_duo_b": "#ffd27f",
               "pt_ramp_near": 0.5, "pt_ramp_far": 2.2, "pt_fog": 0.0,
               "pt_clip_near": 0.0, "pt_clip_far": 8.0,
               "pt_clip_ylo": 0.0, "pt_clip_yhi": 1.0,
               "pt_rotate": 0, "pt_rotate_speed": 10.0, "pt_fov_off": 0.0,
               "pt_pulse": 1, "pt_pulse_speed": 0.35, "pt_sparkle": 0.24,
               "pt_bg": "#000000", "pt_conf_size": 0.0, "pt_conf_alpha": 0.0}
_SAM3HL_PRESET_PATH = Path(__file__).resolve().parent / "sam3hl_preset.json"


def _load_sam3hl_preset():
    """服务启动时读取上次从网页保存的高亮点云预设。"""
    if not _SAM3HL_PRESET_PATH.is_file():
        return
    try:
        saved = json.loads(_SAM3HL_PRESET_PATH.read_text(encoding="utf-8"))
        if not isinstance(saved, dict):
            raise ValueError("preset must be an object")
        for key, (lo, hi, value_type) in _HL_NUM_FIELDS.items():
            if key in saved:
                _sam3hl_cfg[key] = value_type(min(hi, max(lo, float(saved[key]))))
        if saved.get("style") in _HL_STYLE_CN:
            _sam3hl_cfg["style"] = saved["style"]
        if saved.get("color_mode") in ("auto", "custom"):
            _sam3hl_cfg["color_mode"] = saved["color_mode"]
        for key in _HL_HEX_FIELDS:
            value = str(saved.get(key, "")).strip()
            if re.fullmatch(r"#[0-9a-fA-F]{6}", value):
                _sam3hl_cfg[key] = value
        print(f"[da3-web] 已加载高亮点云预设：{_SAM3HL_PRESET_PATH}", flush=True)
    except Exception as exc:
        print(f"[da3-web] 高亮点云预设读取失败：{type(exc).__name__}: {exc}", flush=True)


def _save_sam3hl_preset():
    """原子写入当前高亮点云配置，避免服务中断时留下半个 JSON 文件。"""
    temp_path = _SAM3HL_PRESET_PATH.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(_sam3hl_cfg, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temp_path.replace(_SAM3HL_PRESET_PATH)


_load_sam3hl_preset()
_sam3hl = {"kind": None, "url": None, "bytes": None, "seq": 0, "meta": None, "error": None}


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
# 多设备单路处理：DA3/SAM3/识别 这条重链路同一时刻只跑「当前选中设备」一路，
# 但 _recent_frames / SAM3 流式长记忆等跨帧时序状态都是单流假设——
# 切换选中设备时必须整组重置，防止上一台设备的帧污染新设备的时序语义。
# _stream_gen 是流代号：切设备 +1；在飞的 SAM3 后台任务凭它丢弃过期写回。
# ══════════════════════════════════════════════════════════════════════
_stream_lock = threading.Lock()
_stream_device = None     # 当前处理流的 device_id（None=还没处理过任何帧）
_stream_gen = 0


def _current_stream_gen():
    with _stream_lock:
        return _stream_gen


def _reset_stream_state():
    """清空所有跨帧时序状态（调用方须已判定发生了设备切换）。
    SAM3 流式 session 直接废弃（server 端按空闲自回收），新设备首帧自动重建。"""
    global _sam3_stream_sessions
    with _recent_lock:
        del _recent_frames[:]
    with _sam3_stream_lock:
        _sam3_stream_sessions = {}
    with _sam3hl_lock:
        _sam3hl.update({"kind": None, "url": None, "bytes": None, "meta": None, "error": None})


def _track_stream_device(device_id):
    """processor 每帧调用：设备与上一帧不同则重置时序缓存并推进流代号。返回当前流代号。"""
    global _stream_device, _stream_gen
    with _stream_lock:
        if device_id == _stream_device:
            return _stream_gen
        prev = _stream_device
        _stream_device = device_id
        _stream_gen += 1
        gen = _stream_gen
    if prev is not None:
        print(f"[da3-web] 处理流切换设备：{prev} → {device_id}，重置时序缓存与 SAM3 会话",
              flush=True)
        _reset_stream_state()
    return gen


# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
# 浅体验区展示页（/experience）：IFA 展台品牌化全屏 UI（Figma「IFA 专项 · 浅体验区」）
# - 全屏背景 = 实时点云（默认 SAM3 高亮点云，右下临时按钮在三种点云来源间切换；
#   本页任何情况不展示设备原图，产物未就绪时保持黑场）
# - 右侧状态区：待机「Place your food here」 ↔ 识别成功（名称/英文描述/营养标签/食物信号）
# - 流水视图：临时按钮进入，当日识别记录（名称+时间）+ 实时画面小窗
# - 设计稿 2240×1260，用 rem 等比缩放（1rem=设计稿 100px）；品牌字体走 /static/fonts
# - 手机适配：按视口横竖切版式（@media orientation:portrait），竖屏走 Figma 手机版
#   375×812（1473-2538 / 1473-2656）：logo 顶部居中、待机文案与营养卡贴底通栏；
#   竖屏按宽度等比（1rem=100vw/3.75，clamp 上限防竖屏平板过大）+ 贴边元素锚边 + safe-area
# - 竖屏细调（Figma 1616-4400 / 4397 / 4484 的真机标注）：logo 上提 12；待机文案与营养卡
#   按「距屏底」口径贴底（含 safe-area 分别为 87 / 48）；三宫格与其上分隔线间距 +2；
#   顶底压暗层由贴图换成可调的纵向渐变。注意标注量自「未全屏」的旧截图（底部还留着
#   浏览器工具栏那条），故落地取标注反推的目标距屏底绝对值，而非标注上的位移量
# ══════════════════════════════════════════════════════════════════════
EXPERIENCE_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<!-- PWA「添加到主屏幕」：手机从主屏图标打开即无地址栏全屏（iOS 走 apple-* meta 的
     standalone，Android 走 manifest 的 fullscreen）；浏览器直开不受影响。
     状态栏样式取 black 而非 black-translucent：后者让内容从屏幕最顶开始画（点云能铺到
     刘海），代价是视口整体上移状态栏高度、高度却不加，屏幕最底同样多少像素页面根本
     画不到（iPhone 17 实测 screen 874 / inner 812 / safe-top 62，底部黑掉 62）。顶底
     只能二选一，选把那一条留给状态栏——它本就该是黑的，而底部紧挨营养卡，黑边最扎眼。
     注意 iOS 在「添加到主屏幕」那一刻快照这些 meta，改完必须删图标重加才生效 -->
<link rel="manifest" href="/static/experience-manifest.json">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black">
<meta name="apple-mobile-web-app-title" content="ODYSS">
<link rel="apple-touch-icon" href="/static/icon-odyss-180.png">
<meta name="theme-color" content="#000000">
<title>ODYSS · Experience</title>
<script type="module" src="https://unpkg.com/@google/model-viewer@3.5.0/dist/model-viewer.min.js"></script>
<style>
 @font-face{font-family:'ABC Arizona Serif';src:url('/static/fonts/ABCArizonaSerif-Regular-Trial.otf') format('opentype');font-weight:400;font-display:swap}
 @font-face{font-family:'ABC Arizona Serif';src:url('/static/fonts/ABCArizonaSerif-Light-Trial.otf') format('opentype');font-weight:300;font-display:swap}
 @font-face{font-family:'Seabirds';src:url('/static/fonts/SeabirdsTrial-Book-V1.ttf') format('truetype');font-weight:400;font-display:swap}
 @font-face{font-family:'Seabirds';src:url('/static/fonts/SeabirdsTrial-SemiBold-V1.ttf') format('truetype');font-weight:600;font-display:swap}
 :root{--white:#FFFDF7;
   /* 竖屏压暗层参数（见下方 @media orientation:portrait 的 #shade），可被 URL 参数覆盖 */
   --sh-top:.62;--sh-bot:.62;--sh-span:24;
   /* 视口底距屏底的缺口（JS 实测写入，见 setVpGap）：竖屏贴底元素据此校正 */
   --vpgap:0px}
 /* 设计稿 2240×1260 等比缩放：1rem = 设计稿 100px，按宽高较小者定标（保持构图比例） */
 html{font-size:min(calc(100vw/22.4),calc(100vh/12.6))}
 *{box-sizing:border-box}
 body{margin:0;height:100vh;overflow:hidden;background:#000;color:var(--white);
      font-family:'Seabirds','Century Gothic','Futura',system-ui,sans-serif}
 /* ── 背景层：双 img 交叉淡入（换帧不闪黑） + model-viewer（GLB 产物用） ── */
 #stage{position:fixed;inset:0;background:#000}
 .bg{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;opacity:0;transition:opacity .45s ease}
 .bg.on{opacity:1}
 /* 点云化样式层：图片背景像素格化后经 mask 圆点镂空（形态层，保留原色彩）。
    开启时 img 层隐藏、由本 canvas 呈现（构图同 object-fit:cover） */
 #bgDot{position:absolute;inset:0;width:100%;height:100%;display:none}
 #stage.doton .bg{visibility:hidden}
 /* GLB 背景双缓冲：两个 model-viewer 常驻（display 不可为 none——model-viewer 靠
    IntersectionObserver 决定加载，display:none 的实例永远不触发 load），用 opacity
    切换可见性；隐藏实例后台加载下一个 GLB，load 后一帧切换，消掉换图空窗闪 */
 .bgmv{position:absolute;inset:0;width:100%;height:100%;display:block;opacity:0;
   pointer-events:none;--poster-color:transparent;
   /* 淡出结束后彻底 visibility:hidden：opacity:0 的 WebGL 合成层在图片类来源下
      仍可能被动效重绘透出/抢合成（跨样式泄漏），隐藏后与图片层完全隔离；
      visibility 延迟 .3s 切换保住双缓冲 crossfade 不闪 */
   visibility:hidden;transition:opacity .3s ease,visibility 0s linear .3s}
 .bgmv.on{opacity:1;visibility:visible;transition:opacity .3s ease,visibility 0s}
 /* 压暗层（设计稿导出，2240×1260 RGBA）：中心全透、边缘一圈微光、四周暗角压黑，
    保证两侧文案可读。铺满 100% 100%（不裁切），暗角始终贴合视口四边。
    两版备选，默认 B，加 ?shade=a 现场切回 A 对比（改默认只需换下面这行的文件名）：
      A = bg-shade.png  横椭圆、透光区偏小、四角留 ~28% 透光
      B = bg-shade2.png 正圆、透光区更大、四角压到近全黑 */
 :root{--shade-img:url('/static/bg-shade2.png')}
 #shade{position:absolute;inset:0;pointer-events:none;
   background:var(--shade-img) center/100% 100% no-repeat}
 /* ── UI 定位画布：设计稿 2240×1260（16:9）内接于屏幕并水平居中 ──
    背景点云/遮罩铺满整屏，但 logo 与右侧文案是按 16:9 构图摆的：屏幕比 16:9 宽时
    不能把它们拉到视口两端（会跑出遮罩暗角、左右不对称），而是落在这块居中画布里。
    1rem = 设计稿 100px 且按宽高较小者定标，故 22.4rem×12.6rem 恰是那块内接画布 */
 #ui{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
     width:22.4rem;height:12.6rem;pointer-events:none}
 /* ── 左侧 ODYSS logo（Figma 653-25299：左边距 26px、210×60、垂直居中） ── */
 #logo{position:absolute;left:.26rem;top:50%;transform:translateY(-50%);width:2.1rem;height:auto}
 /* ── 右侧状态区（两态叠放，淡入淡出切换）──
    两态在设计稿里是两张不同的图，宽度与右边距各不相同，故面板本身只负责贴右
    与垂直居中，宽度/右边距由各自的态给：
      待机 Figma 653-25306：515 宽、右边距 43
      成功 Figma 748-1145 ：532 宽、右边距 26 */
 #panel{position:absolute;right:0;top:50%;transform:translateY(-50%);display:grid;justify-items:end}
 /* 两态同格叠放；状态盒在面板高度内纵向居中（待机态只有一行标题，Figma 上为垂直居中） */
 #panel .state{grid-area:1/1;opacity:0;transition:opacity .5s ease;pointer-events:none;
   display:flex;flex-direction:column;justify-content:center}
 #panel .state.on{opacity:1}
 /* 成功卡不吃这条渐显：玻璃底只做拉伸，透明度全程 100%（渐显叠在拉伸上会让矩形
    从半透明"浮"出来）。opacity 仍是 0/1 的开关，只是瞬时切换——切到 on 的那一刻
    玻璃底还是 scaleY(0)、各行也还没推入，看不到任何硬闪 */
 #panel #card{transition:none}
 #idle{width:5.15rem;margin-right:.43rem}
 /* 让位给卡片时快速退场（默认的 .5s 渐隐会和卡片展开撞在一起，两段文案叠着）；
    卡片收完再渐显时把这个类摘掉，走回默认的慢渐显 */
 #idle.fast{transition:opacity .18s ease}
 h1{font-family:'ABC Arizona Serif',Georgia,serif;font-weight:400;font-size:.5rem;line-height:1;
    letter-spacing:-.01rem;margin:0}
 #idle h1{text-align:right}
 .sub{font-size:.2rem;line-height:1.2;letter-spacing:-.002rem;margin:0;color:var(--white)}
 .rule{border:0;border-top:1px solid #54514A;margin:0}
 /* 识别成功卡（Figma 563-1251 里的 nutrition-snapshot 748-1145）：532×536（高度由内容撑）、
    右边距 26、10% 白 + 40px 背景模糊的浅色玻璃面板、直角；内边距 44/24、
    区块间距一律 24（名称+描述 / Calories / 三列宏量 / Food Classification）。
    行距挪到了各行的遮罩层 .rvw 上，内层元素一律不留 margin（否则遮罩会连空白一起裁） */
 #card{position:relative;width:5.32rem;margin-right:.26rem;padding:.44rem .24rem;transition:none}
 /* 库内命中标记：内容来自参考食物库（而非模型现编）。展台上运营一眼可辨，
    观众几乎注意不到——排障时不必再开控制面去翻日志 */
 #creg{position:absolute;right:.1rem;top:.1rem;width:.07rem;height:.07rem;border-radius:50%;
       background:rgba(255,253,247,.55);display:none}
 #creg.on{display:block}
 /* ── 卡片出入场动效 ──
    出现：玻璃底先从中间往上下拉开（只拉伸、不渐显），随后各行依次从下往上推入
          （上一行走到一半下一行就起步）
    消失：内容先一起淡出，再由玻璃底向中间折叠收走（宽度不变、只压高度） */
 #cardbg{position:absolute;inset:0;background:rgba(255,253,247,.1);backdrop-filter:blur(.4rem);
   transform:scaleY(0);transform-origin:center;
   transition:transform .42s cubic-bezier(.22,.61,.36,1)}
 #card.bgin #cardbg{transform:scaleY(1)}
 /* 整卡不再渐显后，库内命中的小圆点得自己跟着玻璃底显出来，否则会先在空卡上冒出来 */
 #creg{opacity:0;transition:opacity .42s ease}
 #card.bgin #creg{opacity:1}
 #card .rvw{position:relative;overflow:hidden}
 #card .rvw+.rvw{margin-top:.24rem}
 #card .rvw.g8{margin-top:.08rem}          /* 名称与描述之间是 8，不是 24 */
 #card .rvw>*{transform:translateY(105%);opacity:0;
   transition:transform .2s cubic-bezier(.22,.61,.36,1),opacity .2s ease}
 /* 标题行不做遮罩裁切：h1 是 line-height:1，"Orange"/"Yogurt" 里 g、y 的降部会伸出
    行盒，被 overflow:hidden 齐根切掉。改成不裁 + 小位移推入（幅度大了会盖到描述行）*/
 #card .rvw.nom{overflow:visible}
 #card .rvw.nom>*{transform:translateY(.12rem)}
 #card .rvw.in>*{transform:none;opacity:1}
 #card.out .rvw>*,#card.out #creg{transition:opacity .24s ease;opacity:0;transform:none}
 .krow{display:flex;justify-content:space-between;align-items:center;gap:.3rem}
 .klab{font-size:.2rem;letter-spacing:-.002rem;color:var(--white);white-space:nowrap}
 .kval{font-size:.36rem;letter-spacing:-.0036rem;white-space:nowrap;
       font-variant-numeric:lining-nums proportional-nums}
 #macros{display:flex;justify-content:center;gap:.9rem;text-align:left;padding:.3rem 0}
 #macros .mlab{display:block;font-size:.2rem;letter-spacing:-.002rem;color:var(--white)}
 #macros .mval{display:block;font-size:.36rem;letter-spacing:-.0036rem;margin-top:.09rem;
       font-variant-numeric:lining-nums proportional-nums}
 /* ── 流水视图：实时画面小窗 + 当日识别记录列表 ── */
 #tl{position:absolute;inset:0;display:none}
 #tl.on{display:block;background:rgba(0,0,0,.62)}   /* 流水态强压暗背景，突出列表 */
 body.tlon #panel{visibility:hidden}                 /* 流水态彻底藏起状态文案（不留渐隐残影） */
 #tlinset{position:absolute;left:17.5%;top:29.8%;width:40.2%;height:39.8%;background:#050505;
          border-radius:.08rem;overflow:hidden;display:flex;align-items:center;justify-content:center}
 #tlinset img{width:100%;height:100%;object-fit:cover;display:none}
 #tlwait{font-size:.16rem;color:rgba(255,253,247,.5)}
 #tllist{position:absolute;left:65%;right:3.3%;top:0;bottom:0;display:flex;flex-direction:column;justify-content:center}
 .trow{display:flex;justify-content:space-between;align-items:baseline;gap:.3rem;
       padding:.205rem 0;border-bottom:1px solid rgba(255,253,247,.34);
       font-family:'ABC Arizona Serif',Georgia,serif;font-size:.4rem;letter-spacing:-.006rem}
 .trow .tname{display:flex;align-items:center;gap:.16rem;min-width:0;overflow:hidden;
       text-overflow:ellipsis;white-space:nowrap}
 .trow .ttime{font-variant-numeric:tabular-nums;flex:none}
 .trow.dim{opacity:.42}
 .spin{flex:none;width:.26rem;height:.26rem;border-radius:50%;
       border:.025rem solid rgba(255,253,247,.35);border-top-color:var(--white);
       animation:spin 1s linear infinite}
 @keyframes spin{to{transform:rotate(360deg)}}
 #tlempty{font-family:'ABC Arizona Serif',Georgia,serif;font-size:.3rem;color:rgba(255,253,247,.55)}
 /* ── 右下临时工具按钮（后期调样式用，刻意低调；尺寸用物理像素不随设计稿缩放） ── */
 #tools{position:fixed;right:20px;bottom:18px;display:flex;gap:8px;z-index:9}
 /* 右键菜单：控制台（右下调试按钮组）显隐开关 */
 #ctxmenu{position:fixed;display:none;z-index:99;background:rgba(24,24,26,.96);
  border:1px solid #3a3a3c;border-radius:10px;padding:5px;backdrop-filter:blur(10px)}
 #ctxmenu .ctxitem{padding:9px 16px;border-radius:7px;cursor:pointer;font-size:13px;
  color:#e8e8ea;white-space:nowrap;user-select:none}
 #ctxmenu .ctxitem:hover{background:#3a3a3c}
 #tools button{background:rgba(255,253,247,.08);border:1px solid rgba(255,253,247,.22);
   color:rgba(255,253,247,.72);font:12px system-ui,sans-serif;padding:6px 14px;border-radius:99px;
   cursor:pointer;backdrop-filter:blur(6px)}
 #tools button:hover{background:rgba(255,253,247,.16)}
 #tools select{background:rgba(255,253,247,.08);border:1px solid rgba(255,253,247,.22);
   color:rgba(255,253,247,.72);font:12px system-ui,sans-serif;padding:6px 10px;border-radius:99px;
   cursor:pointer;backdrop-filter:blur(6px);outline:none}
 #tools select option{background:#1c1c1e;color:#eee}
 /* 隧道状态按钮：绿=已连通 / 橙=重建中 / 红=已断开（红时点击一键重建） */
 #btnTun{display:inline-flex;align-items:center;gap:6px}
 #tunDot{width:7px;height:7px;border-radius:50%;background:#bbb;flex:none}
 #btnTun.down{border-color:rgba(255,59,48,.55);color:#ff8478}
 /* ── 高亮点云样式调节抽屉（临时工程工具；物理像素、不随设计稿缩放；同 /panel 两张调节卡） ── */
 #hlcfg{position:fixed;top:0;right:-340px;bottom:0;width:320px;z-index:20;overflow-y:auto;
   background:rgba(12,12,14,.94);border-left:1px solid rgba(255,253,247,.14);
   backdrop-filter:blur(10px);transition:right .3s ease;
   font:12px system-ui,sans-serif;color:rgba(255,253,247,.85);padding:14px 16px 20px}
 #hlcfg.on{right:0}
 #hlcfg .hd{display:flex;align-items:center;justify-content:space-between;font-size:13px;
   font-weight:600;margin-bottom:4px;color:rgba(255,253,247,.95)}
 #hlcfg .hd button{background:none;border:0;color:rgba(255,253,247,.6);font-size:15px;cursor:pointer;padding:2px 6px}
 #hlcfg .sec{font-weight:600;margin:14px 0 4px;color:rgba(255,253,247,.95);
   border-top:1px solid rgba(255,253,247,.12);padding-top:12px}
 #hlcfg .seg{display:flex;gap:6px;flex-wrap:wrap;margin:6px 0 2px}
 #hlcfg .seg button{font-size:12px;padding:4px 12px;border-radius:99px;cursor:pointer;
   border:1px solid rgba(255,253,247,.25);background:transparent;color:rgba(255,253,247,.75)}
 #hlcfg .seg button.on{background:var(--white);border-color:var(--white);color:#161311}
 #hlcfg .fld{margin:9px 0 2px}
 #hlcfg .fld label{display:block;margin-bottom:2px;color:rgba(255,253,247,.72)}
 #hlcfg .fld b{color:#6ab7ff;font-weight:600;font-variant-numeric:tabular-nums}
 #hlcfg input[type=range]{width:100%;margin:2px 0 0}
 #hlcfg .radios{display:flex;gap:14px;align-items:center;margin:6px 0 2px;flex-wrap:wrap}
 #hlcfg .radios label{cursor:pointer}
 #hlcfg input[type=color]{width:38px;height:24px;border:none;background:none;padding:0;cursor:pointer}
 #hlcfg select{width:100%;background:#1c1c1e;color:#eee;border:1px solid rgba(255,253,247,.25);
   border-radius:6px;padding:4px 6px;font:12px system-ui,sans-serif;margin:2px 0;outline:none}
 #hlcfg .hint{margin-top:12px;font-size:11px;line-height:1.5;color:rgba(255,253,247,.45)}
 /* 配置预设区：保存输入行 + 预设列表行 */
 #hlcfg .prow{display:flex;align-items:center;gap:6px;margin:6px 0 2px}
 #hlcfg .prow input[type=text]{flex:1;min-width:0;background:#1c1c1e;color:#eee;
   border:1px solid rgba(255,253,247,.25);border-radius:6px;padding:4px 6px;
   font:12px system-ui,sans-serif;outline:none}
 #hlcfg .prow button{font-size:12px;padding:4px 10px;border-radius:6px;cursor:pointer;
   border:1px solid rgba(255,253,247,.25);background:transparent;color:rgba(255,253,247,.75);
   white-space:nowrap}
 #hlcfg .prow button:hover{background:rgba(255,253,247,.12)}
 #hlcfg .prow button.confirm{border-color:#ff7a6b;color:#ff7a6b}
 #hlcfg .prow .nm{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;
   white-space:nowrap;color:rgba(255,253,247,.85)}
 #hlcfg .prow .ts{font-size:10px;color:rgba(255,253,247,.4);white-space:nowrap;
   font-variant-numeric:tabular-nums}
 /* ══ 手机竖屏（Figma 手机版 1473-2538 / 1473-2656，设计稿 375×812）══
    切换规则：按视口横竖切版式——竖屏（高>宽）走本块的手机构图，横屏（展台大屏、
    手机横持）维持上方大屏构图；不判 UA/设备。
    缩放规则：竖屏按「宽度」等比（1rem = 100vw/3.75，仍是 1rem=设计稿 100px 口径），
    纵向不做固定内接画布：logo 锚顶、待机文案/营养卡锚底（各带 safe-area），中间由
    背景 cover 自然填充——手机高宽比从 16:9 到 19.5:9+ 不等，内接固定画布会留黑/脱边。
    clamp 上限 115px：竖屏平板/竖置大屏不随宽度无限放大（内容约等效 431px 宽居中） */
 @media (orientation:portrait){
  html{font-size:min(calc(100vw/3.75),115px)}
  /* 竖屏压暗层：纵向 CSS 渐变（不再用 bg-shade-portrait.png，该图留仓备回退）。原贴图顶底 alpha 高到
     0.96、且 20% 高度内就从 0.96 急落到 0.30，顶底观感是死黑一条；而它横向剖面全程
     全透，本就等价于一条纵向渐变，改 CSS 后既减淡又能现场调参。
     三个参数可用 URL 覆盖（手机上改地址栏即可对比）：?shtop=.62&shbot=.62&shspan=24
       shtop/shbot = 顶/底边缘不透明度，shspan = 单侧渐变跨度（%，越大过渡越长越柔）
     写 background 整条覆盖（不走 --shade-img 变量），?shade=a 的 A/B 只作用横屏 */
  #shade{background:linear-gradient(to bottom,
     rgba(0,0,0,var(--sh-top)) 0%,
     rgba(0,0,0,calc(var(--sh-top) * .35)) calc(var(--sh-span) * .5%),
     rgba(0,0,0,0) calc(var(--sh-span) * 1%),
     rgba(0,0,0,0) calc(100% - var(--sh-span) * 1%),
     rgba(0,0,0,calc(var(--sh-bot) * .35)) calc(100% - var(--sh-span) * .5%),
     rgba(0,0,0,var(--sh-bot)) 100%)}
  /* iOS Safari 地址栏收放：100vh 会比可视区高，底部锚定元素被工具栏盖住；dvh 跟随可视区 */
  body{height:100dvh}
  /* UI 画布铺满视口：竖屏构图全部贴边锚定，不再用 16:9 内接画布 */
  #ui{width:100%;height:100%;top:0;left:0;transform:none}
  /* logo 顶部居中（Figma：105×30、原稿状态栏下 25；1616-4400/4397 标注上提 12 → safe-area + .13rem） */
  #logo{left:50%;top:calc(env(safe-area-inset-top,0px) + .13rem);transform:translateX(-50%);width:1.05rem}
  /* 状态区从贴右垂直居中改为贴底通栏，两态各自给底边距 */
  #panel{left:0;right:0;top:auto;bottom:0;transform:none;justify-items:center;align-items:end}
  h1{font-size:.24rem;letter-spacing:-.0048rem}
  .sub{font-size:.12rem;letter-spacing:-.0012rem}
  #idle{width:auto;margin:0 0 max(0px,calc(.8rem - var(--vpgap)))}
  #idle h1{text-align:center}
  /* 营养卡贴底通栏（Figma 343×223：左右 16、距底 54、padding 16、区块间距 12）。
     距底一律按「距屏幕最底」算、不叠加 safe-area（2026-08-27 用户当面定：待机文案 80、
     营养卡 54，即 Figma 原稿值本身）——叠加 safe-area 会凭空多抬 34。
     再扣掉 --vpgap「视口缺口」：iOS standalone 下视口有时短于屏幕（画不到最底那一条，
     实测 iPhone 17 上短 61），此时按屏底算的 54 落到视口里其实是 54+61。扣掉缺口让元素
     真正贴到能画的最低处；缺口为 0（视口铺满）时就退化成标称的 54 */
  #card{width:calc(100% - .32rem);max-width:3.43rem;
        margin:0 0 max(0px,calc(.54rem - var(--vpgap)));padding:.16rem}
  #cardbg{backdrop-filter:blur(.13rem)}
  #card .rvw+.rvw{margin-top:.12rem}
  #card .rvw.g8{margin-top:.04rem}
  /* 三宫格（Protein/Carbs/Fat）与其上方那条分隔线之间比通用区块间距多 2（Figma 1616-4397） */
  #card .rvw:has(#macros){margin-top:.14rem}
  .rule{border-top-color:rgba(255,253,247,.2)}
  .klab{font-size:.12rem;letter-spacing:-.0012rem}
  .kval{font-size:.2rem;letter-spacing:-.002rem}
  #macros{gap:.58rem;padding:.16rem 0}
  #macros .mlab{font-size:.12rem;letter-spacing:-.0012rem}
  #macros .mval{font-size:.2rem;letter-spacing:-.002rem;margin-top:.06rem}
  /* 手机版设计没有 Food Classification 行（连同其上方分隔线一起隐藏） */
  #card .rvw:has(#crule),#card .rvw:has(#cclsrow){display:none}
  /* 工程调试 UI（右下按钮组/调节抽屉/右键菜单）在手机上必挡内容，竖屏一律隐藏；
     display 用 !important 压过 applyConsole 写的 inline style */
  #tools,#hlcfg,#ctxmenu{display:none!important}
  /* 流水视图竖屏改上下排（运营偶尔手机看，保可用即可，不追求精排） */
  #tlinset{left:6%;right:6%;width:auto;top:10%;height:34%}
  #tllist{left:6%;right:6%;top:48%;bottom:6%;justify-content:flex-start}
  .trow{font-size:.2rem}
  #tlempty{font-size:.2rem}
 }
</style></head><body>
<div id="stage">
 <img class="bg" id="bgA" alt=""><img class="bg" id="bgB" alt="">
 <canvas id="bgDot"></canvas>
 <model-viewer id="bgmvA" class="bgmv" touch-action="none" interaction-prompt="none"
   camera-orbit="0deg 90deg 1.5m" field-of-view="55deg" camera-target="0m 0m -1.5m"
   min-camera-orbit="-Infinity 0deg 1%" max-camera-orbit="Infinity 180deg 2000%"
   min-field-of-view="10deg" max-field-of-view="60deg"
   shadow-intensity="0.3" exposure="1.35"></model-viewer>
 <model-viewer id="bgmvB" class="bgmv" touch-action="none" interaction-prompt="none"
   camera-orbit="0deg 90deg 1.5m" field-of-view="55deg" camera-target="0m 0m -1.5m"
   min-camera-orbit="-Infinity 0deg 1%" max-camera-orbit="Infinity 180deg 2000%"
   min-field-of-view="10deg" max-field-of-view="60deg"
   shadow-intensity="0.3" exposure="1.35"></model-viewer>
 <div id="shade"></div>

 <div id="ui">
 <svg id="logo" viewBox="0 0 210 60" fill="none" xmlns="http://www.w3.org/2000/svg">
  <g clip-path="url(#clip0_653_25299)">
   <path d="M114.546 30.014C114.546 17.0107 104.033 6.47683 91.0654 6.47683H83.4596V53.5511H91.0654C104.033 53.5511 114.546 43.0173 114.546 30.014ZM120 30.014C120 46.03 107.046 59.0121 91.0654 59.0121H77.9986V1.01587H91.0654C107.046 1.01587 120 13.998 120 30.014Z" fill="#FFFDF7"/>
   <path d="M165 1.00903H158.594L139.995 30.3317L121.404 1.00903H114.998L137.3 36.1807V59.0123H142.683V36.1807L165 1.00903Z" fill="#FFFDF7"/>
   <path d="M166.312 10.0612C166.524 4.36737 171.216 -0.204595 177.594 0.0141254C179.28 0.0705694 180.826 0.451567 182.18 1.07951L179.577 5.91958C178.942 5.66558 178.215 5.51036 177.404 5.48214C174.031 5.36925 171.929 7.62701 171.83 10.2728C171.731 12.9116 173.417 15.1835 178.201 18.7959C182.752 22.2319 188.276 26.1407 191.289 31.2982C194.372 36.5757 195.289 43.6665 191.606 50.4045C187.874 57.2343 178.554 62.0602 169.41 59.1463C160.379 56.2606 155.651 48.3232 156.025 39.9835C156.272 34.5861 158.558 30.3528 161.077 27.3048L165.36 30.7479C163.413 33.0974 161.719 36.2794 161.543 40.2234C161.268 46.3053 164.633 51.8721 171.103 53.9393C177.467 55.9713 184.156 52.5494 186.752 47.801C189.398 42.961 188.77 37.9022 186.512 34.0358C184.184 30.0494 179.781 26.8674 174.857 23.1491C170.165 19.6073 166.101 15.7761 166.319 10.0753L166.312 10.0612Z" fill="#FFFDF7"/>
   <path fill-rule="evenodd" clip-rule="evenodd" d="M43.095 0C59.1674 0 72.0014 13.5466 72.0014 30C72.0014 46.4534 59.1745 60 43.095 60C40.6609 60 38.2973 59.6896 36.0466 59.1039C33.7888 59.6896 31.4252 60 28.984 60C12.8692 60 0 46.4605 0 30C0 13.5395 12.8692 0 28.984 0C31.4252 0 33.7888 0.310442 36.0466 0.896049C38.2973 0.310442 40.6609 0 43.095 0ZM27.5235 4.71308C15.3246 5.55268 5.49624 16.8697 5.49624 30C5.49624 43.1303 15.3246 54.4403 27.5235 55.2869C19.4802 49.9389 14.1886 40.5691 14.1886 30C14.1886 19.4309 19.4802 10.0682 27.5235 4.71308ZM44.5978 4.72013C52.6623 10.0682 57.968 19.4309 57.968 30C57.968 40.5691 52.6623 49.9318 44.5978 55.2799C56.7333 54.412 66.5052 43.1162 66.5052 30C66.5052 16.8838 56.7262 5.58796 44.5978 4.72013ZM36.6322 6.0889C36.2582 5.94779 35.842 5.94779 35.468 6.0889C26.3311 9.60254 19.6849 19.1839 19.6849 30C19.6849 40.8161 26.3311 50.3975 35.468 53.9111C35.842 54.0522 36.2582 54.0522 36.6322 53.9111C45.8043 50.3975 52.4718 40.8161 52.4718 30C52.4718 19.1839 45.8043 9.60254 36.6322 6.0889Z" fill="#FFFDF7"/>
   <path d="M187.204 8.09268C188.107 3.46427 192.7 -0.232811 197.907 0.0141317C199.431 0.0846867 201.047 0.402184 202.556 1.07246L199.706 5.82787C199.085 5.63031 198.394 5.50331 197.646 5.46804C195.092 5.34809 192.961 7.23897 192.594 9.12984C192.213 11.0983 192.841 13.2996 196.065 16.2912C199.551 19.5226 204.906 24.4826 207.495 29.739C210.12 35.0588 210.924 41.9027 208.751 48.6407C207.615 52.1472 205.562 55.8161 202.408 59.0052H195.621V57.6153C199.861 54.5884 202.352 50.6233 203.53 46.9685C205.252 41.6345 204.595 36.2441 202.571 32.1449C200.517 27.9751 196.009 23.7065 192.326 20.2916C188.382 16.6298 186.322 12.6505 187.211 8.09974L187.204 8.09268Z" fill="#FFFDF7"/></g>
  <defs><clipPath id="clip0_653_25299"><rect width="210" height="60" fill="white"/></clipPath></defs>
 </svg>

 <div id="panel">
  <div class="state" id="idle">
   <h1>Place your food here</h1>
  </div>
  <div class="state" id="card">
   <!-- 玻璃底独立成层：出入场缩放只作用在它身上，不会把文字一起压扁 -->
   <div id="cardbg"></div>
   <span id="creg" title="from reference catalog"></span>
   <!-- 每行外面套一层 .rvw 当遮罩，内容从下往上推入；行距移到 .rvw 上（内层不再留 margin） -->
   <div class="rvw nom"><h1 id="cname"></h1></div>
   <div class="rvw g8"><p class="sub" id="cdesc"></p></div>
   <div class="rvw"><hr class="rule"></div>
   <div class="rvw"><div class="krow" id="ckcalrow"><span class="klab">Calories</span><span class="kval" id="ckcal"></span></div></div>
   <div class="rvw"><hr class="rule" id="mrule"></div>
   <div class="rvw"><div id="macros">
    <div class="m" id="mpro"><span class="mlab">Protein</span><span class="mval"></span></div>
    <div class="m" id="mcarb"><span class="mlab">Carbs</span><span class="mval"></span></div>
    <div class="m" id="mfat"><span class="mlab">Fat</span><span class="mval"></span></div>
   </div></div>
   <div class="rvw"><hr class="rule" id="crule"></div>
   <div class="rvw"><div class="krow" id="cclsrow"><span class="klab">Food Classification</span><span class="kval" id="ccls"></span></div></div>
  </div>
 </div>
 </div>

 <div id="tl">
  <div id="tlinset"><img id="tlraw" alt=""><span id="tlwait">Waiting for camera…</span></div>
  <div id="tllist"></div>
 </div>
</div>

<div id="tools">
 <select id="selDev" style="display:none" title="选择设备"></select>
 <select id="selStyle">
  <option value="devdepth" title="仅 G335 等带硬件深度的相机支持">设备深度图</option>
  <option value="devpc" title="硬件真深度反投影彩色点云，仅 G335 等带硬件深度的相机支持">设备点云</option>
 </select>
 <button id="btnTun" title="Qwen 识别隧道：检测中…"><span id="tunDot"></span><span id="tunTxt">隧道</span></button>
 <button id="btnHlCfg">调节</button>
 <button id="btnTl">流水</button>
 <button id="btnFs" title="整页铺满显示器（Esc 退出）">全屏</button>
</div>

<!-- 右键菜单：控制台显隐（右下调试按钮组默认隐藏，展台观众看不到） -->
<div id="ctxmenu"><div class="ctxitem" id="ctxToggle">显示控制台</div></div>

<div id="hlcfg">
 <div class="hd">展示调节 <button id="hlcfgClose" title="关闭">✕</button></div>
 <div class="sec" style="border-top:0;padding-top:0;margin-top:8px">识别触发（主链路直传 VLM）</div>
 <div class="fld"><label>直传开关</label>
  <div class="radios">
   <label><input type="radio" name="rdon" value="1" checked> 开（每帧直送 VLM，不等 SAM3）</label>
   <label><input type="radio" name="rdon" value="0"> 关（SAM3 先筛，命中食物/饮品才送 VLM）</label>
  </div></div>
 <div class="fld"><label>识别间隔 <b id="v_rd_itv">0.5</b> s（多久直传一帧去识别）</label>
  <input type="range" id="r_rd_itv" min="0.2" max="10" step="0.1" value="0.5"></div>
 <div class="fld"><label>并发上限 <b id="v_rd_conc">1</b> 路</label>
  <input type="range" id="r_rd_conc" min="1" max="6" step="1" value="1"></div>
 <div class="fld"><label>帧新鲜度上限 <b id="v_rd_age">8.0</b> s（帧比这更旧就整轮丢掉，不发请求）</label>
  <input type="range" id="r_rd_age" min="1" max="30" step="0.5" value="8"></div>
 <div class="hint">两种口径共用同一条触发线程与帧源（当前设备的最新 <b>RGB 帧</b>——VLM 与
  SAM3 都吃彩色图，伪彩深度图只是背景呈现）。<b>开</b>=整帧直送 Qwen VLM 问画面里有什么食物；
  <b>关</b>=同一帧先过 SAM3（生产词表 + food 词，直接跑彩色帧），认出食物/饮品才带框送 VLM——
  SAM3 只当「有没有东西」的前置门，命中后照样识别。关比开省 VLM 调用（画面空时不烧），
  代价是每轮多一次 SAM3（约 0.5~0.9s）。<b>并发=1</b> 是串行·最新优先：上一轮没回就丢旧帧用
  最新帧，实际节奏≈VLM 延时（实测约 1.2~1.5s/轮）；调大并发才真按间隔多路齐发，代价是成本 ×N
  且并发轮次拿的是同一份去重候选、可能给同一食物重复建卡。间隔也快不过 RGB 推帧
  （下方「数据源帧率」），同一帧不会重复送。<br>实测：<b id="v_rd_stat">--</b></div>
 <div class="sec" style="border-top:0;padding-top:0;margin-top:8px">数据源帧率（当前设备）</div>
 <div class="fld"><label>RGB 推帧 <b id="v_push_fps">2.0</b> fps · 实测到帧 <b id="v_fps_meas">--</b> fps</label>
  <input type="range" id="r_push_fps" min="0.5" max="30" step="0.5" value="2"></div>
 <div class="fld"><label>点云直传间隔 <b id="v_prod_itv">2.5</b> s</label>
  <input type="range" id="r_prod_itv" min="0.5" max="10" step="0.5" value="2.5"></div>
 <div class="hint" style="margin-top:8px">按设备生效（推流端每 2s 轮询取走，调完看「实测到帧」
  确认生效，约 2~4s 跟上）。<b>RGB 推帧</b>驱动 DA3/SAM3 处理链路，是高亮/SAM3 点云的
  帧率上限（点云画面刷新还受 GPU 处理速度约束，不会跟满推帧率；设备深度图/原始帧则直接
  跟满）；<b>点云直传间隔</b>只对带真深度直传的设备（如 macmini-astra）生效，决定单目
  点云来源的刷新节奏。Mac↔5090 链路实测约 14Mbps：RGB 约 135KB/帧、深度约 40KB/帧
  （质量 72），两台摄像机同推时调密任何一路都会挤占另一路，调完盯一眼实测值是否跟得上。</div>
 <div id="ddonly" style="display:none">
 <div class="sec">深度渲染（写回推流端，约 2~4s 生效）</div>
 <div class="fld"><label>色彩映射</label>
  <select id="dd_cmap">
   <option value="lidar">LIDAR 蓝青（近亮青远深蓝）</option>
   <option value="radar" selected>雷达蓝绿·暖尖（近端绿黄橙点睛，默认）</option>
   <option value="indigo">靛蓝冰晶（近白远靛黑）</option>
   <option value="moss">苔原绿（近卡其远墨绿）</option>
   <option value="lavender">薰衣草雾（柔和低对比）</option>
   <option value="turbo">TURBO</option><option value="jet">JET</option>
   <option value="viridis">VIRIDIS</option><option value="plasma">PLASMA</option>
   <option value="inferno">INFERNO</option><option value="magma">MAGMA</option>
   <option value="hot">HOT</option><option value="bone">BONE</option>
   <option value="ocean">OCEAN</option><option value="hsv">HSV</option>
   <option value="parula">PARULA</option><option value="cividis">CIVIDIS</option>
   <option value="twilight_shifted">TWILIGHT</option><option value="deepgreen">DEEPGREEN</option>
   <option value="gray">灰度</option>
  </select></div>
 <div class="fld"><label>映射方向</label>
  <div class="radios">
   <label><input type="radio" name="ddinv" value="0" checked> 近亮远暗</label>
   <label><input type="radio" name="ddinv" value="1"> 近暗远亮</label>
  </div></div>
 <div class="fld"><label>量程模式</label>
  <div class="radios">
   <label><input type="radio" name="ddar" value="0"> 固定量程</label>
   <label><input type="radio" name="ddar" value="1" checked> 自动分位</label>
  </div></div>
 <div class="fld"><label>量程近端 <b id="v_dd_min">0.05</b>m</label>
  <input type="range" id="r_dd_min" min="0.05" max="3" step="0.05" value="0.05"></div>
 <div class="fld"><label>量程远端 <b id="v_dd_max">1.4</b>m</label>
  <input type="range" id="r_dd_max" min="0.3" max="8" step="0.1" value="1.4"></div>
 <div class="fld"><label>自动分位 低 <b id="v_dd_lo">8</b>%</label>
  <input type="range" id="r_dd_lo" min="0" max="20" step="1" value="8"></div>
 <div class="fld"><label>自动分位 高 <b id="v_dd_hi">89</b>%</label>
  <input type="range" id="r_dd_hi" min="80" max="100" step="1" value="89"></div>
 <div class="fld"><label>Gamma <b id="v_dd_gamma">0.65</b></label>
  <input type="range" id="r_dd_gamma" min="0.2" max="3" step="0.05" value="0.65"></div>
 <div class="fld"><label>直方图均衡</label>
  <div class="radios">
   <label><input type="radio" name="ddeq" value="off"> 关</label>
   <label><input type="radio" name="ddeq" value="global" checked> 全局</label>
   <label><input type="radio" name="ddeq" value="clahe"> CLAHE</label>
  </div></div>
 <div class="fld"><label>CLAHE 强度 <b id="v_dd_clip">2.5</b></label>
  <input type="range" id="r_dd_clip" min="0.5" max="8" step="0.5" value="2.5"></div>
 <div class="fld"><label>无效点颜色</label>
  <div class="radios"><input type="color" id="c_dd_invalid" value="#000000"></div></div>
 <div class="fld"><label>孔洞填充</label>
  <div class="radios">
   <label><input type="radio" name="ddfill" value="off"> 关</label>
   <label><input type="radio" name="ddfill" value="close" checked> 形态学</label>
   <label><input type="radio" name="ddfill" value="inpaint"> 修补</label>
  </div></div>
 <div class="fld"><label>填充半径 <b id="v_dd_fillpx">5</b>px</label>
  <input type="range" id="r_dd_fillpx" min="1" max="15" step="1" value="5"></div>
 <div class="fld"><label>时域平滑 <b id="v_dd_ema">0.20</b></label>
  <input type="range" id="r_dd_ema" min="0" max="0.9" step="0.05" value="0.2"></div>
 <div class="fld"><label>空域滤波</label>
  <div class="radios">
   <label><input type="radio" name="ddsm" value="off" checked> 关</label>
   <label><input type="radio" name="ddsm" value="median"> 中值</label>
   <label><input type="radio" name="ddsm" value="bilateral"> 双边</label>
  </div></div>
 <div class="fld"><label>边缘描边 <b id="v_dd_edge">0.00</b></label>
  <input type="range" id="r_dd_edge" min="0" max="1" step="0.05" value="0"></div>
 <div class="fld"><label>等值线间隔 <b id="v_dd_contour">0.00</b>m（0=关）</label>
  <input type="range" id="r_dd_contour" min="0" max="1" step="0.05" value="0"></div>
 <div class="fld"><label>JPEG 质量 <b id="v_dd_jq">80</b></label>
  <input type="range" id="r_dd_jq" min="30" max="95" step="5" value="80"></div>
 <div class="fld"><label>深度推帧率 <b id="v_dd_fps">30</b>fps（0=跟随RGB）</label>
  <input type="range" id="r_dd_fps" min="0" max="30" step="0.5" value="30"></div>
 <div class="sec">深度显示（本页即时）</div>
 <div class="fld"><label>亮度 ×<b id="v_dc_bright">2.50</b></label>
  <input type="range" id="r_dc_bright" min="0.2" max="4" step="0.05" value="2.5"></div>
 <div class="fld"><label>亮核辉光 <b id="v_dc_glow">0.60</b>（顶格区已到显示上限，只能靠掺白变亮）</label>
  <input type="range" id="r_dc_glow" min="0" max="1.5" step="0.05" value="0.6"></div>
 <div class="fld"><label>对比度 ×<b id="v_dc_contrast">1.00</b></label>
  <input type="range" id="r_dc_contrast" min="0.2" max="2.5" step="0.05" value="1"></div>
 <div class="fld"><label>饱和度 ×<b id="v_dc_sat">1.05</b></label>
  <input type="range" id="r_dc_sat" min="0" max="3" step="0.05" value="1.05"></div>
 <div class="fld"><label>色相旋转 <b id="v_dc_hue">0</b>°</label>
  <input type="range" id="r_dc_hue" min="-180" max="180" step="5" value="0"></div>
 <div class="fld"><label>反色</label>
  <div class="radios">
   <label><input type="radio" name="dcinv" value="0" checked> 关</label>
   <label><input type="radio" name="dcinv" value="1"> 开</label>
  </div></div>
 <div class="fld"><label>模糊 <b id="v_dc_blur">0</b>px</label>
  <input type="range" id="r_dc_blur" min="0" max="10" step="1" value="0"></div>
 <div class="fld"><label>不透明度 ×<b id="v_dc_opacity">1.00</b></label>
  <input type="range" id="r_dc_opacity" min="0.2" max="1" step="0.05" value="1"></div>
 <div class="fld"><label>缩放模式</label>
  <div class="radios">
   <label><input type="radio" name="dcfit" value="cover" checked> 铺满裁边</label>
   <label><input type="radio" name="dcfit" value="contain"> 完整适配</label>
  </div></div>
 <div class="fld"><label>水平镜像</label>
  <div class="radios">
   <label><input type="radio" name="dcmir" value="0"> 关</label>
   <label><input type="radio" name="dcmir" value="1" checked> 开</label>
  </div></div>
 <div class="fld"><label>旋转</label>
  <div class="radios">
   <label><input type="radio" name="dcrot" value="0" checked> 0°</label>
   <label><input type="radio" name="dcrot" value="90"> 90°</label>
   <label><input type="radio" name="dcrot" value="180"> 180°</label>
   <label><input type="radio" name="dcrot" value="270"> 270°</label>
  </div></div>
 <div class="fld"><label>像素化</label>
  <div class="radios">
   <label><input type="radio" name="dcpix" value="0" checked> 关</label>
   <label><input type="radio" name="dcpix" value="1"> 开</label>
  </div></div>
 <div class="hint"><b>深度渲染</b>区写 per-device 配置（/api/frame/device-config，与帧率同通道），
  推流端每 2s 轮询取走后在 mini 端重渲染，约 2~4s 生效；/panel「原设备深度图」格
  是同一张图，会同步变化。量程模式为「固定」时自动分位不生效，反之量程近/远端不生效；
  CLAHE 强度仅均衡=CLAHE 时生效；填充半径仅孔洞填充开启时生效。<b>深度推帧率</b>与
  RGB 各自独立节流（默认 30fps）：0=跟随 RGB 节拍；&gt;0 时按自己的节拍推送，可高于 RGB
  （RGB 节拍之间单独上报深度）。链路预算：Mac↔5090 实测约 14Mbps ≈ 1.75MB/s，
  RGB 约 135KB/帧、深度约 40KB/帧（质量 72），两路合计别把预算打满，否则整体卡顿。
  <b>深度显示</b>区只改本页背景的 CSS，拖动立即生效、只存本浏览器（localStorage）；
  旋转 90°/270° 按视口比例自动放大铺满。</div>
 <div class="sec">配置预设（存服务器）</div>
 <div class="prow"><input type="text" id="ddp_name" maxlength="40" placeholder="预设名（重名覆盖）">
  <button id="ddp_save">保存</button></div>
 <div id="ddp_list"></div>
 <div class="hint">把当前「深度渲染 + 深度显示 + 点云化」整套参数以命名快照存到服务器
  （数量不限、重名覆盖、服务重启不丢，换浏览器/展示端也能恢复）。恢复：深度渲染整套
  写回推流端（约 2~4s 生效），深度显示与点云化立即生效并记入本浏览器；删除需再点一次确认。</div>
 </div>
 <div id="dotonly" style="display:none">
 <div class="sec">点云化样式（图片类背景，本页即时）</div>
 <div class="fld"><label>点阵化</label>
  <div class="radios">
   <label><input type="radio" name="dton" value="0"> 关</label>
   <label><input type="radio" name="dton" value="1" checked> 开</label>
  </div></div>
 <div class="fld"><label>模式</label>
  <div class="radios">
   <label><input type="radio" name="dtmode" value="0"> 网格点阵</label>
   <label><input type="radio" name="dtmode" value="1" checked> 粒子云</label>
  </div></div>
 <div id="dtgrid">
 <div class="fld"><label>点距 <b id="v_dt_pitch">5</b>px</label>
  <input type="range" id="r_dt_pitch" min="3" max="16" step="1" value="5"></div>
 <div class="fld"><label>圆径占比 <b id="v_dt_r">34</b>%</label>
  <input type="range" id="r_dt_r" min="10" max="50" step="1" value="34"></div>
 <div class="fld"><label>点形状</label>
  <div class="radios">
   <label><input type="radio" name="dtgsh" value="1" checked> 圆</label>
   <label><input type="radio" name="dtgsh" value="0"> 方</label>
  </div></div>
 <div class="fld"><label>不规律排布 <b id="v_dt_jit">0</b>%</label>
  <input type="range" id="r_dt_jit" min="0" max="100" step="5" value="0"></div>
 <div class="fld"><label>运动</label>
  <div class="radios">
   <label><input type="radio" name="dtmo" value="0" checked> 关</label>
   <label><input type="radio" name="dtmo" value="1"> 漂移</label>
   <label><input type="radio" name="dtmo" value="2"> 呼吸</label>
   <label><input type="radio" name="dtmo" value="3"> 闪烁</label>
  </div></div>
 </div>
 <div id="dtcloud" style="display:none">
 <div class="fld"><label>粒子数量 <b id="v_dt_pn">50000</b></label>
  <input type="range" id="r_dt_pn" min="5000" max="80000" step="5000" value="50000"></div>
 <div class="fld"><label>粒径 <b id="v_dt_psz">3.5</b>px</label>
  <input type="range" id="r_dt_psz" min="1" max="4" step="0.5" value="3.5"></div>
 <div class="fld"><label>粒子形状</label>
  <div class="radios">
   <label><input type="radio" name="dtpsh" value="0"> 方</label>
   <label><input type="radio" name="dtpsh" value="1" checked> 圆</label>
  </div></div>
 <div class="fld"><label>密度对比 <b id="v_dt_pct">0.5</b></label>
  <input type="range" id="r_dt_pct" min="0.5" max="3" step="0.1" value="0.5"></div>
 <div class="fld"><label>单色模式</label>
  <div class="radios">
   <label><input type="radio" name="dtmono" value="0" checked> 关（取原色彩）</label>
   <label><input type="radio" name="dtmono" value="1"> 开</label>
   <input type="color" id="c_dt_pc" value="#fffdf7">
  </div></div>
 <div class="fld"><label>漂移幅度 <b id="v_dt_drift">3</b></label>
  <input type="range" id="r_dt_drift" min="0" max="3" step="0.1" value="3"></div>
 <div class="fld"><label>重生率 <b id="v_dt_resp">0</b></label>
  <input type="range" id="r_dt_resp" min="0" max="3" step="0.1" value="0"></div>
 </div>
 <div class="fld"><label>漂浮强度 <b id="v_dt_pf">0</b>%</label>
  <input type="range" id="r_dt_pf" min="0" max="100" step="5" value="0"></div>
 <div class="fld"><label>空间纵深 <b id="v_dt_pdep">100</b>%</label>
  <input type="range" id="r_dt_pdep" min="0" max="100" step="5" value="100"></div>
 <div class="fld"><label>运动速度 <b id="v_dt_spd">3</b></label>
  <input type="range" id="r_dt_spd" min="0" max="3" step="0.1" value="3"></div>
 <div class="fld"><label>点阵背景色</label>
  <div class="radios"><input type="color" id="c_dt_bg" value="#000000"></div></div>
 <div class="hint"><b>网格点阵</b>：像素格化取色 + 等距圆点（LiDAR 点阵观感），
  不规律排布=帧间稳定的逐点随机偏移，运动=漂移/呼吸/闪烁。<b>粒子云</b>：数万粒子
  按画面亮度重要性采样落点——主体稠密近连片、边缘稀疏成雾、暗区零星孤点（粉尘爆散
  观感）；粒径大小不一、伪噪声缓慢漂浮，重生率控制粒子闪换与分布跟随画面变化的速度；
  单色模式全部粒子用同一颜色（只用画面定密度），关闭则取原色彩。<b>漂浮强度</b>：
  把不同深度区块交界处的点朝暗侧吹散——边缘点持续剥离、渐隐、回炉再飞（消散观感），
  边界因此起雾不再生硬；两种模式都生效，0=关。<b>空间纵深</b>：把画面明暗当远近
  （配合近亮远暗的色彩映射最准），近点放大并随缓慢镜头摆动多移、远点反向缩小微移——
  层间视差把前后空间拉开；两种模式都生效，0=关。两种模式只作用于
  图片类背景（设备深度图、高亮/SAM3/单目点云的图模式）；GLB 点云背景请用上方
  「点渲染」区。即时生效、只存本浏览器。</div>
 </div>
 <div id="hlonly">
 <div class="sec">点云整体样式</div>
 <div class="fld"><label>俯视角 <b id="v_view_tilt">0</b>°</label>
  <input type="range" id="r_view_tilt" min="0" max="45" step="1" value="0"></div>
 <div class="fld"><label>相机距离 ×<b id="v_view_zoom">1.00</b></label>
  <input type="range" id="r_view_zoom" min="0.6" max="2.0" step="0.05" value="1"></div>
 <div class="fld"><label>附加抬升 ×<b id="v_eye_lift">0.00</b></label>
  <input type="range" id="r_eye_lift" min="0" max="0.5" step="0.05" value="0"></div>
 <div class="fld"><label>附加后撤 ×<b id="v_eye_back">0.00</b></label>
  <input type="range" id="r_eye_back" min="0" max="0.5" step="0.05" value="0"></div>
 <div class="sec">点渲染 · 形态（实时）</div>
 <div class="fld"><label>点大小 <b id="v_pt_size">1.0</b></label>
  <input type="range" id="r_pt_size" min="0.5" max="12" step="0.5" value="1"></div>
 <div class="fld"><label>点形状</label>
  <div class="radios">
   <label><input type="radio" name="ptshape" value="0" checked> 方点</label>
   <label><input type="radio" name="ptshape" value="1"> 圆点</label>
   <label><input type="radio" name="ptshape" value="2"> 柔边</label>
  </div></div>
 <div class="fld"><label>近大远小</label>
  <div class="radios">
   <label><input type="radio" name="ptatten" value="0" checked> 关（固定像素）</label>
   <label><input type="radio" name="ptatten" value="1"> 开（透视缩放）</label>
  </div></div>
 <div class="fld"><label>不透明度 ×<b id="v_pt_opacity">1.00</b></label>
  <input type="range" id="r_pt_opacity" min="0.05" max="1" step="0.05" value="1"></div>
 <div class="fld"><label>发光叠加</label>
  <div class="radios">
   <label><input type="radio" name="ptblend" value="0" checked> 关</label>
   <label><input type="radio" name="ptblend" value="1"> 开（重叠越亮）</label>
  </div></div>
 <div class="fld"><label>点密度 <b id="v_pt_density">100</b>%</label>
  <input type="range" id="r_pt_density" min="5" max="100" step="5" value="100"></div>
 <div class="sec">点渲染 · 色彩（实时）</div>
 <div class="fld"><label>色相偏移 <b id="v_pt_hue">0</b>°</label>
  <input type="range" id="r_pt_hue" min="-180" max="180" step="5" value="0"></div>
 <div class="fld"><label>饱和度 ×<b id="v_pt_sat">1.0</b></label>
  <input type="range" id="r_pt_sat" min="0" max="2.5" step="0.1" value="1"></div>
 <div class="fld"><label>明度 ×<b id="v_pt_val">1.0</b></label>
  <input type="range" id="r_pt_val" min="0.2" max="2.5" step="0.1" value="1"></div>
 <div class="fld"><label>对比度 ×<b id="v_pt_contrast">1.00</b></label>
  <input type="range" id="r_pt_contrast" min="0.5" max="2" step="0.05" value="1"></div>
 <div class="fld"><label>曝光 ×<b id="v_pt_exposure">1.35</b></label>
  <input type="range" id="r_pt_exposure" min="0.3" max="3" step="0.05" value="1.35"></div>
 <div class="fld"><label>反色</label>
  <div class="radios">
   <label><input type="radio" name="ptinvert" value="0" checked> 关</label>
   <label><input type="radio" name="ptinvert" value="1"> 开</label>
  </div></div>
 <div class="fld"><label>着色模式</label>
  <div class="radios">
   <label><input type="radio" name="ptcm" value="0" checked> 原色</label>
   <label><input type="radio" name="ptcm" value="1"> 双色调</label>
   <label><input type="radio" name="ptcm" value="2"> 按深度</label>
   <label><input type="radio" name="ptcm" value="3"> 按高度</label>
  </div></div>
 <div class="fld"><label>双色调 深→浅</label>
  <div class="radios">
   <input type="color" id="c_pt_duo_a" value="#141450">
   <input type="color" id="c_pt_duo_b" value="#ffd27f">
  </div></div>
 <div class="fld"><label>色带/雾 近端 <b id="v_pt_ramp_near">0.5</b>m</label>
  <input type="range" id="r_pt_ramp_near" min="0.2" max="5" step="0.1" value="0.5"></div>
 <div class="fld"><label>色带/雾 远端 <b id="v_pt_ramp_far">2.2</b>m</label>
  <input type="range" id="r_pt_ramp_far" min="0.3" max="6" step="0.1" value="2.2"></div>
 <div class="fld"><label>深度雾 <b id="v_pt_fog">0.00</b></label>
  <input type="range" id="r_pt_fog" min="0" max="1" step="0.05" value="0"></div>
 <div class="fld"><label>背景色</label>
  <div class="radios"><input type="color" id="c_pt_bg" value="#000000"></div></div>
 <div class="sec">点渲染 · 空间裁剪（实时）</div>
 <div class="fld"><label>深度裁剪 近 <b id="v_pt_clip_near">0.0</b>m</label>
  <input type="range" id="r_pt_clip_near" min="0" max="6" step="0.1" value="0"></div>
 <div class="fld"><label>深度裁剪 远 <b id="v_pt_clip_far">8.0</b>m</label>
  <input type="range" id="r_pt_clip_far" min="0.3" max="8" step="0.1" value="8"></div>
 <div class="fld"><label>高度裁剪 下 <b id="v_pt_clip_ylo">0.00</b></label>
  <input type="range" id="r_pt_clip_ylo" min="0" max="1" step="0.02" value="0"></div>
 <div class="fld"><label>高度裁剪 上 <b id="v_pt_clip_yhi">1.00</b></label>
  <input type="range" id="r_pt_clip_yhi" min="0" max="1" step="0.02" value="1"></div>
 <div class="sec">点渲染 · 相机与动效（实时）</div>
 <div class="fld"><label>自动旋转</label>
  <div class="radios">
   <label><input type="radio" name="ptrot" value="0" checked> 关</label>
   <label><input type="radio" name="ptrot" value="1"> 开</label>
  </div></div>
 <div class="fld"><label>旋转速度 <b id="v_pt_rotate_speed">10</b>°/s</label>
  <input type="range" id="r_pt_rotate_speed" min="2" max="60" step="1" value="10"></div>
 <div class="fld"><label>FOV 偏移 <b id="v_pt_fov_off">0</b>°</label>
  <input type="range" id="r_pt_fov_off" min="-20" max="20" step="1" value="0"></div>
 <div class="fld"><label>呼吸脉冲</label>
  <div class="radios">
   <label><input type="radio" name="ptpulse" value="0" checked> 关</label>
   <label><input type="radio" name="ptpulse" value="1"> 明度</label>
   <label><input type="radio" name="ptpulse" value="2"> 点大小</label>
  </div></div>
 <div class="fld"><label>脉冲/闪烁频率 <b id="v_pt_pulse_speed">1.0</b>Hz</label>
  <input type="range" id="r_pt_pulse_speed" min="0.2" max="3" step="0.1" value="1"></div>
 <div class="fld"><label>星光闪烁 <b id="v_pt_sparkle">0.00</b></label>
  <input type="range" id="r_pt_sparkle" min="0" max="1" step="0.05" value="0"></div>
 <div class="sec">点渲染 · 置信度联动</div>
 <div class="fld"><label>置信度→点大小 <b id="v_pt_conf_size">0.00</b></label>
  <input type="range" id="r_pt_conf_size" min="0" max="1" step="0.05" value="0"></div>
 <div class="fld"><label>置信度→透明度 <b id="v_pt_conf_alpha">0.00</b></label>
  <input type="range" id="r_pt_conf_alpha" min="0" max="1" step="0.05" value="0"></div>
 <div class="hint">与 /panel 的两张调节卡同一套配置（/api/sam3hl/config），两边改动互相可见。
  <b>俯视角/相机距离/抬升/后撤</b>是前端相机参数，拖动立即生效；高亮样式、
  <b>饱和度/明度/底色色相/置信度分位/离群裁剪</b>写进 GLB，下一轮 SAM3 结果生效
  （约 1~3 秒）、只调底色不动高亮色。<b>点渲染</b>各区直改浏览器内 three.js
  点材质/着色器，拖动立即生效、对所有 GLB 背景（高亮/SAM3/单目点云）通用，色彩类
  作用于最终画面<b>含高亮色</b>。置信度联动首次开启需等下一轮 GLB（把置信度烘进
  顶点 alpha，仅高亮/SAM3 点云有此数据），之后实时。自动旋转在每次换模时回正。
  默认值均为中性观感。</div>
 </div>
</div>

<script>
const $=id=>document.getElementById(id);
const DEMO=new URLSearchParams(location.search).get('demo');   // ?demo=1：无后端时目检布局用
// 压暗层备选方案切换：?shade=a 用 A 版（横椭圆），缺省用 CSS 里的默认 B 版（正圆）
if(new URLSearchParams(location.search).get('shade')==='a')
  document.documentElement.style.setProperty('--shade-img',"url('/static/bg-shade.png')");
// 视口缺口实测：iOS standalone 下视口偶尔短于屏幕（底部那一条页面画不进去），此时按
// 「距屏底」标注的贴底值落进视口里会凭空多抬一截。把缺口量出来写进 --vpgap，竖屏贴底
// 元素扣掉它就能真正贴到能画的最低处；视口铺满时缺口为 0、退化成标称值。
function setVpGap(){
  // screen.height - innerHeight 只说明「视口比屏幕短了多少」，分不清短在哪一头：
  //   · 状态栏 translucent：视口顶贴屏顶、env(top) 报状态栏高，短的那截全在底部
  //   · 状态栏 opaque      ：视口顶从状态栏下方起、env(top)=0，短的那截全在顶部，
  //                          底部本就贴着屏底，再补偿就会把内容顶起来
  // 故以 env(top) 是否为 0 判向：只有 translucent 那种才需要补。
  const pr=document.createElement('div');
  pr.style.cssText='position:fixed;left:0;top:0;width:0;height:0;visibility:hidden;'+
    'padding-top:env(safe-area-inset-top,0px)';
  document.body.appendChild(pr);
  const safeTop=parseFloat(getComputedStyle(pr).paddingTop)||0;
  pr.remove();
  const off=(window.visualViewport||{}).offsetTop||0;
  const g=safeTop>0?Math.max(0,(screen.height||0)-innerHeight-off):0;
  // 只在「主屏图标打开的竖屏 web app」里校正：这是唯一会出现视口短一截的场景。
  // 浏览器标签页里 screen 是整块显示器、跟视口本就不同口径（窗口没最大化就会算出
  // 一个几十上百 px 的假缺口，把内容凭空顶上去），一律不补
  const app=navigator.standalone===true||matchMedia('(display-mode:standalone)').matches
            ||matchMedia('(display-mode:fullscreen)').matches;
  const on=app&&innerHeight>innerWidth&&g>0&&g<200;   // 200 上限防异常值把内容顶飞
  document.documentElement.style.setProperty('--vpgap',(on?g:0).toFixed(1)+'px');
}
setVpGap();
addEventListener('resize',setVpGap);
addEventListener('orientationchange',()=>setTimeout(setVpGap,300));
if(window.visualViewport) visualViewport.addEventListener('resize',setVpGap);

// 几何自检：?geom=1 把视口、安全区、各锚定元素的真实落点画在屏上。手机上「看着不对」
// 十有八九是视口没铺到屏底（内容画不进去那一条），肉眼分不清是版式偏了还是视口短了，
// 拿这一屏截图回来一算就知道差在哪
function showGeom(){
  const pr=document.createElement('div');
  pr.style.cssText='position:fixed;left:0;top:0;width:0;height:0;visibility:hidden;'+
    'padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px);'+
    'padding-left:env(safe-area-inset-left,0px);padding-right:env(safe-area-inset-right,0px)';
  document.body.appendChild(pr);
  const cs=getComputedStyle(pr),vv=window.visualViewport||{};
  const r=el=>{const b=el.getBoundingClientRect();
    return `top ${b.top.toFixed(1)} bottom↕ ${(innerHeight-b.bottom).toFixed(1)} h ${b.height.toFixed(1)}`;};
  const cv=$('bgDot'),st=$('stage');
  const L=[
    ['standalone', (navigator.standalone===undefined?'-':navigator.standalone)+
      ' / display-mode:'+(matchMedia('(display-mode:standalone)').matches?'standalone':
        matchMedia('(display-mode:fullscreen)').matches?'fullscreen':'browser')],
    ['screen', `${screen.width}x${screen.height}  dpr ${devicePixelRatio}`],
    ['inner', `${innerWidth}x${innerHeight}   outerH ${outerHeight}`],
    ['visualViewport', `${(vv.width||0).toFixed(0)}x${(vv.height||0).toFixed(0)} offTop ${(vv.offsetTop||0).toFixed(0)}`],
    ['clientH', document.documentElement.clientHeight],
    ['视口底距屏底', (screen.height-innerHeight-(vv.offsetTop||0)).toFixed(1)+'  ← 不为 0 就画不到屏底'],
    ['safe-area', `T ${cs.paddingTop} B ${cs.paddingBottom} L ${cs.paddingLeft} R ${cs.paddingRight}`],
    ['1rem', getComputedStyle(document.documentElement).fontSize],
    ['--vpgap', getComputedStyle(document.documentElement).getPropertyValue('--vpgap')],
    ['#stage', r(st)],
    ['#bgDot', `css ${cv.clientWidth}x${cv.clientHeight}  buf ${cv.width}x${cv.height}`],
    ['#logo', r($('logo'))],
    ['#idle', r($('idle'))],
    ['#card', r($('card'))],
  ];
  const box=document.createElement('div');
  box.style.cssText='position:fixed;inset:0;z-index:999;background:rgba(0,0,0,.86);color:#0f0;'+
    'font:13px/1.55 ui-monospace,Menlo,monospace;padding:calc(env(safe-area-inset-top,0px) + 12px) 10px 12px;'+
    'overflow:auto;white-space:pre-wrap;word-break:break-all';
  box.textContent=L.map(([k,v])=>k.padEnd(14)+' '+v).join('\\n')+
    '\\n\\n（bottom↕ = 该元素底边距视口底；点一下关掉）';
  box.onclick=()=>box.remove();
  document.body.appendChild(box);
}
if(new URLSearchParams(location.search).get('geom')!==null) setTimeout(showGeom,1200);
// 主屏图标打开时没有地址栏、加不了 ?geom=1，留一个手势入口：连点 logo 5 下唤出自检。
// #ui 整层 pointer-events:none，故单独把 logo 打开成可点
{let n=0,t=0;
 const lg=$('logo');
 lg.style.pointerEvents='auto';
 lg.addEventListener('click',()=>{
   const now=Date.now();
   n=(now-t>3000)?1:n+1; t=now;
   if(n>=5){n=0;showGeom();}
 });}
// 竖屏压暗层现场调参：?shtop=.62&shbot=.62&shspan=24（顶/底边缘不透明度、单侧渐变跨度%）。
// 展台上手机改地址栏就能比出深浅，定稿后把值写回 CSS 的 :root 默认
{const q=new URLSearchParams(location.search);
 for(const[k,v]of[['shtop','--sh-top'],['shbot','--sh-bot'],['shspan','--sh-span']])
   if(q.get(k)!==null) document.documentElement.style.setProperty(v,q.get(k));}

// ══ 背景层：两种来源下拉框选择（临时工具），默认设备深度图（g335 硬件深度伪彩）══
// 只保留硬件深度的两条：伪彩深度帧(mini 端渲染)与真深度反投影彩色点云(devpc)；
// DA3/SAM3 派生的四种点云来源(高亮/双目高亮/单目/SAM3)已随去冗余下线
let bgSource=localStorage.getItem('exp_bg')||'devdepth';
if(!['devdepth','devpc'].includes(bgSource))bgSource='devdepth';   // 旧版存过 hl/stereo/la/s3/raw 的自动回落默认
const MIN_SWAP_MS=0;               // 换图节流已停用（同 /panel，2026-08-13）；"加载中不打断"守卫仍在
let bgFlip=false,lastBgKey='',lastMvUrl='',lastMvSwap=0,mvFov=55;

// ── GLB 背景双缓冲：front=可见实例，back=后台加载实例；load 后一帧切换 ──
let mvFlip=false;                              // false=bgmvA 为 front
const mvEls=()=>[$('bgmvA'),$('bgmvB')];
const mvFront=()=>mvFlip?$('bgmvB'):$('bgmvA');
const mvBack=()=>mvFlip?$('bgmvA'):$('bgmvB');
const mvVisible=()=>$('bgmvA').classList.contains('on')||$('bgmvB').classList.contains('on');
function hideModelLayer(){mvEls().forEach(el=>el.classList.remove('on'));lastMvUrl='';}

let lastBgUrl='';                     // 当前背景图 url（流水小窗镜像用）
function showImg(url,key){            // 双缓冲交叉淡入：新图解码完成后才切换，不闪黑
  if(!url)return false;
  if(key===lastBgKey)return true;
  lastBgKey=key;
  const im=new Image();
  im.onload=()=>{ if(key!==lastBgKey)return;      // 已被更新的帧超越则丢弃
    bgFlip=!bgFlip;
    const showEl=bgFlip?$('bgA'):$('bgB'),hideEl=bgFlip?$('bgB'):$('bgA');
    showEl.src=url;showEl.classList.add('on');hideEl.classList.remove('on');
    hideModelLayer();lastBgUrl=url;
    dotIm=im;   // 点云化样式开启时用解码完成的这张图刷新取色缓存
    if(+dotCfg.on){
      dotSample();
      $('bgDot').style.display='block';
      // 需要动画循环的模式（粒子云恒开/网格运动开）rAF 自会用新缓存重绘，
      // 中断后此处顺手复活循环；纯静态网格则重绘一次
      if(dotNeedLoop()){if(!dotRaf)dotRaf=requestAnimationFrame(dotLoop);}
      else dotDraw();
    }
  };
  im.src=url;
  return true;
}
function showModel(url,fov){          // GLB 产物：双缓冲——back 实例后台加载，load 后一帧切换（不闪空窗）
  if(fov)mvFov=fov;
  if(url===lastMvUrl){                // 已是当前目标：确保可见（从图片背景切回 GLB 时）
    if(!mvVisible()&&mvFront().loaded)mvFront().classList.add('on');
    return true;
  }
  const back=mvBack();
  // back 还在加载上一个目标时不打断（dataset.pending 在 load 时清；25s 卡死兜底同旧逻辑）
  if(back.dataset.pending&&!back.loaded&&Date.now()-lastMvSwap<25000)return true;
  if(Date.now()-lastMvSwap<MIN_SWAP_MS)return true;
  lastMvSwap=Date.now();lastMvUrl=url;lastBgKey='';
  back.dataset.pending=url;
  back.src=url;                      // 可见性切换发生在该实例的 load 事件里
  return true;
}
// GLB 加载完自动摆「调优视角」（同 /panel：略俯视、拉远，FOV 用真实相机内参）。
// 高亮点云来源时视角参数可调（抽屉「点云整体样式」，与服务端渲染同语义：绕点云中心
// 俯视 tilt、距离=|cz|×zoom、附加抬升/后撤 ×|cz|）；默认 10°/1.25/0/0 与②③取景逐位一致。
const DEF_VIEW={view_tilt:0,view_zoom:1.0,eye_lift:0,eye_back:0};  // 拍摄视角零offset
function applyExpView(mvEl){const mv=mvEl||mvFront();
  try{const c=mv.getBoundingBoxCenter();const cz=(c.z<-0.001)?c.z:-1.5;
    const v=DEF_VIEW;   // 来源只剩硬件深度两条，不再有 hl/stereo 的专用视角
    const t=v.view_tilt*Math.PI/180;
    const dy=v.view_zoom*Math.sin(t)+v.eye_lift, dz=v.view_zoom*Math.cos(t)+v.eye_back;
    mv.cameraTarget='0m 0m '+cz.toFixed(4)+'m';
    mv.cameraOrbit='0deg '+(90-Math.atan2(dy,dz)*180/Math.PI).toFixed(2)+'deg '
      +(Math.abs(cz)*Math.hypot(dy,dz)).toFixed(4)+'m';
    mv.fieldOfView=Math.min(60,Math.max(10,mvFov+(+ptStyle.pt_fov_off)))+'deg';
    mv.jumpCameraToGoal&&mv.jumpCameraToGoal();}catch(e){}}

// ── 点渲染（实时）：穿透 model-viewer 私有 Symbol("scene") 拿内部 three.js 场景，直改
// PointsMaterial + 注入着色器（形态/色彩/裁剪/动效全套调节）。glTF 的 POINTS 图元不携带
// 点大小等渲染参数——model-viewer 公开 API 不暴露，只能穿透（版本钉死 3.5.0，符号描述
// minify 后仍稳定）。每次换 GLB 材质是新实例，须在 load 事件重新应用。──
let mvSceneSym=null,ptStyle={pt_size:1,pt_shape:0,pt_atten:0,pt_opacity:1,pt_blend:0,
  pt_density:100,pt_hue:0,pt_sat:1,pt_val:1,pt_contrast:1,pt_exposure:1.35,pt_invert:0,
  pt_colormode:0,pt_duo_a:'#141450',pt_duo_b:'#ffd27f',pt_ramp_near:0.5,pt_ramp_far:2.2,
  pt_fog:0,pt_clip_near:0,pt_clip_far:8,pt_clip_ylo:0,pt_clip_yhi:1,
  pt_rotate:0,pt_rotate_speed:10,pt_fov_off:0,pt_pulse:0,pt_pulse_speed:1,pt_sparkle:0,
  pt_bg:'#000000',pt_conf_size:0,pt_conf_alpha:0};
// 材质间共享引用的 uniform 表：改 value 即生效、零重编译（vec3 用数组）
const PTU={shape:{value:0},hue:{value:0},sat:{value:1},val:{value:1},contrast:{value:1},
  invert:{value:0},cmode:{value:0},duoa:{value:[0.08,0.08,0.31]},duob:{value:[1,0.82,0.5]},
  rampn:{value:0.5},rampf:{value:2.2},fog:{value:0},bg:{value:[0,0,0]},
  clipn:{value:0},clipf:{value:8},clipylo:{value:0},clipyhi:{value:1},
  ymin:{value:0},ymax:{value:1},pulse:{value:0},pspeed:{value:1},sparkle:{value:0},
  time:{value:0},confsize:{value:0},confalpha:{value:0}};
function ptHex(h){return [parseInt(h.slice(1,3),16)/255,parseInt(h.slice(3,5),16)/255,
  parseInt(h.slice(5,7),16)/255];}
function mvScene(mv){
  if(!mvSceneSym)mvSceneSym=Object.getOwnPropertySymbols(mv).find(s=>s.description==='scene');
  return mvSceneSym?mv[mvSceneSym]:null;
}
// 注入片元：色相/饱和/明度/对比/反色 → 着色模式(双色调/深度/高度色带) → 脉冲/闪烁 →
// 深度雾 → 置信度透明 → 点形状(圆/柔边)。全部由 uniform 驱动，一次编译多路复用。
// （本页嵌在 Python 普通字符串里，JS 转义序列必须写成 \\n，否则整页脚本 SyntaxError）
const PT_FRAG='if(vPtD<uPt_clipn||vPtD>uPt_clipf)discard;\\n'
 +'float ptyn=clamp((vPtY-uPt_ymin)/max(uPt_ymax-uPt_ymin,1e-5),0.0,1.0);\\n'
 +'if(ptyn<uPt_clipylo||ptyn>uPt_clipyhi)discard;\\n'
 +'float ptcf=1.0;\\n#ifdef USE_COLOR_ALPHA\\nptcf=clamp(vColor.a,0.0,1.0);\\n#endif\\n'
 +'vec3 pc=diffuseColor.rgb;\\n'
 +'pc=mix(pc,vec3(1.0)-pc,uPt_invert);\\n'
 +'float pl=dot(pc,vec3(0.2126,0.7152,0.0722));\\n'
 +'pc=mix(vec3(pl),pc,uPt_sat)*uPt_val;\\n'
 +'float pha=radians(uPt_hue);vec3 pk=vec3(0.57735);\\n'
 +'pc=pc*cos(pha)+cross(pk,pc)*sin(pha)+pk*dot(pk,pc)*(1.0-cos(pha));\\n'
 +'pc=(pc-0.5)*uPt_contrast+0.5;\\n'
 +'float plum=dot(clamp(pc,0.0,1.0),vec3(0.2126,0.7152,0.0722));\\n'
 +'if(uPt_cmode>0.5&&uPt_cmode<1.5){pc=mix(uPt_duoa,uPt_duob,plum);}\\n'
 +'else if(uPt_cmode>1.5){\\n'
 +' float ptt=uPt_cmode<2.5?clamp((vPtD-uPt_rampn)/max(uPt_rampf-uPt_rampn,1e-4),0.0,1.0):ptyn;\\n'
 +' float pth=(1.0-ptt)*0.6667;\\n'
 +' vec3 ptr=clamp(abs(fract(pth+vec3(1.0,0.6667,0.3333))*6.0-3.0)-1.0,0.0,1.0);\\n'
 +' pc=ptr*mix(0.35,1.0,plum);}\\n'
 +'if(uPt_pulse>0.5&&uPt_pulse<1.5)pc*=1.0+0.25*sin(6.2832*uPt_pspeed*uPt_time);\\n'
 +'pc*=mix(1.0,0.35+0.65*(0.5+0.5*sin(6.2832*(uPt_pspeed*uPt_time+vPtR))),uPt_sparkle);\\n'
 +'float pfog=clamp((vPtD-uPt_rampn)/max(uPt_rampf-uPt_rampn,1e-4),0.0,1.0);\\n'
 +'pc=mix(pc,uPt_bg,pfog*uPt_fog);\\n'
 +'diffuseColor.rgb=max(pc,vec3(0.0));\\n'
 +'diffuseColor.a=opacity*mix(1.0,ptcf,uPt_confalpha);\\n'
 +'if(uPt_shape>0.5&&uPt_shape<1.5){if(length(gl_PointCoord-vec2(0.5))>0.5)discard;}\\n'
 +'else if(uPt_shape>1.5){float pdd=length(gl_PointCoord-vec2(0.5));if(pdd>0.5)discard;diffuseColor.a*=smoothstep(0.5,0.12,pdd);}\\n';
const PT_FRAG_DECL='uniform float uPt_shape;uniform float uPt_hue;uniform float uPt_sat;'
 +'uniform float uPt_val;uniform float uPt_contrast;uniform float uPt_invert;'
 +'uniform float uPt_cmode;uniform vec3 uPt_duoa;uniform vec3 uPt_duob;'
 +'uniform float uPt_rampn;uniform float uPt_rampf;uniform float uPt_fog;uniform vec3 uPt_bg;'
 +'uniform float uPt_clipn;uniform float uPt_clipf;uniform float uPt_clipylo;uniform float uPt_clipyhi;'
 +'uniform float uPt_ymin;uniform float uPt_ymax;uniform float uPt_pulse;uniform float uPt_pspeed;'
 +'uniform float uPt_sparkle;uniform float uPt_time;uniform float uPt_confalpha;'
 +'varying float vPtD;varying float vPtY;varying float vPtR;\\n';
// 注入顶点：视深/世界高度/逐点随机相位三个 varying + 置信度→点大小
const PT_VERT='vPtD=-mvPosition.z;vPtY=(modelMatrix*vec4(transformed,1.0)).y;'
 +'vPtR=fract(sin(dot(position.xyz,vec3(12.9898,78.233,37.719)))*43758.5453);\\n'
 +'#ifdef USE_COLOR_ALPHA\\ngl_PointSize*=mix(1.0,clamp(vColor.a,0.05,1.0),uPt_confsize);\\n#endif\\n';
const PT_VERT_DECL='uniform float uPt_confsize;varying float vPtD;varying float vPtY;varying float vPtR;\\n';
let ptMats=[],ptTimer=null;
function applyPtStyle(mvEl){
  const S=ptStyle;
  // ── 动效计时器启停先于场景检查：调节作用域收敛——点渲染动效只属于 GLB 来源，
  // 图片类来源（设备深度图）下必须停表，否则 50ms 改材质+queueRender 会持续驱动
  // 已隐藏的 GLB 层（「双目脉冲影响点阵画面」的泄漏根源）。挪到 early-return 之前
  // 是因为图片来源下 mvScene 可能拿不到场景、走不到原来的停表行 ──
  const tick=((+S.pt_pulse>0)||(+S.pt_sparkle>0.001))&&bgSource!=='devdepth';
  if(tick&&!ptTimer)ptTimer=setInterval(()=>{
    if(!mvVisible())return;   // GLB 层不可见（图片背景顶层）时空转不驱动渲染
    PTU.time.value=performance.now()/1000;
    if(+ptStyle.pt_pulse===2){   // 点大小脉冲：直接调材质 size（uniform 每帧自动刷新）
      const b=+ptStyle.pt_atten?ptStyle.pt_size*0.003:+ptStyle.pt_size;
      const f=1.0+0.25*Math.sin(6.2832*ptStyle.pt_pulse_speed*PTU.time.value);
      ptMats.forEach(m=>{m.size=b*f;});
    }
    const s2=mvScene(mvFront());s2&&s2.queueRender&&s2.queueRender();},50);
  if(!tick&&ptTimer){clearInterval(ptTimer);ptTimer=null;}
  const mv=mvEl||mvFront(),sc=mvScene(mv);if(!sc||!sc.traverse)return;
  // uniform 同步
  PTU.shape.value=+S.pt_shape;PTU.hue.value=+S.pt_hue;PTU.sat.value=+S.pt_sat;
  PTU.val.value=+S.pt_val;PTU.contrast.value=+S.pt_contrast;PTU.invert.value=+S.pt_invert;
  PTU.cmode.value=+S.pt_colormode;PTU.duoa.value=ptHex(S.pt_duo_a);PTU.duob.value=ptHex(S.pt_duo_b);
  PTU.rampn.value=+S.pt_ramp_near;PTU.rampf.value=+S.pt_ramp_far;PTU.fog.value=+S.pt_fog;
  PTU.bg.value=ptHex(S.pt_bg);PTU.clipn.value=+S.pt_clip_near;PTU.clipf.value=+S.pt_clip_far;
  PTU.clipylo.value=+S.pt_clip_ylo;PTU.clipyhi.value=+S.pt_clip_yhi;
  PTU.pulse.value=+S.pt_pulse;PTU.pspeed.value=+S.pt_pulse_speed;PTU.sparkle.value=+S.pt_sparkle;
  PTU.confsize.value=+S.pt_conf_size;PTU.confalpha.value=+S.pt_conf_alpha;
  // 元素级：曝光 / 自动旋转 / 背景色（fog 混向同色）
  mv.exposure=+S.pt_exposure;
  if(+S.pt_rotate){mv.setAttribute('auto-rotate','');mv.setAttribute('auto-rotate-delay','0');
    mv.setAttribute('rotation-per-second',S.pt_rotate_speed+'deg');}
  else mv.removeAttribute('auto-rotate');
  $('stage').style.background=S.pt_bg;
  ptMats=[];
  sc.traverse(o=>{
    if(!o.isPoints||!o.material)return;
    const m=o.material,g=o.geometry;
    ptMats.push(m);
    if(g&&g.attributes&&g.attributes.position){   // 点密度：drawRange 抽稀
      const n=g.attributes.position.count;
      g.setDrawRange(0,Math.max(1,Math.round(n*(+S.pt_density)/100)));
      if(!g.boundingBox)g.computeBoundingBox();   // 高度归一化范围（世界系）
      if(g.boundingBox){o.updateWorldMatrix&&o.updateWorldMatrix(true,false);
        const bb=g.boundingBox.clone().applyMatrix4(o.matrixWorld);
        PTU.ymin.value=bb.min.y;PTU.ymax.value=bb.max.y;}
    }
    // 近大远小开启时 size 是世界单位：场景深度约 1.2~1.5m，用 3mm/档 把滑条映射到米
    m.size=+S.pt_atten?S.pt_size*0.003:+S.pt_size;
    m.opacity=+S.pt_opacity;
    const trans=(+S.pt_opacity<0.999)||(+S.pt_shape===2)||(+S.pt_blend===1)||(+S.pt_conf_alpha>0.001);
    m.transparent=trans;m.depthWrite=!trans;      // 半透明点不写深度，避免自遮挡黑斑
    m.blending=(+S.pt_blend===1)?2:1;             // three 数值常量：2=Additive 1=Normal
    const att=!!+S.pt_atten;
    if(m.sizeAttenuation!==att){m.sizeAttenuation=att;m.needsUpdate=true;}  // define 变更须重编译
    if(!m.userData.ptPatched){
      m.userData.ptPatched=true;
      m.onBeforeCompile=sh=>{
        Object.keys(PTU).forEach(k=>sh.uniforms['uPt_'+k]=PTU[k]);
        sh.vertexShader=PT_VERT_DECL+sh.vertexShader.replace(
          '#include <logdepthbuf_vertex>',PT_VERT+'#include <logdepthbuf_vertex>');
        sh.fragmentShader=PT_FRAG_DECL+sh.fragmentShader.replace(
          '#include <color_fragment>','#include <color_fragment>\\n'+PT_FRAG);
      };
      m.needsUpdate=true;
    }
  });
  sc.queueRender&&sc.queueRender();
}
// 双缓冲切换点：back 实例 load 后，先摆视角+注材质，再一帧交换可见性——旧模型
// 显示到新模型完全就绪的瞬间，消掉"卸旧等新"的空窗闪。过期加载（期间目标已更新）丢弃。
mvEls().forEach(el=>el.addEventListener('load',()=>{
  const url=el.dataset.pending;
  delete el.dataset.pending;
  if(url!==lastMvUrl)return;
  applyExpView(el);applyPtStyle(el);
  const other=(el===$('bgmvA'))?$('bgmvB'):$('bgmvA');
  el.classList.add('on');other.classList.remove('on');
  mvFlip=(el===$('bgmvB'));
  $('bgA').classList.remove('on');$('bgB').classList.remove('on');
  $('bgDot').style.display='none';   // GLB 背景不适用点云化样式层，避免遮挡
}));

// 服务重启后配置清零会回落 depth 模式（识别不触发）：本页独立运行时补推默认配置（glb=识别链路）
let lastCfgPush=0;
function pushDefaultConfig(){
  if(Date.now()-lastCfgPush<5000)return;
  lastCfgPush=Date.now();
  fetch('/api/frame/config',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({export_format:'glb',process_res:504,conf_thresh_percentile:40,
                         num_max_points:800000,show_cameras:'0'})}).catch(()=>{});   // 展示页不要相机线框
}

// ── 多设备：下拉选设备（服务端只处理选中设备一路；同 /panel 的下拉逻辑） ──
// 打开默认设备：页面加载后第一次在状态里看到 g335 在线时，若未被选中则切过去一次
// （选中是服务端全局粘性态，可能粘在手机等无深度设备上——如 g335 掉线重连后的回落）。
// 只在页面加载时干预一次，此后的选择权完全交还下拉与其它页面
const PREF_DEVICE='macmini-g335';
let prefDevDone=false;
let lastDevKey='',curDev=null;
function renderDevices(s){
  const devs=s.devices||[],sel=$('selDev');
  sel.style.display=devs.length?'':'none';
  if(document.activeElement!==sel){   // 下拉展开操作中不重建选项，避免选择被打断
    const key=devs.map(d=>d.device_id).join('|')+'#'+(s.selected||'');
    if(key!==lastDevKey){lastDevKey=key;
      sel.innerHTML=devs.map(d=>'<option value="'+d.device_id+'"'
        +(d.device_id===s.selected?' selected':'')+'>'+d.device_id+'</option>').join('');}
  }
}
$('selDev').onchange=()=>{
  fetch('/api/frame/select',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({device_id:$('selDev').value})}).catch(()=>{});
};

// ── 数据源帧率（调节抽屉顶部）：per-device 配置，POST /api/frame/device-config ──
// push_fps=RGB 推帧频率（高亮/SAM3 点云链路的帧率上限）；product_interval=点云直传
// 间隔（Astra 类真深度设备，单目点云来源的刷新节奏）。只作用当前选中设备；推流端每 2s
// 轮询 /api/frame/status 取走。与 /api/frame/config 分开——那边全量覆盖且全局一份
const RATE_SLIDERS={push_fps:['r_push_fps','v_push_fps'],product_interval:['r_prod_itv','v_prod_itv']};
let ratePend={},rateTimer=null,rateTouched=0;
Object.keys(RATE_SLIDERS).forEach(key=>{
  const [rid,vid]=RATE_SLIDERS[key];
  $(rid).addEventListener('input',()=>{
    $(vid).textContent=(+$(rid).value).toFixed(1);
    if(!curDev)return;
    rateTouched=Date.now();
    ratePend[key]=+$(rid).value;
    clearTimeout(rateTimer);
    rateTimer=setTimeout(()=>{
      const cfg=ratePend;ratePend={};
      fetch('/api/frame/device-config',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({device_id:curDev,config:cfg})}).catch(()=>{});
    },300);
  });
});
function renderRate(s){
  const d=(s.devices||[]).find(x=>x.device_id===s.selected);
  // 实测到帧 fps 回显：不受回填守卫影响，随每轮 status 刷新——用户调完滑杆
  // 能直接看到推流端是否真的跟上了
  $('v_fps_meas').textContent=(d&&d.fps)?(+d.fps).toFixed(1):'--';
  // 滑条回填选中设备已下发的值（切设备跟着换）；拖动中或刚下发 1.5s 内不回写，避免打架
  if(Date.now()-rateTouched<1500)return;
  const c=(d&&d.config)||{};
  [['push_fps',c.push_fps!=null?c.push_fps:(s.config||{}).push_fps],
   ['product_interval',c.product_interval]].forEach(([key,val])=>{
    const [rid,vid]=RATE_SLIDERS[key],el=$(rid);
    if(val!=null&&document.activeElement!==el){el.value=val;$(vid).textContent=(+el.value).toFixed(1);}
  });
}

// ══ 设备深度图调节：深度渲染参数（device-config 下发到 mini 端，~2-4s 生效）
//    + 深度显示参数（本页 CSS，即时生效、localStorage 记忆） ══
// 滑条表：配置键 -> [range id, 数值显示 id, 小数位数]
const DD_SLIDERS={depth_min_m:['r_dd_min','v_dd_min',2],depth_max_m:['r_dd_max','v_dd_max',1],
  depth_auto_lo:['r_dd_lo','v_dd_lo',0],depth_auto_hi:['r_dd_hi','v_dd_hi',0],
  depth_gamma:['r_dd_gamma','v_dd_gamma',2],depth_eq_clip:['r_dd_clip','v_dd_clip',1],
  depth_fill_px:['r_dd_fillpx','v_dd_fillpx',0],depth_ema:['r_dd_ema','v_dd_ema',2],
  depth_edge:['r_dd_edge','v_dd_edge',2],depth_contour_m:['r_dd_contour','v_dd_contour',2],
  depth_jpeg_q:['r_dd_jq','v_dd_jq',0],depth_fps:['r_dd_fps','v_dd_fps',1]};
// radio 组：name -> 配置键（数值型 radio 下发数字，枚举型下发字符串）
const DD_RADIOS={ddinv:'depth_invert',ddar:'depth_autorange',ddeq:'depth_eq',
  ddfill:'depth_fill',ddsm:'depth_smooth'};
let ddPend={},ddTimer=null,ddTouched=0;
function ddPush(k,v){
  if(!curDev)return;
  ddTouched=Date.now();
  ddPend[k]=v;
  clearTimeout(ddTimer);
  ddTimer=setTimeout(()=>{
    const cfg=ddPend;ddPend={};
    fetch('/api/frame/device-config',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({device_id:curDev,config:cfg})}).catch(()=>{});
  },300);
}
Object.keys(DD_SLIDERS).forEach(k=>{
  const [rid,vid,dp]=DD_SLIDERS[k];
  $(rid).addEventListener('input',()=>{
    $(vid).textContent=(+$(rid).value).toFixed(dp);
    ddPush(k,+$(rid).value);
  });
});
Object.keys(DD_RADIOS).forEach(nm=>document.querySelectorAll('input[name='+nm+']').forEach(r=>
  r.addEventListener('change',()=>{
    const v=document.querySelector('input[name='+nm+']:checked').value;
    ddPush(DD_RADIOS[nm],isNaN(+v)?v:+v);   // 枚举值传字符串，0/1 开关传数字
  })));
$('dd_cmap').addEventListener('change',()=>ddPush('depth_colormap',$('dd_cmap').value));
$('c_dd_invalid').addEventListener('input',()=>ddPush('depth_invalid_color',$('c_dd_invalid').value));
function renderDdCfg(s){
  // 从 status 回填选中设备已下发的深度渲染配置（切设备跟着换）；
  // 拖动中或刚下发 1.5s 内不回写，避免打架（同帧率滑条的守卫逻辑）
  if(Date.now()-ddTouched<1500)return;
  const d=(s.devices||[]).find(x=>x.device_id===s.selected);
  fillDdControls((d&&d.config)||{});
}
// 深度渲染控件回填（status 轮询与预设恢复两处复用）：只填传入的键，跳过正被操作的控件
function fillDdControls(c){
  Object.keys(DD_SLIDERS).forEach(k=>{
    const [rid,vid,dp]=DD_SLIDERS[k],el=$(rid);
    if(c[k]!=null&&document.activeElement!==el){el.value=c[k];$(vid).textContent=(+el.value).toFixed(dp);}
  });
  Object.keys(DD_RADIOS).forEach(nm=>{
    const k=DD_RADIOS[nm];
    if(c[k]!=null){
      const el=document.querySelector('input[name='+nm+'][value="'+c[k]+'"]');
      if(el&&!el.checked)el.checked=true;
    }
  });
  if(c.depth_colormap&&document.activeElement!==$('dd_cmap'))$('dd_cmap').value=c.depth_colormap;
  if(/^#[0-9a-fA-F]{6}$/.test(c.depth_invalid_color||''))$('c_dd_invalid').value=c.depth_invalid_color;
}
// ── 深度显示（本页 CSS 即时层）：只在 devdepth 来源挂到背景 img，切走来源即摘除 ──
// 默认=服务器「默认」配置预设口径（2026-08-17 调定）：亮度2.5/对比1/饱和1.05/镜像开
let ddCss={br:2.5,gl:0.6,ct:1,sa:1.05,hu:0,inv:0,bl:0,op:1,fit:'cover',mir:1,rot:0,pix:0};
try{Object.assign(ddCss,JSON.parse(localStorage.getItem('exp_dd_css')||'{}'));}catch(e){}
// 深度显示颜色链（对比→饱和→色相→反相，与 CSS filter 同序同义）合成为
// 一个 3×3 颜色矩阵+偏移，供 dotSample 烘进取色缓存。canvas 层因此不挂颜色类
// CSS 滤镜——合成器偶发漏套滤镜的帧不再产生可见亮度跳变（全屏暗闪根因）。
// 亮度不进矩阵：线性乘法在饱和伪彩上截断冲白（亮度拉不动、颜色发灰），改走
// ddBrightLut 的软肩曲线单独烘焙。恒等变换返回 null（零开销）
function ddColorMatrix(){
  if(bgSource!=='devdepth')return null;
  const ct=Math.max(0,+ddCss.ct||1),sa=Math.max(0,+ddCss.sa||1);
  const hu=(+ddCss.hu||0)*Math.PI/180,inv=+ddCss.inv?1:0;
  let M=[1,0,0,0,1,0,0,0,1],off=[0,0,0];
  function mul(A,ao){
    const R=new Array(9),ro=new Array(3);
    for(let r=0;r<3;r++){
      for(let c=0;c<3;c++)R[r*3+c]=A[r*3]*M[c]+A[r*3+1]*M[3+c]+A[r*3+2]*M[6+c];
      ro[r]=A[r*3]*off[0]+A[r*3+1]*off[1]+A[r*3+2]*off[2]+ao[r];
    }
    M=R;off=ro;
  }
  if(ct!==1){const o=255*(0.5-ct/2);mul([ct,0,0,0,ct,0,0,0,ct],[o,o,o]);}
  if(sa!==1)mul([0.213+0.787*sa,0.715-0.715*sa,0.072-0.072*sa,
                 0.213-0.213*sa,0.715+0.285*sa,0.072-0.072*sa,
                 0.213-0.213*sa,0.715-0.715*sa,0.072+0.928*sa],[0,0,0]);
  if(hu){const c=Math.cos(hu),s=Math.sin(hu);
    mul([0.213+c*0.787-s*0.213,0.715-c*0.715-s*0.715,0.072-c*0.072+s*0.928,
         0.213-c*0.213+s*0.143,0.715+c*0.285+s*0.140,0.072-c*0.072-s*0.283,
         0.213-c*0.213-s*0.787,0.715-c*0.715+s*0.715,0.072+c*0.928+s*0.072],[0,0,0]);}
  if(inv)mul([-1,0,0,0,-1,0,0,0,-1],[255,255,255]);
  const ident=M[0]===1&&M[4]===1&&M[8]===1&&!M[1]&&!M[2]&&!M[3]&&!M[5]&&!M[6]&&!M[7]
    &&!off[0]&&!off[1]&&!off[2];
  return ident?null:[M,off];
}
// 亮度软肩曲线查找表（按 max 通道索引）：v'=1-(1-v)^br。br>1 时暗区/中间调
// 实打实抬升、顶端平滑收敛永不硬截断（保住亮部层次）；三通道按 K 等比缩放，
// 色相/饱和不动，不再有线性乘法的截断冲白。
// W=掺白权重：源头 max 通道已顶到 255 的像素（本机深度伪彩里近距离暖色区约
// 占 6.5%）K 恒为 1——那是该色相在 sRGB 的亮度上限，等比缩放一点都推不动，
// 只能靠掺白（过曝辉光）变亮，否则调亮度时那一坨橙色纹丝不动（2026-08-17
// 现场实测：br 1→3 蓝色中间调 +103%，顶格橙色区仅 +13%）。门控用源头 m 而非
// 输出 v'——用 v' 时高亮度下中间调也被判为"亮"而误吃掺白，饱和度掉到 0.60；
// 改用 m^8 后中间调饱和稳在 0.98，辉光只打真正到顶的像素。强度由「亮核辉光」
// 滑杆（ddCss.gl）控制，0=不掺白（橙色区就会回到"不响应亮度"）。
function ddBrightLut(){
  if(bgSource!=='devdepth')return null;
  const br=Math.max(0.1,+ddCss.br||1),hd=Math.max(0,+ddCss.gl||0);
  if(br===1&&!hd)return null;
  const K=new Float32Array(256),W=new Float32Array(256);
  const g=Math.max(0,1-1/br);   // 辉光随亮度增大而增强；br<=1 时恒 0
  for(let m=0;m<256;m++){
    const v2=1-Math.pow(1-m/255,br);
    K[m]=m?255*v2/m:0;
    W[m]=hd*g*Math.pow(m/255,8);
  }
  return [K,W];
}
function applyDdCss(){
  const on=bgSource==='devdepth';
  // 点云化 canvas 层也套同一组显示参数：devdepth+点阵组合时调节仍可见。
  // 例外——颜色类滤镜（亮度/对比/饱和/色相/反相）对 canvas 走像素烘焙（见
  // ddColorMatrix/dotSample），canvas 的 CSS filter 只留模糊与不透明度
  [$('bgA'),$('bgB'),$('bgDot')].forEach(el=>{
    if(!on){el.style.filter='';el.style.objectFit='';el.style.transform='';el.style.imageRendering='';return;}
    // 不透明度用 filter 的 opacity()：inline opacity 会盖掉 .bg.on 的交叉淡入
    if(el.id==='bgDot')
      el.style.filter=((+ddCss.bl?'blur('+ddCss.bl+'px) ':'')
        +(+ddCss.op!==1?'opacity('+ddCss.op+')':'')).trim();
    else
      el.style.filter='brightness('+ddCss.br+') contrast('+ddCss.ct+') saturate('+ddCss.sa
        +') hue-rotate('+ddCss.hu+'deg)'+(+ddCss.inv?' invert(1)':'')
        +(+ddCss.bl?' blur('+ddCss.bl+'px)':'')+' opacity('+ddCss.op+')';
    el.style.objectFit=ddCss.fit;
    // 旋转 90°/270° 宽高互换：按视口长宽比放大补偿，保证铺满不露黑边
    const r=+ddCss.rot;
    const sc=(r===90||r===270)?Math.max(innerWidth/innerHeight,innerHeight/innerWidth):1;
    el.style.transform=(r?'rotate('+r+'deg)':'')+(+ddCss.mir?' scaleX(-1)':'')
      +(sc!==1?' scale('+sc.toFixed(3)+')':'');
    el.style.imageRendering=+ddCss.pix?'pixelated':'';
  });
  // 亮度增益烘焙进取色缓存：换亮度后立即重建缓存并重绘（页面初始化时 dotIm 尚无则跳过）
  if(+dotCfg.on&&on&&dotIm){
    dotSample();
    if(dotNeedLoop()){if(!dotRaf)dotRaf=requestAnimationFrame(dotLoop);}
    else dotDraw(performance.now()/1000);
  }
}
const DC_SLIDERS={br:['r_dc_bright','v_dc_bright',2],gl:['r_dc_glow','v_dc_glow',2],
  ct:['r_dc_contrast','v_dc_contrast',2],
  sa:['r_dc_sat','v_dc_sat',2],hu:['r_dc_hue','v_dc_hue',0],bl:['r_dc_blur','v_dc_blur',0],
  op:['r_dc_opacity','v_dc_opacity',2]};
const DC_RADIOS={dcinv:'inv',dcfit:'fit',dcmir:'mir',dcrot:'rot',dcpix:'pix'};
function saveDdCss(){localStorage.setItem('exp_dd_css',JSON.stringify(ddCss));}
Object.keys(DC_SLIDERS).forEach(k=>{
  const [rid,vid,dp]=DC_SLIDERS[k];
  $(rid).addEventListener('input',()=>{
    ddCss[k]=+$(rid).value;$(vid).textContent=(+$(rid).value).toFixed(dp);
    applyDdCss();saveDdCss();
  });
});
Object.keys(DC_RADIOS).forEach(nm=>document.querySelectorAll('input[name='+nm+']').forEach(r=>
  r.addEventListener('change',()=>{
    const v=document.querySelector('input[name='+nm+']:checked').value;
    ddCss[DC_RADIOS[nm]]=(nm==='dcfit')?v:+v;
    applyDdCss();saveDdCss();
  })));
// ddCss 当前值回填显示层控件（页面加载与预设恢复两处复用）
function fillDcControls(){
  Object.keys(DC_SLIDERS).forEach(k=>{
    const [rid,vid,dp]=DC_SLIDERS[k];
    $(rid).value=ddCss[k];$(vid).textContent=(+ddCss[k]).toFixed(dp);
  });
  Object.keys(DC_RADIOS).forEach(nm=>{
    const el=document.querySelector('input[name='+nm+'][value="'+ddCss[DC_RADIOS[nm]]+'"]');
    if(el)el.checked=true;
  });
}
fillDcControls();   // localStorage 记忆值回填（页面加载一次）
addEventListener('resize',applyDdCss);   // 旋转补偿量随视口比例变化

// ══ 点云化样式（图片类背景通用形态层，两种模式）：
//    · 网格点阵——像素格化取色 + 逐格画圆，可加逐点稳定随机抖动与动效（LiDAR 点阵观感）；
//    · 粒子云——持久粒子池 + 按图像亮度重要性采样定位：主体处粒子稠密近连片、
//      边缘快速稀疏成雾、低权重区留零星孤点（粉尘爆散观感，参考图语义），
//      粒径 1~2px 大小不一、伪噪声缓慢漂移、寿命重生跟随画面内容变化。
//    深度帧更新只刷新取色缓存，与 rAF 动画解耦互不打断。
//    对所有走 showImg 的图片类来源统一生效；GLB 背景不适用（有自己的点渲染区）══
// 默认=服务器「默认」配置预设口径（2026-08-17 调定）：粒子云模式、5万粒子、
// 粒径3.5、圆粒、密度对比0.5、漂移3、重生0、速度3、纵深100%
let dotCfg={on:1,mode:1,pitch:5,r:34,jitter:0,motion:0,speed:3,bg:'#000000',gshape:1,
  pn:50000,psize:3.5,pcontrast:0.5,pmono:0,pcolor:'#fffdf7',pdrift:3,prespawn:0,pfloat:0,pshape:1,
  pdepth:100};
try{Object.assign(dotCfg,JSON.parse(localStorage.getItem('exp_dot')||'{}'));}catch(e){}
let dotIm=null,dotOffCv=null;        // 最近一帧背景 Image / 离屏降采样画布（复用）
let dotData=null,dotCols=0,dotRows=0;// 降采样取色缓存：动画重绘不重复采样
let inkT=0,inkMass=0,inkN=0;         // 诊断记账：每绘制 tick 的墨量(Σ粒径²×α)与粒/点数
function dotSample(){
  // 新帧/视口/点距变化时重建取色缓存：源图按 cover 构图居中裁剪，缩到 cols×rows，
  // getImageData 一次取回全部格子颜色。网格模式格距=点距；粒子云用固定 8px 采样格
  //（颜色/权重查询网格，与粒子密度无关）
  const im=dotIm,cv=$('bgDot');
  if(!im||!im.naturalWidth)return false;
  const dpr=devicePixelRatio||1;
  const W=Math.round(innerWidth*dpr),H=Math.round(innerHeight*dpr);
  if(cv.width!==W||cv.height!==H){cv.width=W;cv.height=H;}
  const P=(+dotCfg.mode===1?8:Math.max(2,+dotCfg.pitch))*dpr;
  dotCols=Math.max(1,Math.round(W/P));dotRows=Math.max(1,Math.round(H/P));
  if(!dotOffCv)dotOffCv=document.createElement('canvas');
  dotOffCv.width=dotCols;dotOffCv.height=dotRows;
  const ar=W/H,iar=im.naturalWidth/im.naturalHeight;
  let sx,sy,sw,sh;
  if(iar>ar){sh=im.naturalHeight;sw=sh*ar;sx=(im.naturalWidth-sw)/2;sy=0;}
  else{sw=im.naturalWidth;sh=sw/ar;sx=0;sy=(im.naturalHeight-sh)/2;}
  const octx=dotOffCv.getContext('2d',{willReadFrequently:true});
  octx.drawImage(im,sx,sy,sw,sh,0,0,dotCols,dotRows);
  dotData=octx.getImageData(0,0,dotCols,dotRows).data;
  // 深度显示颜色链烘进像素（Uint8ClampedArray 自动截断），canvas 层不再挂
  // 颜色类 CSS 滤镜——合成器偶发"漏套滤镜"的帧从此与正常帧无差别（全屏暗闪
  // 根因，2026-08-14）。对比/饱和/色相/反相走矩阵；亮度走软肩曲线+高光辉光
  //（ddBrightLut，2026-08-17）——顺序：先矩阵后亮度
  const cm=ddColorMatrix();
  if(cm){const d=dotData,M=cm[0],off=cm[1];
    for(let i=0;i<d.length;i+=4){
      const r=d[i],g=d[i+1],b=d[i+2];
      d[i]  =M[0]*r+M[1]*g+M[2]*b+off[0];
      d[i+1]=M[3]*r+M[4]*g+M[5]*b+off[1];
      d[i+2]=M[6]*r+M[7]*g+M[8]*b+off[2];
    }}
  const bl=ddBrightLut();
  if(bl){const d=dotData,K=bl[0],W=bl[1];
    for(let i=0;i<d.length;i+=4){
      const r=d[i],g=d[i+1],b=d[i+2],m=Math.max(r,g,b);
      const k=K[m],w=W[m];
      d[i]  =r*k+(255-r*k)*w;
      d[i+1]=g*k+(255-g*k)*w;
      d[i+2]=b*k+(255-b*k)*w;
    }}
  dotEdge=null;
  if(+dotCfg.pfloat>0)dotEdgeBuild();   // 漂浮强度开启时随取色缓存一并重建边缘场
  return true;
}
// ── 边缘场：每格 [强度, 外散方向x, 外散方向y]。强度=与四邻的色差（不同深度区块的
//    交界处高），方向=亮度梯度反方向（朝更暗一侧；梯度太小则逐格稳定随机方向）。
//    供「漂浮强度」把边缘的点吹散——消散观感的几何基础 ──
let dotEdge=null;
function dotEdgeBuild(){
  const d=dotData;if(!d)return;
  dotEdge=new Float32Array(dotCols*dotRows*3);
  const lum=o=>d[o]*0.2126+d[o+1]*0.7152+d[o+2]*0.0722;
  for(let y=1;y<dotRows-1;y++)for(let x=1;x<dotCols-1;x++){
    const i=y*dotCols+x,o=i*4,oL=o-4,oR=o+4,oU=o-dotCols*4,oD=o+dotCols*4;
    let diff=0;
    for(let c=0;c<3;c++)diff+=Math.abs(d[oR+c]-d[oL+c])+Math.abs(d[oD+c]-d[oU+c]);
    const e=Math.min(1,diff/220);
    if(e<0.05)continue;
    let nx=lum(oL)-lum(oR),ny=lum(oU)-lum(oD);
    const m=Math.hypot(nx,ny);
    if(m>6){nx/=m;ny/=m;}
    else{const a=dotHash(i)*6.28318;nx=Math.cos(a);ny=Math.sin(a);}
    const b=i*3;dotEdge[b]=e;dotEdge[b+1]=nx;dotEdge[b+2]=ny;
  }
  // 膨胀两轮：把边缘场向两侧各扩一格（强度按 0.55/格衰减、方向继承最强邻居）——
  // 参与消散的是一条带而不是一条线，量感和「炸」的规模由此而来
  for(let pass=0;pass<2;pass++){
    const src=dotEdge;dotEdge=new Float32Array(src.length);dotEdge.set(src);
    for(let y=1;y<dotRows-1;y++)for(let x=1;x<dotCols-1;x++){
      const i=y*dotCols+x,b=i*3;
      let best=src[b]; let bi=-1;
      for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){
        if(!dx&&!dy)continue;
        const nb=((y+dy)*dotCols+x+dx)*3,v=src[nb]*0.55;
        if(v>best){best=v;bi=nb;}
      }
      if(bi>=0){dotEdge[b]=best;dotEdge[b+1]=src[bi+1];dotEdge[b+2]=src[bi+2];}
    }
  }
}
function dotHash(i){
  // 逐点稳定伪随机 [0,1)：只随下标变、帧间不变——抖动排布不逐帧乱跳
  const x=Math.sin(i*127.1+311.7)*43758.5453;
  return x-Math.floor(x);
}
// ── 网格点阵模式 ──
function dotDraw(tSec){
  const cv=$('bgDot');
  if(!dotData&&!dotSample())return;
  const ctx=cv.getContext('2d');
  const dpr=devicePixelRatio||1,P=Math.max(2,+dotCfg.pitch)*dpr;
  const baseR=P*(+dotCfg.r)/100,jit=P*(+dotCfg.jitter)/100*0.5;  // 抖动上限=半格
  const mode=+dotCfg.motion,spd=+dotCfg.speed,t=tSec||0,TAU=6.28318,d=dotData;
  const pf=(+dotCfg.pfloat)/100;
  // 空间纵深：把亮度当远近（配合近亮远暗色彩映射），缓慢镜头摆动做层间视差
  const pd=(+dotCfg.pdepth)/100;
  const swx=pd?Math.sin(t*0.35)*70*dpr*pd:0,swy=pd?Math.cos(t*0.22)*42*dpr*pd:0;
  let ink=0,inkn=0;
  ctx.clearRect(0,0,cv.width,cv.height);
  for(let ry=0;ry<dotRows;ry++)for(let rx=0;rx<dotCols;rx++){
    const idx=ry*dotCols+rx,o=idx*4;
    const h1=dotHash(idx),h2=dotHash(idx+7919);
    let x=(rx+0.5)*P+(h1-0.5)*2*jit,y=(ry+0.5)*P+(h2-0.5)*2*jit;
    let r=baseR,al=1;
    if(mode===1){        // 漂移：逐点相位的缓慢圆游（幅度约 1/4 格）
      x+=Math.sin(t*spd+h1*TAU)*P*0.25;y+=Math.cos(t*spd*0.9+h2*TAU)*P*0.25;
    }else if(mode===2){  // 呼吸：半径正弦，逐点相位错开
      r*=1+0.35*Math.sin(t*spd*2+h1*TAU);
    }else if(mode===3){  // 闪烁：透明度逐点随机相位
      al=0.45+0.55*(0.5+0.5*Math.sin(t*spd*3+h1*TAU));
    }
    if(pf>0&&dotEdge){   // 边缘消散：边缘格的点沿外散方向飘出，越远越淡越小（缓慢起伏）
      const eb=idx*3,e=dotEdge[eb];
      if(e>0.05){
        // 飞行上限随强度非线性抬升：低档轻雾（~3格），拉满真炸（~11格）
        const h3=dotHash(idx+3571),F=2+9*pf;
        const fly=e*pf*F*(0.25+0.75*h3)*(1+0.3*Math.sin(t*spd*0.8+h3*TAU));
        x+=dotEdge[eb+1]*fly*P;y+=dotEdge[eb+2]*fly*P;
        al*=1-Math.min(0.85,fly/F*0.9);r*=1-0.35*Math.min(1,fly/F);
      }
    }
    if(pd){   // 近点放大并随摆动多移、远点反向缩小微移——前后层滑开
      const dz=(d[o]*0.2126+d[o+1]*0.7152+d[o+2]*0.0722)/255-0.45;
      x+=dz*swx;y+=dz*swy;r*=Math.max(0.3,1+dz*1.3*pd);
    }
    if(r<=0.2)continue;
    ink+=r*r*al;inkn++;   // 诊断记账：亮度不计入——只量"画了多少"
    ctx.fillStyle='rgba('+d[o]+','+d[o+1]+','+d[o+2]+','+al.toFixed(3)+')';
    if(+dotCfg.gshape===1){ctx.beginPath();ctx.arc(x,y,r,0,TAU);ctx.fill();}
    else ctx.fillRect(x-r,y-r,r*2,r*2);
  }
  inkT=performance.now();inkMass=ink;inkN=inkn;
}
// ── 粒子云模式：持久粒子池（x,y,相位,粒径因子 ×4 float）──
let pcPool=null,pcN=0,pcSig='',pcW=0,pcH=0;
function pcLum(gx,gy){   // 格子亮度权重 [0,1]：亮/有效处高，黑/无效处 0
  const o=(gy*dotCols+gx)*4;
  return (dotData[o]*0.2126+dotData[o+1]*0.7152+dotData[o+2]*0.0722)/255;
}
function pcSpawn(i){
  // 重要性采样定位：拒绝采样 10 次——亮处高概率落点（稠密近连片），全不中则
  // 落随机位置（低权重区零星孤点，构成雾状边缘）
  const b=i*4,cv=$('bgDot'),gamma=+dotCfg.pcontrast;
  let x=Math.random()*cv.width,y=Math.random()*cv.height;
  if(dotData){
    for(let k=0;k<10;k++){
      const gx=Math.floor(Math.random()*dotCols),gy=Math.floor(Math.random()*dotRows);
      if(Math.random()<Math.pow(pcLum(gx,gy),gamma)){
        x=(gx+Math.random())*cv.width/dotCols;
        y=(gy+Math.random())*cv.height/dotRows;break;}
    }
  }
  pcPool[b]=x;pcPool[b+1]=y;
  pcPool[b+2]=Math.random()*6.28318;          // 漂移相位（逐粒子随机、恒定）
  pcPool[b+3]=0.35+Math.random()*0.65;        // 粒径因子：大小不一
}
function pcEnsure(){
  // 池分配/全量重生：数量或密度对比变化时整池重建（分布立即跟新参数）
  const cv=$('bgDot');
  const sig=Math.round(+dotCfg.pn)+'|'+(+dotCfg.pcontrast);
  if(pcPool&&pcSig===sig){
    // 画布尺寸变化（进出全屏/改窗口）：粒子位置等比缩放到新尺寸立即铺满。
    // 不能指望寿命重生补位——重生率可调到 0，粒子会永远留在旧窗口区域
    if(cv.width!==pcW||cv.height!==pcH){
      const kx=cv.width/(pcW||cv.width),ky=cv.height/(pcH||cv.height);
      for(let i=0;i<pcN;i++){pcPool[i*4]*=kx;pcPool[i*4+1]*=ky;}
      pcW=cv.width;pcH=cv.height;
    }
    return;
  }
  pcSig=sig;pcN=Math.round(+dotCfg.pn);
  pcPool=new Float32Array(pcN*4);
  for(let i=0;i<pcN;i++)pcSpawn(i);
  pcW=cv.width;pcH=cv.height;
}
// ── 粒子云 ImageData 直写渲染：逐粒子 fillStyle 字符串解析+fillRect（外加拖尾三倍
//    过绘）在粒子数拉高后一帧要十几万次 Canvas 状态切换，主线程被打满、帧率崩塌，
//    深度帧轮询/取色缓存全被饿死——画面整体"卡住"。改成 typed array 手写像素、
//    整帧一次 putImageData，粒子数拉满也稳住帧率 ──
let pcImg=null,pcBuf=null,pcMaskCache={},pcPrevT=0;
function pcMask(s){
  // 圆形蒙版偏移缓存（整数粒径 s → 圆内像素 [dx,dy] 扁平表）；s<3 时圆=方不走蒙版
  let m=pcMaskCache[s];if(m)return m;
  m=[];const c=(s-1)/2,r2=(s/2+0.1)*(s/2+0.1);
  for(let dy=0;dy<s;dy++)for(let dx=0;dx<s;dx++){
    const ax=dx-c,ay=dy-c;
    if(ax*ax+ay*ay<=r2){m.push(dx);m.push(dy);}
  }
  pcMaskCache[s]=m;return m;
}
function pcPx(idx,cr,cg,cb,a){
  // 单像素 src-over：不透明或落在空像素直接写；半透明叠加时按通道混合
  const dst=pcBuf[idx];
  if(a===255||dst===0){pcBuf[idx]=(a<<24)|(cb<<16)|(cg<<8)|cr;return;}
  const da=dst>>>24,ia=255-a;
  const dr=dst&255,dg=(dst>>>8)&255,db=(dst>>>16)&255;
  pcBuf[idx]=(Math.max(da,a)<<24)|(((cb*a+db*ia)/255|0)<<16)|(((cg*a+dg*ia)/255|0)<<8)|((cr*a+dr*ia)/255|0);
}
function pcStamp(x,y,s,cr,cg,cb,al,round,W,H){
  // 以 (x,y) 为中心盖一颗 s×s 的粒子章（方/圆），越界部分逐像素裁剪
  const x0=Math.round(x-s/2),y0=Math.round(y-s/2);
  if(x0+s<=0||y0+s<=0||x0>=W||y0>=H)return;
  const a=al>=1?255:(al<=0?0:(al*255)|0);
  if(!a)return;
  if(round&&s>=3){
    const m=pcMask(s);
    for(let k=0;k<m.length;k+=2){
      const px=x0+m[k],py=y0+m[k+1];
      if(px>=0&&px<W&&py>=0&&py<H)pcPx(py*W+px,cr,cg,cb,a);
    }
  }else{
    for(let dy=0;dy<s;dy++){
      const py=y0+dy;if(py<0||py>=H)continue;
      const row=py*W;
      for(let dx=0;dx<s;dx++){
        const px=x0+dx;if(px>=0&&px<W)pcPx(row+px,cr,cg,cb,a);
      }
    }
  }
}
function pcDraw(t){
  const cv=$('bgDot');
  if(!dotData&&!dotSample())return;
  pcEnsure();
  const ctx=cv.getContext('2d'),dpr=devicePixelRatio||1;
  const W=cv.width,H=cv.height;
  if(!pcImg||pcImg.width!==W||pcImg.height!==H){
    pcImg=ctx.createImageData(W,H);
    pcBuf=new Uint32Array(pcImg.data.buffer);
  }
  pcBuf.fill(0);
  let ink=0,inkn=0;
  const mono=+dotCfg.pmono===1,spd=+dotCfg.speed;
  const amp=(+dotCfg.pdrift)*1.6*dpr,psz=(+dotCfg.psize)*dpr;
  const round=+dotCfg.pshape===1;
  const gw=W/dotCols,gh=H/dotRows,d=dotData;
  // 寿命重生按真实时间走（基础 12%/秒×重生率）：旧实现按帧固定 0.4%，帧率一低
  // 每秒重生数等比暴跌，画面内容变了粒子还赖在旧位置——"部分点停在原地"的根源
  const dt=Math.min(0.25,Math.max(0.001,t-pcPrevT));pcPrevT=t;
  const re=Math.max(1,Math.round(pcN*0.12*(+dotCfg.prespawn)*dt));
  for(let k=0;k<re;k++)pcSpawn(Math.floor(Math.random()*pcN));
  const pf=(+dotCfg.pfloat)/100;
  // 空间纵深：亮度当远近，缓慢镜头摆动做层间视差（近多移远反向）+近大远小
  const pd=(+dotCfg.pdepth)/100;
  const swx=pd?Math.sin(t*0.35)*70*dpr*pd:0,swy=pd?Math.cos(t*0.22)*42*dpr*pd:0;
  let mr=255,mg=253,mb=247;
  if(mono){const c=dotCfg.pcolor;
    mr=parseInt(c.slice(1,3),16);mg=parseInt(c.slice(3,5),16);mb=parseInt(c.slice(5,7),16);}
  for(let i=0;i<pcN;i++){
    const b=i*4,ph=pcPool[b+2];
    // 伪噪声漂移：两组不同频正弦叠加、逐粒子相位——悬浮微尘的缓慢无序运动
    const x=pcPool[b]+Math.sin(t*spd*0.7+ph)*amp+Math.sin(t*spd*0.31+ph*2.7)*amp*0.6;
    const y=pcPool[b+1]+Math.cos(t*spd*0.6+ph*1.7)*amp+Math.cos(t*spd*0.23+ph*3.1)*amp*0.5;
    let cr=mr,cg=mg,cb=mb,dz=0;
    if(!mono||pd){   // 保留原色彩（取飘散前位置）；空间纵深也需取格子亮度当深度
      const gx=Math.min(dotCols-1,Math.max(0,(x/gw)|0));
      const gy=Math.min(dotRows-1,Math.max(0,(y/gh)|0));
      const o=(gy*dotCols+gx)*4;
      if(!mono){cr=d[o];cg=d[o+1];cb=d[o+2];}
      if(pd)dz=(d[o]*0.2126+d[o+1]*0.7152+d[o+2]*0.0722)/255-0.45;
    }
    let al=1,ex=0,ey=0;
    if(pf>0&&dotEdge){
      // 边缘消散：落在边缘格的粒子沿外散方向持续剥离——锯齿进度循环
      //（飞出→渐隐→回炉再飞），带着起点颜色飘进暗侧，深度区块交界因此起雾
      const gx0=Math.min(dotCols-1,Math.max(0,(pcPool[b]/gw)|0));
      const gy0=Math.min(dotRows-1,Math.max(0,(pcPool[b+1]/gh)|0));
      const eb=(gy0*dotCols+gx0)*3,e=dotEdge[eb];
      if(e>0.05){
        const h=ph*0.159155;   // 相位归一化 [0,1) 作逐粒子稳定随机
        const prog=(t*spd*(0.1+0.2*h)+h*7)%1;
        // 飞行距离随强度非线性抬升（拉满约 300px×dpr），加垂直向摆动让轨迹带弧度
        const dist=e*(30+270*pf*pf+120*pf*h)*dpr*prog;
        const sway=Math.sin(prog*9+ph)*dist*0.18;
        ex=dotEdge[eb+1]*dist-dotEdge[eb+2]*sway;
        ey=dotEdge[eb+2]*dist+dotEdge[eb+1]*sway;
        al=1-prog*0.9;
      }
    }
    const pxo=dz*swx,pyo=dz*swy;   // 视差偏移只挪位置，不当作消散（不触发拖尾）
    const s=Math.max(1,Math.round(pcPool[b+3]*psz*(pd?Math.max(0.35,1+dz*1.3*pd):1)));
    ink+=s*s*al;inkn++;   // 诊断记账
    pcStamp(x+ex+pxo,y+ey+pyo,s,cr,cg,cb,al,round,W,H);
    if(ex||ey){   // 拖尾：沿来路补两颗渐淡渐小的点，扫出消散的流线感
      pcStamp(x+ex*0.75+pxo,y+ey*0.75+pyo,Math.max(1,Math.round(s*0.85)),cr,cg,cb,al*0.5,round,W,H);
      pcStamp(x+ex*0.5+pxo,y+ey*0.5+pyo,Math.max(1,Math.round(s*0.7)),cr,cg,cb,al*0.25,round,W,H);
    }
  }
  inkT=performance.now();inkMass=ink;inkN=inkn;
  ctx.putImageData(pcImg,0,0);
}
// ── rAF 动画循环（~30fps 节流）：粒子云常开（speed=0 时粒子静止但仍重生闪换）；
//    网格模式仅运动开启时循环。深度帧到达只刷 dotSample 缓存，循环不重置 ──
let dotRaf=0,dotLastT=0;
function dotNeedLoop(){return +dotCfg.on&&(+dotCfg.mode===1||+dotCfg.motion>0||+dotCfg.pfloat>0||+dotCfg.pdepth>0);}
function dotLoop(ts){
  dotRaf=0;
  if(!dotNeedLoop()||$('bgDot').style.display==='none')return;
  if(ts-dotLastT>=33){dotLastT=ts;
    if(+dotCfg.mode===1)pcDraw(ts/1000);else dotDraw(ts/1000);}
  dotRaf=requestAnimationFrame(dotLoop);
}
function applyDotCfg(){
  const on=+dotCfg.on===1,cv=$('bgDot'),cloud=+dotCfg.mode===1;
  $('stage').classList.toggle('doton',on);   // 开启时 img 层隐藏、canvas 呈现
  // 仅图片类背景显示 canvas：GLB（model-viewer）背景时不遮挡
  cv.style.display=(on&&!mvVisible())?'block':'none';
  $('dtgrid').style.display=cloud?'none':'';
  $('dtcloud').style.display=cloud?'':'none';
  if(on){
    $('stage').style.background=dotCfg.bg;
    dotSample();                             // 模式/点距/视口可能变了：重建取色缓存
    if(cloud){pcDraw(performance.now()/1000);}
    else dotDraw(performance.now()/1000);
    if(dotNeedLoop()&&!dotRaf)dotRaf=requestAnimationFrame(dotLoop);
  }else{
    $('stage').style.background='';   // 交还默认黑底（GLB 来源由点渲染区自设背景色）
  }
}
function saveDot(){localStorage.setItem('exp_dot',JSON.stringify(dotCfg));}
const DT_RADIOS={dton:'on',dtmode:'mode',dtmo:'motion',dtmono:'pmono',
  dtgsh:'gshape',dtpsh:'pshape'};
Object.keys(DT_RADIOS).forEach(nm=>document.querySelectorAll('input[name='+nm+']').forEach(r=>
  r.addEventListener('change',()=>{
    dotCfg[DT_RADIOS[nm]]=+document.querySelector('input[name='+nm+']:checked').value;
    applyDotCfg();saveDot();})));
const DT_SLIDERS={pitch:['r_dt_pitch','v_dt_pitch'],r:['r_dt_r','v_dt_r'],
  jitter:['r_dt_jit','v_dt_jit'],speed:['r_dt_spd','v_dt_spd'],
  pn:['r_dt_pn','v_dt_pn'],psize:['r_dt_psz','v_dt_psz'],
  pcontrast:['r_dt_pct','v_dt_pct'],pdrift:['r_dt_drift','v_dt_drift'],
  prespawn:['r_dt_resp','v_dt_resp'],pfloat:['r_dt_pf','v_dt_pf'],
  pdepth:['r_dt_pdep','v_dt_pdep']};
Object.keys(DT_SLIDERS).forEach(k=>{
  const [rid,vid]=DT_SLIDERS[k];
  $(rid).addEventListener('input',()=>{
    dotCfg[k]=+$(rid).value;$(vid).textContent=$(rid).value;
    applyDotCfg();saveDot();});
});
[['c_dt_bg','bg'],['c_dt_pc','pcolor']].forEach(([id,k])=>
  $(id).addEventListener('input',()=>{
    dotCfg[k]=$(id).value;applyDotCfg();saveDot();}));
// dotCfg 当前值回填点云化控件（页面加载与预设恢复两处复用）
function fillDtControls(){
  Object.keys(DT_RADIOS).forEach(nm=>{
    const el=document.querySelector('input[name='+nm+'][value="'+(+dotCfg[DT_RADIOS[nm]])+'"]');
    if(el)el.checked=true;});
  Object.keys(DT_SLIDERS).forEach(k=>{
    const [rid,vid]=DT_SLIDERS[k];
    $(rid).value=dotCfg[k];$(vid).textContent=''+dotCfg[k];});
  $('c_dt_bg').value=dotCfg.bg;$('c_dt_pc').value=dotCfg.pcolor;
}
fillDtControls();   // localStorage 记忆值回填（页面加载一次）
addEventListener('resize',()=>{if(+dotCfg.on)applyDotCfg();});

// ══ 配置预设（存服务器）：深度渲染 + 深度显示 + 点云化 整套参数的命名快照 ══
// 保存：从控件当前值收集完整参数 POST 到 /api/frame/depth-presets（重名覆盖、数量不限，
// 服务端落盘 depth_presets.json 重启不丢）；恢复：深度渲染整套写回当前选中设备
// （device-config 通道，~2-4s 生效），深度显示/点云化本页即时生效并记入 localStorage
function ddpCollect(){
  const cfg={};
  Object.keys(DD_SLIDERS).forEach(k=>{cfg[k]=+$(DD_SLIDERS[k][0]).value;});
  Object.keys(DD_RADIOS).forEach(nm=>{
    const el=document.querySelector('input[name='+nm+']:checked');
    if(el)cfg[DD_RADIOS[nm]]=isNaN(+el.value)?el.value:+el.value;});
  cfg.depth_colormap=$('dd_cmap').value;
  cfg.depth_invalid_color=$('c_dd_invalid').value;
  return {config:cfg,display:Object.assign({},ddCss),dot:Object.assign({},dotCfg)};
}
async function ddpLoad(){
  try{
    const r=await(await fetch('/api/frame/depth-presets',{cache:'no-store'})).json();
    ddpRender(r.presets||[]);
  }catch(e){/* 列表加载失败保持现状，下次打开抽屉重试 */}
}
function ddpRender(list){
  const box=$('ddp_list');box.textContent='';
  if(!list.length){
    const d=document.createElement('div');d.className='hint';d.style.marginTop='4px';
    d.textContent='（暂无预设）';box.appendChild(d);return;
  }
  list.forEach(p=>{
    const row=document.createElement('div');row.className='prow';
    const nm=document.createElement('span');nm.className='nm';nm.textContent=p.name;nm.title=p.name;
    const ts=document.createElement('span');ts.className='ts';
    if(p.saved_at){
      const d=new Date(p.saved_at*1000);
      ts.textContent=(d.getMonth()+1)+'/'+d.getDate()+' '
        +String(d.getHours()).padStart(2,'0')+':'+String(d.getMinutes()).padStart(2,'0');
    }
    const bR=document.createElement('button');bR.textContent='恢复';
    bR.addEventListener('click',()=>ddpApply(p));
    const bD=document.createElement('button');bD.textContent='删除';
    bD.addEventListener('click',()=>{
      // 两次点击确认（不用 confirm() 弹窗：展台全屏页不弹系统对话框）
      if(!bD.classList.contains('confirm')){
        bD.classList.add('confirm');bD.textContent='确认删除';
        setTimeout(()=>{bD.classList.remove('confirm');bD.textContent='删除';},3000);
        return;
      }
      fetch('/api/frame/depth-presets/delete',{method:'POST',
        headers:{'Content-Type':'application/json'},body:JSON.stringify({name:p.name})})
        .then(()=>ddpLoad()).catch(()=>{});
    });
    row.appendChild(nm);row.appendChild(ts);row.appendChild(bR);row.appendChild(bD);
    box.appendChild(row);
  });
}
function ddpApply(p){
  // 深度渲染：整套写回推流端（预设是完整快照，所有可控键全量覆盖），并即时回填控件
  //（ddTouched 守卫 1.5s，防 status 轮询用旧值回写打架）
  if(p.config&&Object.keys(p.config).length&&curDev){
    ddTouched=Date.now();
    fetch('/api/frame/device-config',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({device_id:curDev,config:p.config})}).catch(()=>{});
    fillDdControls(p.config);
  }
  // 深度显示 + 点云化：与手动拖滑杆同一条路径（应用 + localStorage + 控件回填）
  if(p.display){Object.assign(ddCss,p.display);saveDdCss();applyDdCss();fillDcControls();}
  if(p.dot){Object.assign(dotCfg,p.dot);saveDot();applyDotCfg();fillDtControls();}
}
$('ddp_save').addEventListener('click',()=>{
  const name=$('ddp_name').value.trim();
  if(!name)return;
  fetch('/api/frame/depth-presets',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(Object.assign({name:name},ddpCollect()))})
    .then(r=>r.json()).then(r=>{if(r&&r.ok){$('ddp_name').value='';ddpLoad();}}).catch(()=>{});
});
ddpLoad();

let lastInset='';
let bgFps=0;   // 选中设备实测入帧 fps（bgTick 每轮更新，驱动轮询自适应）
async function bgTick(){
 if(DEMO)return;
 try{
  const s=await(await fetch('/api/frame/status',{cache:'no-store'})).json();
  if(s.processor&&s.config_gen===0)pushDefaultConfig();
  // 打开默认设备：首次见到 g335 在线即（按需）选中它，本轮先返回、下一轮按新选中渲染
  if(!prefDevDone&&(s.devices||[]).some(d=>d.device_id===PREF_DEVICE)){
    prefDevDone=true;
    if(s.selected!==PREF_DEVICE){
      await fetch('/api/frame/select',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({device_id:PREF_DEVICE})}).catch(()=>{});
      return;
    }
  }
  renderDevices(s);
  renderRate(s);
  // 供轮询自适应排程用：选中设备的实测入帧 fps（无帧/无效时 0）
  bgFps=(s.fps&&s.fps>0)?+s.fps:0;
  if(s.device&&s.device!==curDev){   // 切设备：清背景与卡片缓存，等新设备的帧/产物
    if(curDev!==null){lastBgKey='';lastMvUrl='';lastBgUrl='';lastInset='';curCard=null;lastCardKey='';}
    curDev=s.device;
  }
  // 本页任何情况都不展示设备原图：所选来源暂无产物（服务重启/切设备/首轮构建中）时
  // 保持黑场，等第一份点云产物就绪再上画
  // 设备深度图/设备点云来源仅带硬件深度的相机（G335 等）可用：无深度能力的设备置灰
  const ddOpt=$('selStyle').querySelector('option[value=devdepth]');
  if(ddOpt)ddOpt.disabled=s.has_depth!==true;
  const pcOpt=$('selStyle').querySelector('option[value=devpc]');
  if(pcOpt)pcOpt.disabled=s.has_depth!==true;
  renderDdCfg(s);
  if(bgSource==='devpc'){
    // 设备点云（硬件真深度反投影彩色点云 GLB）：轮询即点播——status 请求按 10s TTL
    // 续期服务端 demand，推流端 ~2-4s 内开始推原料、切走后自动停推
    if(s.has_depth){
      const pc=await(await fetch('/api/devpc/status?device='
        +encodeURIComponent(s.device||''),{cache:'no-store'})).json();
      // device 比对：切设备后旧设备的点云 GLB 不上画（保持黑场等新产物）
      if(pc.url&&pc.device===s.device)showModel(pc.url,pc.meta&&pc.meta.fov_deg);
    }else{
      // 无硬件深度能力的设备：清残留画面回黑场
      hideModelLayer();
      $('bgA').classList.remove('on');$('bgB').classList.remove('on');
    }
  }else{
    // 设备深度图（默认来源，仅 G335 等带硬件深度的相机）：直接展示 mini 端伪彩深度帧。
    // 深度是相机产物不是设备 RGB 原图，不违反「本页不展示设备原图」的产品红线
    if(s.has_depth&&s.depth_seq){
      showImg('/api/frame/latest-depth?device='+encodeURIComponent(s.device||'')
        +'&t='+s.depth_seq,'dd:'+s.depth_seq);
    }else{
      // 无深度能力/深度未就绪：清残留画面回黑场
      hideModelLayer();
      $('bgA').classList.remove('on');$('bgB').classList.remove('on');
    }
  }
  // 流水小窗：镜像当前背景画面——图片背景直接复用 url；GLB 背景截取 model-viewer
  // 画布（toDataURL）做镜像（不受流水态压暗影响）。GLB 换模加载中（loaded=false）
  // 保持上一帧镜像不动；镜像未就绪时留等待占位，绝不垫设备原帧（本页不展示原图）。
  if($('tl').classList.contains('on')){
    const mv=mvFront(),mvOn=mvVisible(),tlr=$('tlraw');
    if(mvOn){
      if(mv.loaded&&mv.src){
        try{tlr.src=mv.toDataURL('image/jpeg',0.8);lastInset='';
          tlr.style.display='block';$('tlwait').style.display='none';}catch(e){}
      }
    }else{
      const insetUrl=lastBgUrl;
      if(insetUrl&&insetUrl!==lastInset){lastInset=insetUrl;
        tlr.src=insetUrl;
        tlr.style.display='block';$('tlwait').style.display='none';}
    }
  }
 }catch(e){/* 单次轮询失败忽略 */}
}

// ══ 状态机：待机 ↔ 识别成功（识别失败态暂不做）══
// 两个时钟，口径不同，别再合成一个（旧实现合成了一个 4s，是"卡只显示一秒"的根因）：
//   · DWELL_MS   驻留：按**上屏时刻**算——人眼读完一张卡要多久，与链路延时无关。
//                同一批的后续识别只续期不重画，所以持续识别到同一个食物就一直停着。
//   · FRESH_TTL_MS 新鲜度：按**帧时刻**算——now-last_ts 就是端到端延时（帧到 8060
//                → 结果落卡）。超了说明这份结果描述的画面早过去了，整条丢弃。
// 旧实现两者共用一个 FRESH_MS 且都从帧时刻起算，于是
//   屏上驻留 = FRESH_MS − 端到端延时 = 4s − 2.1s ≈ 1.9s（延时一抖就只剩零点几秒，
//   延时 ≥ 4s 更是完全不上屏）——这是数学上的必然，不是偶发。
// 两个都是写死的常量：现场调定后不再暴露旋钮——它们是链路行为不是观感偏好，
// 按展示端各调各的只会让"这台屏为什么和那台不一样"变成排障噪声。要改改这里。
// 新鲜度必须明显大于端到端延时（5090 实测中位约 2.1s、抖动时 3.5s+），
// 调太小会把慢轮整轮丢掉，表现为「这次识别凭空消失」——比停留短更糟。
const DWELL_MS=3000;        // 驻留：上屏后停多久
const FRESH_TTL_MS=5000;    // 新鲜度：帧到服务器 → 结果落卡，超了整条丢弃
// 历史遗留的展示端持久化值一律清掉：早先这两个量挂在抽屉滑杆上、写进 localStorage，
// 留着会让"我明明改了代码怎么没生效"重演一遍（滑杆已经没了，值却还在）。
['exp_card_fresh_s','exp_card_dwell_s','exp_card_ttl_s']
  .forEach(k=>{try{localStorage.removeItem(k)}catch(e){}});

// ══ 识别触发（主链路直传 VLM）：读写 /api/recog/direct/config（服务端全局配置，
//    与卡片驻留那种纯展示端 localStorage 不同——直传节奏是链路行为，所有页面共享）══
let rdTimer=null;
function rdLabels(){
  $('v_rd_itv').textContent=(+$('r_rd_itv').value).toFixed(1);
  $('v_rd_conc').textContent=$('r_rd_conc').value;
  $('v_rd_age').textContent=(+$('r_rd_age').value).toFixed(1);
}
function pushRdCfg(){
  clearTimeout(rdTimer);
  rdTimer=setTimeout(()=>{
    fetch('/api/recog/direct/config',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({on:document.querySelector('input[name=rdon]:checked').value==='1',
        interval_s:+$('r_rd_itv').value,concurrency:+$('r_rd_conc').value,
        max_frame_age_s:+$('r_rd_age').value})})
      .then(r=>r.json()).then(j=>{if(j&&j.stats)rdStat(j.stats);}).catch(()=>{});
  },250);
}
function rdStat(s){
  const el=$('v_rd_stat');if(!el||!s)return;
  const age=s.last_ts?Math.max(0,Math.round(Date.now()/1000-s.last_ts)):null;
  const off=document.querySelector('input[name=rdon]:checked').value==='0';
  // 关（SAM3 门控）时把门控耗时与命中率一并回显——空轮多说明画面里确实没东西
  const gate=off?('SAM3 门控 '+(s.gate_ms||0)+'ms · 命中 '+(s.gate_hits||0)
    +'/'+((s.gate_hits||0)+(s.gate_misses||0))+' 轮 · '):'';
  // 陈旧帧丢弃：非零就显示，它是"积压有没有被截住"的直接读数
  const drop=(s.dropped_stale?(' · 陈旧丢弃 '+s.dropped_stale+' 轮'
    +(s.last_drop_age_ms?'（最近 '+(s.last_drop_age_ms/1000).toFixed(1)+'s）':'')):'')
    +(s.dropped_out_of_order?' · 乱序丢弃 '+s.dropped_out_of_order+' 轮':'');
  el.textContent=gate+'VLM '+(s.last_ms||0)+'ms · 在飞 '+(s.in_flight||0)+' 路 · 累计 '
    +(s.rounds||0)+' 轮'+(age!==null?' · 最近一轮 '+age+'s 前':'')+drop
    +(s.last_error?' · 错误：'+s.last_error:'');
}
async function loadRdCfg(){
  try{
    const j=await(await fetch('/api/recog/direct/config',{cache:'no-store'})).json();
    const c=j.config||{};
    const on=document.querySelector('input[name=rdon][value="'+(c.on?1:0)+'"]');
    if(on)on.checked=true;
    if(c.interval_s!==undefined)$('r_rd_itv').value=c.interval_s;
    if(c.concurrency!==undefined)$('r_rd_conc').value=c.concurrency;
    if(c.max_frame_age_s!==undefined)$('r_rd_age').value=c.max_frame_age_s;
    rdLabels();rdStat(j.stats);
  }catch(e){/* 读取失败保持面板默认值 */}
}
$('r_rd_itv').addEventListener('input',()=>{rdLabels();pushRdCfg();});
$('r_rd_conc').addEventListener('input',()=>{rdLabels();pushRdCfg();});
$('r_rd_age').addEventListener('input',()=>{rdLabels();pushRdCfg();});
document.querySelectorAll('input[name=rdon]').forEach(r=>r.addEventListener('change',pushRdCfg));
loadRdCfg();
setInterval(()=>{if($('hlcfg').classList.contains('on'))loadRdCfg();},3000);  // 抽屉开着才刷实测
let lastCardKey='',cardShownAt=0,curCard=null;
// ══ 展示批次：同一个物体在后端常被登记成多张卡（模型没认领候选就另起一张，
//    实测一瓶酒能散成 14 张），每张卡自己的内容是冻结的，但屏幕取 cards[0] 就会
//    在这些卡之间来回跳——描述从 "glass bottle" 变 "transparent bottle"、
//    卡路里从 35 变 50。这里不动后端，只在展示层把「归一名+类型」相同的卡看成
//    同一批，内容一律取批次里**最早建立**的那张：_recog_id 全局自增，
//    min(id) 就是第一次识别到它时给的那份描述与营养。
//    注：批次锚点若被 RECOG_MAX_CARDS(200) 淘汰，锚点会前移一次、内容跟着变一次。
//    不额外做内存快照兜底——那会让「真的换了另一个橙子」也永远显示旧内容。
function grpKey(c){return String(c&&c.name||'').trim().toLowerCase()+'|'+(c&&c.type||'');}
function grpAnchor(cards,k){
  let a=null;
  for(const c of cards||[]) if(grpKey(c)===k && (!a || c.id<a.id)) a=c;
  return a;
}
// ══ 成功卡出入场动效 ══
// 出现：玻璃底先拉开(CARD_BG_IN)，再逐行推入——每个元素走 CARD_ROW，相邻两行只错峰
//       CARD_STAGGER(= 行时长的一半)，前一行走到一半后一行就起步，不必等它落地
// 消失：内容先一起淡出(CARD_OUT_TXT)，再折叠玻璃底(CARD_OUT_BG)，收完才让待机文案渐显
// 节奏账：每个元素 200ms、错峰 100ms。分割线不占节拍（见 cardRowsIn），内容行就 5 行——
// 80+4×100+200=680ms 全部落地，分割线再统一花 200ms，入场共 ~0.9s；卡片驻留
// DWELL_MS(3s) 里留出 2s 给人读，退场再花 0.66s。
const CARD_BG_IN=420,CARD_ROW_DELAY=80,CARD_ROW=200,CARD_STAGGER=100,CARD_OUT_TXT=240,CARD_OUT_BG=420;
let cardTimers=[];
function cardClear(){cardTimers.forEach(clearTimeout);cardTimers=[];}
function cardRows(){   // 只取当前可见的行（无营养数据的行整行隐藏，不该占动画节拍）
  return [...$('card').querySelectorAll('.rvw')].filter(w=>w.style.display!=='none');
}
// 内容行逐个错峰推入；分割线不参与逐行节拍——它们只是分区的线，跟着内容一条条爬
// 反而拖慢观感，统一等内容全部落地后一起出现
function cardRowsIn(delay0){
  const rows=cardRows(),isRule=w=>!!w.querySelector('hr');
  const body=rows.filter(w=>!isRule(w)),rules=rows.filter(isRule);
  body.forEach((w,i)=>cardTimers.push(
    setTimeout(()=>w.classList.add('in'),delay0+i*CARD_STAGGER)));
  const tail=delay0+Math.max(0,body.length-1)*CARD_STAGGER+CARD_ROW;   // 末行落地的时刻
  rules.forEach(w=>cardTimers.push(setTimeout(()=>w.classList.add('in'),tail)));
}
// 下一帧执行；页面被切到后台时 rAF 会停摆，用定时器兜底一次（只会跑一次），
// 免得运营切走窗口那会儿的状态切换把卡片卡在半路
function nextTick(fn){let done=false;const run=()=>{if(done)return;done=true;fn();};
  requestAnimationFrame(run);setTimeout(run,32);}
function cardIn(){
  cardClear();
  const card=$('card');
  card.classList.remove('out');
  card.querySelectorAll('.rvw').forEach(w=>w.classList.remove('in'));
  card.classList.add('on');
  // 下一帧再挂 bgin：同帧加 on+bgin 会被浏览器合成成"直接是终态"，看不到拉开过程
  nextTick(()=>{card.classList.add('bgin');cardRowsIn(CARD_ROW_DELAY);});
}
function cardOut(done){
  cardClear();
  const card=$('card');
  card.classList.add('out');                                     // 内容一起淡出
  cardTimers.push(setTimeout(()=>card.classList.remove('bgin'),CARD_OUT_TXT));   // 玻璃底折叠
  cardTimers.push(setTimeout(()=>{
    card.classList.remove('on','out');
    card.querySelectorAll('.rvw').forEach(w=>w.classList.remove('in'));
    done&&done();},CARD_OUT_TXT+CARD_OUT_BG));
}
let uiState=null;   // 'card' | 'idle' | 'none'(流水态两者都不显示)
function setState(st){
  const want=$('tl').classList.contains('on')?'none':st;
  if(want===uiState)return;                    // 每秒都会调，状态没变就不重播动画
  const prev=uiState;uiState=want;
  if(want==='card'){$('idle').classList.add('fast');$('idle').classList.remove('on');cardIn();return;}
  if(prev==='card'){                           // 先把卡片收干净，再让待机文案渐显
    cardOut(()=>{if(uiState==='idle'){$('idle').classList.remove('fast');$('idle').classList.add('on');}});
    return;
  }
  $('idle').classList.toggle('on',want==='idle');
}
function rowShow(id,on){   // 整行显隐：作用在该行的遮罩层上，隐藏行不占位也不占动画节拍
  const e=$(id),w=(e&&e.closest('.rvw'))||e;
  if(w)w.style.display=on?'':'none';
}
function renderCard(c){
  $('cname').textContent=c.name||'';
  $('cdesc').textContent=c.description_en||'';
  // 营养数字与分级：VLM 偶发不给（guardrail 置 null/空）→ 对应行整行隐藏
  const kcal=(c.calories_kcal!=null)?c.calories_kcal+' kcal':'';
  rowShow('ckcalrow',!!kcal);$('ckcal').textContent=kcal;
  let anyMac=false;
  [['mpro',c.protein_g],['mcarb',c.carbs_g],['mfat',c.fat_g]].forEach(([id,v])=>{
    const on=v!=null;$(id).style.display=on?'':'none';
    if(on){$(id).querySelector('.mval').textContent=v+'g';anyMac=true;}});
  rowShow('macros',anyMac);rowShow('mrule',anyMac);
  const cls=c.classification||'';
  rowShow('cclsrow',!!cls);rowShow('crule',!!cls);
  $('ccls').textContent=cls;
  // 库内命中：整组内容来自参考食物库（名称/描述/营养/分级都是录入的定值）
  $('creg').classList.toggle('on',c.source==='catalog');
  // 卡片在场时换了一批（换了个食物）：玻璃底不动，只把各行重推一遍
  if(uiState==='card'){cardClear();
    $('card').querySelectorAll('.rvw').forEach(w=>w.classList.remove('in'));
    nextTick(()=>cardRowsIn(0));}
}
function renderTimeline(cards){
  const list=$('tllist');
  // 与主面板同口径按批次去重：cards 最新在前，每批留先遇到的那条(时间最新、同批同名)，
  // 否则流水里会连着列出十几条 Wine
  const seen=new Set(),uniq=[];
  for(const c of cards||[]){const k=grpKey(c);if(!seen.has(k)){seen.add(k);uniq.push(c);}}
  const rows=uniq.slice(0,10).reverse();   // 最新 10 条，按时间升序排布（同设计稿）
  if(!rows.length){list.innerHTML='<span id="tlempty">No records yet today.</span>';return;}
  list.innerHTML=rows.map(c=>{
    const busy=c.status&&c.status!=='done';
    return '<div class="trow'+(busy?' dim':'')+'"><span class="tname">'
      +(busy?'<span class="spin"></span>':'')
      +(c.name||'')+(busy?'…':'')+'</span><span class="ttime">'
      +String(c.t||'').slice(0,5)+'</span></div>';}).join('');
}
// 识别结果由 SSE 推送驱动（不再定时轮询）：后端出卡/合并/清空/切设备时推全量列表
function applyRecog(r){
  const cards=r.cards||[];
  renderTimeline(cards);
  const head=cards[0];                       // 后端最后碰过的那条：决定"现在该看哪一批"
  // ── 第一层·新鲜度闸（帧时刻口径）──────────────────────────────────
  // last_ts 是**帧时刻**，Date.now()-last_ts 就是这轮的端到端延时。超过 FRESH_TTL_MS
  // 说明它描述的画面早过去了（多半是实物已离场、请求飞得久才回来）——整条静默丢弃。
  // 不只是不上屏：curCard/lastCardKey/cardShownAt 一律不动，否则屏幕当下虽判 idle，
  // 下一次任何更新到来时会先闪一下这份旧内容。这就是"离场保护"的落点。
  // 后端也有一道帧龄闸（发请求前，max_frame_age_s），但那道管的是"值不值得发"，
  // 挡不住"发出去之后飞太久"；两道口径不同，都要有。
  // 这一层同时兜住刷新场景（2026-08-18 修过的老坑）：刷新后 lastCardKey 内存清零，
  // 存量卡都会被判成"新的一批"，但半小时前的卡在这里就被判过期整条丢掉，
  // 不会再"刷新先闪一下旧卡才回落待机"。
  const ts=head?(head.last_ts?head.last_ts*1000:Date.now()):0;
  if(head&&Date.now()-ts<=FRESH_TTL_MS){
    const key=grpKey(head);
    // ── 第二层·驻留计时（上屏时刻口径）────────────────────────────
    // 只有换了一批（换了个食物）才重画；同批次内的后续命中一个字都不改，只续期。
    // 内容取批次锚点(最早那张)，不取 head——head 可能是这批里第 14 张，
    // 拿它的描述/营养上屏正是"跳"的来源。
    if(key!==lastCardKey){lastCardKey=key;curCard=grpAnchor(cards,key)||head;renderCard(curCard);}
    // 起点 = 真正 render/续期的这一刻，与端到端延时彻底解耦：调 2s 就是屏上 2s。
    // 续期而非"key 变才刷"：批次身份稳定后 key 不再变，跟着 key 走会导致明明还在
    // 持续识别却到期回落待机。画面稳不稳由响应内容决定（还认得出同一个食物就一直
    // 续着），不靠压请求。
    cardShownAt=Date.now();
  }
  setState(curCard&&Date.now()-cardShownAt<DWELL_MS?'card':'idle');
}

// ══ 右下临时工具：来源下拉框 + 高亮调节抽屉开关 + 流水视图开关 ══
function syncStyleUI(){
  $('selStyle').value=bgSource;
  // 抽屉常驻可开（帧率/识别触发区对所有来源有意义）；点渲染区仅 GLB 类来源（设备点云）
  // 展示；设备深度图调节区仅该来源展示
  $('hlonly').style.display=(bgSource==='devpc')?'':'none';
  $('ddonly').style.display=(bgSource==='devdepth')?'':'none';
  // 点云化样式区只对图片类背景（设备深度图）有意义（devpc 恒为 GLB 不列）
  $('dotonly').style.display=(bgSource!=='devpc')?'':'none';
  applyDdCss();   // 切来源即时挂上/摘掉深度显示的 CSS（其它来源不受深度显示参数影响）
  applyDotCfg();  // 点云化样式层随来源重估显隐（GLB 来源不遮挡）
  // 调节作用域收敛：图片类来源下停掉 GLB 侧一切动效——脉冲/闪烁 timer（applyPtStyle
  // 内按来源重估）与自动旋转都只属于 GLB 来源，切回时 applyPtStyle 会按配置重新挂上
  if(bgSource==='devdepth')mvEls().forEach(el=>el.removeAttribute('auto-rotate'));
  applyPtStyle();
}
$('selStyle').onchange=()=>{
  bgSource=$('selStyle').value;
  localStorage.setItem('exp_bg',bgSource);
  lastBgKey='';lastMvUrl='';   // 立刻允许下一轮加载新来源
  applyExpView();              // 切来源后按当前来源重摆相机
  syncStyleUI();bgTick();
};
$('btnTl').onclick=()=>{
  const on=!$('tl').classList.contains('on');
  $('tl').classList.toggle('on',on);
  document.body.classList.toggle('tlon',on);   // 流水态：状态文案彻底隐藏 + 背景压暗
  $('btnTl').textContent=on?'实时':'流水';
  setState(curCard&&Date.now()-cardShownAt<DWELL_MS?'card':'idle');
};
// ══ 控制台显隐：右下调试按钮组默认隐藏（观众看不到），右键菜单「显示/隐藏控制台」
// 切换，选择记进 localStorage（刷新保持；从未设置过=默认隐藏）══
let consoleOn=localStorage.getItem('exp_console')==='1';
function applyConsole(){
  $('tools').style.display=consoleOn?'':'none';
  if(!consoleOn)$('hlcfg').classList.remove('on');   // 隐藏控制台时顺手收起调节抽屉
  $('ctxToggle').textContent=consoleOn?'隐藏控制台':'显示控制台';
}
document.addEventListener('contextmenu',e=>{
  e.preventDefault();
  const m=$('ctxmenu');
  m.style.display='block';
  // 先显示再量尺寸，钳制进视口（右/下边缘右键时菜单不出界）
  m.style.left=Math.max(4,Math.min(e.clientX,innerWidth-m.offsetWidth-8))+'px';
  m.style.top=Math.max(4,Math.min(e.clientY,innerHeight-m.offsetHeight-8))+'px';
});
document.addEventListener('click',()=>{$('ctxmenu').style.display='none';});
$('ctxToggle').addEventListener('click',e=>{
  e.stopPropagation();
  consoleOn=!consoleOn;
  localStorage.setItem('exp_console',consoleOn?'1':'0');
  applyConsole();
  $('ctxmenu').style.display='none';
});
applyConsole();

// ══ 全屏：Fullscreen API 整页铺满显示器（展台演示用；须由用户点击手势触发）══
function fsOn(){return !!(document.fullscreenElement||document.webkitFullscreenElement);}
$('btnFs').onclick=()=>{
  const root=document.documentElement;
  if(fsOn())(document.exitFullscreen||document.webkitExitFullscreen).call(document);
  else (root.requestFullscreen||root.webkitRequestFullscreen).call(root);
};
// 状态跟随浏览器事件（Esc/系统手势退出也能同步按钮文案）
['fullscreenchange','webkitfullscreenchange'].forEach(ev=>
  document.addEventListener(ev,()=>{$('btnFs').textContent=fsOn()?'退出全屏':'全屏';}));
// ══ Qwen 识别隧道状态 + 一键重建（复用 /panel 那套 /api/tunnel/*）══
// 链路：5090:8011 ←反向SSH← Mac:18011 ←IAP← GCP gpu-g4-01 的 vLLM。GCP 凭证只在 Mac，
// 所以网页只负责「下发重建指令」，真正重建由 Mac 上常驻的 qwen_tunnel_keeper 心跳领走执行。
let tunReqAt=0;        // 本页最近一次下发重建的时刻，用于在探测到通之前显示「重建中…」
let tunKeeper=false;   // Mac 守护是否在线；不在线时网页下发也没人执行
async function tunTick(){
 try{
  const t=await(await fetch('/api/tunnel/status',{cache:'no-store'})).json();
  const dot=$('tunDot'),txt=$('tunTxt'),btn=$('btnTun');
  tunKeeper=!!t.keeper_alive;
  const rebuilding=!t.up&&(t.pending||Date.now()-tunReqAt<90000)&&tunReqAt;
  const msg=t.keeper_msg?('（'+t.keeper_msg+'）'):'';
  btn.classList.toggle('down',!t.up&&!rebuilding);
  if(t.up){
   dot.style.background='#34c759';txt.textContent='隧道';
   btn.title='Qwen 识别隧道：已连通'+(t.rtt_ms?(' · RTT '+Math.round(t.rtt_ms)+'ms'):'')+'（点击可强制重建）';
  }else if(rebuilding){
   dot.style.background='#ff9f0a';txt.textContent='重建中…';
   btn.title='Qwen 识别隧道：重建指令已下发，等 Mac 守护执行'+msg;
  }else{
   dot.style.background='#ff3b30';
   txt.textContent=tunKeeper?'隧道断开 · 重建':'隧道断开';
   btn.title=tunKeeper?('Qwen 识别隧道：已断开，点击一键重建'+msg)
     :'Qwen 识别隧道：已断开，且 Mac 守护不在线，网页无法远程重建（需在 Mac 上拉起 qwen_tunnel_keeper）'+msg;
  }
 }catch(e){/* 单次轮询失败忽略，下轮重试 */}
}
$('btnTun').onclick=async()=>{
 if(!tunKeeper){   // 守护离线：就地提示 2s 再回落，避免下发一条没人领的指令
  $('tunTxt').textContent='Mac 守护离线';setTimeout(tunTick,2000);return;}
 tunReqAt=Date.now();
 $('tunDot').style.background='#ff9f0a';$('tunTxt').textContent='已下发…';
 try{await fetch('/api/tunnel/rebuild',{method:'POST'});}catch(e){}
 setTimeout(tunTick,1500);
};
setInterval(tunTick,5000);tunTick();

syncStyleUI();

// ══ 点云样式调节抽屉：读写 /api/sam3hl/config（与 /panel 的点渲染卡同一套配置） ══
// 打开时从服务端读当前配置回填（不主动推初值，避免覆盖 /panel 已调好的参数），改动才下发。
// 本页只保留「前端实时生效」的两类：view_* 相机参数、pt_* 点材质参数——它们对当前唯一的
// GLB 背景（设备点云）有效。服务端烘焙类（strength/dim/sat/val/conf/hue/outlier_mad）随
// DA3 派生点云来源一起从本页下线，配置本身仍在，/panel 那张卡照旧可调。
const HL_KEYS=['view_tilt','view_zoom','eye_lift','eye_back',
  'pt_size','pt_opacity','pt_density','pt_hue','pt_sat','pt_val',
  'pt_contrast','pt_exposure','pt_ramp_near','pt_ramp_far','pt_fog',
  'pt_clip_near','pt_clip_far','pt_clip_ylo','pt_clip_yhi',
  'pt_rotate_speed','pt_fov_off','pt_pulse_speed','pt_sparkle','pt_conf_size','pt_conf_alpha'];
const _f1=v=>(+v).toFixed(1),_f2=v=>(+v).toFixed(2);
const HL_FMT={view_zoom:_f2,eye_lift:_f2,eye_back:_f2,
  pt_size:_f1,pt_opacity:_f2,pt_sat:_f1,pt_val:_f1,pt_contrast:_f2,pt_exposure:_f2,
  pt_ramp_near:_f1,pt_ramp_far:_f1,pt_fog:_f2,pt_clip_near:_f1,pt_clip_far:_f1,
  pt_clip_ylo:_f2,pt_clip_yhi:_f2,pt_pulse_speed:_f1,pt_sparkle:_f2,
  pt_conf_size:_f2,pt_conf_alpha:_f2};
// radio 组（name → 配置字段）与颜色控件（元素 id → 配置字段）
const PT_RADIOS={ptshape:'pt_shape',ptatten:'pt_atten',ptblend:'pt_blend',
  ptinvert:'pt_invert',ptcm:'pt_colormode',ptrot:'pt_rotate',ptpulse:'pt_pulse'};
const PT_COLORS={c_pt_duo_a:'pt_duo_a',c_pt_duo_b:'pt_duo_b',c_pt_bg:'pt_bg'};
const VIEW_KEYS=['view_tilt','view_zoom','eye_lift','eye_back'];
let hlView={view_tilt:0,view_zoom:1.0,eye_lift:0,eye_back:0};  // 拍摄视角零offset（服务端配置可调）
let hlPushTimer=null;
function hlLabel(k){$('v_'+k).textContent=(HL_FMT[k]||(v=>v))($('r_'+k).value);}
function pushHlCfg(){
  clearTimeout(hlPushTimer);
  hlPushTimer=setTimeout(()=>{
    const body={};
    HL_KEYS.forEach(k=>body[k]=+$('r_'+k).value);
    Object.keys(PT_RADIOS).forEach(nm=>
      body[PT_RADIOS[nm]]=+document.querySelector('input[name='+nm+']:checked').value);
    Object.keys(PT_COLORS).forEach(id=>body[PT_COLORS[id]]=$(id).value);
    fetch('/api/sam3hl/config',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify(body)}).catch(()=>{});
  },250);
}
async function loadHlCfg(){
  try{
    const st=await(await fetch('/api/sam3hl/status',{cache:'no-store'})).json();
    const c=st.cfg||{};
    HL_KEYS.forEach(k=>{if(c[k]!==undefined){$('r_'+k).value=c[k];hlLabel(k);}});
    VIEW_KEYS.forEach(k=>{if(c[k]!==undefined)hlView[k]=+c[k];});
    applyExpView();   // 服务端可能有 /panel 调过的视角参数，回填后立即应用
    HL_KEYS.forEach(k=>{if(k.indexOf('pt_')===0&&c[k]!==undefined)ptStyle[k]=+c[k];});
    Object.keys(PT_RADIOS).forEach(nm=>{const k=PT_RADIOS[nm];
      if(c[k]!==undefined){ptStyle[k]=Math.round(+c[k]);
        const el=document.querySelector('input[name='+nm+'][value="'+ptStyle[k]+'"]');
        if(el)el.checked=true;}});
    Object.keys(PT_COLORS).forEach(id=>{const k=PT_COLORS[id];
      if(/^#[0-9a-fA-F]{6}$/.test(c[k]||'')){$(id).value=c[k];ptStyle[k]=c[k];}});
    applyPtStyle();   // 点渲染参数回填后立即应用到当前 GLB
  }catch(e){/* 读取失败保持面板默认值 */}
}
HL_KEYS.forEach(k=>$('r_'+k).addEventListener('input',()=>{hlLabel(k);
  if(VIEW_KEYS.includes(k)){hlView[k]=+$('r_'+k).value;applyExpView();}   // 相机参数立即生效
  if(k.indexOf('pt_')===0){ptStyle[k]=+$('r_'+k).value;applyPtStyle();
    if(k==='pt_fov_off')applyExpView();}                                  // 点渲染参数立即生效
  pushHlCfg();}));
// radio 组与颜色控件：改动立即应用到当前 GLB 并下发持久化
document.querySelectorAll(
  Object.keys(PT_RADIOS).map(n=>'input[name='+n+']').join(',')).forEach(r=>
  r.addEventListener('change',()=>{
    Object.keys(PT_RADIOS).forEach(nm=>{
      ptStyle[PT_RADIOS[nm]]=+document.querySelector('input[name='+nm+']:checked').value;});
    applyPtStyle();pushHlCfg();}));
Object.keys(PT_COLORS).forEach(id=>$(id).addEventListener('input',()=>{
  ptStyle[PT_COLORS[id]]=$(id).value;applyPtStyle();pushHlCfg();}));
$('btnHlCfg').onclick=()=>{
  const on=!$('hlcfg').classList.contains('on');
  $('hlcfg').classList.toggle('on',on);
  if(on){loadHlCfg();ddpLoad();loadRdCfg();}   // 每次打开都回填服务端当前值 + 刷新配置预设列表
};
loadHlCfg();   // 启动即回填持久化配置：点渲染/视角参数在首个 GLB 加载前就绪，刷新页面不丢样式
$('hlcfgClose').onclick=()=>$('hlcfg').classList.remove('on');

if(DEMO){  // 演示模式：不连后端，用假数据目检三个视图的布局
  curCard={name:'Banana',description_en:'A quick source of everyday energy.',
           calories_kcal:89,protein_g:1.1,carbs_g:22.8,fat_g:0.3,
           classification:'Good'};
  renderCard(curCard);cardShownAt=Date.now();lastCardKey='demo';
  setInterval(()=>{cardShownAt=Date.now();},5000);   // 常驻成功态
  renderTimeline([
    {name:'Green tea latte',t:'20:45:00',status:'done'},
    {name:'Pasta carbonara',t:'19:20:00',status:'done'},
    {name:'Mango smoothie',t:'18:10:00',status:'done'},
    {name:'Chicken soup',t:'17:30:00',status:'done'},
    {name:'Blueberry yogurt',t:'14:50:00',status:'done'},
    {name:'Avocado toast',t:'12:35:00',status:'done'},
    {name:'Salmon sushi',t:'12:00:00',status:'done'},
    {name:'Apple',t:'10:15:00',status:'done'},
    {name:'Grilled steak',t:'08:20:00',status:'pending'},
    {name:'Grilled steak',t:'07:45:00',status:'pending'}]);
  setState('card');
}else{
  setState('idle');
  // 背景轮询：串行自排程（每轮结束按 delay 排下一轮，慢请求不堆积并发，同 /panel
  // f91e6f7 的自适应节奏）。设备深度图来源直接展示入帧，轮询按实测 fps 自适应
  // （30fps 时 ~33ms 一轮，显示帧率不再被固定 500ms 封顶）；GLB 类来源产物本来就是
  // 秒级一轮，维持 500ms 不空转高频轮询
  (function bgLoop(){
    Promise.resolve(bgTick()).catch(()=>{}).then(()=>{
      const delay=(bgSource==='devdepth'&&bgFps>0)
        ?Math.max(33,Math.min(500,1000/(bgFps*1.5))):500;
      setTimeout(bgLoop,delay);
    });
  })();
  // 识别结果 SSE 推送：连上即收全量快照，之后有变更即刻到达（断线 EventSource 自动重连）
  new EventSource('/api/recog/events').onmessage=ev=>{try{applyRecog(JSON.parse(ev.data))}catch(e){}};
  // 成功态驻留到期回落待机是纯本地状态，用轻量本地定时器驱动（不发请求）。
  // 250ms 而不是 1s：驻留可短到 1s，1s 的 tick 会把它拖成最多 2s（量化误差和驻留同量级）
  setInterval(()=>setState(curCard&&Date.now()-cardShownAt<DWELL_MS?'card':'idle'),250);
}

// ══ 临时诊断（屏闪排障）v2：URL 加 ?trace=1 开启逐帧采样——每个动画帧在
//    点云 canvas 的 5 个固定补丁（四角+中心，各 48×48 物理像素）直接 getImageData
//    逐像素平均亮度（无缩放无插值，规避极限缩小的采样混叠），连同当前帧 key 与
//    点云化配置回传 /api/flicker-report。诊断结束整段可移除 ══
if(new URLSearchParams(location.search).get('trace')==='1'){
  // v3：零扰动——不读 canvas 像素，每个动画帧只记录绘制函数自己记的账
  //（墨量 inkMass=Σ粒径²×α、粒点数 inkN、上次绘制时刻 inkT）+ 当前帧 key。
  // 墨量恒定而肉眼仍闪 ⇒ 闪在 canvas 之后（合成/显示）；墨量骤降 ⇒ 绘制逻辑
  const tbuf=[];
  (function tSample(){
    tbuf.push({t:Math.round(performance.now()),k:lastBgKey,
      it:Math.round(inkT),im:Math.round(inkMass),n:inkN});
    if(tbuf.length>2400)tbuf.shift();
    requestAnimationFrame(tSample);
  })();
  setInterval(()=>{
    fetch('/api/flicker-report',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({page_ts:Date.now(),total:tbuf.length,v:3,bld:'bake3-softknee',
        cfg:Object.assign({},dotCfg),dd:Object.assign({},ddCss),
        fdot:getComputedStyle($('bgDot')).filter,
        samples:tbuf.slice(-900)})
    }).catch(()=>{});
  },5000);
}
</script>
</body></html>"""


@app.get("/")
def home():
    """根路径直接进浅体验区展示页（主链路）；调试页从 /panel 的导航进。"""
    return RedirectResponse("/experience")


@app.get("/experience", response_class=HTMLResponse)
def experience():
    """IFA 浅体验区展示页：品牌化全屏实时识别 UI。no-store 防止排障期拿到缓存旧页。"""
    return HTMLResponse(EXPERIENCE_PAGE, headers={"Cache-Control": "no-store"})


@app.get("/panel", response_class=HTMLResponse)
def panel():
    """原 DA3 扩展面板已随 DA3 退役（2026-08-25），保留路由给出去向提示。"""
    return ('<!doctype html><html lang="zh"><head><meta charset="utf-8">'
            '<title>面板已下线</title></head><body style="font-family:sans-serif;'
            'max-width:36em;margin:4em auto;color:#1f2329">'
            '<h2>DA3 扩展面板已下线（2026-08-25）</h2>'
            '<p>DA3 深度/点云/网格能力已整体退役。设备帧与全部调节请用 '
            '<a href="/experience">/experience</a>（右下「调节」抽屉）；'
            'SAM3 调优见 <a href="/sam3tune">/sam3tune</a>。</p></body></html>')


# ── 临时诊断（屏闪排障）：/experience 页加 ?trace=1 开启逐帧亮度采样并回传，
#    这里内存单槽存最近一份报文供拉取分析；诊断结束整段可移除 ──
_flicker_trace: dict = {}


@app.post("/api/flicker-report")
async def flicker_report_post(body: dict = Body(...)):
    """存展示端回传的逐帧采样报文（新报覆盖旧报）。"""
    _flicker_trace["received_at"] = time.time()
    _flicker_trace["report"] = body
    return {"ok": True}


@app.get("/api/flicker-report")
def flicker_report_get():
    """读最近一份采样报文（含服务器收到时刻）。"""
    return _flicker_trace or {"report": None}


# ── 设备实时帧中继：接收 mobile 直发的帧 → 缓存展示 + DA3 深度处理 ──────────
# ══════════════════════════════════════════════════════════════════════
# 实时识别（Qwen3-VL 多模态）：把某帧送 GCP g4-01 的 Qwen3-VL（OpenAI 兼容）识别
# 「具体是什么」，产出 名称 + 类型(食物/液体)。触发有两种口径，二选一（见下方直传区块）：
#   · 直传（默认，主链路）：按固定间隔取选中设备最新 RGB 帧直接送，只发原图；
#   · SAM3（直传关闭时）：SAM3 命中某帧 → 取原图 + 带框图送。
#   · 识别是慢操作（多模态 LLM 数秒），放独立后台线程 + 节流，绝不阻塞 DA3 产线。
#   · 结果进 _recog_cards，前端 /recog 轮询 /api/recog/list 渲染卡片、名称流式打字。
#   · endpoint 经环境变量配置（默认空=不接入，右栏显示「识别服务未接入」）：
#       RECOG_ENDPOINT / RECOG_API_KEY / RECOG_MODEL / RECOG_MIN_INTERVAL
#   · 双识别目标可切（页面按钮，免重启、下一轮生效）：
#       qwen 目标=上面三个变量（历史名不动，兼容既有 .env）；
#       gemini 目标=RECOG_ENDPOINT_GEMINI / RECOG_API_KEY_GEMINI / RECOG_MODEL_GEMINI，
#       未配置 endpoint 的目标在页面上置灰不可切；RECOG_TARGET 指定启动默认目标。
# ══════════════════════════════════════════════════════════════════════
RECOG_ENDPOINT = os.environ.get("RECOG_ENDPOINT", "").strip()
RECOG_API_KEY = os.environ.get("RECOG_API_KEY", "").strip()
RECOG_MODEL = os.environ.get("RECOG_MODEL", "Qwen3.6-35B-A3B-FP8").strip()
# 识别目标注册表：页面切换按钮在两套 endpoint/key/model 间切，label 供前端展示
RECOG_TARGETS = {
    "qwen": {"label": "Qwen", "endpoint": RECOG_ENDPOINT,
             "api_key": RECOG_API_KEY, "model": RECOG_MODEL},
    "gemini": {"label": "Gemini Pro",
               "endpoint": os.environ.get("RECOG_ENDPOINT_GEMINI", "").strip(),
               "api_key": os.environ.get("RECOG_API_KEY_GEMINI", "").strip(),
               "model": os.environ.get("RECOG_MODEL_GEMINI", "gemini-3.1-pro-preview").strip()},
}
_recog_target = os.environ.get("RECOG_TARGET", "qwen").strip()
if _recog_target not in RECOG_TARGETS or not RECOG_TARGETS[_recog_target]["endpoint"]:
    _recog_target = "qwen"   # 非法或未配置的默认目标一律回退 qwen（与历史行为一致）
RECOG_MIN_INTERVAL = float(os.environ.get("RECOG_MIN_INTERVAL", "4.0"))  # 两次识别最小间隔(秒)，节流防刷屏
RECOG_TIMEOUT = 30.0
# 识别触发整轮周期下限（秒）：一轮 = 一次 SAM3 门控（词表 2 个词 = 2 个请求）+ 至多一次
# VLM，1s 下限即 SAM3 ≤2 qps、VLM ≤1 qps。关思考后识别只要 ~0.2s/轮，仅靠 interval_s(0.2)
# 节拍会空转到 SAM3 ~6 qps（空桌上 SAM3 误触发、VLM 秒答空数组也拦不住节奏），白烧
# 每轮 ~3900 token 的 prefill 与门控算力，日志也被灌满。
RECOG_MIN_CYCLE_S = float(os.environ.get("RECOG_MIN_CYCLE_S", "1.0"))
# 流式返回：开=stream=true 逐块收 SSE。收满才建卡的链路不变，但能把 http 段拆成
# 「网络往返+prefill」(ttft) 与「decode」两截——之前这两截混在一个 http_ms 里，
# 「这轮慢是慢在传图还是慢在生成」只能靠猜。出问题设 RECOG_STREAM=0 立刻回非流式。
RECOG_STREAM = os.environ.get("RECOG_STREAM", "1").strip() not in ("0", "false", "False")
RECOG_MAX_CARDS = 200
RECOG_ACTIVE_WINDOW = 30.0   # 去重活跃窗口(秒)：只在最后出现≤此窗口的卡里找重复(滑动窗口)
RECOG_MAX_CANDIDATES = 3     # 每次最多带几张候选参考图喂 VLM。8 → 3：候选越多，
                             # 「桌上有这类东西」的先验越强，线上实测过 Orange 与
                             # Clementine 同时在候选里互相强化，把爆米花袋一起带偏
# 卡片最长存活(秒)：合并会刷新 last_ts，错卡因此永远掉不出 30s 活跃窗口，
# 线上出现过一张卡自我确认 252 次（约 17 分钟）。first_ts 是绝对上限，到点必退场。
RECOG_CARD_MAX_LIFE = 120.0
# 强制复检周期：某张卡每被合并这么多次，就有一轮把它从候选里剔除、并关掉粘性，
# 让模型在**没有任何提示**的情况下重新看一次画面。仍报同名才认，否则这张卡就此老化——
# 这是唯一能主动打断「误报 → 自我确认 → 再误报」闭环的机制。
RECOG_RECHECK_EVERY = 15

_recog_lock = threading.Lock()
_recog_cv = threading.Condition(_recog_lock)
# 卡片变更通知（SSE 推送用）：任何新卡/合并/清空都把版本号 +1 并唤醒推送线程
_recog_gen = 0
_recog_evt_cv = threading.Condition(_recog_lock)
# device_id -> [{id,status,name,type,glb_url,frame,t}]，每设备一条独立卡片流（append 顺序=时间顺序）。
# 切换设备时卡片不清空：/api/recog/list 默认下发当前选中设备的桶，切回来卡片还在。
_recog_cards = {}
_recog_id = 0              # 卡片自增 id（全设备共用一个计数器，保证 id 全局唯一）
_recog_last_ts = 0.0       # 上次触发识别的时刻(节流用)
# device_id -> {"name","type","card_id","ts"}：上一轮真正上屏的那个物品。
# 展示端一次只显示一个，若每轮各挑各的，屏幕就会在几样东西之间反复闪；把上次
# 选中的告诉模型、让它「还在画面里就继续选它」，粘性从 prompt 层就建立了。
# 超过 RECOG_ACTIVE_WINDOW 视为过期（人早换东西吃了，再提上次没意义）。
_recog_last_pick = {}
# device_id -> 已落卡的最新**帧时刻**（wall clock）。识别链路原先没有任何时序保护
# （SAM3 流那侧早有 _stream_gen 代次机制，这边一直没有），并发>1 或隧道抖动时，
# 先发的请求可能后回来，把一张更新的卡覆盖成更旧的画面。落卡前拿它当水位线：
# 比水位线还旧的结果直接丢弃，不落卡、不更新 last_pick、不推 SSE。
# 用 frame_recv_at 而不是 stage["seq"]：设备桶闲置 60s 被清后 seq 会从 1 重计
# （见 frame_relay 撞键那条），拿它当水位线会永久卡死；wall clock 不受影响。
_recog_applied_at = {}
_recog_pending = []        # 待识别任务队列(worker 池消费，最新优先、丢弃积压)
_recog_workers = 0         # 已起的 worker 线程数（按并发配置只增不减；多余线程自行退出）

# ══════════════════════════════════════════════════════════════════════
# 主链路直传 VLM 识别（/experience 的「设备深度图」链路）
#   动机：识别触发历来挂 SAM3——单目链取 SAM3 流式液体框、双目链取左目 IR 的 SAM3
#   命中，SAM3 没定位到就完全不识别。直传去掉这层前置：按固定间隔取选中设备的最新
#   RGB 帧直接送 Qwen VLM 问「画面里可能有什么食物」。
#   · 开=整帧直送 VLM；关=同一帧先过 SAM3 门控（_sam3_gate_dets，直接跑 RGB 彩色帧），
#     认出食物/饮品才带框送 VLM——SAM3 只当"有没有东西"的前置门，命中后照样识别。
#   · 两种口径共用同一条触发线程与帧源，切换即时生效、互不干扰。
#   · 并发=1 时「串行·最新优先」（上一轮没回就丢旧帧用最新帧，实际节奏≈VLM 延时）；
#     调大并发即真按间隔多路齐发，代价是 GPU 成本 ×N 且并发轮次拿同一份去重候选、
#     可能给同一食物重复建卡。
# 配置：GET/POST /api/recog/direct/config（落盘 recog_direct_cfg.json，全局一份）
# ══════════════════════════════════════════════════════════════════════
_recog_direct = recog_direct.DirectConfig(
    Path(__file__).resolve().parent / "recog_direct_cfg.json")
_recog_direct_stats = {"rounds": 0, "in_flight": 0, "last_ms": 0,
                       "last_ts": 0.0, "skipped_same_frame": 0, "last_error": "",
                       # SAM3 门控（直传关闭时）观测：单轮耗时 + 命中/空轮计数
                       "gate_ms": 0, "gate_hits": 0, "gate_misses": 0,
                       # 陈旧帧截断：丢弃轮次数 + 最近一次丢弃时的帧龄（排积压时看这两个）
                       "dropped_stale": 0, "last_drop_age_ms": 0,
                       # 乱序丢弃：结果回来时发现比已落卡的帧还旧
                       "dropped_out_of_order": 0}
print(f"[da3-web] 直传识别配置：{_recog_direct.snapshot()}", flush=True)

# 食物健康分级的合法枚举（静态 guardrail 白名单：模型输出只保留集合内的值、其余置空）。
FOOD_CLASSIFICATIONS = ["Good", "Neutral", "Bad"]
_CLS_CANON = {c.lower(): c for c in FOOD_CLASSIFICATIONS}   # 小写→规范写法，校验大小写不敏感
RECOG_DESC_MAX = 60         # 英文描述最大字符数


def _recog_num(v, lo, hi, as_int=False):
    """营养数字字段 guardrail：转数字并夹进 [lo, hi]；非法/NaN 返回 None（前端隐藏该行）。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:   # NaN
        return None
    f = min(hi, max(lo, f))
    return int(round(f)) if as_int else round(f, 1)

def _img_data_uri(rgb):
    """RGB ndarray → data URI（JPEG）。"""
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return ("data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()) if ok else None


SHOT_DIR = GLB_DIR.parent / "recog_shots"   # 识别缩略图目录（点云服务端渲染图）
SHOT_KEEP = 160                             # 磁盘最多保留的缩略图数量


def _save_cloud_shot(pred, dets, conf):
    """识别缩略图：用该帧 DA3 pred 走与②/③相同的服务端点云渲染链路，
    存盘返回 /shotimg url；渲染失败返回 None。
    dets: [(label, nx1, ny1, nx2, ny2)]，坐标 0-1 归一化（左上、右下）；空则渲无框点云。
    识别链路自模型输出去掉 box 起一律传空，框由别的调用方（若将来有）自带。"""
    try:
        # 相机模型已改为「调优视角」（对准点云中心、按场景深度定距，场景相对不跳帧）：
        # 缩略图取稍近距离(1.0×) + 俯视 20°；splat=1 + out_size=1140 保持点状离散
        # （点距>点径才处处离散成"碎点"，760 时中近距离会糊成片）
        img = _render_pointcloud_image(pred, dets or None, conf_thresh_percentile=conf,
                                       view_tilt=20.0, view_zoom=1.0, splat=1,
                                       out_size=1140, eye_lift=0.0, eye_back=0.0,
                                       color_grade=(0.0, 0.75))   # 饱和0=纯黑白点云
    except Exception as e:
        print(f"[da3-web] 识别缩略图渲染失败：{type(e).__name__}: {e}", flush=True)
        return None
    if img is None:
        return None
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        return None
    SHOT_DIR.mkdir(parents=True, exist_ok=True)
    name = uuid.uuid4().hex + ".jpg"
    (SHOT_DIR / name).write_bytes(buf.tobytes())
    files = sorted(SHOT_DIR.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in files[SHOT_KEEP:]:             # 只留最近 SHOT_KEEP 张
        try:
            p.unlink()
        except OSError:
            pass
    return f"/shotimg/{name}"


# ══════════════════════════════════════════════════════════════════════
# 参考食物库：录入的实物图 → 每轮请求最前面那段固定「参考清单」
#   动机：识别链路原本让 VLM 从零现编名称/描述/营养，同一根香蕉两轮之间卡路里
#   会从 89 跳到 105，展台上肉眼可见。参考库把展台真正要摆的那几种食物先录进来
#   （实物图 + 营养 + 描述），识别时把这批图钉在请求最前面，VLM 只回答「桌上那个
#   是不是清单里第几项」，命中就由服务端查库回填——数字不再由模型编，逐轮恒定。
#   · 前缀必须逐字节稳定：参考图在**录入时**就按当前档位规范化并落盘，之后每轮
#     复用同一份文件字节（不重新编码），vLLM 的 prefix cache 才能命中；顺序恒按
#     id 升序（见 foodref.Catalog.menu_items）。
#   · 换档位（edge/quality）不丢已有产物：规范化结果按 (id, 序号, 边长, 质量) 落盘
#     缓存，换回旧档位直接命中老文件，不必重新处理原图。
#   · 目录与配置态在 foodref.py（纯逻辑可单测），这里只管图片处理与接口。
# ══════════════════════════════════════════════════════════════════════
FOODREF_DIR = Path(__file__).resolve().parent / "food_ref"
FOODREF_CACHE_DIR = FOODREF_DIR / "cache"      # 规范化产物（送 VLM 的那一份）
FOODREF_ORIG_MAX = 2048        # 上传原图存盘前的长边上限：换档位重算够用，又不撑爆磁盘
FOODREF_ORIG_Q = 92            # 原图副本的 JPEG 质量（它只是重算的源，不进请求）
_foodref = foodref.Catalog(Path(__file__).resolve().parent / "food_catalog.json")
_foodref_lock = threading.Lock()
_foodref_uri_cache = {}        # 缓存文件名 → dataURI（避免每轮读盘 + base64 40 张）
_foodref_menu_cache = {"version": -1}   # 整段参考区（blocks + 元信息）的内存缓存
print(f"[da3-web] 参考食物库：{len(_foodref.menu_items())} 种参与识别，"
      f"配置 {_foodref.config()}", flush=True)


def _foodref_orig_path(item_id, n):
    return FOODREF_DIR / ("%d_%d.orig.jpg" % (int(item_id), int(n)))


def _foodref_cache_path(item_id, n, edge, quality):
    return FOODREF_CACHE_DIR / ("%d_%d_e%dq%d.jpg" % (int(item_id), int(n),
                                                      int(edge), int(quality)))


def _make_menu_img(bgr, edge, quality):
    """录入原图 → 送 VLM 的参考图：等比缩放 + 顶部 REFERENCE 横幅 + 灰边框，返回 JPEG 字节。

    横幅与边框是防幻觉的视觉防线：实测过画面空无一物时模型照着参考图报 Snickers，
    参考区又被放到了请求最前面，风险更高——图上写死「不是当前画面」，配合 prompt
    里的明令与服务端的证据校验，三道一起用。
    横幅不写清单编号：编号会随增删变动，写进图里会让所有缓存产物在删一项后全部失效；
    编号由紧挨着的文字标签行承担。"""
    h, w = bgr.shape[:2]
    nw, nh = foodref.fit_size(w, h, edge)
    out = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    bh = max(20, nh // 10)
    cv2.rectangle(out, (0, 0), (nw, bh), (90, 90, 90), -1)
    cv2.putText(out, "REFERENCE - NOT CURRENT FRAME", (6, int(bh * 0.74)),
                cv2.FONT_HERSHEY_SIMPLEX, bh / 46.0, (255, 255, 255),
                max(1, bh // 18), cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (nw - 1, nh - 1), (90, 90, 90), max(2, nw // 110))
    ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return buf.tobytes() if ok else None


def _foodref_decode_upload(raw):
    """解码一张上传图为 BGR 数组：cv2 直解（jpg/png），失败再走 Pillow 兜底（HEIC 等）。

    iPhone 相册默认 HEIC，cv2 不认；pillow-heif 注册 opener 后 PIL 能开。Pillow 路径
    顺带按 EXIF 方向转正（cv2.imdecode 本就不理 EXIF，jpg/png 老行为保持不变）。
    pillow-heif 未安装时只影响 HEIC，jpg/png 不受牵连。"""
    arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if arr is not None:
        return arr
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except ImportError:
        print("[da3-web] 参考食物库：未安装 pillow-heif，HEIC 上传不可用", flush=True)
    try:
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception:
        return None
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def _foodref_save_original(item_id, n, raw):
    """存一张上传原图（长边限 FOODREF_ORIG_MAX），返回图片元信息；失败返回 None。"""
    arr = _foodref_decode_upload(raw)
    if arr is None:
        return None
    h, w = arr.shape[:2]
    if max(w, h) > FOODREF_ORIG_MAX:
        s = FOODREF_ORIG_MAX / float(max(w, h))
        arr = cv2.resize(arr, (max(1, round(w * s)), max(1, round(h * s))),
                         interpolation=cv2.INTER_AREA)
        h, w = arr.shape[:2]
    ok, buf = cv2.imencode(".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), FOODREF_ORIG_Q])
    if not ok:
        return None
    FOODREF_DIR.mkdir(parents=True, exist_ok=True)
    _foodref_orig_path(item_id, n).write_bytes(buf.tobytes())
    return {"n": int(n), "w": int(w), "h": int(h), "bytes": len(buf.tobytes())}


def _foodref_ref_uri(item_id, n, edge, quality):
    """取某张参考图的 dataURI：内存缓存 → 磁盘缓存 → 从原图重算并落盘。

    三级都命中不了（原图丢了）返回 None，该项本轮不进参考区。"""
    path = _foodref_cache_path(item_id, n, edge, quality)
    key = path.name
    hit = _foodref_uri_cache.get(key)
    if hit:
        return hit
    data = None
    if path.exists():
        try:
            data = path.read_bytes()
        except OSError:
            data = None
    if data is None:
        src = _foodref_orig_path(item_id, n)
        if not src.exists():
            return None
        arr = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if arr is None:
            return None
        data = _make_menu_img(arr, edge, quality)
        if not data:
            return None
        FOODREF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".jpg.tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
        print("[da3-web] 参考图规范化：%s（%d 字节）" % (path.name, len(data)), flush=True)
    uri = "data:image/jpeg;base64," + base64.b64encode(data).decode()
    _foodref_uri_cache[key] = uri
    return uri


def _foodref_menu():
    """本轮要发的参考区：返回 (items, content_blocks, meta)。

    items 的下标 +1 就是模型要回填的 ref_id，落卡时也拿同一份做命中回填——
    两处必须是同一个快照，否则并发换库时编号会错位。
    blocks 是 OpenAI content 数组片段：开场白 → [标签行 + 该项的图]×N。"""
    if not _foodref.enabled():
        return [], [], {"items": 0, "images": 0, "tokens": 0, "bytes": 0, "version": 0}
    version = _foodref.version()
    with _foodref_lock:
        cached = _foodref_menu_cache
        if cached.get("version") == version:
            return cached["items"], cached["blocks"], cached["meta"]
    cfg = _foodref.config()
    edge, quality = cfg["edge"], cfg["quality"]
    items, blocks, n_images = foodref.build_blocks(
        _foodref.menu_items(),
        lambda item_id, n: _foodref_ref_uri(item_id, n, edge, quality))
    n_bytes = sum(len(b["image_url"]["url"]) for b in blocks
                  if b.get("type") == "image_url") * 3 // 4    # base64 → 原始字节
    meta = {"items": len(items), "images": n_images,
            "tokens": n_images * foodref.est_tokens(edge),
            "bytes": n_bytes, "edge": edge, "version": version}
    with _foodref_lock:
        _foodref_menu_cache.clear()
        _foodref_menu_cache.update({"version": version, "items": items,
                                    "blocks": blocks, "meta": meta})
    print("[da3-web] 参考区重建（版本 %d）：%d 种 / %d 张 / 约 %d token / %.0f KB" % (
        version, meta["items"], meta["images"], meta["tokens"], meta["bytes"] / 1024.0),
        flush=True)
    return items, blocks, meta


def _foodref_drop_cache(item_id):
    """作废某条的规范化产物（重传图片后必须调用）。

    缓存文件名只含 (id, 序号, 边长, 质量)，不含内容哈希——换了原图而文件名不变，
    不清缓存就会一直发着旧图。这里连内存里的 dataURI 一起清。"""
    for path in list(FOODREF_CACHE_DIR.glob("%d_*.jpg" % int(item_id))):
        try:
            path.unlink()
        except OSError:
            pass
        _foodref_uri_cache.pop(path.name, None)


def _foodref_drop_files(item_id):
    """删条目时清掉它的原图与所有档位的缓存产物（含内存里的 dataURI）。"""
    _foodref_drop_cache(item_id)
    for path in list(FOODREF_DIR.glob("%d_*.orig.jpg" % int(item_id))):
        try:
            path.unlink()
        except OSError:
            pass


REF_IMG_EDGE = 256      # 历史参考图的长边：整帧 900 token → 裁剪后约 64~100 token


def _make_ref_img(rgb, box=None):
    """历史参考图：按检测框裁出物体特写 → 缩到 REF_IMG_EDGE → 打 HISTORY REF 横幅与灰边框。

    两条防线各治一个病：
      · **裁剪**治「证据码全 1」——参考图过去是整帧，与当前画面九成像素相同，
        四项对照必然一致；裁成特写后才重新有区分度（box=None 时退化为中心 60% 裁剪）；
      · **缩到 256** 治「注意力预算」——8 张整帧参考图曾占掉七成以上视觉 token，
        当前画面反而只占一成；缩完之后比例回到 1:1 量级。
    横幅与边框沿用原防线：实测画面空无一物时，模型会照着候选参考图把 Snickers
    当当前画面物品输出，视觉标记 + prompt 明令双管齐下把两者硬隔离。"""
    H0, W0 = rgb.shape[:2]
    if box is None:                      # 没有检测框（直传口径）：退化成中心 60% 裁剪
        box = (0.2, 0.2, 0.8, 0.8)
    x1 = max(0, min(W0 - 1, int(box[0] * W0)))
    y1 = max(0, min(H0 - 1, int(box[1] * H0)))
    x2 = max(x1 + 1, min(W0, int(box[2] * W0)))
    y2 = max(y1 + 1, min(H0, int(box[3] * H0)))
    out = rgb[y1:y2, x1:x2].copy()
    h, w = out.shape[:2]
    if max(w, h) > REF_IMG_EDGE:
        scale = REF_IMG_EDGE / float(max(w, h))
        out = cv2.resize(out, (max(32, round(w * scale)), max(32, round(h * scale))),
                         interpolation=cv2.INTER_AREA)
    H, W = out.shape[:2]
    bh = max(28, H // 12)
    cv2.rectangle(out, (0, 0), (W, bh), (90, 90, 90), -1)
    cv2.putText(out, "HISTORY REF - NOT CURRENT FRAME", (10, int(bh * 0.72)),
                cv2.FONT_HERSHEY_SIMPLEX, bh / 44.0, (255, 255, 255),
                max(1, bh // 16), cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (W - 1, H - 1), (90, 90, 90), max(4, W // 100))
    return out


def _draw_boxes(rgb, detections):
    """在原图上画 food(红)/drink(蓝) 检测框，返回带框图(RGB)。"""
    out = rgb.copy()
    H, W = out.shape[:2]
    for (label, nx1, ny1, nx2, ny2) in detections:
        color = (222, 52, 52) if label == "food" else (46, 120, 235)  # RGB
        cv2.rectangle(out, (int(nx1 * W), int(ny1 * H)), (int(nx2 * W), int(ny2 * H)),
                      color, max(2, W // 300))
    return out


def _parse_recog(content):
    """从模型输出里抽 items 并对每个字段做静态 guardrail 校验（容错：截第一个 JSON 对象）。

    校验规则（每字段一个静态 guardrail）：
      name           非空、限 40 字，否则丢弃该 item；
      edible         必须为布尔 true（宽容字符串 "true"），否则丢弃该 item——
                     这是「只展示能进嘴的食物/饮料」的服务端硬闸：识别模型迁到
                     5090 本机 Qwen3.6 后约束遵循变弱，出现过把手机/电脑/空容器
                     当 item 输出的漏网，prompt 约束之外必须有这道兜底；
      type           枚举 → 归一到 “食物”/“液体”（模型给的是英文 food/drink）；
      description_en  字符串、去换行、限 RECOG_DESC_MAX 字符；
      calories_kcal  整数、夹到 [0,5000]，非法置 None；
      protein_g/carbs_g/fat_g  数字（1 位小数）、夹到 [0,500]，非法置 None；
      classification 枚举白名单 FOOD_CLASSIFICATIONS，非法值置空；
      match_evidence 只截长度（证据码合法性由 recog_match 在闸门里判）；
      seen/cur_text/diff  同样只截长度，判定交给 recog_match.check_self_evidence。"""
    try:
        obj = json.loads(content[content.index("{"): content.rindex("}") + 1])
    except Exception:
        return []
    items = obj.get("items") if isinstance(obj, dict) else None
    if not isinstance(items, list):
        return []
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        name = str(it.get("name", "")).strip()[:40]
        if not name:
            continue
        # edible 硬闸：模型没明确说 true 就不落卡（缺字段、false、乱值一律丢）。
        # 原始返回仍完整进观测日志（raw），被丢的条目在日志里可见，不影响排障。
        if it.get("edible") not in (True, "true", "True"):
            continue
        typ = str(it.get("type", "")).strip().lower()
        is_liquid = ("液" in typ or "饮" in typ or "drink" in typ or "liquid" in typ)
        desc_en = str(it.get("description_en", "")).strip().replace("\n", " ")[:RECOG_DESC_MAX]
        kcal = _recog_num(it.get("calories_kcal"), 0, 5000, as_int=True)
        protein = _recog_num(it.get("protein_g"), 0, 500)
        carbs = _recog_num(it.get("carbs_g"), 0, 500)
        fat = _recog_num(it.get("fat_g"), 0, 500)
        cls = _CLS_CANON.get(str(it.get("classification", "")).strip().lower(), "")  # 非法→置空
        try:
            match = int(it.get("match"))     # 命中的候选编号；范围校验在 worker（要对齐候选数）
        except (TypeError, ValueError):
            match = None
        # 证据码最长 8 字符（BxCxSxVx）/ NONE；留点余量让非法码进日志可读，
        # 合法性交给 recog_match.check_evidence，这里不做语义判断
        evidence = str(it.get("match_evidence", "")).strip().replace("\n", " ")[:16]
        mname = str(it.get("matched_name") or "").strip()[:40]
        # match_confidence 已从输出契约移除（2026-08-24）：它是证据码的确定性函数，
        # 置信度改由 recog_match.derive_confidence 在闸门四现场推导，这里不再解析
        # 当前画面自证三件套（治「照着参考图编」）：seen 是任务一强制先写的观察，
        # cur_text 是照抄的包装文字（B 位填 1 的抵押物），diff 是与候选的否定证据。
        # 这里只做长度/换行的 guardrail，合法性交给 recog_match.check_self_evidence。
        seen = str(it.get("seen", "")).strip().replace("\n", " ")[:recog_prompt.SEEN_MAX]
        cur_text = str(it.get("cur_text", "")).strip().replace("\n", " ")[:recog_prompt.CUR_TEXT_MAX]
        diff = str(it.get("diff", "")).strip().replace("\n", " ")[:recog_prompt.DIFF_MAX]
        # 参考食物库的命中三件套（没开参考库时模型不会给，一律留空/None）：
        # 范围校验放在 foodref.resolve_hit（要对齐本轮清单长度），这里只做类型归一
        try:
            ref_id = int(it.get("ref_id"))
        except (TypeError, ValueError):
            ref_id = None
        ref_conf = str(it.get("ref_confidence") or "").strip().lower()
        ref_conf = ref_conf if ref_conf in foodref.CONFIDENCE_ORDER else ""
        ref_ev = str(it.get("ref_evidence") or "").strip().replace("\n", " ")[:120]
        out.append({"seen": seen, "cur_text": cur_text, "diff": diff,
                    "ref_id": ref_id, "ref_confidence": ref_conf, "ref_evidence": ref_ev,
                    "name": name, "type": "液体" if is_liquid else "食物",
                    "description_en": desc_en,
                    "calories_kcal": kcal, "protein_g": protein,
                    "carbs_g": carbs, "fat_g": fat, "classification": cls,
                    "match": match, "match_evidence": evidence,
                    "matched_name": mname})
    return out


# ══════════════════════════════════════════════════════════════════════
# VLM 识别观测日志：每一轮识别请求的「请求图 + prompt + 模型原始返回 + 解析结果
# + 去重判定」整轮留痕，供浅体验区控制面（superadmin /ifa-support/experience）
# 可视化排障。服务端内存环形缓冲，不落盘。
#   · 与卡片流的区别：卡片是「桌上现在有什么」的结果态，日志是「这一轮发了什么图、
#     问了什么、模型答了什么、五道闸门怎么判的」的过程态——漏识别 / 幻觉 / 错并
#     只能在过程态里看出来，stdout 那份审计日志上了展台没人去 ssh 翻。
#   · 缓冲与投影是纯逻辑，收在 recog_log.py（零 cv2/torch 依赖，可单测）；
#     图片编码留在本文件（_thumb_uri），日志模块只搬运字符串。
# ══════════════════════════════════════════════════════════════════════
VLMLOG_MAX = 30            # 环形条数上限（每条含请求图 dataURI，别放太大）
VLMLOG_RAW_MAX = 20000     # 单条留存的模型原始返回上限（字符），超长截断
_vlmlog = recog_log.RecogLog(VLMLOG_MAX, VLMLOG_RAW_MAX)


def _vlmlog_begin(device, trigger, candidates, n_food, n_drink, orig_rgb, boxed_rgb):
    """开一条识别日志（请求侧快照）：识别调用填 req/resp，worker 补 outcome 后 commit。"""
    entry = _vlmlog.begin(device, trigger, candidates, n_food, n_drink,
                          img_orig=_thumb_uri(orig_rgb),
                          img_boxed=_thumb_uri(boxed_rgb) if boxed_rgb is not None else None)
    entry["ts"] = time.time()
    return entry


def _recognize_dedup(orig_rgb, boxed_rgb, candidates, n_food=0, n_drink=0, target=None,
                     log=None, last_pick=None, refs=None):
    """调多模态 VLM 识别 + 去重。一次多图请求：图1原图 + 图2带框图 + 各候选参考图(带横幅标记)。
    candidates: [{"id","name","type","desc","ref_img"}...]（顺序即参考图编号）。
    n_food/n_drink：当前画面检测器命中数，作为软接地信息进 prompt。
    boxed_rgb=None → 主链路直传口径：无 SAM3 框，只发图1 + 参考图，prompt 同步换成
    直传版（参考图编号从图2起）。
    target：识别目标预设（RECOG_TARGETS 里的一项，缺省=当前选中目标）。
    log：观测日志条目（_vlmlog_begin 建的 dict），非空则把请求与原始返回填回去。
    返回 [{name,type,description,...,match,matched_name}]；任何失败返回 []。"""
    cfg = target or RECOG_TARGETS[_recog_target]
    if not cfg["endpoint"]:
        _vlmlog.set_response(log, False, error="识别服务未接入（endpoint 为空）")
        return []
    direct = boxed_rgb is None
    _t_enc = time.time()      # 本地编码段起点：两张原图 JPEG + base64 + JSON 序列化
    u1 = _img_data_uri(orig_rgb)
    u2 = None if direct else _img_data_uri(boxed_rgb)
    if not u1 or (not direct and not u2):
        _vlmlog.set_response(log, False, error="请求图编码失败")
        return []
    # 参考食物库：清单图独占**第一条消息**，且逐字节固定（顺序按 id 升序、图片是录入时
    # 编码好的同一份字节）——vLLM 的 prefix cache 按前缀命中，任何每轮会变的东西
    # （当前帧、去重候选、上次选中）都必须留在后面那条消息里，否则整段前缀白算。
    ref_items, ref_blocks, ref_meta = refs if refs else ([], [], {})
    cands = [c for c in (candidates or []) if c.get("ref_img")]   # 无图的不进清单，编号才对得齐
    # ── 本条消息的排布（2026-08-18 重排，治「参考图污染当前画面」）──────────
    #   [固定段] 指称约定 + 硬约束 + 任务零 + 任务一 + 判同流程 + JSON 示例
    #   [可变段] 任务二清单 → 逐个「标签 + 参考图」→ 当前画面 → 带框图 → 收尾
    # 两条要害：每张图前面都有方括号标签（旧版靠数序数，十几张图根本对不齐）；
    # 当前画面排在所有参考图之后，成为离生成最近的那张图（旧版它在最前面）。
    content = [{"type": "text", "text": recog_prompt.fixed_head(
        direct, ref_items, _foodref.config()["min_confidence"])}]
    content.append({"type": "text", "text": recog_prompt.candidates_intro(cands)})
    for i, c in enumerate(cands):
        content.append({"type": "text", "text": recog_prompt.candidate_label(i + 1, c)})
        content.append({"type": "image_url", "image_url": {"url": c["ref_img"]}})
    content.append({"type": "text", "text": recog_prompt.current_label(bool(cands))})
    content.append({"type": "image_url", "image_url": {"url": u1}})
    if u2:
        content.append({"type": "text", "text": recog_prompt.boxed_label(n_food, n_drink)})
        content.append({"type": "image_url", "image_url": {"url": u2}})
    content.append({"type": "text", "text": recog_prompt.tail(last_pick, cands)})
    prompt = recog_prompt.render_for_log(content)   # 控制面读的是这一个字符串
    # temperature=0：判同结果下游有硬合并动作，消除轮间抖动、日志巡检可复现；
    # max_tokens 1536：每 item 的输出字段不少（营养四数字
    # + classification），多物品帧 1024 有截断风险——截断会被 _parse_recog 的容错
    # 整体吞掉 items，表现为静默丢识别。
    messages = []
    if ref_blocks:
        # 三段式（清单 → 固定回执 → 当前画面）而不是一条消息塞完：前缀边界正好落在
        # 回执那句上，缓存块对齐更干净，语义上也把「参考清单」与「当前画面」硬隔开。
        messages.append({"role": "user", "content": ref_blocks})
        messages.append({"role": "assistant", "content": foodref.menu_ack(len(ref_items))})
    messages.append({"role": "user", "content": content})
    # enable_thinking=False：5090 本机的 Qwen3.6 是混合思考模型且**默认开思考**，
    # 思考散文会顶满 max_tokens 把 JSON 挤没（截断被 _parse_recog 容错吞掉，表现为
    # 静默 0 项）。GCP 时代跑的是 Instruct（非思考）权重，这里显式关掉思考对齐当时表现。
    payload = {"model": cfg["model"], "messages": messages,
               "max_tokens": 1536, "temperature": 0,
               "chat_template_kwargs": {"enable_thinking": False}}
    if RECOG_STREAM:
        # include_usage：流式下 token 数只在最后一个 usage-only chunk 里给
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
    body = json.dumps(payload).encode()      # 序列化几百 KB base64 也要算进本地耗时
    encode_ms = (time.time() - _t_enc) * 1000.0
    if log is not None:
        # img_full/img_boxed_full = 真正送进请求体的那两张图（原帧原尺寸、cv2 默认 q95），
        # 直接引用同一个 dataURI 字符串：不重复编码、不额外占内存。详情页据此判断
        # 「模型是不是因为图太糊/没拍到才认错」——列表里的缩略图回答不了这个问题。
        log["req"] = {"label": cfg["label"], "model": cfg["model"],
                      "endpoint": cfg["endpoint"], "direct": direct,
                      "n_images": sum(1 for b in content if b.get("type") == "image_url"),
                      "prompt": prompt,
                      # 参考区规模单独留一格：排障时「这一轮到底带没带清单、带了多大」
                      # 是第一个要问的问题，混进 n_images 里就分不出来了
                      "ref": dict(ref_meta) if ref_meta else None,
                      "ref_prompt": (foodref.menu_intro(len(ref_items), ref_meta.get("images", 0))
                                     + "\n" + "\n".join(
                                         foodref.item_label(i + 1, it)
                                         for i, it in enumerate(ref_items))
                                     if ref_items else None),
                      "max_tokens": payload["max_tokens"],
                      "temperature": payload["temperature"],
                      "stream": bool(payload.get("stream")),
                      "img_full": u1, "img_boxed_full": u2,
                      "img_full_px": "%dx%d" % (orig_rgb.shape[1], orig_rgb.shape[0])}
    headers = {"Content-Type": "application/json"}
    if cfg["api_key"]:
        headers["Authorization"] = f"Bearer {cfg['api_key']}"
    _t_http = time.time()
    ttft_ms = None          # 首个内容块到达耗时（仅流式有）：网络往返 + 服务端 prefill
    try:
        req = urllib.request.Request(cfg["endpoint"], data=body, headers=headers)
        with urllib.request.urlopen(req, timeout=RECOG_TIMEOUT) as r:
            if payload.get("stream"):
                out, usage, ttft_ms = recog_sse.read_sse_completion(
                    r, _t_http, RECOG_TIMEOUT, time.time)
            else:
                data = json.loads(r.read().decode())
                out = data["choices"][0]["message"]["content"]
                # vLLM 的 OpenAI 兼容响应**不带服务端耗时**，只有 token 数——拿它解释
                # 「这轮为什么慢」：输出 token 越多解码越久，是 http 段里最可归因的一项
                usage = data.get("usage") or {}
    except Exception as e:
        print(f"[da3-web] 识别调用失败：{type(e).__name__}: {e}", flush=True)
        _vlmlog.set_response(log, False, error=f"{type(e).__name__}: {e}",
                             timings={"encode_ms": round(encode_ms, 1),
                                      "http_ms": round((time.time() - _t_http) * 1000.0, 1)})
        return []
    http_ms = (time.time() - _t_http) * 1000.0
    _t_parse = time.time()
    items = _parse_recog(out)
    # 三段的边界说明（前端照此展示，别再靠猜）：
    #   encode = 本机 CPU（两张原图 JPEG q95 + base64 + JSON 序列化，几百 KB）
    #   http   = 发请求到收完响应：网络往返(经隧道) + 服务端排队 + 模型推理 + 回传
    #   ttft   = http 里到「首个内容块」为止的那截：上行传图 + 排队 + prefill；
    #            http - ttft 即 decode 段（仅流式；非流式为 None，拆不开）
    #   parse  = 解析模型输出 + 字段 guardrail
    _vlmlog.set_response(log, True, raw=out, items=items,
                         timings={"encode_ms": round(encode_ms, 1),
                                  "http_ms": round(http_ms, 1),
                                  "parse_ms": round((time.time() - _t_parse) * 1000.0, 1),
                                  "req_bytes": len(body),
                                  "prompt_tokens": usage.get("prompt_tokens"),
                                  "completion_tokens": usage.get("completion_tokens"),
                                  "ttft_ms": (round(ttft_ms, 1) if ttft_ms is not None else None)})
    return items


def _name_tokens(s):
    """名称 → 词面 token 集（拉丁按词、中文按字符 2-gram，单字退化为单字）。
    用于「合并名称零重叠」闸门：模型合并时被要求照抄候选名，实际输出名也强烈
    倾向沿用候选名，零重叠+high 是罕见且高危的错并组合，确定性拦截、误杀极少。"""
    s = (s or "").strip().lower()
    toks = set(re.findall(r"[a-z0-9]+", s))
    cjk = re.findall(r"[一-鿿]", s)
    toks |= {a + b for a, b in zip(cjk, cjk[1:])} or set(cjk)
    return toks


def _recog_worker(idx=0):
    """后台 worker：取最新识别任务、丢弃积压，识别 + 去重：
      duplicate 命中且通过全部闸门 → 合并到那张卡（只追加缩略图、刷新 last_ts、rev+1，内容不改）；
      否则 → 新建卡（ref_img=整帧带框图；模型输出已无 box，不再裁本物品特写）。
    合并闸门共五道（宁拒勿并，错并比重复建卡严重）：回显校验 → 类型一致 →
    证据自洽 → high 置信 → 名称零重叠拦截。

    idx=本 worker 在池中的序号：直传并发上限调小后，序号越界的 worker 干完当前
    一轮自行退出（池子只在需要时增长，不留常驻空转线程）。"""
    global _recog_id, _recog_gen, _recog_workers
    while True:
        with _recog_cv:
            while not _recog_pending:
                _recog_cv.wait()
            if idx >= max(1, _recog_direct.concurrency()):
                _recog_workers -= 1      # 并发被调小：多余 worker 退场
                _recog_cv.notify_all()   # 把刚才领到的唤醒还回去，别把任务闷死
                return
            orig, boxed, glb_url, frame, t, candidates, n_food, n_drink, enq_ts, pred, conf, dev, \
                stage, extra = _recog_pending.pop()     # 最新优先
            dets = extra.get("dets") or []            # 裁参考图用的本轮检测框
            recheck_ids = extra.get("recheck") or []  # 本轮被强制复检的卡
            _recog_pending.clear()                                             # 丢弃积压，防慢识别拖垮
            # 上次命中：过期的不再喂给模型（人早换东西吃了，再提上次只会误导）
            lp = _recog_last_pick.get(dev)
            if lp and time.time() - lp.get("ts", 0) > RECOG_ACTIVE_WINDOW:
                lp = None
            if lp and lp.get("card_id") in (extra.get("recheck") or []):
                lp = None      # 复检轮：连"上一轮选中的是什么"都不告诉模型
            _recog_direct_stats["in_flight"] += 1
        wait_ms = (time.time() - enq_ts) * 1000.0     # 在队列里等 worker 的时长
        # ── 陈旧帧截断：帧太旧就整轮丢掉，别发出去 ──────────────────────
        # 一轮慢请求（隧道抖一下就是十几秒）会让后面的帧在队列里干等；等轮到它们
        # 时画面早过去了。识别一张十几秒前的画面既没有展示价值，还要占住隧道让
        # 后面更堵——一次抖动就这样被放大成持续积压。阈值走控制面配置。
        # 卡在这里而不是更靠后：下面 _make_ref_img + JPEG 编码是实打实的 CPU，
        # 注定要丢的轮次不该先烧一遍。
        age_ms = ((time.time() - stage["frame_recv_at"]) * 1000.0
                  if stage.get("frame_recv_at") else None)
        max_age_ms = _recog_direct.max_frame_age_s() * 1000.0
        if age_ms is not None and age_ms > max_age_ms:
            with _recog_lock:
                _recog_direct_stats["in_flight"] -= 1
                _recog_direct_stats["dropped_stale"] += 1
                _recog_direct_stats["last_drop_age_ms"] = int(age_ms)
            print("[da3-web] 陈旧帧截断：帧龄 %.1fs > 上限 %.1fs（排队 %.0fms），本轮不发" % (
                age_ms / 1000.0, max_age_ms / 1000.0, wait_ms), flush=True)
            continue          # 回去领下一个（大概率是刚入队的新鲜帧）
        # 新卡代表图：整帧带框图加 HISTORY REF 横幅。同轮所有新卡共享这一张——
        # 模型输出去掉 box 后没法再裁本物品特写，这条曾是 over-merge 的根因路径，
        # 属去 box 的已知代价。直传口径没有带框图（boxed=None），用原帧。
        # 新卡代表图：按本轮最大的检测框，从**原图**（不是带框图，红蓝框会干扰判同）
        # 裁出物体特写。取框规则在 recog_match.pick_ref_box，无框时中心裁剪。
        boxed_uri = _img_data_uri(_make_ref_img(orig, recog_match.pick_ref_box(dets)))
        tgt = RECOG_TARGETS[_recog_target]   # 快照本轮识别目标：切换只影响后续轮次
        # 控制面观测：整轮留痕（请求图/prompt/原始返回/解析结果/去重判定），
        # 失败轮也要 commit——「这一轮压根没调通」正是最该被看见的一种日志
        vlog = _vlmlog_begin(dev, "direct" if boxed is None else "sam3",
                             candidates, n_food, n_drink, orig, boxed)
        # 参考清单快照：请求与落卡回填必须用**同一份**，否则并发轮次撞上换库时编号会错位
        refs = _foodref_menu()
        _tq = time.time()
        try:
            items = _recognize_dedup(orig, boxed, candidates, n_food, n_drink, tgt,
                                     log=vlog, last_pick=lp, refs=refs)
        finally:
            with _recog_lock:
                _recog_direct_stats["in_flight"] -= 1
        llm_ms = (time.time() - _tq) * 1000.0
        # ── 门禁：一轮只放行一个 item ──────────────────────────────────
        # 展示端一次只显示一张卡，所以一轮也只该动一张。prompt 已明令只输出一个，
        # 这里是兜底：模型不守规矩多给了，取第一个（prompt 要求把最有把握的排第一），
        # 其余只记进观测日志——不建卡、不上屏、也不刷新任何卡的 last_ts。
        # 代价是已知的：桌上其他东西不再被本轮续命，掉出候选窗口后会被判成新物品。
        gate_dropped = [str(i.get("name") or "") for i in items[1:]]
        if gate_dropped:
            items = items[:1]
            print("[da3-web] 门禁截断：模型返回 %d 项，只放行『%s』，丢弃 %s" % (
                len(gate_dropped) + 1, items[0].get("name"), "、".join(gate_dropped)), flush=True)
        # ── 参考食物库命中：模型只负责给编号，内容一律查库回填 ──────────────
        # 命中项的名称/描述/营养/分级整组由库覆盖，模型编的数字直接丢弃——这正是
        # 参考库的意义：同一根香蕉的卡路里不该在 89 和 105 之间来回跳。
        ref_items = refs[0]
        if ref_items and items:
            min_conf = _foodref.config()["min_confidence"]
            resolved = []
            for it in items:
                hit, src, reason = foodref.resolve_hit(it, ref_items, min_conf)
                if hit is not None:
                    print("[da3-web] 参考库命中：『%s』→ [%d] %s（%s，%s）｜证据 %s" % (
                        it.get("name"), hit["id"], hit.get("name"), src, reason,
                        it.get("ref_evidence") or "（模型未给）"), flush=True)
                    it = foodref.apply_hit(it, hit, src)
                else:
                    it = dict(it, source="vlm")
                    if it.get("ref_id") is not None:
                        print("[da3-web] 参考库未命中（模型给了编号 %s）：%s" % (
                            it.get("ref_id"), reason), flush=True)
                it["ref_reason"] = reason
                resolved.append(it)
            items = resolved
        if vlog.get("resp") is not None:
            vlog["resp"]["ref"] = [
                {"name": i.get("name"), "ref_id": i.get("ref_id"),
                 "hit_id": i.get("ref_hit_id"), "source": i.get("source", "vlm"),
                 "confidence": i.get("ref_confidence", ""),
                 "evidence": i.get("ref_evidence", ""), "reason": i.get("ref_reason", "")}
                for i in items]
            vlog["resp"]["llm_ms"] = int(llm_ms)
            vlog["resp"]["wait_ms"] = int(wait_ms)
            vlog["resp"].setdefault("n_items", len(items))
            vlog["resp"]["gate_dropped"] = gate_dropped
        # 链路分段：触发线程测到的三段 + 本 worker 的排队/调用，落卡后再补 post/total
        vlog.setdefault("timings", {}).update({
            "frame_age_ms": (round((stage["pick_at"] - stage["frame_recv_at"]) * 1000.0, 1)
                             if stage.get("frame_recv_at") and stage.get("pick_at") else None),
            # 触发线程里「门控跑完 → 任务入队」的零头，单独留一格免得并进别人头上
            "enqueue_ms": (round((enq_ts - stage["pick_at"]) * 1000.0, 1)
                           - (stage.get("decode_ms") or 0) - (stage.get("gate_ms") or 0)
                           if stage.get("pick_at") else None),
            "decode_ms": stage.get("decode_ms"),
            "gate_ms": stage.get("gate_ms"),
            "wait_ms": round(wait_ms, 1),
            "llm_ms": round(llm_ms, 1),
        })
        _t_post = time.time()
        with _recog_lock:                     # 观测：抽屉里回显实测延时与轮次
            _recog_direct_stats["rounds"] += 1
            _recog_direct_stats["last_ms"] = int(llm_ms)
            _recog_direct_stats["last_ts"] = time.time()
        # 全链路审计日志：每轮（含空返回）都落盘——耗时分段 + 原始判定，可定位慢在哪/错在哪
        print("[da3-web] 识别一轮（%s）：排队%.0fms · %s %.0fms · 返回 %d 项（去重候选 %d 个：%s）" % (
            "直传" if boxed is None else "SAM3", wait_ms, tgt["label"], llm_ms, len(items),
            len(candidates), "、".join(c["name"] for c in candidates) or "无"), flush=True)
        now = time.time()
        # ── 保序闸门：比已落卡的帧还旧的结果，一律不落卡 ──────────────────
        # 检查与推进必须在同一把锁内，否则并发多路会同时通过检查。
        frame_at = stage.get("frame_recv_at")
        with _recog_lock:
            stale = (frame_at is not None
                     and frame_at < _recog_applied_at.get(dev, 0.0))
            if stale:
                behind_s = _recog_applied_at[dev] - frame_at
                _recog_direct_stats["dropped_out_of_order"] += 1
            elif frame_at is not None:
                _recog_applied_at[dev] = frame_at
        if stale:
            print("[da3-web] 乱序丢弃：本轮帧比已落卡的旧 %.1fs，结果不落卡（%d 项）" % (
                behind_s, len(items)), flush=True)
            if vlog.get("resp") is not None:
                vlog["resp"]["dropped_reason"] = "乱序：帧比已落卡的旧 %.1fs" % behind_s
            items = []       # 后面的落卡循环自然空转；resp.n_items 仍保留模型真实返回数
        with _recog_lock:
            cards = _recog_cards.setdefault(dev, [])   # 该设备自己的卡片流
            for it in items:
                target = None
                gate = ""            # 非空=被某道闸门拦下（进日志的拒合并原因）
                m = it.get("match")   # 命中候选编号；还要过五道闸才允许合并（宁拒勿并）
                # 证据码 → (判定, 中文对照文本)。解码一次复用：闸门三判它，
                # 日志/合并史/控制面展示的是解码后的中文——码是给机器省 token 的，
                # 不该让人去背 B1C1S1V1
                ev_verdict, ev_text = recog_match.check_evidence(it.get("match_evidence"))
                self_ok, self_reason = recog_match.check_self_evidence(it)
                # 置信度由证据码推导（B=1 或 C/S/V 全 1 才 high）——match_confidence
                # 字段已从输出契约移除，模型不再自报，闸门四与日志都用这个推导值
                m_conf = recog_match.derive_confidence(it.get("match_evidence"))
                ref_hit = it.get("ref_hit_id")
                if ref_hit:
                    # 参考库命中项不走证据码那五道闸：库里的 id 本身就是「同一种东西」的
                    # 权威判据，比让模型逐项对照品牌/颜色/形状/容器可靠得多，也少一次错并
                    # 的机会。活跃窗口内已有同一 ref 的卡就并过去，否则建新卡。
                    # 名称一致性 + 绝对存活上限：ref_id 并卡绕过了五道闸门，一次错的命中
                    # 会永久自我确认；这两条是它的兜底（库改名/换项时 ref_id 可能对不上人）
                    target = next((c for c in reversed(cards)
                                   if c.get("ref_hit_id") == ref_hit
                                   and foodref.name_key(c.get("name")) == foodref.name_key(it["name"])
                                   and now - c.get("last_ts", 0) <= RECOG_ACTIVE_WINDOW
                                   and now - c.get("first_ts", c.get("last_ts", 0))
                                       <= RECOG_CARD_MAX_LIFE),
                                  None)
                    if target is not None:
                        ev_text = "参考库同项（[%d] %s）" % (ref_hit, it["name"])
                elif isinstance(m, int) and 1 <= m <= len(candidates):
                    cand = candidates[m - 1]
                    if (it.get("matched_name") or "").strip().casefold() \
                            != cand["name"].strip().casefold():
                        gate = "回显不一致：回显『%s』≠候选『%s』" % (
                            it.get("matched_name") or "空", cand["name"])
                        # 闸门一·回显校验：matched_name 必须照抄候选名——只校验编号↔名称
                        # 指称对齐（抓编号幻觉/错位），零语义判断、零语义误伤
                        print("[da3-web] 识别拒合并（回显不一致）：『%s』match=%s 回显『%s』≠候选『%s』→ 按新卡处理" % (
                            it["name"], m, it.get("matched_name") or "空", cand["name"]), flush=True)
                    elif (cand.get("type") or "食物") != it["type"]:
                        gate = "类型不一致：本轮 %s / 候选『%s』%s" % (
                            it["type"], cand["name"], cand.get("type") or "食物")
                        # 闸门二·类型一致：食物↔液体跨类合并是强错并信号，确定性拦截；
                        # 误杀代价只是多一张卡（演示场景可接受的错误方向）
                        print("[da3-web] 识别拒合并（类型不一致）：『%s』(%s) match=%s 候选『%s』(%s) → 按新卡处理" % (
                            it["name"], it["type"], m, cand["name"],
                            cand.get("type") or "食物"), flush=True)
                    elif ev_verdict != "ok":
                        gate = "证据不通过（%s）：%s" % (ev_verdict, ev_text)
                        # 闸门三·证据自洽：证据码里有 0（自称某项不一致却仍要合并）、
                        # 自称 NONE 却给了 match、或码本身不合法——三者都是自相矛盾。
                        # 旧版判的是「文本里含『不一致』三个字」，模型换个措辞就绕过去了；
                        # 码可校验，格式不合法本身就是拒合并的理由（宁拒勿并）。
                        print("[da3-web] 识别拒合并（证据不通过·%s）：『%s』match=%s 证据『%s』→ 按新卡处理" % (
                            ev_verdict, it["name"], m, it.get("match_evidence")), flush=True)
                    elif m_conf != "high":
                        gate = "低置信：证据码 %s 推导为 low" % (it.get("match_evidence") or "空")
                        # 闸门四·置信度：由证据码推导（B=1 或 C/S/V 全 1 才 high），
                        # 语义与旧的模型自报字段一致 → 宁拒勿并
                        print("[da3-web] 识别拒合并（低置信）：『%s』match=%s 证据码 %s 推导为 low → 按新卡处理" % (
                            it["name"], m, it.get("match_evidence") or "空"), flush=True)
                    elif not (_name_tokens(it["name"]) & _name_tokens(cand["name"])):
                        gate = "名称零重叠：『%s』vs 候选『%s』" % (it["name"], cand["name"])
                        # 闸门五·名称零重叠：合并时模型被要求照抄候选名，本轮识别名与候选名
                        # 连一个词面 token 都不重叠还要合并，几乎必是错并——从 WARN 升级为拦截
                        print("[da3-web] 识别拒合并（名称零重叠）：『%s』→候选『%s』 → 按新卡处理" % (
                            it["name"], cand["name"]), flush=True)
                    elif not self_ok:
                        # 闸门六/七·当前画面自证 + B 位抵押（纯格式判定，见 recog_match）：
                        # 说不出画面里它长什么样、说不出与候选的任何差别、或者证据码 B=1
                        # 却没照抄出任何包装文字 —— 三者都说明这次合并没有真实证据支撑。
                        # 线上那张自我确认 252 次的卡，合并史清一色「四项全一致」，
                        # 正是因为填 1 零成本；这两道闸把成本加了回去。
                        gate = "自证不通过：" + self_reason
                        print("[da3-web] 识别拒合并（自证不通过）：『%s』match=%s %s → 按新卡处理" % (
                            it["name"], m, self_reason), flush=True)
                    else:
                        target = next((c for c in cards if c["id"] == cand["id"]), None)
                        if target is None:
                            gate = "候选卡已不在流里（被淘汰/清空）"
                if target is None and recheck_ids and not ref_hit:
                    # 复检回收：本轮被强制剔除候选的卡，如果模型在**没有任何提示**的情况下
                    # 仍报出同一个名字，说明它当初的判断站得住 —— 回收合并，rev 继续累加。
                    # 这一条只在复检轮生效，平时不放宽去重口径。
                    back = next((c for c in reversed(cards)
                                 if c["id"] in recheck_ids
                                 and foodref.name_key(c.get("name")) == foodref.name_key(it["name"])
                                 and now - c.get("last_ts", 0) <= RECOG_ACTIVE_WINDOW), None)
                    if back is not None:
                        target = back
                        ev_text = "复检通过：无提示下仍报『%s』" % it["name"]
                        print("[da3-web] 复检回收：卡%s『%s』无提示复检通过 → 合并" % (
                            back["id"], back.get("name")), flush=True)
                if target is not None:
                    print("[da3-web] 识别去重：『%s』match=%s(high) → 合并到卡%s『%s』｜证据 %s：%s" % (
                        it["name"], m, target["id"], target["name"],
                        it.get("match_evidence") or "（模型未给）", ev_text), flush=True)
                else:
                    print("[da3-web] 识别新卡：『%s』(%s) match=%s｜证据 %s：%s" % (
                        it["name"], it["type"], m,
                        it.get("match_evidence") or "（模型未给）", ev_text), flush=True)
                # 缩略图 = 该帧点云（与②/③同一渲染链路）。模型输出已去掉 box，
                # 3D 框无从画起，一律渲无框点云——去 box 是明确的产品取舍，不是回退。
                shot_url = _save_cloud_shot(pred, [], conf) if pred is not None else None
                if target is not None:           # 去重命中：合并（显示名不改，本轮名称进合并史）
                    target.setdefault("merge_history", []).append({
                        "t": t, "name": it["name"],
                        "evidence": ev_text,
                        "seen": it.get("seen", ""),     # 这一轮模型自己写的观察
                        # ref 命中并卡看 ref_confidence，判同并卡看证据码推导值
                        "confidence": (it.get("ref_confidence") or "") if ref_hit else m_conf})
                    del target["merge_history"][:-20]    # 只留最近 20 条
                    if shot_url:
                        target["shots"].append(shot_url)
                    # shots 只留最近 8 张：磁盘只保留最近 SHOT_KEEP 张，太多前端也摆不下
                    del target["shots"][:-8]
                    # 用**帧时刻**而不是落卡时刻：一个飞了 20 秒才回来的结果，带的是
                    # 20 秒前的画面，不该盖一个"此刻"的时间戳——前端正是按这个值判
                    # 卡片还新不新鲜，盖成此刻就会把待机态("Please place your food")顶掉
                    target["last_ts"] = frame_at or now
                    target["t"] = t
                    target["frame"] = frame
                    # 最近一次识别到该食物的 VLM 延时（合并也刷新：卡片右侧展示）
                    target["latency_ms"] = int(llm_ms)
                    target["latency_model"] = tgt["label"]
                    target["rev"] = target.get("rev", 0) + 1
                    # 有更新的卡移到列表末尾（下发时反转=置顶），顺序=最近更新在前
                    cards.remove(target)
                    cards.append(target)
                else:                            # 新食物：新建卡
                    _recog_id += 1
                    # 参考图=整帧（带框图，直传口径下是原帧）。模型输出去掉 box 后
                    # 没法再裁本物品特写——同轮多张新卡会共享同一张整帧参考图，
                    # 下一轮去重的逐项对照因此变弱（曾是 over-merge 的根因），已知取舍。
                    ref_uri = boxed_uri
                    cards.append({
                        "id": _recog_id, "status": "done",
                        "name": it["name"], "type": it["type"],
                        "description_en": it.get("description_en", ""),
                        "calories_kcal": it.get("calories_kcal"),
                        "protein_g": it.get("protein_g"), "carbs_g": it.get("carbs_g"),
                        "fat_g": it.get("fat_g"), "classification": it.get("classification", ""),
                        # 参考库命中标记：展示端据此打「库内命中」的角标，
                        # 下一轮也拿它做去重（同一 ref 直接并卡，不再问模型）
                        "ref_hit_id": it.get("ref_hit_id"),
                        "source": it.get("source", "vlm"),
                        "shots": [shot_url] if shot_url else [], "ref_img": ref_uri,
                        "merge_history": [],
                        "latency_ms": int(llm_ms), "latency_model": tgt["label"],
                        "frame": frame, "t": t, "last_ts": frame_at or now,
                        # first_ts 只在建卡时写一次：last_ts 会被合并刷新，它不会，
                        # 卡片凭它到点退场（见 RECOG_CARD_MAX_LIFE）
                        "first_ts": frame_at or now, "rev": 0})
                # 逐项判定进观测日志：控制面据此回答「这一项为什么建了新卡/并到了哪张」
                vlog["outcome"].append({
                    "name": it["name"], "type": it["type"], "match": m,
                    # 自证三件套进日志：控制面据此分辨「没看清」还是「看清了但被清单带跑」
                    "seen": it.get("seen", ""), "cur_text": it.get("cur_text", ""),
                    "diff": it.get("diff", ""),
                    "action": "merge" if target is not None else "new",
                    "card_id": target["id"] if target is not None else _recog_id,
                    "gate": gate,
                    "confidence": (it.get("ref_confidence") or "") if ref_hit else m_conf,
                    "evidence": ev_text})
                # 记下本轮真正上屏的这一个，下一轮 prompt 拿它做「还在就别换」的粘性锚
                _recog_last_pick[dev] = {
                    "name": it["name"], "type": it["type"], "ts": now,
                    "card_id": target["id"] if target is not None else _recog_id}
            if len(cards) > RECOG_MAX_CARDS:
                del cards[:len(cards) - RECOG_MAX_CARDS]
            if items:                    # 本轮有卡片变更（新卡/合并）→ 通知 SSE 推送线程
                _recog_gen += 1
                _recog_evt_cv.notify_all()
        now2 = time.time()
        vlog["timings"]["post_ms"] = round((now2 - _t_post) * 1000.0, 1)
        # 端到端 = 帧到 8060 → 本轮结果落卡。拿不到帧时刻（旧链路/手动灌帧）时留空，
        # 不用"从触发算起"冒充端到端——那会把帧在缓存里等触发的时间藏掉
        vlog["timings"]["total_ms"] = (round((now2 - stage["frame_recv_at"]) * 1000.0, 1)
                                       if stage.get("frame_recv_at") else None)
        vlog["timings"]["tunnel_rtt_ms"] = _tunnel_rtt_ms()
        _vlmlog.commit(vlog)


def _recog_ensure_workers_locked():
    """按当前并发配置把 worker 池补齐（须在 _recog_lock 内调用）。
    池子只增不减，多余线程在并发调小后自行退场（见 _recog_worker 的 idx 判断）。"""
    global _recog_workers
    want = max(1, _recog_direct.concurrency())
    while _recog_workers < want:
        threading.Thread(target=_recog_worker, args=(_recog_workers,),
                         daemon=True, name=f"recog-worker-{_recog_workers}").start()
        _recog_workers += 1


def _maybe_recognize(orig_rgb, detections, glb_url, frame, pred=None, conf=40.0,
                     device=UNKNOWN_DEVICE, direct=False, stage=None):
    """detections 非空 + 节流通过 → 取该设备活跃卡快照作去重候选、提交异步识别。
    processor 里调用，不阻塞产线。去重候选与新卡都落在 device 自己的卡片桶里。
    不再加 pending 占位卡（去重后归属不定），识别中用前端顶部 live 指示。

    direct=True 是主链路直传口径：无 SAM3 检测框（detections 传空即可），节奏由
    直传触发线程按配置间隔控制，不吃 RECOG_MIN_INTERVAL 节流。
    direct=False 是 SAM3 触发口径：证据来自触发线程的 SAM3 门控（_sam3_gate_dets，
    跑 RGB 彩色帧），单目链启用时它的液体证据也走这条——直传开启时整条闸掉。"""
    global _recog_last_ts
    if not RECOG_TARGETS[_recog_target]["endpoint"]:
        return
    if not recog_direct.should_trigger(direct, bool(detections), _recog_direct.enabled()):
        return                      # SAM3 触发：无证据、或已被直传接管
    now = time.time()
    with _recog_lock:
        if not direct and now - _recog_last_ts < RECOG_MIN_INTERVAL:
            return
        _recog_last_ts = now
        _recog_ensure_workers_locked()
        # 该设备活跃卡快照(最后出现≤30s、有代表图)作去重候选，取最近的至多 N 张（顺序=参考图编号）
        # 另加绝对存活上限：合并刷新 last_ts 会让错卡永不老化，first_ts 到点强制退场。
        active = [c for c in _recog_cards.get(device, [])
                  if c.get("status") == "done" and c.get("ref_img")
                  and now - c.get("last_ts", 0) <= RECOG_ACTIVE_WINDOW
                  and now - c.get("first_ts", c.get("last_ts", 0)) <= RECOG_CARD_MAX_LIFE]
        active.sort(key=lambda c: c.get("last_ts", 0), reverse=True)
        # 复检轮：合并次数到周期的卡，本轮不进候选、也不做粘性锚——让模型在无提示下
        # 重看一次。仍报同名的会在落卡时被"复检回收"并回原卡，否则它就此老化。
        recheck = [c["id"] for c in active
                   if c.get("rev", 0) >= RECOG_RECHECK_EVERY
                   and c.get("rev", 0) % RECOG_RECHECK_EVERY == 0]
        if recheck:
            print("[da3-web] 强制复检：卡%s 本轮不进候选、关闭粘性" % recheck, flush=True)
        # 同名去重：Orange 与 Clementine、两张同款零食同时在候选里会互相提供交叉证据，
        # 只留同名里最近的那一张
        seen_keys, uniq = set(), []
        for c in active:
            if c["id"] in recheck:
                continue
            key = foodref.name_key(c.get("name"))
            if key and key in seen_keys:
                continue
            seen_keys.add(key)
            uniq.append(c)
        # type 进快照：候选清单文字带类型，且供 worker 的「类型不一致禁并」闸门用
        candidates = [{"id": c["id"], "name": c.get("name", ""), "type": c.get("type", "食物"),
                       "desc": c.get("description_en", ""), "ref_img": c["ref_img"]}
                      for c in uniq[:RECOG_MAX_CANDIDATES]]
    t = time.strftime("%H:%M:%S")
    # 直传口径不画框、也不发图2（boxed=None 一路透传到 _recognize_dedup）
    boxed = None if direct else _draw_boxes(orig_rgb, detections)
    n_food = sum(1 for d in detections if d[0] == "food")
    n_drink = len(detections) - n_food
    with _recog_cv:
        _recog_pending.append((orig_rgb.copy(), boxed, glb_url, frame, t, candidates,
                               n_food, n_drink, time.time(), pred, conf, device,
                               dict(stage or {}),
                               {"dets": list(detections or []), "recheck": recheck}))
        _recog_cv.notify_all()   # worker 池可能不止一个线程在等（并发>1）


def _sam3_gate_dets(rgb, device=None):
    """SAM3 门控（直传关闭时用）：**直接在 RGB 彩色帧上**跑生产词表，返回归一化检测框。

    与历史口径的差别只在"跑在哪张图上"：以前 SAM3 挂在 DA3 链上（单目跑 RGB 但只把
    液体当证据、双目跑左目 IR 灰度图），现在直接跑触发线程手里的这张 RGB——彩色图上
    找食物/饮品比带散斑的 IR 灰度准，也不必先跑 DA3 才轮到 SAM3。
    food 与 drink 都算证据（历史单目口径只收 drink，食物因此从不触发）。

    返回 (dets, 耗时ms)；dets=[(food|drink, nx1, ny1, nx2, ny2)…] 归一化 0-1。
    SAM3 整体不可用（每个词都拿不到结果）时抛异常，由调用方记 last_error。

    每轮结果写回控制面观测（_sam3tune_record_prod，src="gate"）：这是现网唯一在跑的
    SAM3 生产链路，不写回的话控制面 SAM3 区就是空的。写回只做投影（画定位图 + 存
    分数），不发起额外推理；流式步进本就带 debug 捕获（server 端是前向 hook 读已有
    输出，不多跑推理），presence 与 top-K 原始分因此白拿。"""
    cfg = _get_score_cfg()
    targets = [(w["word"], w.get("label") or "drink") for w in cfg["words"]]
    n_prod = len(targets)      # 前 n_prod 个是口径词，其后是系统补跑的 food 词（不计口径统计）
    # 补跑食物词的判据是**词面**没被覆盖，不是「没有 label=food 的词」：现网口径词表
    # 是 food/drink 但两个都标了 label=drink，按 label 判会把 food 这个词原样再跑一遍
    # ——同一张图、同一个词、同样的 presence，纯白跑一次 SAM3；命中时还会让同一个物体
    # 出两个框（一次 drink 一次 food）。词面已在词表里就不补。
    if not any(w == SAM3_TEXT_DEFAULT for (w, _lbl) in targets):
        targets = targets + [(SAM3_TEXT_DEFAULT, "food")]
    H, W = rgb.shape[:2]
    t0 = time.time()

    def one(ql):
        word, label = ql
        insts, _gidx, impl, dbg = _sam3_stream_frame(word, rgb)   # 流式优先（服务端长记忆、每步一帧）
        if insts is None:
            insts, dbg = _sam3_segment_debug(rgb, word, topk=10, alpha=cfg["alpha"],
                                             det_thresh=cfg["thresh"])   # 老 server 回退无状态
            impl = "segment"
        return word, label, insts, impl, dbg

    with ThreadPoolExecutor(max_workers=len(targets)) as ex:
        results = list(ex.map(one, targets))
    gate_ms = (time.time() - t0) * 1000.0
    if all(insts is None for (_w, _lbl, insts, _im, _d) in results):
        raise RuntimeError(f"SAM3 无应答（{SAM3_ENDPOINT}）")
    # 控制面观测写回：统一成单目链的六元组 (query, label, instances, gidx, impl, debug)
    try:
        _sam3tune_record_prod(rgb, [(w, lbl, insts or [], None, im, dbg)
                                    for (w, lbl, insts, im, dbg) in results],
                              gate_ms, src="gate", device=device, n_prod=n_prod)
    except Exception as e:
        print(f"[da3-web] SAM3 门控观测写回失败（忽略）：{type(e).__name__}: {e}", flush=True)
    dets = []
    for _word, label, insts, _impl, _dbg in results:
        for ins in insts or []:
            box = ins.get("box_xywh_px")
            if not (box and len(box) == 4):
                continue
            x, y, bw, bh = (float(v) for v in box)
            if bw <= 1 or bh <= 1:
                continue
            dets.append((label, max(0.0, x / W), max(0.0, y / H),
                         min(1.0, (x + bw) / W), min(1.0, (y + bh) / H)))
    return dets, gate_ms


# ── 识别触发线程：按配置间隔取选中设备最新 RGB 帧 → 直传送 VLM 或 先过 SAM3 门控 ──
def _recog_direct_loop():
    """常驻线程：按 interval_s 节拍取「当前选中设备」的最新 RGB 帧，按开关走两条口径。

      · 直传开：整帧直送 VLM，不看 SAM3；
      · 直传关：同一帧先过 _sam3_gate_dets，认出食物/饮品才带框送 VLM（历史口径的
        本意——SAM3 只是"有没有东西"的前置门，命中后照样识别）。

    其余要点：
      · 帧源用 frame_relay 的最新帧（RGB 彩色帧——VLM 与 SAM3 都吃彩色图，伪彩深度图
        识别不了食物；「设备深度图」只是背景呈现，识别始终走同设备的 RGB 帧）；
      · seq 没变说明推帧比触发间隔还慢，跳过不重复送（省一次多图/SAM3 请求）；
      · 并发=1 时队列 latest-wins 天然退化成「串行·最新优先」，实际节奏≈VLM 延时。"""
    last_seq, last_dev = 0, None
    while True:
        try:
            itv = _recog_direct.interval_s()
            dev = get_selected_device()
            if not dev:
                time.sleep(min(1.0, itv))
                continue
            raw, seq, recv_at = get_latest_frame_seq(dev)
            if raw is None or (dev == last_dev and seq == last_seq):
                with _recog_lock:      # 推帧比触发慢：本轮无新图，不重复送
                    if raw is not None:
                        _recog_direct_stats["skipped_same_frame"] += 1
                time.sleep(min(0.2, itv))
                continue
            last_dev, last_seq = dev, seq
            _t = time.time()          # 触发线程拿起这一帧、开始本轮处理的时刻
            arr = np.array(ImageOps.exif_transpose(
                Image.open(io.BytesIO(raw))).convert("RGB"))
            # 本轮的链路计时随任务下传，worker 合并出端到端分段（见 _recog_worker）。
            # frame_recv_at 是帧到 8060 的时刻：端到端从它起算，才盖得住「帧在缓存里
            # 等触发线程」这段——只从触发算起会把推帧慢造成的延时藏起来。
            # pick_at 与 frame_recv_at 分开记：帧龄只能算「帧在缓存里等触发线程」那段。
            # 用入队时刻减帧到达时刻的话，解码与 SAM3 门控（都在入队之前）会被算两遍，
            # 各段之和就会大于端到端（实测多出 183ms，正是 decode+gate）。
            stage = {"frame_recv_at": recv_at, "pick_at": _t, "seq": seq,
                     "decode_ms": round((time.time() - _t) * 1000.0, 1), "gate_ms": None}
            if _recog_direct.enabled():
                _maybe_recognize(arr, [], None, f"d{seq}", pred=None, conf=40.0,
                                 device=dev, direct=True, stage=stage)
            else:
                _track_stream_device(dev)      # 切设备重置 SAM3 流式会话（幂等）
                dets, gate_ms = _sam3_gate_dets(arr, device=dev)
                stage["gate_ms"] = round(gate_ms, 1)
                with _recog_lock:
                    _recog_direct_stats["gate_ms"] = int(gate_ms)
                    _recog_direct_stats["gate_hits" if dets else "gate_misses"] += 1
                if dets:
                    _maybe_recognize(arr, dets, None, f"g{seq}", pred=None, conf=40.0,
                                     device=dev, direct=False, stage=stage)
            with _recog_lock:
                _recog_direct_stats["last_error"] = ""
        except Exception as e:         # 触发线程必须长命：任何异常只记录不退出
            msg = f"{type(e).__name__}: {e}"
            with _recog_lock:
                _recog_direct_stats["last_error"] = msg
            print(f"[da3-web] 直传识别触发异常（已忽略）：{msg}", flush=True)
            time.sleep(1.0)
            continue
        # 节流：从拿起帧（_t）起算整轮周期，不足 RECOG_MIN_CYCLE_S 补齐——上限 qps 见常量注释
        time.sleep(max(0.05, _recog_direct.interval_s(),
                       RECOG_MIN_CYCLE_S - (time.time() - _t)))


threading.Thread(target=_recog_direct_loop, daemon=True, name="recog-direct").start()


# ══ 设备点云（/experience「设备点云」来源）：硬件真深度 + RGB → 彩色点云 GLB ══
# 与 DA3 点云平行的独立链路：mini 端仅在页面选中该来源时才推原料（demand 由本处
# 以 10s TTL 维护，经 frame_relay 状态接口的 devices[].pc_want 下发给推流端），
# 构建纯 CPU（devpc.build_points 反投影 + 手写 GLB 导出，毫秒级），不占 GPU、
# 不碰单目/双目产物槽与伪彩深度链路。
_devpc_lock = threading.Lock()
_devpc = {"seq": 0, "url": None, "device": None, "meta": None, "error": None}
_devpc_demand: dict = {}          # device_id -> demand 过期时刻（time.time() 秒）
_DEVPC_DEMAND_TTL = 10.0
# 构建串行化：推流快于构建时直接丢新帧（latest 语义，不排队不积压）
_devpc_build_lock = threading.Lock()


def _devpc_wanted(device_id: str) -> bool:
    """frame_relay 状态接口的按需标志回调：该设备是否有页面正在看设备点云。"""
    with _devpc_lock:
        return _devpc_demand.get(device_id, 0.0) > time.time()


set_pc_want_provider(_devpc_wanted)


@app.get("/api/devpc/status")
def devpc_status(device: Optional[str] = None):
    """设备点云状态（/experience 轮询）：url 为最新 GLB；device 供前端做陈旧守卫
    （切设备后旧点云不上画）。轮询本身就是「点播」信号——按 device 续期 demand
    （10s TTL），推流端经 /api/frame/status 的 devices[].pc_want 在 ~2-4s 内跟上，
    页面切走后 demand 过期即自动停推，链路零常驻带宽。"""
    dev = (device or "").strip() or get_selected_device()
    now = time.time()
    with _devpc_lock:
        if dev:
            _devpc_demand[dev] = now + _DEVPC_DEMAND_TTL
            # 顺手清过期 demand 条目（设备下线/改名不留垃圾）
            for d in [d for d, t in _devpc_demand.items() if t <= now]:
                _devpc_demand.pop(d, None)
        return JSONResponse(dict(_devpc))


@app.post("/api/devpc/frame")
def devpc_ingest(depth: UploadFile = File(...), rgb: UploadFile = File(...),
                 meta: str = Form(...)):
    """接收 mini 端设备点云原料（对齐 uint16 深度 PNG + 同帧 RGB JPEG + 内参 meta），
    同步反投影构建彩色点云 GLB（sync 端点跑在 FastAPI 线程池，纯 CPU 毫秒级）。
    构建被占用（推流快于构建）时直接丢本帧返回 busy，保持 latest 语义。"""
    try:
        mj = json.loads(meta) or {}
        m = devpc.parse_meta(mj)
    except (ValueError, TypeError) as e:
        return JSONResponse({"ok": False, "error": f"meta 非法：{e}"}, status_code=400)
    dev = str(mj.get("device_id") or "").strip() or UNKNOWN_DEVICE
    depth_b = depth.file.read()
    rgb_b = rgb.file.read()
    if not depth_b or not rgb_b:
        return JSONResponse({"ok": False, "error": "depth 与 rgb 必须非空"},
                            status_code=400)
    if not _devpc_build_lock.acquire(blocking=False):
        return JSONResponse({"ok": True, "busy": True})
    t0 = time.time()
    try:
        d16 = cv2.imdecode(np.frombuffer(depth_b, np.uint8), cv2.IMREAD_UNCHANGED)
        if d16 is None or d16.ndim != 2 or d16.dtype != np.uint16:
            raise ValueError("depth 不是 16 位单通道 PNG")
        bgr = cv2.imdecode(np.frombuffer(rgb_b, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("rgb 解码失败")
        # RGB 分辨率与 meta 标称（=对齐深度的全分辨率）不一致时缩放对齐取色坐标
        if bgr.shape[0] != m["height"] or bgr.shape[1] != m["width"]:
            bgr = cv2.resize(bgr, (m["width"], m["height"]),
                             interpolation=cv2.INTER_AREA)
        pts, cols = devpc.build_points(d16, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), m)
        if pts.shape[0] == 0:
            raise ValueError("有效深度点为空（画面全在量程外/全是空洞）")
        token = uuid.uuid4().hex
        outdir = GLB_DIR / token
        outdir.mkdir(parents=True, exist_ok=True)
        glb = outdir / "scene.glb"
        _write_pointcloud_glb(str(glb), pts, cols, None, quantize=GLB_QUANTIZE)
        _prune_glb()
        with _devpc_lock:
            _devpc["seq"] += 1
            _devpc["url"] = f"/glb/{token}/scene.glb"
            _devpc["device"] = dev
            _devpc["meta"] = {"fov_deg": devpc.fov_y_deg(m),
                              "points": int(pts.shape[0]),
                              "build_ms": round((time.time() - t0) * 1000, 1),
                              "shape": [m["width"], m["height"]],
                              "stride": m["stride"]}
            _devpc["error"] = None
            seq = _devpc["seq"]
        return JSONResponse({"ok": True, "seq": seq, "points": int(pts.shape[0])})
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        with _devpc_lock:
            _devpc["error"] = err
        return JSONResponse({"ok": False, "error": err}, status_code=400)
    finally:
        _devpc_build_lock.release()


app.include_router(frame_router)
# DA3 单目产物链路已于 2026-08-25 整体退役（模型/权重从服务器移除）：frame_relay 对
# RGB 帧为纯中继（原图/硬件深度展示照常），识别走独立的直传/SAM3 门控触发线程。


def _recog_list_payload(dev):
    """识别卡片列表负载（/api/recog/list 与 SSE 推送共用）：最新在前。"""
    with _recog_lock:
        # 剔除 ref_img（大 dataURI，仅后端去重比对用）；shots(点云 glb 列表) + rev 下发给前端
        cards = [{k: v for k, v in c.items() if k != "ref_img"}
                 for c in _recog_cards.get(dev, []) if c.get("status") != "empty"]
    cards.reverse()   # 最新在前，前端 prepend 到顶部
    tgt = RECOG_TARGETS[_recog_target]
    return {"enabled": bool(tgt["endpoint"]), "device": dev, "cards": cards,
            "target": _recog_target, "model": tgt["model"],
            "targets": {k: bool(v["endpoint"]) for k, v in RECOG_TARGETS.items()}}


@app.get("/api/recog/list")
def recog_list(device: Optional[str] = None):
    """指定设备（缺省=当前选中设备）的识别卡片列表（最新在前）；
    enabled=false 表示识别服务未接入。响应带 device 字段，前端据此判断是否发生了设备切换。"""
    dev = (device or "").strip() or get_selected_device()
    return JSONResponse(_recog_list_payload(dev))


@app.get("/api/recog/events")
def recog_events():
    """识别卡片 SSE 推送（/experience 用，取代 1.5s 轮询）：连上先发一次全量快照，
    此后卡片变更（新卡/合并/清空）或选中设备切换时即刻推最新列表（负载与
    /api/recog/list 相同）；无变更时约 15s 发一次注释心跳兼探测断连。
    同步生成器跑在线程池里（每个连接占一个线程），展台量级（个位数页面）没有压力。"""
    def gen():
        last_gen, last_dev, beats = -1, None, 0
        while True:
            cur_dev = get_selected_device()
            with _recog_lock:
                if _recog_gen == last_gen and cur_dev == last_dev:
                    # 变更靠 notify 即时唤醒；1s 超时兜底顺带检查设备切换
                    _recog_evt_cv.wait(timeout=1.0)
                cur_gen = _recog_gen
            cur_dev = get_selected_device()
            if cur_gen != last_gen or cur_dev != last_dev:
                last_gen, last_dev, beats = cur_gen, cur_dev, 0
                yield "data: " + json.dumps(_recog_list_payload(cur_dev),
                                            ensure_ascii=False) + "\n\n"
            else:
                beats += 1
                if beats >= 15:
                    beats = 0
                    yield ": ping\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-store",
                                      "X-Accel-Buffering": "no"})


@app.get("/api/recog/target")
def recog_target_get():
    """当前识别目标 + 各目标是否可用（endpoint 已配置）。"""
    tgt = RECOG_TARGETS[_recog_target]
    return JSONResponse({"target": _recog_target, "model": tgt["model"],
                         "targets": {k: bool(v["endpoint"]) for k, v in RECOG_TARGETS.items()}})


@app.post("/api/recog/target")
def recog_target_set(body: dict = Body(default=None)):
    """切换识别目标（qwen|gemini）：免重启，下一轮识别即用新目标；未配置 endpoint 的目标拒绝。"""
    global _recog_target
    want = str((body or {}).get("target", "")).strip()
    if want not in RECOG_TARGETS:
        return JSONResponse({"error": "target 只支持 " + "|".join(RECOG_TARGETS)}, status_code=400)
    if not RECOG_TARGETS[want]["endpoint"]:
        return JSONResponse({"error": f"目标 {want} 未配置 endpoint（.env 缺 RECOG_ENDPOINT"
                             + ("_GEMINI" if want == "gemini" else "") + "）"}, status_code=400)
    _recog_target = want
    print(f"[da3-web] 识别目标切换 → {want}（{RECOG_TARGETS[want]['model']}）", flush=True)
    return JSONResponse({"ok": True, "target": want, "model": RECOG_TARGETS[want]["model"]})


@app.get("/api/recog/direct/config")
def recog_direct_config_get():
    """直传识别配置 + 观测（抽屉回显用）。"""
    with _recog_lock:
        stats = dict(_recog_direct_stats)
    return JSONResponse({"config": _recog_direct.snapshot(), "stats": stats,
                         "limits": recog_direct.LIMITS})


@app.post("/api/recog/direct/config")
def recog_direct_config_set(body: dict = Body(default=None)):
    """更新直传识别配置（merge-patch：on / interval_s / concurrency），立即生效并落盘。
    并发调大时补齐 worker 池；调小时多余 worker 干完当前一轮自行退场。"""
    cfg = _recog_direct.update(body or {})
    with _recog_lock:
        _recog_ensure_workers_locked()
        stats = dict(_recog_direct_stats)
    print(f"[da3-web] 直传识别配置更新：{cfg}", flush=True)
    return JSONResponse({"ok": True, "config": cfg, "stats": stats})


# ── 参考食物库接口（浅体验区控制面「参考食物库」页签）──────────────────────
# 录入即生效：任何写操作都会 bump 目录版本，下一轮识别自动重建参考区，无需重启。
@app.get("/api/foodref/list")
def foodref_list():
    """目录全量 + 配置 + 参考区规模。控制面据此渲染列表与顶部预算条。

    不下发图片本体（40 张 dataURI 有几百 KB），控制面按 /api/foodref/image 取。"""
    snap = _foodref.snapshot()
    items = [foodref.item_public(it) for it in snap["items"]]
    menu = _foodref.menu_items()
    cfg = snap["config"]
    return JSONResponse({
        "config": cfg, "version": snap["version"],
        "items": items,
        "budget": foodref.budget(menu, cfg["edge"]),
        "limits": {"edge_choices": foodref.EDGE_CHOICES,
                   "max_items": foodref.MAX_ITEMS,
                   "max_images_per_item": foodref.MAX_IMAGES_PER_ITEM,
                   "confidences": foodref.CONFIDENCE_ORDER,
                   "types": foodref.TYPES,
                   "classifications": foodref.CLASSIFICATIONS},
    })


@app.post("/api/foodref/config")
def foodref_config_set(body: dict = Body(default=None)):
    """更新参考库配置（merge-patch：on / edge / quality / min_confidence）。

    edge/quality 变了等于参考图字节变了：版本 +1、下一轮重建；老档位的产物留在
    磁盘上不删，切回去立刻命中，不必重新处理原图。"""
    cfg = _foodref.set_config(body or {})
    print(f"[da3-web] 参考食物库配置更新：{cfg}", flush=True)
    return JSONResponse({"ok": True, "config": cfg, "version": _foodref.version()})


@app.post("/api/foodref/item")
async def foodref_item_save(meta: str = Form(...), item_id: Optional[str] = Form(None),
                            keep: Optional[str] = Form(None),
                            images: List[UploadFile] = File(default=[])):
    """新增或更新一条参考食物。

    meta 是 JSON 串（名称/别名/类型/外观/营养/分级/描述）。图片语义分两档：
      · 不传 keep（旧客户端）：images 传了就**整组替换**该条的实物图，不传就全保留
        ——控制面改个营养数字不必重新上传照片；
      · 传 keep（JSON 数组，要保留的旧图序号，按展示顺序）：最终图组 = 保留的旧图
        + 新上传的 images，逐张增删都走这一条，落盘后序号重排为 0..k-1。"""
    try:
        patch = json.loads(meta or "{}")
    except ValueError:
        return JSONResponse({"error": "meta 不是合法 JSON"}, status_code=400)
    keep_ns = None
    if keep is not None:
        try:
            keep_ns = json.loads(keep)
        except ValueError:
            keep_ns = "bad"
        if not isinstance(keep_ns, list):
            return JSONResponse({"error": "keep 不是合法的序号数组"}, status_code=400)
    try:
        item = _foodref.upsert(patch, int(item_id) if item_id else None)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except KeyError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    files = [f for f in (images or []) if f is not None and f.filename]
    if keep_ns is not None or files:
        kept = (foodref.select_kept(item.get("images") or [], keep_ns)
                if keep_ns is not None else [])
        if len(kept) + len(files) > foodref.MAX_IMAGES_PER_ITEM:
            return JSONResponse({"error": "每种最多 %d 张实物图"
                                 % foodref.MAX_IMAGES_PER_ITEM}, status_code=400)
        # 新图先全部读进来并验证能解码，再动磁盘——避免写了一半才发现坏图，
        # 留下目录态与文件对不上的中间状态
        raws = []
        for i, f in enumerate(files):
            raw = await f.read()
            if _foodref_decode_upload(raw) is None:
                return JSONResponse({"error": "第 %d 张新图解不出来（只收 jpg/png/heic）" % (i + 1)},
                                    status_code=400)
            raws.append(raw)
        # 保留的旧原图先整张读进内存：序号重排会覆盖同名文件，不能边读边写
        kept_srcs = []
        for im in kept:
            try:
                kept_srcs.append((dict(im), _foodref_orig_path(item["id"], im["n"]).read_bytes()))
            except OSError:
                pass                          # 原图丢了就当这张不存在，别让整次保存失败
        _foodref_drop_cache(item["id"])       # 换图必须作废旧产物，否则一直发老图
        metas = []
        FOODREF_DIR.mkdir(parents=True, exist_ok=True)
        for im, raw in kept_srcs:             # 旧原图直接搬字节，不重编码（免二次画质损失）
            im["n"] = len(metas)
            _foodref_orig_path(item["id"], im["n"]).write_bytes(raw)
            metas.append(im)
        for raw in raws:
            got = _foodref_save_original(item["id"], len(metas), raw)
            if got is None:
                return JSONResponse({"error": "新图编码失败"}, status_code=500)
            metas.append(got)
        for stale in FOODREF_DIR.glob("%d_*.orig.jpg" % item["id"]):
            try:                              # 张数变少时，把多出来的原图删掉
                if int(stale.stem.split(".")[0].split("_")[1]) >= len(metas):
                    stale.unlink()
            except (OSError, ValueError, IndexError):
                pass
        item = _foodref.set_images(item["id"], metas)
    print("[da3-web] 参考食物库保存：[%d] %s（%d 张图）" % (
        item["id"], item.get("name"), len(item.get("images") or [])), flush=True)
    return JSONResponse({"ok": True, "item": foodref.item_public(item),
                         "version": _foodref.version()})


@app.delete("/api/foodref/item/{item_id}")
def foodref_item_delete(item_id: int):
    """删除一条参考食物（连同原图与所有档位的缓存产物）。"""
    if not _foodref.delete(item_id):
        return JSONResponse({"error": "条目不存在"}, status_code=404)
    _foodref_drop_files(item_id)
    print("[da3-web] 参考食物库删除：[%d]" % item_id, flush=True)
    return JSONResponse({"ok": True, "version": _foodref.version()})


@app.get("/api/foodref/image/{item_id}/{n}")
def foodref_image(item_id: int, n: int, kind: str = "ref"):
    """取某张参考图：kind=ref 是真正送 VLM 的那一份（按当前档位规范化、带横幅），
    kind=orig 是上传的原图副本。控制面用 ref 做「模型看到的到底是什么」的目检。"""
    if kind == "orig":
        path = _foodref_orig_path(item_id, n)
        if not path.exists():
            return JSONResponse({"error": "图片不存在"}, status_code=404)
        return FileResponse(str(path), media_type="image/jpeg")
    cfg = _foodref.config()
    uri = _foodref_ref_uri(item_id, n, cfg["edge"], cfg["quality"])
    if not uri:
        return JSONResponse({"error": "图片不存在"}, status_code=404)
    return Response(content=base64.b64decode(uri.split(",", 1)[1]),
                    media_type="image/jpeg")


@app.post("/api/recog/clear")
def recog_clear(device: Optional[str] = None):
    """清空指定设备（缺省=当前选中设备）的识别卡片；其他设备的卡片不受影响。"""
    global _recog_gen
    dev = (device or "").strip() or get_selected_device()
    with _recog_lock:
        _recog_cards.pop(dev, None)
        _recog_last_pick.pop(dev, None)   # 卡都清了，再拿上次命中指称就是悬空引用
        _recog_gen += 1              # 清空也算变更，SSE 立即推空列表
        _recog_evt_cv.notify_all()
    return JSONResponse({"ok": True, "device": dev})


# ── VLM 识别观测日志接口（浅体验区控制面用）────────────────────────────────
# 长轮询挂起上限（秒）：必须小于 superadmin nginx 对 /da3-api 的 proxy_read_timeout(30s)，
# 否则反代会先掐断连接、前端看到的是一串报错而不是"还没有新日志"
RECOGLOG_HOLD_MAX = 25.0


@app.get("/api/recoglog/list")
def recoglog_list(device: Optional[str] = None, limit: int = 12,
                  since_gen: int = -1, wait_s: float = 0.0):
    """识别日志列表（最新在前）。列表态不含 prompt 全文 / 原始返回 / 候选参考图——
    那三样在详情里取，避免秒级轮询把响应撑成几 MB。

    wait_s>0 = 长轮询：当 since_gen 已是最新版本时把请求挂起，有新日志立刻返回，
    否则等到 wait_s（上限 RECOGLOG_HOLD_MAX）超时返回。控制面据此做到"来一条画一条"，
    而不是按固定间隔猜。挂起走同步端点跑在线程池里（与 /api/recog/events 同款做法），
    展台量级（个位数页面）没有压力；用长轮询而不是 SSE 是因为它对反代零要求——
    每次仍是一个普通的一次性响应，nginx 的缓冲与超时都不用改。"""
    if wait_s and wait_s > 0:
        _vlmlog.wait_for(since_gen, min(float(wait_s), RECOGLOG_HOLD_MAX))
    items, total = _vlmlog.list(device=device, limit=limit)
    # 顺带刷新网络基线：控制面在看日志时才探（探测自带 2s 缓存），没人看就不探。
    # 探测跑在这条 HTTP 请求线程里而不是识别 worker 里——worker 里补发请求会把
    # 被测对象自己搅乱。基线没有值的话，时序图的 HTTP 段就永远拆不出「网络 / 服务端」。
    try:
        _tunnel_probe()
    except Exception:       # 探测失败不影响日志本身
        pass
    tgt = RECOG_TARGETS[_recog_target]
    return JSONResponse({"items": items, "total": total, "max": VLMLOG_MAX,
                         "gen": _vlmlog.gen(),
                         "target": tgt["label"], "model": tgt["model"],
                         "direct": _recog_direct.snapshot(),
                         "tunnel_rtt_ms": _tunnel_rtt_ms()})


@app.get("/api/recoglog/{entry_id}")
def recoglog_detail(entry_id: int):
    """单条识别日志全文：prompt 原文、模型原始返回、解析出的每一项、候选参考图。"""
    hit = _vlmlog.get(entry_id)
    if hit is None:
        return JSONResponse({"ok": False, "error": "日志已滚出缓冲（只保留最近 %d 条）"
                             % VLMLOG_MAX}, status_code=404)
    return JSONResponse({"ok": True, "item": hit})


@app.get("/api/recoglog/{entry_id}/image/{kind}")
def recoglog_image(entry_id: int, kind: str):
    """这一轮**真正送进 VLM 请求体**的那张图（kind=orig 图1 / boxed 图2），原尺寸原质量。
    只留最近若干条（见 RecogLog.full_keep），滑出后 404。"""
    uri = _vlmlog.full_image(entry_id, kind)
    if not uri or "," not in uri:
        return JSONResponse({"ok": False, "error": "原图已滚出留存窗口或该轮没有这张图"},
                            status_code=404)
    try:
        raw = base64.b64decode(uri.split(",", 1)[1])
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"{type(e).__name__}: {e}"}, status_code=500)
    return Response(content=raw, media_type="image/jpeg",
                    headers={"Cache-Control": "public, max-age=3600"})


@app.post("/api/recoglog/clear")
def recoglog_clear():
    """清空识别日志缓冲。"""
    _vlmlog.clear()
    return JSONResponse({"ok": True})


# ══════════════════════════════════════════════════════════════════════
# Qwen 识别隧道监控 / 一键重建
# 链路：5090:8011 ←反向SSH← Mac:18011 ←IAP← gpu-g4-01:8000(vllm)。
# 隧道进程都在 Mac 上（GCP Workforce 凭证只在 Mac，5090 没有 GCP 身份），
# 本服务只能做两件事：
#   · 探测：请求本机 8011，凡有 HTTP 应答（含 401）即视为全链路通；
#   · 转发指令：网页点「一键重建」→ 置 rebuild_requested 标志，Mac 上的守护
#     进程（本仓 tools/qwen_tunnel_keeper.py，launchd 常驻）心跳时领走执行。
# ══════════════════════════════════════════════════════════════════════
# 2026-08-21 起识别端点已切到 5090 本机 vLLM(8000)，探测地址跟随 RECOG_ENDPOINT 推导
# （其 origin + /v1/models）；RECOG_ENDPOINT 缺省时才回落旧的 8011 反向隧道探测。
def _tunnel_probe_url() -> str:
    _ep = os.environ.get("RECOG_ENDPOINT", "").strip()
    if "/v1/" in _ep:
        return _ep.rsplit("/v1/", 1)[0] + "/v1/models"
    return "http://127.0.0.1:8011/v1/models"


TUNNEL_PROBE_URL = _tunnel_probe_url()
TUNNEL_PROBE_CACHE = 2.0     # 探测结果缓存(秒)：多个前端轮询共享，避免每次都真探一枪
# 探测超时：原来 3s。识别满负载时单个请求体 600KB+、并发 2，隧道上行只有 ~400KB/s，
# 探测小包要排在几 MB 数据后面——3s 必然探不通，而链路其实好好的。放宽到 8s。
TUNNEL_PROBE_TIMEOUT = 8.0
TUNNEL_KEEPER_ALIVE = 15.0   # 守护心跳超时(秒)：超过视为 Mac 守护不在线

_tunnel_lock = threading.Lock()
_tunnel = {"up": False, "state": "down", "checked_at": 0.0, "last_ok": 0.0,
           "rtt_ms": None, "rebuild_requested": False, "keeper_seen": 0.0,
           "keeper_msg": ""}


def _tunnel_rtt_ms():
    """最近一次隧道探测（GET 8011 /v1/models）的往返耗时，毫秒。

    识别日志拿它当**网络基线**：VLM 那段的 http_ms 里网络往返占多少，靠这个数对照——
    同一条隧道、同样的 5090→Mac→IAP→GCP 路径，只是请求体极小、服务端不做推理。
    只读缓存不主动探测：worker 里再发一枪网络请求会把被测对象自己搅乱。"""
    with _tunnel_lock:
        return _tunnel["rtt_ms"]


def _tunnel_probe():
    """探一次本机 8011（缓存期内复用上次结果），返回三态 up / busy / down。

    为什么必须区分「堵」和「断」（2026-08-18 实发）：识别满负载时隧道被 600KB 的
    请求体连续占满，探测小包排在后面必然超时。守护把「超时」当成「断了」，就会
    kill 掉一条**正在正常工作**的隧道去重建——重建那几十秒才是真空期，日志里那
    一串 Connection refused 全是守护自己制造的。隧道本来没断，是守护把它拆了。

    判据：真断是**立刻**失败（端口没了→连接被拒 / 对端关闭→连接被重置），
    拥塞是**超时**。据此分流：
      up   有应答（含 4xx，说明链路通到了 vllm）
      busy 超时——链路多半还在，只是排队，**不构成重建理由**
      down 硬失败——反向段或 IAP 段真没了，该重建
    """
    now = time.time()
    with _tunnel_lock:
        if now - _tunnel["checked_at"] < TUNNEL_PROBE_CACHE:
            return _tunnel["state"]
    _t = time.time()
    try:
        with urllib.request.urlopen(TUNNEL_PROBE_URL, timeout=TUNNEL_PROBE_TIMEOUT):
            state = "up"
    except urllib.error.HTTPError:
        state = "up"         # 服务有应答（如未带 key 的 401），链路本身是通的
    except socket.timeout:
        state = "busy"
    except urllib.error.URLError as e:
        # urllib 会把超时包一层，剥开看真正的原因；其余（拒绝/重置/无路由）算真断
        state = "busy" if isinstance(e.reason, (socket.timeout, TimeoutError)) else "down"
    except TimeoutError:
        state = "busy"
    except Exception:
        state = "down"
    rtt_ms = round((time.time() - _t) * 1000.0, 1)
    with _tunnel_lock:
        _tunnel["state"] = state
        # up 是给前端状态灯用的布尔值：busy 时沿用上次，别让拥塞把灯打成"已断开"
        _tunnel["up"] = _tunnel["up"] if state == "busy" else (state == "up")
        _tunnel["rtt_ms"] = rtt_ms if state == "up" else None  # 超时/失败的耗时没有参考意义
        _tunnel["checked_at"] = time.time()
        if state == "up":
            _tunnel["last_ok"] = _tunnel["checked_at"]
        # 注意：重建指令只由守护心跳领取消费，这里绝不代为撤销——
        # 探测(2s缓存)比守护心跳(5s)勤，恢复瞬间撤销会把刚下发的指令静默吞掉。
    return state


@app.get("/api/tunnel/status")
def tunnel_status():
    """识别隧道状态（网页轮询）：up=5090→GCP Qwen 全链路是否通；keeper_alive=Mac 守护是否在线。

    state 比 up 多一档 busy（链路在、只是被大请求塞住）——守护据它决定要不要重建，
    网页据它把状态灯从"已断开"改说成"拥塞"，别让满负载看起来像故障。"""
    state = _tunnel_probe()
    now = time.time()
    with _tunnel_lock:
        return JSONResponse({
            "up": _tunnel["up"], "state": state,
            "last_ok": _tunnel["last_ok"], "rtt_ms": _tunnel["rtt_ms"],
            "keeper_alive": now - _tunnel["keeper_seen"] <= TUNNEL_KEEPER_ALIVE,
            "keeper_msg": _tunnel["keeper_msg"],
            "pending": _tunnel["rebuild_requested"],
        })


@app.post("/api/tunnel/rebuild")
def tunnel_rebuild():
    """网页一键重建：置标志，等 Mac 守护下次心跳（≤5s）领走执行。"""
    with _tunnel_lock:
        _tunnel["rebuild_requested"] = True
        alive = time.time() - _tunnel["keeper_seen"] <= TUNNEL_KEEPER_ALIVE
    return JSONResponse({"ok": True, "keeper_alive": alive})


@app.post("/api/tunnel/keeper")
def tunnel_keeper(body: dict = Body(default=None)):
    """Mac 守护心跳：上报状态文案、领取重建指令（领走即清标志），并带回 5090 侧探测结果。"""
    with _tunnel_lock:
        _tunnel["keeper_seen"] = time.time()
        _tunnel["keeper_msg"] = str((body or {}).get("msg", ""))[:120]
        req = _tunnel["rebuild_requested"]
        _tunnel["rebuild_requested"] = False
    state = _tunnel_probe()
    with _tunnel_lock:
        up = _tunnel["up"]
    # state 是给守护做重建决策的；up 仅供展示（busy 时沿用上次，不打成断开）
    return JSONResponse({"rebuild": req, "up": up, "state": state})


@app.get("/api/sam3/health")
def sam3_health():
    """SAM3 服务健康（经隧道）。"""
    try:
        with urllib.request.urlopen(SAM3_ENDPOINT + "/health", timeout=6) as r:
            return JSONResponse({"ok": True, "endpoint": SAM3_ENDPOINT,
                                 "health": json.loads(r.read().decode())})
    except Exception as e:
        return JSONResponse({"ok": False, "endpoint": SAM3_ENDPOINT,
                             "error": f"{type(e).__name__}: {e}"})


# ══════════════════════════════════════════════════════════════════════
# SAM3 调优页（/sam3tune）：每次运行 = 当前帧对每个词各跑一次带 debug 的 /v1/segment，
# 拿到 presence 分与 top-K query 原始分做漏检归因；运行历史养在服务端（翻页/多端可见）。
# ══════════════════════════════════════════════════════════════════════
SAM3TUNE_HISTORY_MAX = 30      # 历史条数上限（缩略图 dataURI 常驻内存，别放太大）
# 生产观测写历史的最小间隔（秒）：识别触发线程按 interval_s（默认 0.5s）一轮一跑门控，
# 不采样的话 30 条历史只覆盖十几秒、翻页还没看完就被冲掉。实时区（live）不受此限。
SAM3TUNE_HIST_MIN_GAP = 1.5
# 留原尺寸帧（供控制面点开看大图）的条数。存的是 ndarray 引用不是 JPEG：
# 常态零编码开销（编码只在有人点开时发生），代价是每条约 5MB 内存——对 5090 无所谓，
# 而识别触发线程的 CPU 节拍才是展台上真正稀缺的东西。
SAM3TUNE_FULL_KEEP = 4
# 观测写回总开关（`.env` 里 SAM3_OBS_LOG=0 关掉）：写回本身只画定位图 + 编码两张缩略图，
# 但它跑在识别触发线程里，展台上真嫌它占节拍时可以一键关（关掉后控制面 SAM3 区空）
SAM3TUNE_OBS_ON = os.environ.get("SAM3_OBS_LOG", "1").strip() not in ("0", "false", "off")
_sam3tune_history = []         # 最新在前；条目结构同 /api/sam3tune/run 返回（图为缩略图）
_sam3tune_live = {}            # device -> 生产流式最近一帧的观测条目（控制面实时区数据源）
_sam3tune_lock = threading.Lock()
_sam3tune_seq = 0
_sam3tune_hist_ts = 0.0        # 上一条生产观测写进历史的时刻（采样节流用）


def _tune_drop_frames(entry):
    """丢掉条目持有的原尺寸帧引用（滑出留存窗口后，图片端点对它降级为 404）。"""
    entry.pop("_raw", None)
    entry.pop("_seg", None)


def _tune_public(entry):
    """观测条目 → 响应体：剥掉 `_` 开头的内部字段（ndarray 不可序列化），
    并告诉前端这条还能不能点开看原图。"""
    out = {k: v for k, v in entry.items() if not k.startswith("_")}
    out["has_full"] = entry.get("_raw") is not None
    return out


def _sam3tune_record_prod(rgb, results, ms, src="prod", device=None, n_prod=None):
    """生产 SAM3 每轮的观测写回：per-device 最新状态 + 滚动历史。
    results 元素 = (query, label, instances, gidx, impl, debug)（历史上来自单目链，现仅 gate 路径产出）
    （单目 DA3 链 src="prod"）或 _sam3_gate_dets（识别门控 src="gate"，现网唯一在跑的）。
    这里只做观测投影（画定位图 + 存分数），不发起任何额外推理——忠实生产结果。

    n_prod=前几个词属于生产口径词（其后是补给识别的 food 词）。补跑词同样进日志
    （控制面要看到「这一轮 SAM3 到底认出了什么」），但标 role="highlight"、不计进
    n_inst——口径统计只认配置词，与 keep 口径语义保持一致。
    device=None 时取当前选中设备。"""
    global _sam3tune_seq, _sam3tune_hist_ts
    if not SAM3TUNE_OBS_ON:
        return
    cfg = _get_score_cfg()
    dev = device or get_selected_device()
    if n_prod is None:
        n_prod = len(results)
    img, word_infos, n_inst = rgb, [], 0
    for wi, (query, label, inst, _g, _im, dbg) in enumerate(results):
        col = _SAM3_COLORS[wi % len(_SAM3_COLORS)]
        img = _draw_instances(img, inst, color=col, label_prefix=query)
        if wi < n_prod:
            n_inst += len(inst)
        word_infos.append({
            "word": query, "label": label, "color": "#%02x%02x%02x" % col, "n": len(inst),
            "role": "cfg" if wi < n_prod else "highlight",
            "instances": [{"obj_id": it.get("obj_id"),
                           "score": round(float(it.get("score", 0)), 4)} for it in inst],
            "debug": dbg,
        })
    now = time.time()
    with _sam3tune_lock:
        _sam3tune_seq += 1
        entry_id = _sam3tune_seq
    # _raw/_seg 前缀下划线=内部字段，投影时剥掉（ndarray 不可 JSON 序列化）
    entry = {"_raw": rgb, "_seg": img,
             "ok": True, "id": entry_id, "ts": now, "src": src,
             "text": ", ".join(q for (q, _l, _i, _g2, _im2, _d) in results[:n_prod]),
             "alpha": cfg["alpha"], "thresh": cfg["thresh"] or 0.5,
             "seg_ms": round(float(ms), 1), "n_inst": n_inst, "words": word_infos,
             "device": dev, "endpoint": SAM3_ENDPOINT,
             "raw": _thumb_uri(rgb), "seg": _thumb_uri(img)}
    with _sam3tune_lock:
        prev = _sam3tune_live.get(dev)
        _sam3tune_live[dev] = entry
        if now - _sam3tune_hist_ts >= SAM3TUNE_HIST_MIN_GAP:
            _sam3tune_hist_ts = now
            _sam3tune_history.insert(0, entry)
            del _sam3tune_history[SAM3TUNE_HISTORY_MAX:]
        elif prev is not None and prev not in _sam3tune_history:
            _tune_drop_frames(prev)      # 上一条没进历史：它的原图没人能再取到，立刻释放
        for old in _sam3tune_history[SAM3TUNE_FULL_KEEP:]:
            if old not in _sam3tune_live.values():
                _tune_drop_frames(old)


@app.get("/api/sam3tune/config")
def sam3tune_config_get():
    """生产 SAM3 打分口径（keep = presence^α × cond > thresh；thresh=0 表示模型默认 0.5）。"""
    return JSONResponse(_get_score_cfg())


@app.post("/api/sam3tune/config")
def sam3tune_config_set(body: dict = Body(default=None)):
    """控制面写口径：下一帧生产流式即生效。alpha∈[0,1]；thresh∈[0,0.95]，0=恢复模型默认。"""
    global _sam3_score_cfg
    try:
        alpha = min(max(float((body or {}).get("alpha", 1.0)), 0.0), 1.0)
        thresh = min(max(float((body or {}).get("thresh", 0.0)), 0.0), 0.95)
    except (TypeError, ValueError):
        return JSONResponse({"error": "alpha/thresh 必须是数字"}, status_code=400)
    # label 决定这个词的命中算 food 还是 drink，一路影响：送 VLM 的检测框颜色、
    # prompt 里的「食物框×N、液体框×M」软接地信息、以及单目链的液体证据过滤。
    # 老客户端只发词面字符串（控制面 tags 输入曾如此），历史上一律兜底成 drink——
    # 于是 food 这个词被标成 drink，每轮都在告诉模型「画面里没有食物」。现在：
    #   给了 label 就用给的；只发词面就**沿用该词现有的 label**（幂等，不打翻已配好的）；
    #   都没有才落到 SAM3_TEXT_DEFAULT 判 food、其余 drink 这条最后兜底。
    known = {w["word"]: w.get("label") for w in _get_score_cfg().get("words") or []}
    words = []
    for w in ((body or {}).get("words") or [])[:4]:   # 识别词最多 4 个；空则回落默认词表
        word = (w.get("word") if isinstance(w, dict) else str(w)).strip()
        if not word:
            continue
        label = (w.get("label") or "").strip() if isinstance(w, dict) else ""
        if label not in ("food", "drink"):
            label = known.get(word) or ("food" if word == SAM3_TEXT_DEFAULT else "drink")
        words.append({"word": word, "label": label})
    with _sam3_score_lock:
        _sam3_score_cfg = {"alpha": alpha, "thresh": thresh, "words": words}
        try:
            _SCORE_CFG_PATH.write_text(json.dumps(_sam3_score_cfg, ensure_ascii=False))
        except Exception as e:
            print(f"[da3-web] 口径配置落盘失败：{type(e).__name__}: {e}", flush=True)
    print("[da3-web] 生产 SAM3 口径更新：alpha=%s thresh=%s words=%s" % (
        alpha, thresh, [f"{w['word']}({w['label']})" for w in words]), flush=True)
    return JSONResponse({"ok": True, "alpha": alpha, "thresh": thresh, "words": words})


@app.get("/api/sam3tune/state")
def sam3tune_state():
    """控制面实时区数据源：口径配置 + 每设备生产流式最近一帧的观测（原图/定位缩略图+分数）。"""
    with _sam3tune_lock:
        live = {d: _tune_public(e) for d, e in _sam3tune_live.items()}
    return JSONResponse({"cfg": _get_score_cfg(), "selected": get_selected_device(),
                         # 生产链路当前在识别的词（随口径配置可调，空=默认词表）
                         "words": _get_score_cfg()["words"],
                         "live": live, "endpoint": SAM3_ENDPOINT})


# 观测缩略图口径（`.env` 可覆盖）：420px/q72 在控制面上肉眼可见糊，看不出食物细节；
# 640px/q80 每张约 30KB（原 9KB），列表按 limit 截断后总体积仍在几百 KB 量级。
# 注意这只是「日志预览图」的口径——送 VLM 的始终是设备原帧（见 _img_data_uri）。
THUMB_W = int(os.environ.get("OBS_THUMB_W", "640"))
THUMB_Q = int(os.environ.get("OBS_THUMB_Q", "80"))


def _thumb_uri(rgb, width=None, quality=None):
    """RGB → 缩略图 JPEG dataURI（历史区用小图，控内存与响应体积）。"""
    width = THUMB_W if width is None else width
    quality = THUMB_Q if quality is None else quality
    h, w = rgb.shape[:2]
    if w > width:
        rgb = cv2.resize(rgb, (width, max(1, round(h * width / w))),
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return ("data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()) if ok else ""


@app.post("/api/sam3tune/run")
def sam3tune_run(body: dict = Body(default=None)):
    """跑一次调优分析：当前帧 + 每词一次带 debug 的单图分割。
    返回原图/定位图（全尺寸）+ 每词 presence 分与 top-K query 原始分，并写入服务端历史。"""
    global _sam3tune_seq
    text = str(((body or {}).get("text") or SAM3_TEXT_DEFAULT)).strip() or SAM3_TEXT_DEFAULT
    try:
        topk = min(max(int((body or {}).get("topk") or 10), 1), 50)
    except (TypeError, ValueError):
        topk = 10
    try:
        alpha = min(max(float((body or {}).get("alpha", 1.0)), 0.0), 1.0)
    except (TypeError, ValueError):
        alpha = 1.0
    try:
        thresh = min(max(float((body or {}).get("thresh", 0.5)), 0.05), 0.95)
    except (TypeError, ValueError):
        thresh = 0.5
    words = [w.strip() for w in re.split(r"[,，;；\n]+", text) if w.strip()][:4] or [SAM3_TEXT_DEFAULT]
    frames = _get_recent_frames(1)
    if not frames:
        # 纯中继模式下 _recent_frames 不再被填充（DA3 单目链已退役），直接取中继最新帧
        dev = get_selected_device()
        raw = get_latest_frame(dev) if dev else None
        if raw is not None:
            try:
                frames = [np.array(ImageOps.exif_transpose(
                    Image.open(io.BytesIO(raw))).convert("RGB"))]
            except Exception:
                frames = []
    if not frames:
        return JSONResponse({"ok": False, "error": "还没有设备帧（等设备把帧推上来再试）"},
                            status_code=400)
    cur = frames[-1]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(words)) as ex:
        results = list(ex.map(lambda w: (w, *_sam3_segment_debug(cur, w, topk, alpha, thresh)), words))
    seg_ms = round((time.time() - t0) * 1000.0, 1)

    img, word_infos, n_inst = cur, [], 0
    for wi, (w, inst, dbg) in enumerate(results):
        col = _SAM3_COLORS[wi % len(_SAM3_COLORS)]
        img = _draw_instances(img, inst, color=col, label_prefix=w)
        n_inst += len(inst)
        word_infos.append({
            "word": w, "color": "#%02x%02x%02x" % col, "n": len(inst),
            "instances": [{"obj_id": it.get("obj_id"),
                           "score": round(float(it.get("score", 0)), 4)} for it in inst],
            "debug": dbg,   # None=SAM3 调用失败或老版 server 不支持 debug
        })
    with _sam3tune_lock:
        _sam3tune_seq += 1
        entry_id = _sam3tune_seq
    common = {"ok": True, "id": entry_id, "ts": time.time(), "src": "manual",
              "text": text, "topk": topk,
              "alpha": alpha, "thresh": thresh,   # 阈值口径：keep = presence^α×cond > thresh
              "seg_ms": seg_ms, "n_inst": n_inst, "words": word_infos,
              "device": get_selected_device(),   # 分析帧来自当前选中设备的处理流
              "endpoint": SAM3_ENDPOINT}
    hist = dict(common)
    hist["raw"] = _thumb_uri(cur)
    hist["seg"] = _thumb_uri(img)
    with _sam3tune_lock:
        _sam3tune_history.insert(0, hist)
        del _sam3tune_history[SAM3TUNE_HISTORY_MAX:]
    full = dict(common)
    full["raw"] = "data:image/jpeg;base64," + (_b64_jpg(cur) or "")
    full["seg"] = "data:image/jpeg;base64," + (_b64_jpg(img) or "")
    return JSONResponse(full)


@app.get("/api/sam3tune/history")
def sam3tune_history(device: Optional[str] = None, limit: int = 0):
    """调优运行历史（最新在前，服务端缓存最近 SAM3TUNE_HISTORY_MAX 条，图为缩略图）。
    device=按设备过滤（控制面单设备视图用）；limit>0=只回最新若干条——每条自带两张
    缩略图 dataURI，控制面秒级轮询全量会几 MB 起步，按需截断。"""
    with _sam3tune_lock:
        items = list(_sam3tune_history)
    if device:
        items = [it for it in items if it.get("device") == device]
    total = len(items)
    if limit and limit > 0:
        items = items[:limit]
    return JSONResponse({"items": [_tune_public(it) for it in items],
                         "total": total, "max": SAM3TUNE_HISTORY_MAX})


@app.get("/api/sam3tune/image/{entry_id}/{kind}")
def sam3tune_image(entry_id: int, kind: str):
    """观测条目的原尺寸帧（kind=raw 原图 / seg 定位图），点开大图时才拉。
    留存窗口只有最近 SAM3TUNE_FULL_KEEP 条 + 各设备实时条，滑出后 404。
    编码在这里做（懒编码）：没人点开就不花这份 CPU。"""
    with _sam3tune_lock:
        hit = next((e for e in list(_sam3tune_history) + list(_sam3tune_live.values())
                    if e.get("id") == entry_id), None)
        arr = None if hit is None else hit.get("_seg" if kind == "seg" else "_raw")
    if arr is None:
        return JSONResponse({"ok": False, "error": "原图已滚出留存窗口（只留最近 %d 条）"
                             % SAM3TUNE_FULL_KEEP}, status_code=404)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        return JSONResponse({"ok": False, "error": "编码失败"}, status_code=500)
    return Response(content=buf.tobytes(), media_type="image/jpeg",
                    headers={"Cache-Control": "public, max-age=3600"})


@app.post("/api/sam3tune/clear")
def sam3tune_clear():
    """清空调优历史。"""
    with _sam3tune_lock:
        _sam3tune_history.clear()
    return JSONResponse({"ok": True})


@app.get("/api/sam3hl/status")
def sam3hl_status():
    """第四图（SAM3 高亮点云·无框）状态：kind=model → url 为高亮 GLB（与②③同管线直渲）；
    kind=image → seq 变化且 meta 非空时拉 /api/sam3hl/latest；cfg=当前高亮配置。"""
    with _sam3hl_lock:
        return JSONResponse({"kind": _sam3hl["kind"], "url": _sam3hl["url"],
                             "seq": _sam3hl["seq"], "meta": _sam3hl["meta"],
                             "error": _sam3hl["error"], "cfg": dict(_sam3hl_cfg)})


@app.get("/api/sam3hl/latest")
def sam3hl_latest():
    """第四图最新 JPEG 字节。"""
    with _sam3hl_lock:
        data, seq = _sam3hl["bytes"], _sam3hl["seq"]
    if not data:
        return JSONResponse({"error": "暂无高亮点云图"}, status_code=404)
    return Response(content=data, media_type="image/jpeg",
                    headers={"Cache-Control": "no-store", "X-Sam3hl-Seq": str(seq)})


@app.post("/api/sam3hl/config")
def sam3hl_config(body: dict = Body(default=None)):
    """更新第四图高亮配置（只影响第四图，下一轮 SAM3 任务即生效）。逐字段校验，非法值忽略。"""
    body = body or {}
    with _sam3hl_lock:
        if str(body.get("style", "")) in _HL_STYLE_CN:
            _sam3hl_cfg["style"] = str(body["style"])
        for k, (lo, hi, typ) in _HL_NUM_FIELDS.items():
            if k not in body:
                continue
            try:
                _sam3hl_cfg[k] = typ(min(hi, max(lo, float(body[k]))))
            except Exception:
                pass
        if str(body.get("color_mode", "")) in ("auto", "custom"):
            _sam3hl_cfg["color_mode"] = str(body["color_mode"])
        for hk in _HL_HEX_FIELDS:
            c = str(body.get(hk, "")).strip()
            if re.fullmatch(r"#[0-9a-fA-F]{6}", c):
                _sam3hl_cfg[hk] = c
        try:
            _save_sam3hl_preset()
        except OSError as exc:
            return JSONResponse({"ok": False, "error": f"预设保存失败：{exc}",
                                 "cfg": dict(_sam3hl_cfg)}, status_code=500)
        return JSONResponse({"ok": True, "cfg": dict(_sam3hl_cfg)})


@app.get("/shotimg/{name}")
def shot_img(name: str):
    """识别缩略图（该帧点云的服务端渲染 JPEG）。"""
    if not re.fullmatch(r"[0-9a-f]{32}\.jpg", name):
        return JSONResponse({"error": "非法文件名"}, status_code=400)
    p = SHOT_DIR / name
    if not p.exists():
        return JSONResponse({"error": "不存在"}, status_code=404)
    return FileResponse(str(p), media_type="image/jpeg")


@app.get("/glb/{token}/{name}")
def serve_glb(token: str, name: str):
    """按 token 提供生成的 GLB（校验为 32 位 hex，仅允许 scene.glb，防目录穿越）。"""
    if len(token) != 32 or any(c not in "0123456789abcdef" for c in token) or name != "scene.glb":
        return JSONResponse({"error": "非法路径"}, status_code=400)
    p = GLB_DIR / token / "scene.glb"
    if not p.exists():
        return JSONResponse({"error": "产物已过期或不存在"}, status_code=404)
    return FileResponse(str(p), media_type="model/gltf-binary", filename="scene.glb")


# ══════════════════════════════════════════════════════════════════════
# 实时识别卡片流页：轮询 /api/recog/list，最近新建/更新的卡在最上；
# 缩略图=该帧点云的服务端渲染图（/shotimg/*）；名称流式打字；食物红 / 液体蓝。
# ══════════════════════════════════════════════════════════════════════
RECOG_PAGE = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>实时识别</title>
<style>
 :root{--bg:#f4f5f7;--panel:#fff;--panel2:#f6f7f9;--ink:#1b1e24;--muted:#69707b;--faint:#98a0ac;
   --line:#e5e8ec;--accent:#0071e3;--accent-soft:#e7f1fd;--food:#de3434;--food-soft:#fdeaea;
   --liquid:#2e78eb;--liquid-soft:#e9f0fd;--mono:ui-monospace,"SF Mono",Menlo,monospace;
   --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,"PingFang SC",sans-serif;}
 *{box-sizing:border-box}
 html,body{height:100%}
 body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.5;
   -webkit-font-smoothing:antialiased;display:flex;flex-direction:column}
 .nav{display:flex;gap:16px;align-items:center;padding:10px 16px;background:var(--panel);
   border-bottom:1px solid var(--line);font-size:13px}
 .nav a{color:var(--muted);text-decoration:none}.nav a.active{color:var(--accent);font-weight:600}
 .nav a.home{margin-left:auto;color:var(--faint)}
 .head{padding:14px 18px 12px;background:var(--panel);border-bottom:1px solid var(--line)}
 .head .l1{display:flex;align-items:center;gap:9px}
 .head h2{margin:0;font-size:16px;font-weight:650}
 .live{display:inline-flex;align-items:center;gap:6px;font-size:11px;font-weight:600;color:#1a9e5f;
   background:rgba(26,158,95,.12);padding:3px 9px;border-radius:999px;margin-left:auto}
 .live.off{color:var(--faint);background:rgba(120,130,145,.12)}
 .clr{margin-left:10px;font-size:12px;font-weight:600;color:var(--muted);background:var(--panel2);border:1px solid var(--line);padding:5px 13px;border-radius:8px;cursor:pointer}
 .clr:hover{color:var(--food);border-color:var(--food)}
 /* 识别目标切换（Qwen / Gemini Pro）：分段按钮，未配置 endpoint 的目标置灰 */
 .seg{display:inline-flex;border:1px solid var(--line);border-radius:9px;overflow:hidden;background:var(--panel2)}
 .seg button{font-size:12px;font-weight:600;padding:5px 13px;border:0;background:transparent;color:var(--muted);cursor:pointer}
 .seg button.sel{background:var(--accent);color:#fff}
 .seg button:disabled{opacity:.4;cursor:not-allowed}
 .live i{width:7px;height:7px;border-radius:50%;background:currentColor;animation:pulse 1.4s infinite}
 @keyframes pulse{0%,100%{opacity:1}50%{opacity:.25}}
 .head .sub{font-size:12px;color:var(--muted);margin-top:3px}
 .head .sub code{font-family:var(--mono);font-size:11px;color:var(--accent);background:var(--accent-soft);padding:1px 6px;border-radius:5px}
 .feed{flex:1;overflow-y:auto;padding:14px;display:flex;flex-direction:column;gap:12px}
 .empty{color:var(--faint);font-size:13px;text-align:center;margin:36px 10px}
 .rcard{position:relative;display:grid;grid-template-columns:118px 1fr auto;gap:15px;padding:12px;border:1px solid var(--line);
   border-radius:13px;background:var(--panel2);animation:rise .34s cubic-bezier(.2,.7,.3,1)}
 /* 卡片右侧：最近一次识别该食物的 VLM 延时 + 所用模型 */
 .rlat{display:flex;flex-direction:column;align-items:flex-end;justify-content:center;gap:3px;min-width:72px}
 .rlat .lv{font-size:15px;font-weight:650;font-family:var(--mono);font-variant-numeric:tabular-nums}
 .rlat .lm{font-size:10.5px;color:var(--faint)}
 @keyframes rise{from{opacity:0;transform:translateY(-10px)}to{opacity:1;transform:none}}
 .thumb{position:relative;width:104px;aspect-ratio:3/4;border-radius:9px;overflow:hidden;border:1px solid var(--line);
   background-color:#10141a;
   background-image:radial-gradient(58% 42% at 50% 60%,rgba(196,204,218,.5),rgba(120,130,150,.14) 55%,transparent 74%),
     radial-gradient(rgba(205,214,228,.5) .6px,transparent .8px);background-size:auto,4px 4px}
 .thumb img{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;opacity:0;transition:opacity .3s}
 .thumb .tag{position:absolute;left:6px;bottom:6px;font-size:9px;font-family:var(--mono);color:#fff;
   background:rgba(10,14,20,.62);padding:1px 6px;border-radius:4px;z-index:2}
 .rbody{min-width:0;display:flex;flex-direction:column;gap:11px}
 .fld{display:flex;flex-direction:column;gap:3px;min-width:0}
 .flab{font-size:10px;letter-spacing:.3px;color:var(--faint)}
 .name{font-size:16px;font-weight:650;line-height:1.25;display:flex;align-items:center;gap:7px;min-width:0}
 .name.pending .nm{color:var(--faint)}
 .tdot{width:8px;height:8px;border-radius:50%;background:var(--faint);flex:0 0 auto}
 .tdot.food{background:var(--food)}.tdot.liquid{background:var(--liquid)}
 .desc{font-size:13px;color:var(--muted);line-height:1.5}
 .tags{display:flex;flex-wrap:wrap;gap:6px}
 .chip{font-size:11.5px;font-weight:600;color:var(--accent);background:var(--accent-soft);padding:2px 9px;border-radius:999px}
 .sig{font-size:14px;font-weight:650;color:var(--ink)}
 .hide{display:none}
 .meta{margin-top:2px;font-size:10.5px;color:var(--faint);font-family:var(--mono);font-variant-numeric:tabular-nums;display:flex;gap:12px}
 .caret{display:inline-block;width:2px;height:1em;background:var(--accent);vertical-align:-1px;margin-left:1px;animation:blink 1s step-end infinite}
 @keyframes blink{50%{opacity:0}}
 /* 扇形叠卡缩略图（一食物 30 秒内去重合并的多帧点云截图） */
 .stack{position:relative;width:118px;height:150px;cursor:pointer;flex:0 0 auto}
 .stack .sh{position:absolute;left:9px;top:3px;width:100px;height:133px;border-radius:9px;border:2px solid var(--panel);
   background-color:#10141a;overflow:hidden;box-shadow:0 2px 8px rgba(15,22,34,.22);transition:transform .28s cubic-bezier(.2,.7,.3,1);
   background-image:radial-gradient(58% 42% at 50% 60%,rgba(196,204,218,.4),transparent 72%),radial-gradient(rgba(205,214,228,.4) .6px,transparent .8px);background-size:auto,4px 4px}
 .stack .sh img{width:100%;height:100%;object-fit:cover;display:block;opacity:0;transition:opacity .3s}
 .stack .cnt{position:absolute;right:-2px;bottom:-2px;z-index:30;font-size:11px;font-weight:700;font-family:var(--mono);color:#fff;background:var(--accent);padding:2px 8px;border-radius:999px;box-shadow:0 1px 4px rgba(0,0,0,.3)}
 .stack .shint{position:absolute;left:0;right:0;bottom:-16px;text-align:center;font-size:9.5px;color:var(--faint)}
 .stack:hover .sh{box-shadow:0 4px 14px rgba(15,22,34,.3)}
 /* 去重命中：卡片闪 + 右上角“更新”提示 */
 .updated{position:absolute;top:10px;right:10px;z-index:6;font-size:11px;font-weight:700;color:#fff;background:var(--accent);padding:3px 10px;border-radius:999px;opacity:0;transform:translateY(-4px);transition:opacity .25s,transform .25s;pointer-events:none}
 .updated.on{opacity:1;transform:none}
 .flash{animation:cardflash .85s ease}
 @keyframes cardflash{0%{box-shadow:0 0 0 0 var(--accent-soft)}35%{box-shadow:0 0 0 3px var(--accent)}100%{box-shadow:0 0 0 0 transparent}}
 /* lightbox 看图集 */
 .lb{position:fixed;inset:0;background:rgba(10,14,20,.82);display:none;flex-direction:column;align-items:center;justify-content:center;gap:13px;z-index:100;padding:24px}
 .lb.on{display:flex}
 .lb .big{width:min(88vw,600px);max-height:74vh;aspect-ratio:3/4;border-radius:14px;overflow:hidden;border:2px solid rgba(255,255,255,.15);background:#10141a}
 .lb .big img{width:100%;height:100%;object-fit:contain;display:block;cursor:zoom-in;transform-origin:center center;will-change:transform;transition:transform .12s ease-out}
 .lb .cap{color:#dfe4ea;font-size:13px;font-family:var(--mono)}
 .lb .strip{display:flex;gap:8px;flex-wrap:wrap;justify-content:center;max-width:min(82vw,420px)}
 .lb .th{width:50px;height:66px;border-radius:7px;overflow:hidden;border:2px solid transparent;cursor:pointer;opacity:.55;background:#10141a}
 .lb .th img{width:100%;height:100%;object-fit:cover;display:block}
 .lb .th.sel{border-color:var(--accent);opacity:1}
 .lb .x{position:absolute;top:16px;right:20px;color:#fff;font-size:26px;cursor:pointer;opacity:.85;line-height:1}
 /* 智能跳顶：下翻时不打断，顶部浮「↑ 有新卡」按钮 */
 .newpill{position:fixed;top:104px;left:50%;transform:translateX(-50%) translateY(-8px);z-index:60;
   font-size:12px;font-weight:700;color:#fff;background:var(--accent);padding:6px 16px;border-radius:999px;
   box-shadow:0 4px 14px rgba(0,0,0,.28);cursor:pointer;opacity:0;pointer-events:none;transition:opacity .25s,transform .25s}
 .newpill.on{opacity:1;transform:translateX(-50%);pointer-events:auto}
 @media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style></head><body>
<div class="nav"><a class="active" href="/recog">实时识别</a><a href="/sam3tune">SAM3 调优</a><a class="home" href="/experience" target="_top">↗ 浅体验区</a></div>
<div class="head">
  <div class="l1"><h2>实时识别 · Live Recognition</h2>
    <span class="seg" id="seg"><button data-t="qwen">Qwen</button><button data-t="gemini">Gemini Pro</button></span>
    <span class="live" id="live"><i></i>识别中</span><button class="clr" id="clr">清空</button></div>
  <div class="sub">food/drink 命中某帧 → <code id="model">Qwen3-VL</code> 识别四字段 · 同一物 30 秒内去重合并（缩略图叠加，点击看图集）· 有更新的卡自动置顶 · 当前设备 <code id="devlab">–</code></div>
</div>
<div class="feed" id="feed"><div class="empty" id="empty">等待 food/drink 命中…</div></div>
<div class="newpill" id="newpill">↑ 有新卡</div>
<div class="lb" id="lb"><span class="x" id="lbx">✕</span>
 <div class="big"><img id="lbBig" alt=""></div>
 <div class="cap" id="lbCap"></div>
 <div class="strip" id="lbStrip"></div>
</div>
<script>
const $=id=>document.getElementById(id);
const feed=$('feed');
const state=new Map();   // id -> {el, rev, shots:[img_url...]}
const typeCls=t=> t==='液体' ? 'liquid' : 'food';

// —— 打字机 ——
function typeInto(node,text){ let i=0; node.textContent='';
  const caret=document.createElement('span'); caret.className='caret'; node.appendChild(caret);
  const timer=setInterval(()=>{ i++; caret.remove(); node.textContent=text.slice(0,i); node.appendChild(caret);
    if(i>=text.length){clearInterval(timer); caret.remove();}}, 46);
}

// —— 扇形叠卡：每张 = Qwen box 裁剪图（/shotimg/*），扇形叠，>5 张右下角显示数字，点击开 lightbox ——
function renderStack(el, name, shots){
  const stack=el.querySelector('.stack');
  stack.querySelectorAll('.sh,.cnt,.shint').forEach(n=>n.remove());
  const show=shots.slice(-5);                 // 最多露 5 张边角
  show.forEach((url,i)=>{
    const isTop=i===show.length-1, off=show.length-1-i, dir=(off%2?-1:1);
    const d=document.createElement('div'); d.className='sh';
    d.style.transform='rotate('+(isTop?0:dir*(4+off*3))+'deg) translate('+(isTop?0:dir*(5+off*3))+'px,'+(off*2)+'px)';
    d.style.zIndex=String(i+1);
    const img=document.createElement('img'); d.appendChild(img);
    img.onload=()=>{img.style.opacity=1;}; img.src=url;
    stack.appendChild(d);
  });
  if(shots.length>5){ const b=document.createElement('span'); b.className='cnt'; b.textContent='×'+shots.length; stack.appendChild(b); }
  const h=document.createElement('span'); h.className='shint'; h.textContent='点击看 '+shots.length+' 张'; stack.appendChild(h);
  stack.onclick=()=>openLB(name, shots);
}

function cardEl(c){
  const el=document.createElement('div'); el.className='rcard';
  el.innerHTML='<div class="updated">更新</div><div class="stack"></div>'
    +'<div class="rbody">'
    +'  <div class="fld"><div class="flab">识别对象 / Detected Food</div>'
    +'    <div class="name"><span class="tdot '+typeCls(c.type)+'"></span><span class="nm"></span></div></div>'
    +'  <div class="fld f-desc hide"><div class="flab">一句话描述 / Description (EN)</div><div class="desc"></div></div>'
    +'  <div class="fld f-nutr hide"><div class="flab">卡路里与营养（按可见份量估算）/ Nutrition (est.)</div><div class="tags nutr"></div></div>'
    +'  <div class="fld f-cls hide"><div class="flab">健康分级 / Classification</div><div class="sig cls"></div></div>'
    +'  <div class="meta"><span>帧 '+(c.frame||'')+'</span><span>'+(c.t||'')+'</span></div>'
    +'</div>'
    +'<div class="rlat"><div class="flab">识别延时 / Latency</div><div class="lv"></div><div class="lm"></div></div>';
  latFill(el, c);
  typeInto(el.querySelector('.nm'), c.name||'');
  if(c.description_en){ el.querySelector('.f-desc').classList.remove('hide'); el.querySelector('.desc').textContent=c.description_en; }
  const nutr=[];   // 老卡（改版前识别的）没有这些字段 → 整段隐藏，不炸
  if(c.calories_kcal!=null)nutr.push(c.calories_kcal+' kcal');
  if(c.protein_g!=null)nutr.push('蛋白 '+c.protein_g+'g');
  if(c.carbs_g!=null)nutr.push('碳水 '+c.carbs_g+'g');
  if(c.fat_g!=null)nutr.push('脂肪 '+c.fat_g+'g');
  if(nutr.length){ el.querySelector('.f-nutr').classList.remove('hide');
    el.querySelector('.nutr').innerHTML=nutr.map(t=>'<span class="chip">'+t+'</span>').join(''); }
  if(c.classification){ el.querySelector('.f-cls').classList.remove('hide'); el.querySelector('.cls').textContent=c.classification; }
  return el;
}
// —— 卡片右侧延时：最近一次识别该食物的 VLM 耗时 + 所用模型（新建与去重合并都会刷新）——
function latFill(el,c){
  const v=el.querySelector('.rlat .lv'), m=el.querySelector('.rlat .lm');
  if(c.latency_ms!=null){ v.textContent=(c.latency_ms/1000).toFixed(1)+'s'; m.textContent=c.latency_model||''; }
  else { v.textContent='–'; m.textContent=''; }
}
function flashUpdate(el){
  el.classList.remove('flash'); void el.offsetWidth; el.classList.add('flash');
  const u=el.querySelector('.updated'); u.classList.add('on');
  clearTimeout(el._ut); el._ut=setTimeout(()=>u.classList.remove('on'), 1600);
}

// —— lightbox 看图集 ——
let lbShots=[], lbIdx=0, lbName='';
let lbScale=1, lbX=0, lbY=0, lbDrag=false, lbSx=0, lbSy=0;
function lbApply(){ const b=$('lbBig'); b.style.transform='translate('+lbX+'px,'+lbY+'px) scale('+lbScale+')';
  b.style.cursor= lbScale>1 ? (lbDrag?'grabbing':'grab') : 'zoom-in'; }
function lbReset(){ lbScale=1; lbX=0; lbY=0; lbApply(); }
function openLB(name, shots){ if(!shots.length)return;
  lbShots=shots.slice(); lbIdx=lbShots.length-1; lbName=name||'';
  const strip=$('lbStrip'); strip.innerHTML='';
  lbShots.forEach((url,i)=>{ const t=document.createElement('div'); t.className='th'+(i===lbIdx?' sel':'');
    const img=document.createElement('img'); img.src=url; t.appendChild(img);
    t.onclick=()=>{lbIdx=i; renderLB();}; strip.appendChild(t); });
  $('lb').classList.add('on'); renderLB();
}
function renderLB(){ const url=lbShots[lbIdx], big=$('lbBig');
  big.src=url;
  $('lbCap').textContent=lbName+' · '+(lbIdx+1)+' / '+lbShots.length+' 张 · 滚轮缩放·拖动·双击';
  [...$('lbStrip').children].forEach((t,i)=>t.classList.toggle('sel',i===lbIdx));
  lbReset();
}
// 缩放：滚轮缩放(1~6x)、放大后拖动平移、双击放大/还原
$('lbBig').addEventListener('wheel',e=>{ e.preventDefault();
  lbScale=Math.max(1,Math.min(6, lbScale*(e.deltaY<0?1.18:0.85)));
  if(lbScale===1){lbX=0;lbY=0;} lbApply(); },{passive:false});
$('lbBig').addEventListener('mousedown',e=>{ if(lbScale<=1)return; lbDrag=true; lbSx=e.clientX-lbX; lbSy=e.clientY-lbY; lbApply(); e.preventDefault(); });
window.addEventListener('mousemove',e=>{ if(!lbDrag)return; lbX=e.clientX-lbSx; lbY=e.clientY-lbSy; lbApply(); });
window.addEventListener('mouseup',()=>{ if(lbDrag){lbDrag=false; lbApply();} });
$('lbBig').addEventListener('dblclick',()=>{ lbScale=lbScale>1?1:2.5; lbX=0; lbY=0; lbApply(); });
$('lbx').onclick=()=>$('lb').classList.remove('on');
$('lb').onclick=e=>{ if(e.target===$('lb'))$('lb').classList.remove('on'); };
document.addEventListener('keydown',e=>{ if(!$('lb').classList.contains('on'))return;
  if(e.key==='Escape')$('lb').classList.remove('on');
  if(e.key==='ArrowRight'){lbIdx=(lbIdx+1)%lbShots.length;renderLB();}
  if(e.key==='ArrowLeft'){lbIdx=(lbIdx-1+lbShots.length)%lbShots.length;renderLB();} });

// 识别目标切换：POST 后端换 endpoint/model，免重启、下一轮识别生效；按钮状态由 tick 对账
$('seg').addEventListener('click', async e=>{
  const b=e.target.closest('button');
  if(!b||b.disabled||b.classList.contains('sel'))return;
  try{
    const r=await fetch('/api/recog/target',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify({target:b.dataset.t})});
    if(r.ok) tick(); else console.log('识别目标切换失败：', await r.text());
  }catch(e){/* 网络失败忽略，按钮状态由下个 tick 恢复 */}
});

// 清空：后端清卡 + 前端清 feed/state + 关 lightbox
$('clr').onclick=async()=>{
  try{ await fetch('/api/recog/clear',{method:'POST'}); }catch(e){}
  feed.querySelectorAll('.rcard').forEach(n=>n.remove()); state.clear();
  $('lb').classList.remove('on'); $('empty').style.display='block';
};

// —— 智能跳顶：接近顶部(<100px)时新内容自动回顶；正在下翻则浮出「↑ 有新卡」不打断 ——
function bumpScroll(){
  if(feed.scrollTop<100){ feed.scrollTop=0; $('newpill').classList.remove('on'); }
  else $('newpill').classList.add('on');
}
$('newpill').onclick=()=>{ feed.scrollTo({top:0,behavior:'smooth'}); $('newpill').classList.remove('on'); };
feed.addEventListener('scroll',()=>{ if(feed.scrollTop<10) $('newpill').classList.remove('on'); });

let curDev=null;   // 卡片流跟随服务端选中设备：切换时列表自动换成新设备的桶（旧卡不丢，切回来还在）
async function tick(){
 try{
  const d=await(await fetch('/api/recog/list',{cache:'no-store'})).json();
  const live=$('live');
  if(!d.enabled){ live.className='live off'; live.innerHTML='<i></i>未接入';
    $('empty').textContent='识别服务未接入（RECOG_ENDPOINT 未配置）'; }
  else { live.className='live'; live.innerHTML='<i></i>识别中'; }
  // 识别目标：模型名 + 分段按钮状态（选中高亮；未配置 endpoint 的目标置灰）
  if(d.model) $('model').textContent=d.model;
  const tg=d.targets||{};
  [...$('seg').children].forEach(b=>{ const t=b.dataset.t;
    b.classList.toggle('sel', t===d.target);
    b.disabled=!tg[t];
    b.title=tg[t]?'':'未配置该目标的 endpoint（服务器 .env）'; });
  // 设备切换（在 /panel 下拉触发）：关掉图集弹层，卡片由下方 seen 对账逻辑自动整体换桶
  const dev=d.device||null;
  if(curDev!==null && dev!==curDev) $('lb').classList.remove('on');
  curDev=dev; $('devlab').textContent=dev||'–';
  const cards=d.cards||[];
  const seen=new Set(cards.map(c=>c.id));
  for(const [id,st] of state){ if(!seen.has(id)){ st.el.remove(); state.delete(id); } }
  $('empty').style.display = cards.length ? 'none' : 'block';
  let changed=false;
  // 新卡：逆序 prepend 使最新在最上
  cards.filter(c=>!state.has(c.id)).reverse().forEach(c=>{
    const el=cardEl(c); feed.insertBefore(el, feed.firstChild);
    const shots=c.shots||[]; state.set(c.id,{el, rev:c.rev||0, shots});
    renderStack(el, c.name||'', shots); changed=true;
  });
  // 已存在卡：rev 变化=去重合并了新缩略图 → 置顶 + 更新叠卡/帧信息 + 闪 + “更新”提示（内容不改）
  cards.forEach(c=>{ const st=state.get(c.id); if(!st)return;
    if((c.rev||0)!==st.rev){ st.rev=c.rev||0; st.shots=c.shots||[];
      if(feed.firstChild!==st.el) feed.insertBefore(st.el, feed.firstChild);
      const ms=st.el.querySelectorAll('.meta span');
      if(ms[0]) ms[0].textContent='帧 '+(c.frame||''); if(ms[1]) ms[1].textContent=c.t||'';
      latFill(st.el, c);
      renderStack(st.el, c.name||'', st.shots); flashUpdate(st.el); changed=true; } });
  if(changed) bumpScroll();
 }catch(e){}
}
setInterval(tick,700); tick();
</script>
</body></html>"""


@app.get("/recog", response_class=HTMLResponse)
def recog_page():
    """实时识别卡片流页。"""
    return RECOG_PAGE


# ══════════════════════════════════════════════════════════════════════
# SAM3 调优页：动态原图 + 定位 + presence / top-K query 原始分 + 运行历史。
# 漏检归因口径（SAM3 的分数分解 p(query匹配) = p(query匹配|概念在图中) × p(概念在图中)）：
#   · presence 低(<0.5) → 概念/词表问题，换词；
#   · presence 高但联合分被阈值砍 → 纯阈值问题，降阈值；
#   · top-K 里没有覆盖目标区域的框 → 定位问题（目标太小/被裁），切图。
# ══════════════════════════════════════════════════════════════════════
SAM3TUNE_PAGE = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SAM3 调优</title>
<style>
 :root{--bg:#f4f5f7;--panel:#fff;--ink:#1b1e24;--muted:#69707b;--faint:#98a0ac;--line:#e5e8ec;
   --accent:#0071e3;--accent-soft:#e7f1fd;--ok:#1a9e5f;--warn:#c77b12;--err:#de3434;
   --mono:ui-monospace,"SF Mono",Menlo,monospace;--sans:-apple-system,BlinkMacSystemFont,system-ui,"PingFang SC",sans-serif}
 @media (prefers-color-scheme:dark){:root{--bg:#0c0e11;--panel:#15181d;--ink:#e8eaed;--muted:#98a1ad;--faint:#6b7480;--line:#262b32;--accent:#3b9bff;--accent-soft:#132436}}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.5}
 .nav{display:flex;gap:16px;align-items:center;padding:10px 16px;background:var(--panel);border-bottom:1px solid var(--line);font-size:13px}
 .nav a{color:var(--muted);text-decoration:none}.nav a.active{color:var(--accent);font-weight:600}
 .wrap{padding:10px 14px 30px;max-width:1800px;margin:0 auto}
 h1{font-size:16px;margin:2px 0 3px;font-weight:650}
 h2{font-size:13px;margin:14px 0 6px;font-weight:650}
 .sub{font-size:12px;color:var(--muted);margin-bottom:8px;line-height:1.45}
 .sub code{font-family:var(--mono);font-size:11px}
 .bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:8px 12px;margin-bottom:10px}
 .bar label{font-size:12.5px;color:var(--muted)}
 .bar input[type=text]{font-size:13px;padding:6px 10px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--ink);width:200px}
 .bar input[type=number]{font-size:13px;padding:6px 8px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--ink);width:62px}
 .bar select{font-size:13px;padding:6px 8px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--ink);min-width:150px}
 .btn{font-size:13px;font-weight:600;color:#fff;background:var(--accent);border:0;padding:7px 16px;border-radius:8px;cursor:pointer}
 .btn:disabled{opacity:.5;cursor:default}
 .btn2{background:var(--panel);color:var(--muted);border:1px solid var(--line)}
 .btn.on{background:var(--ok)}
 .st{font-size:12px;font-family:var(--mono);color:var(--faint);margin-left:auto}
 .st .ok{color:var(--ok)}.st .err{color:var(--err)}
 .live{display:flex;gap:10px;align-items:stretch;flex-wrap:wrap}
 .live figure{flex:0 0 auto}
 figure{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden}
 figure .cap{padding:6px 10px;font-size:12px;font-weight:600;border-bottom:1px solid var(--line);display:flex;gap:8px;align-items:center}
 figure .cap span{font-weight:400;color:var(--faint);font-family:var(--mono);font-size:10.5px;margin-left:auto}
 figure .box{background:#10141a;display:flex;align-items:center;justify-content:center;height:300px;min-width:200px}
 figure img{height:100%;width:auto;max-width:46vw;display:block;object-fit:contain}
 .empty{color:var(--faint);font-size:12.5px;padding:20px 10px;text-align:center}
 .scores{flex:1 1 320px;min-width:300px;display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:10px;align-content:start}
 .wcard{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:9px 12px}
 .whead{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}
 .wchip{font-size:12.5px;font-weight:700;padding:0 10px;border-radius:980px;color:#fff}
 .pres{font-family:var(--mono);font-size:18px;font-weight:700}
 .pres.lo{color:var(--err)}.pres.hi{color:var(--ok)}
 .wmeta{font-size:11px;color:var(--faint);font-family:var(--mono)}
 table.tk{width:100%;border-collapse:collapse;margin-top:5px;font-family:var(--mono);font-size:11px}
 table.tk th{text-align:right;font-weight:600;color:var(--muted);padding:1px 6px;border-bottom:1px solid var(--line)}
 table.tk td{text-align:right;padding:1px 6px;color:var(--ink)}
 table.tk th:first-child,table.tk td:first-child{text-align:left}
 table.tk tr.cut td{color:var(--faint)}
 table.tk tr.clamp td{color:var(--err)}
 .hist{display:grid;grid-template-columns:repeat(auto-fill,minmax(520px,1fr));gap:8px}
 .hcard{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:7px 9px;display:flex;gap:8px;align-items:flex-start}
 .hcard img{width:148px;max-width:22vw;border-radius:6px;display:block;background:#10141a}
 .hmeta{flex:1;min-width:0;font-size:11.5px}
 .hmeta .t{font-family:var(--mono);color:var(--faint);font-size:10.5px}
 .hline{margin:2px 0;font-family:var(--mono);font-size:11px}
 .hline b{font-family:var(--sans)}
 details summary{cursor:pointer;font-size:11px;color:var(--accent);margin-top:2px}
 .badge{display:inline-block;padding:0 7px;border-radius:980px;font-size:11px;font-weight:600}
 .badge.lo{background:#fdecec;color:var(--err)}
 .badge.hi{background:#e5f6ed;color:var(--ok)}
 @media (prefers-color-scheme:dark){.badge.lo{background:#3a1717}.badge.hi{background:#12301f}}
</style></head><body>
<div class="nav"><a href="/recog">实时识别</a><a class="active" href="/sam3tune">SAM3 调优</a><a href="/experience" style="margin-left:auto">↗ 浅体验区</a></div>
<div class="wrap">
 <h1>SAM3 调优 · presence 分 / top-K query 原始分</h1>
 <div class="sub">分数分解：<code>p(query匹配) = p(query匹配 | 概念在图中) × p(概念在图中·presence)</code>，presence 是全局乘性门控。
  归因口径：<b>presence 低(&lt;0.5)</b>→概念/词表问题，换词；<b>presence 高但联合分被阈值砍</b>→纯阈值问题；<b>top-K 里没有覆盖目标区域的框</b>→定位问题，切图。
  联合分 = 最终检测置信度（NMS/阈值前，logit clamp ±10）；原始分 = 直取 dot_prod_scoring 的 <b>pre-sigmoid logit</b>（未经 presence 加权，负样本是 −8/−12 这类明确负数，非除法反推）；联合 logit 触到 clamp 的行<span style="color:var(--err)">标红</span>。
  <b>阈值口径可调</b>：检测保留改为 <code>presence<sup>α</sup> × cond &gt; 阈值</code>——α=1 即原始行为（联合分口径），α=0 完全忽略 presence（纯 cond 口径），中间值为指数软化的连续旋钮；表中 score(α) 列即该分，过阈值行正常显示、未过灰显。</div>
 <div class="bar">
  <label id="devlbl" style="display:none">设备</label>
  <select id="devsel" style="display:none"></select>
  <label>text（英文名词，逗号分隔最多 4 词，每词一色）</label>
  <input type="text" id="text" value="food" placeholder="如 food / bowl of rice, bottle">
  <label>top-K</label>
  <input type="number" id="topk" value="10" min="1" max="50">
  <span style="display:inline-flex;gap:8px;align-items:center;border:1px dashed var(--line);border-radius:8px;padding:4px 10px">
   <label title="score = presence^α × cond：α=1 原始行为(联合分)，α=0 完全忽略 presence，中间为指数软化">presence α</label>
   <input type="number" id="alpha" value="1" min="0" max="1" step="0.05">
   <label title="检测保留与 NMS 阈值，卡在 presence^α × cond 上">阈值</label>
   <input type="number" id="thr" value="0.5" min="0.05" max="0.95" step="0.05">
  </span>
  <button class="btn" id="run">运行一次</button>
  <button class="btn btn2" id="auto">自动（新帧驱动）</button>
  <button class="btn btn2" id="clear">清空历史</button>
  <span class="st" id="st">就绪</span>
 </div>
 <div class="live">
  <figure><div class="cap">① 实时原图 <span id="m1"></span></div>
   <div class="box"><div class="empty" id="e1">等待设备帧…</div><img id="i1" style="display:none"></div></figure>
  <figure><div class="cap">② SAM3 定位 <span id="m2"></span></div>
   <div class="box"><div class="empty" id="e2">点「运行一次」或开自动</div><img id="i2" style="display:none"></div></figure>
  <div class="scores" id="scores"><div class="empty">—</div></div>
 </div>
 <h2>历史（原图 / 定位 / 分数，服务端保留最近 30 条）</h2>
 <div class="hist" id="hist"><div class="empty">暂无历史</div></div>
</div>
<script>
const $=id=>document.getElementById(id);
let auto=false, busy=false, lastSeq=-1, lastRunSeq=-1, lastHistId=-1, curDev=null, lastDevKey='';

// 设备下拉：与主面板同一套 /api/frame/select（选中是全局的，影响处理线程与本页分析帧）
const devsel=$('devsel');
devsel.addEventListener('change',()=>{
  fetch('/api/frame/select',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({device_id:devsel.value})}).catch(()=>{});
});
function renderDevices(s){
  const devs=s.devices||[];
  devsel.style.display=devs.length?'':'none'; $('devlbl').style.display=devs.length?'':'none';
  if(document.activeElement!==devsel){   // 下拉展开操作中不重建选项，避免选择被打断
    const key=devs.map(d=>d.device_id).join('|')+'#'+(s.selected||'');
    if(key!==lastDevKey){ lastDevKey=key;
      devsel.innerHTML=devs.map(d=>'<option value="'+d.device_id+'"'
        +(d.device_id===s.selected?' selected':'')+'>'+d.device_id+'</option>').join('');
    }
  }
}
function resetForSwitch(){   // 切设备：清掉旧设备的原图/定位/分数，等新设备的帧与下一次运行
  lastSeq=-1;
  $('i1').style.display='none'; $('e1').style.display='';
  $('i2').style.display='none'; $('e2').style.display=''; $('m2').textContent='';
  $('scores').innerHTML='<div class="empty">—</div>';
}

// ① 原图动态刷新：与主面板同一套 /api/frame/status + /api/frame/latest 轮询
async function frameTick(){
  try{
    const s=await(await fetch('/api/frame/status',{cache:'no-store'})).json();
    renderDevices(s);
    if(s.device && s.device!==curDev){ if(curDev!==null) resetForSwitch(); curDev=s.device; }
    if(s.has_frame && s.seq!==lastSeq){ lastSeq=s.seq;
      $('i1').src='/api/frame/latest?t='+s.seq; $('i1').style.display='block'; $('e1').style.display='none'; }
    $('m1').textContent = s.has_frame ? ((s.device?s.device+' · ':'')+'帧 '+s.seq
      +(s.interval?(' · 间隔 '+s.interval.toFixed(1)+'s'):'')) : '';
    // 自动模式（新帧驱动）：仅当设备帧 seq 前进且上一次分析已结束时才触发 SAM3
    if(auto && !busy && !document.hidden && s.has_frame && s.seq!==lastRunSeq){ lastRunSeq=s.seq; run(); }
  }catch(e){}
}
setInterval(frameTick, 500); frameTick();

function presClass(p){ return p==null ? '' : (p<0.5?'lo':'hi'); }
function fmt(x,d){ return x==null ? '—' : Number(x).toFixed(d==null?4:d); }

function effScore(dbg,q,alpha){  // 阈值口径分：presence^α × cond（与服务端 keep 同一公式）
  if(!dbg || dbg.presence_score==null || q.cond_score==null || alpha==null) return null;
  return Math.pow(dbg.presence_score,alpha)*q.cond_score;
}

function tkTable(dbg,alpha,thr){
  if(!dbg || !dbg.topk || !dbg.topk.length) return '<div class="wmeta">无 top-K 数据（SAM3 server 需支持 debug）</div>';
  let h='<table class="tk"><tr><th>#</th><th>query</th><th>score(α)</th><th>原始分</th><th>原始logit</th><th>联合分</th><th>联合logit</th></tr>';
  for(const q of dbg.topk){
    const es=effScore(dbg,q,alpha);
    // clamp 标红优先（联合 logit 触 ±10 限幅，联合列数值不可信）；score(α) 未过阈值灰显
    const cls = q.clamped ? 'clamp'
      : (es!=null&&thr!=null ? (es>thr?'':'cut') : (q.joint_score!=null&&q.joint_score<0.5?'cut':''));
    h+='<tr class="'+cls+'"><td>'+q.rank+'</td><td>q'+q.query_idx+'</td><td><b>'+fmt(es)+'</b></td><td>'+fmt(q.cond_score)
      +'</td><td>'+fmt(q.cond_logit,2)+'</td><td>'+fmt(q.joint_score)+'</td><td>'+fmt(q.joint_logit,2)+(q.clamped?' ⛔':'')+'</td></tr>';
  }
  return h+'</table>';
}

function wordCard(w,alpha,thr){
  const dbg=w.debug||null, p=dbg?dbg.presence_score:null;
  const inst=(w.instances||[]).map(it=>'#'+it.obj_id+' '+fmt(it.score,3)).join('  ')||'无';
  return '<div class="wcard">'
    +'<div class="whead"><span class="wchip" style="background:'+w.color+'">'+w.word+'</span>'
    +'<span class="pres '+presClass(p)+'">presence '+fmt(p)+'</span>'
    +'<span class="wmeta">logit '+(dbg?fmt(dbg.presence_logit,2):'—')+' · queries '+(dbg&&dbg.num_queries!=null?dbg.num_queries:'—')
    +(alpha!=null?' · α='+alpha+' thr='+thr:'')+'</span></div>'
    +'<div class="wmeta" style="margin-top:4px">过阈值实例('+w.n+')：'+inst+'</div>'
    +tkTable(dbg,alpha,thr)+'</div>';
}

function renderScores(d){
  $('scores').innerHTML = (d.words||[]).map(w=>wordCard(w,d.alpha,d.thresh)).join('') || '<div class="empty">—</div>';
}

function histLine(w){
  const dbg=w.debug||null, p=dbg?dbg.presence_score:null;
  const top = dbg&&dbg.topk&&dbg.topk.length ? dbg.topk[0] : null;
  return '<div class="hline"><b style="color:'+w.color+'">'+w.word+'</b> '
    +'<span class="badge '+(p==null?'':(p<0.5?'lo':'hi'))+'">presence '+fmt(p,3)+'</span> '
    +'实例 '+w.n
    +(top?(' · top1 联合 '+fmt(top.joint_score,3)+' / 原始 '+fmt(top.cond_score,3)):'')+'</div>';
}

function renderHist(items){
  if(!items.length){ $('hist').innerHTML='<div class="empty">暂无历史</div>'; return; }
  $('hist').innerHTML = items.map(it=>{
    const t=new Date(it.ts*1000).toLocaleTimeString('zh-CN',{hour12:false});
    return '<div class="hcard"><img src="'+it.raw+'" alt="原图"><img src="'+it.seg+'" alt="定位">'
      +'<div class="hmeta"><div class="t">#'+it.id+' · '+t+(it.device?' · '+it.device:'')+' · text="'+it.text+'"'
      +(it.alpha!=null?' · α='+it.alpha+' thr='+it.thresh:'')+' · '+it.seg_ms+'ms · 实例 '+it.n_inst+'</div>'
      +(it.words||[]).map(histLine).join('')
      +'<details open><summary>top-K 明细</summary>'+(it.words||[]).map(w=>'<div class="hline"><b style="color:'+w.color+'">'+w.word+'</b></div>'+tkTable(w.debug,it.alpha,it.thresh)).join('')+'</details>'
      +'</div></div>';
  }).join('');
}

async function loadHist(){
  try{
    const d=await(await fetch('/api/sam3tune/history',{cache:'no-store'})).json();
    const items=d.items||[];
    const newest=items.length?items[0].id:-1;
    if(newest!==lastHistId){ lastHistId=newest; renderHist(items); }
  }catch(e){}
}
setInterval(loadHist, 5000); loadHist();

async function run(){
  if(busy) return; busy=true; $('run').disabled=true; $('st').textContent='跑 SAM3 中…';
  try{
    const r=await fetch('/api/sam3tune/run',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({text:$('text').value||'food', topk:parseInt($('topk').value)||10,
        alpha:parseFloat($('alpha').value), thresh:parseFloat($('thr').value)})});
    const d=await r.json();
    if(!d.ok){ $('st').innerHTML='<span class="err">'+(d.error||'失败')+'</span>'; }
    else{
      $('i2').src=d.seg; $('i2').style.display='block'; $('e2').style.display='none';
      $('m2').textContent='#'+d.id+(d.device?' · '+d.device:'')+' · α='+d.alpha+' thr='+d.thresh+' · '+d.seg_ms+'ms · 实例 '+d.n_inst;
      renderScores(d);
      $('st').innerHTML='<span class="ok">OK</span> · #'+d.id+' · '+d.seg_ms+'ms';
      loadHist();
    }
  }catch(e){ $('st').innerHTML='<span class="err">请求失败：'+e+'</span>'; }
  busy=false; $('run').disabled=false;
}
$('run').onclick=run;
// 自动模式 = 新帧驱动：不做定时连跑，frameTick 里检测到设备帧 seq 变化才触发一次分析
// （同一帧不会重复分析；改词/改 α/阈值后想立刻看效果，点「运行一次」即可）
$('auto').onclick=()=>{ auto=!auto; $('auto').textContent=auto?'停止自动':'自动（新帧驱动）';
  $('auto').classList.toggle('on',auto); if(auto){ lastRunSeq=lastSeq; run(); } };
$('clear').onclick=async()=>{ await fetch('/api/sam3tune/clear',{method:'POST'}); lastHistId=-1; loadHist(); };
// 首次探活
fetch('/api/sam3/health').then(r=>r.json()).then(d=>{
  $('st').innerHTML = d.ok ? '<span class="ok">SAM3 已接通</span> · '+d.endpoint
                           : '<span class="err">SAM3 未接通</span> · '+d.endpoint+' · '+(d.error||'');
}).catch(()=>{});
</script>
</body></html>"""


@app.get("/sam3tune", response_class=HTMLResponse)
def sam3tune_page():
    """SAM3 调优页：动态原图+定位、presence/top-K query 原始分、运行历史。"""
    return SAM3TUNE_PAGE
