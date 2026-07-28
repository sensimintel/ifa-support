# -*- coding: utf-8 -*-
"""设备实时帧中继模块（零重依赖，可脱离 DA3/torch 独立运行与自测）。

用途：mobile 端把从设备取到的照片按现有上传链路的 multipart 形态直发到本服务，
本模块接收后：
  1. 按 device_id（从 camera_info JSON 解析，缺省归 "unknown"）分桶，在内存里缓存
     每台设备的"最新一帧"（原始字节 + 元信息），供 /panel 面板动态刷新展示；
  2. 服务端维护一个"当前选中设备"（面板可切换）：若 app.py 注入了 DA3 处理回调
     （set_processor），后台单线程用"最新优先"策略把**选中设备**的最新帧按「当前
     控件配置」跑一次 DA3，产物（深度图字节 或 GLB 模型 url）写回该设备的桶。
     非选中设备的帧只更新自己的桶（可供缩略图预览），不触发 GPU 处理。

多设备语义：
  · 选中是粘性的——首台发帧的设备自动被选中，此后除非用户显式切换或该设备过期
    下线，不会因另一台设备发帧而跳走；
  · 切换设备（POST /api/frame/select）会立刻唤醒处理线程对新设备最新帧重算；
  · 设备条目超过 DEVICE_TTL 没有新帧即过期清理；选中设备过期后自动回落到
    最近有帧的设备。

刻意不 import torch / cv2 / depth_anything_3：DA3 能力通过回调注入。配置（产物类型/分辨率
等控件值）由 /panel 经 POST /api/frame/config 下发，本模块只当作不透明字典存下并转交回调，
因此本模块在本地无 GPU、无模型的环境下也能被 include_router 起来并跑通收图 + 展示链路。
"""
import json
import threading
import time
from typing import Callable, Optional

from fastapi import APIRouter, Body, File, Form, Query, UploadFile
from fastapi.responses import JSONResponse, Response

router = APIRouter()

DEVICE_TTL = 60.0        # 设备条目过期时间（秒）：超过没有新帧即视为下线并清理
UNKNOWN_DEVICE = "unknown"   # camera_info 缺失/无 device_id 时的兜底桶

# ── 共享状态：一把 Condition 同时兜住状态互斥与"新帧/新配置/切设备"的唤醒 ──────
_cv = threading.Condition()

# device_id -> 设备桶。每桶：
#   seq                帧单调递增序号（每设备独立计数）
#   image / content_type / received_at / prev_received_at / camera_info / timestamp
#   first_seen         该设备首次出现时间（列表排序用，保证下拉顺序稳定）
#   product/product_seq/product_gen/product_error   该设备最新一次 DA3 产物
_devices: dict = {}
_selected: Optional[str] = None       # 当前选中设备（粘性；None=还没有任何设备）

# 处理配置（由 /panel 下发的不透明字典，如 export_format/process_res/conf 等）
_config: dict = {}
_config_gen = 0                       # 配置版本号：变更时 +1，用于触发对最新帧的重算

# DA3 处理回调：fn(image_bytes, config, device_id) -> 产物描述字典；由 app.py 在有模型时注入
#   {"kind":"image","bytes":b"...","content_type":"image/jpeg","meta":{...}}  # 深度图
#   {"kind":"model","url":"/glb/<token>/scene.glb","meta":{...}}              # 点云/网格 GLB
_processor: Optional[Callable[[bytes, dict, str], dict]] = None
_worker_started = False


def set_processor(fn: Callable[[bytes, dict, str], dict]) -> None:
    """注入 DA3 处理回调并按需启动后台处理线程。

    app.py 里在模型可用时调用；本地自测不调用即为纯中继（只收图 + 展示原图）。
    """
    global _processor
    with _cv:
        _processor = fn
        _start_worker_locked()


def get_selected_device() -> Optional[str]:
    """当前生效的选中设备 id（无任何设备时 None）。供 app.py 的识别卡片等模块取默认设备。"""
    with _cv:
        return _effective_device_locked()


def _parse_device_id(camera_info: Optional[str]) -> str:
    """从随帧上报的 camera_info JSON 字符串里取 device_id；解析失败/缺失归 unknown 桶。"""
    if camera_info:
        try:
            dev = str((json.loads(camera_info) or {}).get("device_id") or "").strip()
            if dev:
                return dev
        except (ValueError, TypeError, AttributeError):
            pass
    return UNKNOWN_DEVICE


def _new_bucket(now: float) -> dict:
    return {
        "seq": 0, "image": None, "content_type": "image/jpeg",
        "received_at": 0.0, "prev_received_at": 0.0,
        "camera_info": None, "timestamp": None, "first_seen": now,
        "product": None, "product_seq": 0, "product_gen": 0, "product_error": None,
    }


def _prune_locked(now: float) -> None:
    """清理超过 DEVICE_TTL 没有新帧的设备条目（选中设备也会过期，随后自动回落）。"""
    stale = [d for d, st in _devices.items() if now - st["received_at"] > DEVICE_TTL]
    for d in stale:
        del _devices[d]


def _effective_device_locked() -> Optional[str]:
    """当前生效的选中设备：粘性选中仍在线则用之；否则回落到最近有帧的设备并粘住。"""
    global _selected
    if _selected is not None and _selected in _devices:
        return _selected
    if _devices:
        _selected = max(_devices, key=lambda d: _devices[d]["received_at"])
        return _selected
    _selected = None
    return None


def _start_worker_locked() -> None:
    """在持有 _cv 的前提下懒启动后台处理线程（只启动一次）。"""
    global _worker_started
    if _worker_started or _processor is None:
        return
    _worker_started = True
    threading.Thread(target=_worker_loop, daemon=True, name="frame-da3-worker").start()


def _worker_loop() -> None:
    """后台处理线程：始终按「当前配置」处理**选中设备**的"最新一帧"，中间帧直接丢弃。

    触发条件为 (选中设备, 该设备帧序号, 配置版本) 变化——来了新帧、控件改了配置、
    或用户切换了设备，都会对选中设备的最新帧重算。重活放锁外执行，不阻塞收图与展示。
    """
    last_key = None
    while True:
        with _cv:
            while True:
                dev = _effective_device_locked()
                st = _devices.get(dev) if dev else None
                if _processor is not None and st is not None and st["image"] is not None:
                    key = (dev, st["seq"], _config_gen)
                    if key != last_key:
                        break
                _cv.wait()
            seq, gen, img, proc = st["seq"], _config_gen, st["image"], _processor
            config = dict(_config)
        try:
            product = proc(img, config, dev)
            err = None
        except Exception as e:  # 处理失败不影响原图展示，仅记录错误
            product, err = None, f"{type(e).__name__}: {e}"
        with _cv:
            st2 = _devices.get(dev)
            if st2 is not None:      # 设备可能在处理期间过期下线，桶没了就丢弃产物
                st2["product"] = product if product is not None else st2["product"]
                st2["product_seq"] = seq
                st2["product_gen"] = gen
                st2["product_error"] = err
                _cv.notify_all()
        last_key = key


def _target_device_locked(device: Optional[str]):
    """按 query 参数解析目标设备桶：不传=当前选中设备。返回 (device_id, 桶或None)。"""
    dev = (device or "").strip() or _effective_device_locked()
    if dev is None:
        return None, None
    return dev, _devices.get(dev)


@router.post("/api/frame")
async def ingest_frame(
    image: UploadFile = File(...),
    camera_info: Optional[str] = Form(None),
    timestamp: Optional[str] = Form(None),
):
    """接收设备实时帧（与现有 /device/media/upload/image 同款 multipart 字段）。

    字段：image（二进制图片）、camera_info（可选 JSON 字符串，其中 device_id 用于分桶）、
    timestamp（可选）。仅更新该设备桶的最新帧并唤醒处理线程，立即返回，不阻塞在 GPU 推理上。
    """
    data = await image.read()
    if not data:
        return JSONResponse({"ok": False, "error": "空图片"}, status_code=400)
    dev = _parse_device_id(camera_info)
    now = time.time()
    with _cv:
        st = _devices.get(dev)
        if st is None:
            st = _devices[dev] = _new_bucket(now)
        st["seq"] += 1
        seq = st["seq"]
        st["image"] = data
        st["content_type"] = image.content_type or "image/jpeg"
        st["prev_received_at"] = st["received_at"]
        st["received_at"] = now
        st["camera_info"] = camera_info
        st["timestamp"] = timestamp
        _prune_locked(now)
        _effective_device_locked()    # 首台设备出现即自动粘性选中
        _cv.notify_all()
    return JSONResponse({"ok": True, "seq": seq, "bytes": len(data),
                         "received_at": now, "device": dev})


@router.post("/api/frame/select")
async def select_device(body: dict = Body(...)):
    """切换当前选中设备（/panel 下拉触发）。切换立即唤醒处理线程对新设备最新帧重算。"""
    global _selected
    dev = str((body or {}).get("device_id") or "").strip()
    if not dev:
        return JSONResponse({"ok": False, "error": "缺少 device_id"}, status_code=400)
    with _cv:
        if dev not in _devices:
            return JSONResponse({"ok": False, "error": f"设备不存在或已下线：{dev}"},
                                status_code=404)
        _selected = dev
        _cv.notify_all()
    return JSONResponse({"ok": True, "device": dev})


@router.post("/api/frame/config")
async def set_frame_config(config: dict = Body(...)):
    """由 /panel 下发处理配置（产物类型/分辨率/置信度/点数/相机线框等），
    变更配置版本号以触发对最新帧的重算。配置内容对本模块不透明，交由 DA3 回调解读。"""
    global _config, _config_gen
    with _cv:
        _config = dict(config or {})
        _config_gen += 1
        gen = _config_gen
        _cv.notify_all()
    return JSONResponse({"ok": True, "config_gen": gen, "config": _config})


@router.get("/api/frame/latest")
def latest_frame(device: Optional[str] = Query(None)):
    """返回指定设备（缺省=选中设备）最新一帧原始图片字节（供左框 <img> 直接展示）。"""
    with _cv:
        _dev, st = _target_device_locked(device)
        img = st["image"] if st else None
        ct = st["content_type"] if st else "image/jpeg"
        seq = st["seq"] if st else 0
    if img is None:
        return JSONResponse({"error": "暂无帧"}, status_code=404)
    return Response(
        content=img,
        media_type=ct,
        headers={"Cache-Control": "no-store", "X-Frame-Seq": str(seq)},
    )


@router.get("/api/frame/latest-product")
def latest_product(device: Optional[str] = Query(None)):
    """返回指定设备（缺省=选中设备）最新的 DA3 图片类产物（如彩色深度图）字节。
    模型类产物（GLB）不走这里——它经 /glb/<token>/scene.glb 由 model-viewer 加载，
    url 在 /api/frame/status 的 product_url 字段给出。"""
    with _cv:
        _dev, st = _target_device_locked(device)
        product = st["product"] if st else None
        seq = st["product_seq"] if st else 0
    if not product or product.get("kind") != "image" or not product.get("bytes"):
        return JSONResponse({"error": "暂无图片类产物"}, status_code=404)
    return Response(
        content=product["bytes"],
        media_type=product.get("content_type", "image/jpeg"),
        headers={"Cache-Control": "no-store", "X-Frame-Seq": str(seq)},
    )


@router.get("/api/frame/status")
def frame_status(device: Optional[str] = Query(None)):
    """返回指定设备（缺省=选中设备）最新帧的元信息、到达速率、当前配置与产物状态，
    以及在线设备列表 + 当前选中设备，供面板轮询渲染设备下拉并判断是否刷新。"""
    with _cv:
        now = time.time()
        _prune_locked(now)
        selected = _effective_device_locked()
        dev, st = _target_device_locked(device)
        devices = []
        for d in sorted(_devices, key=lambda x: _devices[x]["first_seen"]):
            b = _devices[d]
            itv = (b["received_at"] - b["prev_received_at"]) if b["prev_received_at"] else 0.0
            devices.append({
                "device_id": d, "seq": b["seq"],
                "age": (now - b["received_at"]) if b["received_at"] else None,
                "fps": (1.0 / itv) if itv > 0 else None,
                "selected": d == selected,
            })
        if st is None:
            return JSONResponse({
                "seq": 0, "has_frame": False, "device": dev, "selected": selected,
                "devices": devices, "processor": _processor is not None,
                "config": _config, "config_gen": _config_gen,
                "received_at": None, "age": None, "interval": None, "fps": None,
                "content_type": None, "camera_info": None, "timestamp": None,
                "product_kind": None, "product_url": None, "product_meta": None,
                "product_seq": 0, "product_gen": 0, "product_error": None,
            })
        interval = (st["received_at"] - st["prev_received_at"]) if st["prev_received_at"] else 0.0
        product = st["product"] or {}
        return JSONResponse({
            "seq": st["seq"],
            "has_frame": st["image"] is not None,
            "device": dev,
            "selected": selected,
            "devices": devices,
            "received_at": st["received_at"] or None,
            "age": (now - st["received_at"]) if st["received_at"] else None,
            "interval": interval or None,
            "fps": (1.0 / interval) if interval > 0 else None,
            "content_type": st["content_type"],
            "camera_info": st["camera_info"],
            "timestamp": st["timestamp"],
            # 产物状态
            "processor": _processor is not None,
            "config": _config,
            "config_gen": _config_gen,
            "product_kind": product.get("kind"),
            "product_url": product.get("url"),
            "product_meta": product.get("meta"),
            "product_seq": st["product_seq"],
            "product_gen": st["product_gen"],
            "product_error": st["product_error"],
        })
