# -*- coding: utf-8 -*-
"""设备实时帧中继模块（零重依赖，可脱离 DA3/torch 独立运行与自测）。

用途：mobile 端把从设备取到的照片按现有上传链路的 multipart 形态直发到本服务，
本模块接收后：
  1. 在内存里缓存"最新一帧"（原始字节 + 元信息），供 /panel 面板动态刷新展示；
  2. 若 app.py 注入了 DA3 处理回调（set_processor），后台单线程用"最新优先"策略把
     最新帧按「当前控件配置」跑一次 DA3，产物（深度图字节 或 GLB 模型 url）也缓存供展示。

刻意不 import torch / cv2 / depth_anything_3：DA3 能力通过回调注入。配置（产物类型/分辨率
等控件值）由 /panel 经 POST /api/frame/config 下发，本模块只当作不透明字典存下并转交回调，
因此本模块在本地无 GPU、无模型的环境下也能被 include_router 起来并跑通收图 + 展示链路。
"""
import threading
import time
from typing import Callable, Optional

from fastapi import APIRouter, Body, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response

router = APIRouter()

# ── 共享状态：一把 Condition 同时兜住状态互斥与"新帧/新配置到达"的唤醒 ──────────
_cv = threading.Condition()

_seq = 0                              # 已接收帧的单调递增序号
_image: Optional[bytes] = None        # 最新一帧原始字节
_content_type = "image/jpeg"          # 最新一帧 MIME
_received_at = 0.0                    # 最新一帧到达的服务器时间戳
_prev_received_at = 0.0              # 上一帧到达时间（用于估算到达间隔/帧率）
_camera_info: Optional[str] = None    # 随帧上报的 camera_info（原样透传的 JSON 字符串）
_timestamp: Optional[str] = None      # 随帧上报的设备时间戳字段

# 处理配置（由 /panel 下发的不透明字典，如 export_format/process_res/conf 等）
_config: dict = {}
_config_gen = 0                       # 配置版本号：变更时 +1，用于触发对最新帧的重算

# 最新一帧的 DA3 处理产物。回调返回一个描述字典：
#   {"kind":"image","bytes":b"...","content_type":"image/jpeg","meta":{...}}  # 深度图
#   {"kind":"model","url":"/glb/<token>/scene.glb","meta":{...}}              # 点云/网格 GLB
_product: Optional[dict] = None
_product_seq = 0                      # 产物对应的帧序号
_product_gen = 0                      # 产物对应的配置版本号
_product_error: Optional[str] = None  # 最近一次处理的错误（None 表示成功）

# DA3 处理回调：fn(image_bytes, config) -> 产物描述字典；由 app.py 在有模型时注入
_processor: Optional[Callable[[bytes, dict], dict]] = None
_worker_started = False


def set_processor(fn: Callable[[bytes, dict], dict]) -> None:
    """注入 DA3 处理回调并按需启动后台处理线程。

    app.py 里在模型可用时调用；本地自测不调用即为纯中继（只收图 + 展示原图）。
    """
    global _processor
    with _cv:
        _processor = fn
        _start_worker_locked()


def _start_worker_locked() -> None:
    """在持有 _cv 的前提下懒启动后台处理线程（只启动一次）。"""
    global _worker_started
    if _worker_started or _processor is None:
        return
    _worker_started = True
    threading.Thread(target=_worker_loop, daemon=True, name="frame-da3-worker").start()


def _worker_loop() -> None:
    """后台处理线程：始终按「当前配置」处理"最新一帧"，中间帧直接丢弃。

    触发条件为 (帧序号, 配置版本) 变化——即来了新帧、或控件改了配置，都会对最新帧重算，
    这样面板改产物类型/分辨率等能立刻在右框看到效果。重活放锁外执行，不阻塞收图与展示。
    """
    last_seq, last_gen = 0, -1
    while True:
        with _cv:
            while _processor is None or (_seq == last_seq and _config_gen == last_gen) or _image is None:
                _cv.wait()
            seq, gen, img, proc = _seq, _config_gen, _image, _processor
            config = dict(_config)
        try:
            product = proc(img, config)
            err = None
        except Exception as e:  # 处理失败不影响原图展示，仅记录错误
            product, err = None, f"{type(e).__name__}: {e}"
        with _cv:
            global _product, _product_seq, _product_gen, _product_error
            _product = product if product is not None else _product
            _product_seq = seq
            _product_gen = gen
            _product_error = err
            _cv.notify_all()
        last_seq, last_gen = seq, gen


@router.post("/api/frame")
async def ingest_frame(
    image: UploadFile = File(...),
    camera_info: Optional[str] = Form(None),
    timestamp: Optional[str] = Form(None),
):
    """接收设备实时帧（与现有 /device/media/upload/image 同款 multipart 字段）。

    字段：image（二进制图片）、camera_info（可选 JSON 字符串）、timestamp（可选）。
    仅缓存最新帧并唤醒处理线程，立即返回，不阻塞在 GPU 推理上。
    """
    data = await image.read()
    if not data:
        return JSONResponse({"ok": False, "error": "空图片"}, status_code=400)
    global _seq, _image, _content_type, _received_at, _prev_received_at
    global _camera_info, _timestamp
    now = time.time()
    with _cv:
        _seq += 1
        seq = _seq
        _image = data
        _content_type = image.content_type or "image/jpeg"
        _prev_received_at = _received_at
        _received_at = now
        _camera_info = camera_info
        _timestamp = timestamp
        _cv.notify_all()
    return JSONResponse({"ok": True, "seq": seq, "bytes": len(data), "received_at": now})


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
def latest_frame():
    """返回最新一帧原始图片字节（供左框 <img> 直接展示）。"""
    with _cv:
        img, ct, seq = _image, _content_type, _seq
    if img is None:
        return JSONResponse({"error": "暂无帧"}, status_code=404)
    return Response(
        content=img,
        media_type=ct,
        headers={"Cache-Control": "no-store", "X-Frame-Seq": str(seq)},
    )


@router.get("/api/frame/latest-product")
def latest_product():
    """返回最新一帧的 DA3 图片类产物（如彩色深度图）字节。
    模型类产物（GLB）不走这里——它经 /glb/<token>/scene.glb 由 model-viewer 加载，
    url 在 /api/frame/status 的 product_url 字段给出。"""
    with _cv:
        product, seq = _product, _product_seq
    if not product or product.get("kind") != "image" or not product.get("bytes"):
        return JSONResponse({"error": "暂无图片类产物"}, status_code=404)
    return Response(
        content=product["bytes"],
        media_type=product.get("content_type", "image/jpeg"),
        headers={"Cache-Control": "no-store", "X-Frame-Seq": str(seq)},
    )


@router.get("/api/frame/status")
def frame_status():
    """返回最新帧的元信息、到达速率、当前配置与产物状态，供面板轮询判断是否刷新。"""
    with _cv:
        now = time.time()
        interval = (_received_at - _prev_received_at) if _prev_received_at else 0.0
        product = _product or {}
        return JSONResponse({
            "seq": _seq,
            "has_frame": _image is not None,
            "received_at": _received_at or None,
            "age": (now - _received_at) if _received_at else None,
            "interval": interval or None,
            "fps": (1.0 / interval) if interval > 0 else None,
            "content_type": _content_type,
            "camera_info": _camera_info,
            "timestamp": _timestamp,
            # 产物状态
            "processor": _processor is not None,
            "config": _config,
            "config_gen": _config_gen,
            "product_kind": product.get("kind"),
            "product_url": product.get("url"),
            "product_meta": product.get("meta"),
            "product_seq": _product_seq,
            "product_gen": _product_gen,
            "product_error": _product_error,
        })
