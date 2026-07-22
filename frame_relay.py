# -*- coding: utf-8 -*-
"""设备实时帧中继模块（零重依赖，可脱离 DA3/torch 独立运行与自测）。

用途：mobile 端把从设备取到的照片按现有上传链路的 multipart 形态直发到本服务，
本模块接收后：
  1. 在内存里缓存"最新一帧"（原始字节 + 元信息），供 /live 面板动态刷新展示；
  2. 若 app.py 注入了 DA3 处理回调（set_processor），后台单线程用"最新优先"策略
     把最新帧跑一次深度推理，产物也缓存供面板展示。

刻意不 import torch / cv2 / depth_anything_3：DA3 能力通过回调注入，因此本模块在本地
无 GPU、无模型的环境下也能被 include_router 起来并跑通收图 + 展示链路，便于自测。
"""
import threading
import time
from typing import Callable, Optional

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response

router = APIRouter()

# ── 共享状态：一把 Condition 同时兜住状态互斥与"新帧到达"的唤醒 ──────────────
_cv = threading.Condition()

_seq = 0                              # 已接收帧的单调递增序号
_image: Optional[bytes] = None        # 最新一帧原始字节
_content_type = "image/jpeg"          # 最新一帧 MIME
_received_at = 0.0                    # 最新一帧到达的服务器时间戳
_prev_received_at = 0.0              # 上一帧到达时间（用于估算到达间隔/帧率）
_camera_info: Optional[str] = None    # 随帧上报的 camera_info（原样透传的 JSON 字符串）
_timestamp: Optional[str] = None      # 随帧上报的设备时间戳字段

_depth: Optional[bytes] = None        # 最新一帧的 DA3 处理产物（如彩色深度图字节）
_depth_content_type = "image/jpeg"    # 处理产物 MIME
_depth_seq = 0                        # 处理产物对应的帧序号
_depth_error: Optional[str] = None    # 最近一次处理的错误（None 表示成功）

# DA3 处理回调：输入原始帧字节，返回处理后图片字节；由 app.py 在有模型时注入
_processor: Optional[Callable[[bytes], bytes]] = None
_worker_started = False


def set_processor(fn: Callable[[bytes], bytes], content_type: str = "image/jpeg") -> None:
    """注入 DA3 处理回调并按需启动后台处理线程。

    app.py 里在模型可用时调用；本地自测不调用即为纯中继（只收图 + 展示原图）。
    """
    global _processor, _depth_content_type
    with _cv:
        _processor = fn
        _depth_content_type = content_type
        _start_worker_locked()


def _start_worker_locked() -> None:
    """在持有 _cv 的前提下懒启动后台处理线程（只启动一次）。"""
    global _worker_started
    if _worker_started or _processor is None:
        return
    _worker_started = True
    threading.Thread(target=_worker_loop, daemon=True, name="frame-da3-worker").start()


def _worker_loop() -> None:
    """后台处理线程：始终只处理"最新一帧"，中间帧直接丢弃。

    这样在帧率高于 GPU 处理能力时不会积压，实时展示只落后一帧，符合演示诉求。
    """
    last = 0
    while True:
        with _cv:
            # 等到有新帧且回调仍在
            while _processor is None or _seq == last:
                _cv.wait()
            seq, img, proc = _seq, _image, _processor
        # 重活（GPU 推理）放在锁外执行，避免阻塞收图与展示
        try:
            out = proc(img)
            err = None
        except Exception as e:  # 处理失败不影响原图展示，仅记录错误
            out, err = None, f"{type(e).__name__}: {e}"
        with _cv:
            global _depth, _depth_seq, _depth_error
            # 只在不落后于已发布产物时更新（并发下保序）
            if seq >= _depth_seq:
                if out is not None:
                    _depth = out
                _depth_seq = seq
                _depth_error = err
            _cv.notify_all()
        last = seq


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


@router.get("/api/frame/latest")
def latest_frame():
    """返回最新一帧原始图片字节（供 /live 面板 <img> 直接展示）。"""
    with _cv:
        img, ct, seq = _image, _content_type, _seq
    if img is None:
        return JSONResponse({"error": "暂无帧"}, status_code=404)
    return Response(
        content=img,
        media_type=ct,
        headers={"Cache-Control": "no-store", "X-Frame-Seq": str(seq)},
    )


@router.get("/api/frame/latest-depth")
def latest_depth():
    """返回最新一帧的 DA3 处理产物（彩色深度图）字节；无产物时 404。"""
    with _cv:
        img, ct, seq = _depth, _depth_content_type, _depth_seq
    if img is None:
        return JSONResponse({"error": "暂无处理产物"}, status_code=404)
    return Response(
        content=img,
        media_type=ct,
        headers={"Cache-Control": "no-store", "X-Frame-Seq": str(seq)},
    )


@router.get("/api/frame/status")
def frame_status():
    """返回最新帧的元信息与到达速率，供面板轮询判断是否需要刷新。"""
    with _cv:
        now = time.time()
        interval = (_received_at - _prev_received_at) if _prev_received_at else 0.0
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
            "has_depth": _depth is not None,
            "depth_seq": _depth_seq,
            "depth_error": _depth_error,
            "processor": _processor is not None,
        })


# ── 实时展示面板 /live：轮询 status，seq 变化即换图（与 /weight 的轮询同构）────
_LIVE_PAGE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>设备实时帧 · Live</title>
<style>
 body{margin:0;background:#0f1113;color:#e6e6e6;font:14px/1.5 -apple-system,system-ui,sans-serif}
 header{padding:12px 16px;background:#17191c;border-bottom:1px solid #26292d;display:flex;
        align-items:center;gap:16px;flex-wrap:wrap}
 header h1{font-size:16px;margin:0;font-weight:600}
 .meta{color:#9aa0a6;font-size:13px}
 .meta b{color:#e6e6e6;font-weight:600}
 .wrap{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:16px}
 @media(max-width:820px){.wrap{grid-template-columns:1fr}}
 figure{margin:0;background:#000;border-radius:10px;overflow:hidden;border:1px solid #26292d}
 figcaption{padding:8px 12px;background:#17191c;color:#c5c9cd;font-size:13px;border-bottom:1px solid #26292d}
 img{width:100%;display:block;background:#000;min-height:120px}
 .off{color:#f0a020}
</style></head><body>
<header>
 <h1>设备实时帧</h1>
 <span class="meta">帧号 <b id="seq">—</b></span>
 <span class="meta">到达间隔 <b id="fps">—</b></span>
 <span class="meta">距上帧 <b id="age">—</b></span>
 <span class="meta" id="dstat">深度：—</span>
</header>
<div class="wrap">
 <figure><figcaption>接收到的原始帧</figcaption><img id="raw" alt="等待帧…"></figure>
 <figure><figcaption>DA3 深度图（若已接入模型）</figcaption><img id="depth" alt="等待处理…"></figure>
</div>
<script>
 let lastSeq=-1, lastDepthSeq=-1;
 const $=id=>document.getElementById(id);
 async function tick(){
  try{
   const r=await fetch('/api/frame/status',{cache:'no-store'});
   const s=await r.json();
   $('seq').textContent = s.seq || '—';
   $('fps').textContent = s.interval ? (s.interval.toFixed(2)+'s / '+ (s.fps?s.fps.toFixed(1):'—')+'fps') : '—';
   $('age').textContent = (s.age!=null) ? (s.age.toFixed(1)+'s') : '—';
   if(s.has_frame && s.seq!==lastSeq){ lastSeq=s.seq; $('raw').src='/api/frame/latest?t='+s.seq; }
   if(s.has_depth && s.depth_seq!==lastDepthSeq){ lastDepthSeq=s.depth_seq; $('depth').src='/api/frame/latest-depth?t='+s.depth_seq; }
   const d=$('dstat');
   if(!s.processor){ d.innerHTML='深度：<span class="off">未接入模型（纯中继）</span>'; }
   else if(s.depth_error){ d.innerHTML='深度：<span class="off">'+s.depth_error+'</span>'; }
   else { d.textContent='深度：帧号 '+(s.depth_seq||'—'); }
  }catch(e){/* 忽略单次轮询失败，下个周期重试 */}
 }
 setInterval(tick, 300); tick();
</script>
</body></html>"""


@router.get("/live", response_class=HTMLResponse)
def live_page():
    """设备实时帧展示面板：每 300ms 轮询，帧号变化即动态换图。"""
    return _LIVE_PAGE
