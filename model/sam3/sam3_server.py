# -*- coding: utf-8 -*-
# SAM3 推理 HTTP 服务（FastAPI）。图像概念分割 + 视频短 tracking + 流式长记忆 session。
# 依据实测 API：build_sam3_predictor -> Sam3VideoPredictorMultiGPU，session 式调用，use_fa3=False 走 torch SDPA。
# 推理统一包在 torch.autocast(bf16)+inference_mode，规避 bf16/fp32 conv bias 不一致。
#
# 流式长记忆（/v1/stream/*）的设计取舍：
#   sam3 包的 inference_state 是「帧数在 init_state 定死」的批式结构（input_batch/per_frame_* 列表
#   全按 num_frames 预分配），公开 API 没有逐帧 append——真·增量 memory 需 fork 上游内部结构，
#   跟随升级成本高。故用纯公开 API 实现「服务端滚动窗口 + 跨窗口身份缝合」：
#     · server 常驻保存每个 session 最近 window 帧（JPEG，CPU 内存），客户端每步只传 1 帧；
#     · 每步对窗口整体重跑 add_prompt+propagate（GPU 瞬时占用与耗时 ∝ window，这就是
#       「用长度控制显存/算力」的旋钮）；
#     · 用「公共 obj_id 注册表 + 窗口内同帧 mask IoU 贪心匹配」把每步会重排的内部 id 缝合成
#       跨请求稳定的公共 id——对象持续在场就一直是同一个 id，离场超 forget_frames 帧才遗忘。
import io, os, base64, shutil, tempfile, threading, time, uuid, logging, contextlib
import numpy as np, torch
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from pycocotools import mask as mask_utils
import sam3

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sam3-server")

CKPT = os.environ.get("SAM3_CKPT", "/home/odyss/models/sam3/sam3.pt")
VERSION = os.environ.get("SAM3_VERSION", "sam3")
HOST = os.environ.get("SAM3_HOST", "127.0.0.1")
PORT = int(os.environ.get("SAM3_PORT", "8013"))

# 流式 session 参数（可被 /v1/stream/start 的入参覆盖，并被下列上限钳住）
STREAM_WINDOW_DEFAULT = int(os.environ.get("SAM3_STREAM_WINDOW", "5"))
STREAM_WINDOW_MAX = int(os.environ.get("SAM3_STREAM_WINDOW_MAX", "16"))
STREAM_FORGET_DEFAULT = int(os.environ.get("SAM3_STREAM_FORGET", "30"))   # 离场多少帧后遗忘该对象
STREAM_TTL_SEC = float(os.environ.get("SAM3_STREAM_TTL", "300"))          # 空闲多久回收 session
STREAM_MAX_SESSIONS = int(os.environ.get("SAM3_STREAM_MAX_SESSIONS", "8"))
STREAM_MATCH_IOU = float(os.environ.get("SAM3_STREAM_MATCH_IOU", "0.4"))  # 身份缝合的 IoU 门槛

_LOCK = threading.Lock()
_pred = None
_err = None

_streams = {}                     # session_id -> 流式 session 状态字典
_streams_lock = threading.Lock()  # 保护 _streams 的增删查（推理仍由 _LOCK 串行）

@contextlib.contextmanager
def _infer_ctx():
    # 统一推理上下文：bf16 自动混合精度 + 关闭梯度
    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            yield

def _load():
    global _pred, _err
    try:
        _mf = float(os.environ.get("SAM3_MEM_FRACTION", "0"))
        if _mf > 0 and torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(_mf, 0)  # 固定显存上限（9G≈0.28，与 5090 其他服务隔离）
            logger.info("SAM3 显存上限 fraction=%.3f（本进程最多用这么多）", _mf)
        logger.info("正在加载 SAM3 predictor: %s (version=%s, use_fa3=False)", CKPT, VERSION)
        _pred = sam3.build_sam3_predictor(checkpoint_path=CKPT, version=VERSION, use_fa3=False)
        logger.info("SAM3 加载完成")
    except Exception as e:
        _err = repr(e)
        logger.exception("SAM3 加载失败: %s", e)

def _rle(mask_bool):
    m = np.asfortranarray(np.asarray(mask_bool).astype(np.uint8))
    r = mask_utils.encode(m)
    r["counts"] = r["counts"].decode("utf-8")
    return {"size": r["size"], "counts": r["counts"]}

def _rle_iou(r1, r2):
    """两个（counts 为 str 的）RLE 的 mask IoU。"""
    a = {"size": r1["size"], "counts": r1["counts"].encode()}
    b = {"size": r2["size"], "counts": r2["counts"].encode()}
    return float(mask_utils.iou([a], [b], [0])[0][0])

def _pack(outputs, W=None, H=None):
    obj_ids = np.asarray(outputs["out_obj_ids"]).tolist()
    probs = np.asarray(outputs["out_probs"]).tolist()
    boxes = np.asarray(outputs["out_boxes_xywh"])
    masks = np.asarray(outputs["out_binary_masks"])
    if masks.ndim == 4:
        masks = masks[:, 0]
    inst = []
    for i in range(len(obj_ids)):
        b = [float(v) for v in boxes[i]]
        item = {"obj_id": int(obj_ids[i]), "score": float(probs[i]),
                "box_xywh_norm": b, "mask_rle": _rle(masks[i].astype(bool))}
        if W is not None and H is not None:
            item["box_xywh_px"] = [b[0]*W, b[1]*H, b[2]*W, b[3]*H]
        inst.append(item)
    return inst

def _decode(b64):
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

def _run_window(frames_jpeg, text):
    """对一段 JPEG 帧序列跑一次 add_prompt(帧0)+propagate，返回 (per_frame, W, H)。
    per_frame: {帧下标: [instance...]}（instance 的 obj_id 是本次运行的内部 id，跨次会重排）。"""
    d = tempfile.mkdtemp(prefix="sam3_stream_")
    try:
        W = H = None
        for i, jb in enumerate(frames_jpeg):
            im = Image.open(io.BytesIO(jb)).convert("RGB")
            if W is None:
                W, H = im.size
            im.save(os.path.join(d, "%05d.jpg" % i), quality=95)
        per_frame = {}
        with _LOCK:
            sid = _pred.start_session(resource_path=d)["session_id"]
            try:
                with _infer_ctx():
                    r = _pred.add_prompt(session_id=sid, frame_idx=0, text=text)
                    per_frame[0] = _pack(r["outputs"], W, H)
                    if len(frames_jpeg) > 1:
                        for res in _pred.propagate_in_video(session_id=sid):
                            per_frame[int(res["frame_index"])] = _pack(res["outputs"], W, H)
            finally:
                _pred.close_session(sid)
        return per_frame, W, H
    finally:
        shutil.rmtree(d, ignore_errors=True)

def _stitch_ids(s, per_frame, newest_idx, base_global):
    """身份缝合：把本次运行的内部 obj_id 映射成跨请求稳定的公共 obj_id。

    注册表 s["registry"]: {pub_id: {"rle","last_global","score"}}——每个公共对象最后一次被看到
    时的 mask 与全局帧号。匹配：若某公共对象的 last_global 落在当前窗口内，则拿本次运行在
    「同一全局帧」上的各内部对象 mask 与它算 IoU，全对齐后按 IoU 降序贪心一一配对（门槛
    STREAM_MATCH_IOU）；配不上的内部对象注册为新公共 id。随后用最新帧刷新注册表并遗忘
    离场超 forget_frames 的对象。返回 {内部id: 公共id}。"""
    registry = s["registry"]
    newest_global = base_global + newest_idx
    # 内部 id → 各帧 mask（同一次运行内 id 稳定）
    inst_by_id = {}
    for fi, insts in per_frame.items():
        for it in insts:
            inst_by_id.setdefault(it["obj_id"], {})[fi] = it
    # 候选配对：(iou, 内部id, 公共id)
    pairs = []
    for pub_id, ent in registry.items():
        fi = ent["last_global"] - base_global          # 该公共对象最后现身帧在当前窗口的下标
        if fi < 0 or fi > newest_idx:
            continue
        for iid, by_frame in inst_by_id.items():
            it = by_frame.get(fi)
            if it is None:
                continue
            iou = _rle_iou(it["mask_rle"], ent["rle"])
            if iou >= STREAM_MATCH_IOU:
                pairs.append((iou, iid, pub_id))
    pairs.sort(reverse=True)
    id_map, used_pub = {}, set()
    for iou, iid, pub_id in pairs:
        if iid in id_map or pub_id in used_pub:
            continue
        id_map[iid] = pub_id
        used_pub.add(pub_id)
    for iid in inst_by_id:
        if iid not in id_map:                          # 全新对象：发新公共 id
            id_map[iid] = s["next_pub"]
            s["next_pub"] += 1
    # 用最新帧刷新注册表；离场超 forget_frames 的对象遗忘
    for iid, by_frame in inst_by_id.items():
        it = by_frame.get(newest_idx)
        if it is not None:
            registry[id_map[iid]] = {"rle": it["mask_rle"], "last_global": newest_global,
                                     "score": it["score"]}
    for pub_id in [p for p, e in registry.items()
                   if newest_global - e["last_global"] > s["forget_frames"]]:
        del registry[pub_id]
    return id_map

def _sweep_streams():
    """回收空闲超时的流式 session（daemon 线程，60s 一轮）。"""
    while True:
        time.sleep(60)
        now = time.time()
        with _streams_lock:
            dead = [sid for sid, s in _streams.items() if now - s["last_ts"] > STREAM_TTL_SEC]
            for sid in dead:
                del _streams[sid]
        if dead:
            logger.info("回收空闲流式 session：%s", dead)

threading.Thread(target=_sweep_streams, daemon=True).start()

app = FastAPI(title="SAM3 Inference Server", version="2.0.0")

@app.on_event("startup")
def _startup():
    _load()

@app.get("/health")
def health():
    with _streams_lock:
        n_stream = len(_streams)
    return {"status": "ok" if _pred is not None else "error",
            "cuda": torch.cuda.is_available(), "ckpt": CKPT, "version": VERSION,
            "load_error": _err, "stream_sessions": n_stream}

@app.get("/v1/models")
def models():
    return {"data": [{"id": "sam3", "ckpt": CKPT, "version": VERSION,
                      "capabilities": ["segment", "track", "stream"]}]}

class SegReq(BaseModel):
    image_b64: str
    text: str

class TrackReq(BaseModel):
    frames_b64: list
    text: str
    prompt_frame_index: int = 0

class StreamStartReq(BaseModel):
    text: str
    window: int = 0            # 0=用服务端默认；上限 STREAM_WINDOW_MAX
    forget_frames: int = 0     # 0=用服务端默认

class StreamFrameReq(BaseModel):
    session_id: str
    image_b64: str

@app.post("/v1/segment")
def segment(req: SegReq):
    if _pred is None:
        raise HTTPException(503, _err or "model not loaded")
    img = _decode(req.image_b64)
    W, H = img.size
    d = tempfile.mkdtemp(prefix="sam3_seg_")
    try:
        img.save(os.path.join(d, "00000.jpg"), quality=95)
        with _LOCK:
            sid = _pred.start_session(resource_path=d)["session_id"]
            try:
                with _infer_ctx():
                    r = _pred.add_prompt(session_id=sid, frame_idx=0, text=req.text)
                inst = _pack(r["outputs"], W, H)
            finally:
                _pred.close_session(sid)
        return {"width": W, "height": H, "num_instances": len(inst), "instances": inst}
    finally:
        shutil.rmtree(d, ignore_errors=True)

@app.post("/v1/track")
def track(req: TrackReq):
    if _pred is None:
        raise HTTPException(503, _err or "model not loaded")
    if not req.frames_b64:
        raise HTTPException(400, "frames_b64 is empty")
    d = tempfile.mkdtemp(prefix="sam3_track_")
    try:
        W = H = None
        for i, b in enumerate(req.frames_b64):
            im = _decode(b)
            if W is None:
                W, H = im.size
            im.save(os.path.join(d, "%05d.jpg" % i), quality=95)
        frames = {}
        with _LOCK:
            sid = _pred.start_session(resource_path=d)["session_id"]
            try:
                with _infer_ctx():
                    _pred.add_prompt(session_id=sid, frame_idx=req.prompt_frame_index, text=req.text)
                    for res in _pred.propagate_in_video(session_id=sid):
                        frames[int(res["frame_index"])] = _pack(res["outputs"], W, H)
            finally:
                _pred.close_session(sid)
        return {"num_frames": len(req.frames_b64), "frames": frames}
    finally:
        shutil.rmtree(d, ignore_errors=True)

# ── 流式长记忆 session ──────────────────────────────────────────────────
@app.post("/v1/stream/start")
def stream_start(req: StreamStartReq):
    """建常驻流式 session：之后每步只传 1 帧，窗口与身份注册表都养在服务端。"""
    if _pred is None:
        raise HTTPException(503, _err or "model not loaded")
    window = min(max(int(req.window) or STREAM_WINDOW_DEFAULT, 1), STREAM_WINDOW_MAX)
    forget = max(int(req.forget_frames) or STREAM_FORGET_DEFAULT, window)
    sid = uuid.uuid4().hex
    with _streams_lock:
        if len(_streams) >= STREAM_MAX_SESSIONS:
            # 满了先回收最久未用的，保证新 session 能建（demo 场景可接受）
            oldest = min(_streams, key=lambda k: _streams[k]["last_ts"])
            del _streams[oldest]
            logger.info("流式 session 数达上限，回收最久未用：%s", oldest)
        _streams[sid] = {
            "text": req.text, "window": window, "forget_frames": forget,
            "frames": [],            # 最近 window 帧的 JPEG 字节（CPU 内存）
            "next_global": 0,        # 全局帧计数（session 生命周期内单调递增）
            "registry": {},          # 公共对象注册表：pub_id -> {rle,last_global,score}
            "next_pub": 1,
            "last_ts": time.time(),
            "step_lock": threading.Lock(),   # 同一 session 的步进串行
        }
    return {"session_id": sid, "text": req.text, "window": window, "forget_frames": forget}

@app.post("/v1/stream/frame")
def stream_frame(req: StreamFrameReq):
    """流式步进：追加 1 帧 → 滚动窗口重跑 → 身份缝合 → 返回最新帧实例（obj_id 跨请求稳定）。"""
    if _pred is None:
        raise HTTPException(503, _err or "model not loaded")
    with _streams_lock:
        s = _streams.get(req.session_id)
    if s is None:
        raise HTTPException(404, "session 不存在或已被回收，请重新 /v1/stream/start")
    img = _decode(req.image_b64)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    with s["step_lock"]:
        s["last_ts"] = time.time()
        s["frames"].append(buf.getvalue())
        if len(s["frames"]) > s["window"]:
            del s["frames"][:len(s["frames"]) - s["window"]]
        newest_idx = len(s["frames"]) - 1
        base_global = s["next_global"] - newest_idx      # 窗口第 0 帧的全局帧号
        t0 = time.time()
        per_frame, W, H = _run_window(s["frames"], s["text"])
        run_ms = (time.time() - t0) * 1000.0
        id_map = _stitch_ids(s, per_frame, newest_idx, base_global)
        global_index = s["next_global"]
        s["next_global"] += 1
        # 最新帧实例：内部 id 换成公共 id 下发
        inst = []
        for it in per_frame.get(newest_idx, []):
            out = dict(it)
            out["obj_id"] = id_map.get(it["obj_id"], it["obj_id"])
            inst.append(out)
        n_reg = len(s["registry"])
    return {"session_id": req.session_id, "global_index": global_index,
            "width": W, "height": H, "window_frames": newest_idx + 1,
            "num_instances": len(inst), "instances": inst,
            "active_objects": n_reg, "run_ms": round(run_ms, 1)}

@app.get("/v1/stream")
def stream_list():
    """列出存活的流式 session（排障/看板用）。"""
    now = time.time()
    with _streams_lock:
        return {"sessions": [
            {"session_id": sid, "text": s["text"], "window": s["window"],
             "frames_seen": s["next_global"], "active_objects": len(s["registry"]),
             "idle_sec": round(now - s["last_ts"], 1)} for sid, s in _streams.items()]}

@app.delete("/v1/stream/{session_id}")
def stream_close(session_id: str):
    with _streams_lock:
        existed = _streams.pop(session_id, None) is not None
    return {"closed": existed}

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, workers=1)
