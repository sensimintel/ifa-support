# -*- coding: utf-8 -*-
# SAM3 推理 HTTP 服务（FastAPI）。图像概念分割 + 视频短 tracking。
# 依据实测 API：build_sam3_predictor -> Sam3VideoPredictorMultiGPU，session 式调用，use_fa3=False 走 torch SDPA。
# 推理统一包在 torch.autocast(bf16)+inference_mode，规避 bf16/fp32 conv bias 不一致。
import io, os, base64, shutil, tempfile, threading, logging, contextlib
import numpy as np, torch
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from pycocotools import mask as mask_utils
import sam3

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sam3-server")

CKPT = os.environ.get("SAM3_CKPT", "/home/yu_ji/models/sam3/sam3.pt")
VERSION = os.environ.get("SAM3_VERSION", "sam3")
HOST = os.environ.get("SAM3_HOST", "127.0.0.1")
PORT = int(os.environ.get("SAM3_PORT", "8001"))

_LOCK = threading.Lock()
_pred = None
_err = None

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

app = FastAPI(title="SAM3 Inference Server", version="1.0.0")

@app.on_event("startup")
def _startup():
    _load()

@app.get("/health")
def health():
    return {"status": "ok" if _pred is not None else "error",
            "cuda": torch.cuda.is_available(), "ckpt": CKPT, "version": VERSION, "load_error": _err}

@app.get("/v1/models")
def models():
    return {"data": [{"id": "sam3", "ckpt": CKPT, "version": VERSION, "capabilities": ["segment", "track"]}]}

class SegReq(BaseModel):
    image_b64: str
    text: str

class TrackReq(BaseModel):
    frames_b64: list
    text: str
    prompt_frame_index: int = 0

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

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, workers=1)
