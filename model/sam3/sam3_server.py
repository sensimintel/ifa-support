# -*- coding: utf-8 -*-
# SAM3 推理 HTTP 服务（FastAPI）。图像概念分割 + 视频短 tracking + 流式长记忆 session。
# 依据实测 API：build_sam3_predictor -> Sam3VideoPredictorMultiGPU，session 式调用，use_fa3=False 走 torch SDPA。
# 推理统一包在 torch.autocast(bf16)+inference_mode，规避 bf16/fp32 conv bias 不一致。
#
# 流式长记忆（/v1/stream/*）——真·增量实现（v3）：
#   模型本身天然增量：propagate 算第 t 帧时只对历史帧的 memory bank（tracker_inference_states 里的
#   SAM2 系 spatial memory + object pointer）做 cross-attention，不回头重算旧帧。缺的只是库层
#   「往已建 session 追加一帧」的容器 API（inference_state 的 input_batch/per_frame_* 列表在
#   init_state 时按 num_frames 定长分配）。本服务在 predictor 之上补上这一层：
#     · 每个流式 session 持有一个"活着的" sam3 session（memory bank 跨 HTTP 请求不销毁）；
#     · 每步 _append_frame_to_state（拼 img_batch + 各逐帧列表追加空位）→ 只 propagate 新帧
#       （forward、max_frame_num_to_track=1）→ 单帧增量耗时（vs 全窗口重放的 ∝window）；
#     · 显存封顶三件套：逐步修剪 feature_cache/cached_frame_outputs 的旧帧条目（backbone 特征是
#       大头）、旧帧 previous_stages_out 置 None、每 SAM3_STREAM_REBUILD_EVERY 帧整体重建一次
#       session（重建种子=最近 window 帧，公共 obj_id 靠 mask IoU 注册表跨代缝合，外部无感）；
#     · 内部 id 在代内由 tracker 原生稳定；跨代/新对象经注册表映射成跨请求稳定的公共 obj_id，
#       离场超 forget_frames 帧才遗忘。
#   风险面：_append_frame_to_state 耦合 sam3 内部字段名（Sam3VideoInference 的 state 结构），
#   上游升级若变动，运行时会抛错——自动回退到 v2 的"滚动窗口全量重放"路径（replay），功能不断。
import io, os, base64, shutil, tempfile, threading, time, uuid, logging, contextlib
import numpy as np, torch
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from pycocotools import mask as mask_utils
import sam3

# 增量路径依赖的 sam3 内部结构（上游若重构导致 import 失败，整体回退 replay，不影响服务可用性）
try:
    from sam3.model.data_misc import convert_my_tensors, FindStage
    from sam3.model.utils.misc import copy_data_to_device
    _INCR_IMPORTS_OK = True
except Exception:  # noqa: BLE001
    _INCR_IMPORTS_OK = False

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
# 增量路径开关与整代重建周期（重建=显存兜底：img_batch 每帧 ~6MB 线性涨，到期整体重来一代）
STREAM_INCREMENTAL = os.environ.get("SAM3_STREAM_INCREMENTAL", "1") not in ("0", "false", "False")
STREAM_REBUILD_EVERY = int(os.environ.get("SAM3_STREAM_REBUILD_EVERY", "60"))

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

def _run_window(frames_pil, text, keep_session=False):
    """对一段 PIL 帧序列跑一次 add_prompt(帧0)+propagate，返回 (per_frame, W, H, session_id)。
    per_frame: {帧下标: [instance...]}（instance 的 obj_id 是本次运行的内部 id）。
    keep_session=True 时不销毁 sam3 session（增量路径的"开代"用），由调用方负责关闭。
    resource_path 直接传 PIL 列表（io_utils 原生支持），不落盘。"""
    W, H = frames_pil[0].size
    per_frame = {}
    with _LOCK:
        sid = _pred.start_session(resource_path=list(frames_pil))["session_id"]
        try:
            with _infer_ctx():
                r = _pred.add_prompt(session_id=sid, frame_idx=0, text=text)
                per_frame[0] = _pack(r["outputs"], W, H)
                if len(frames_pil) > 1:
                    for res in _pred.propagate_in_video(session_id=sid,
                                                        propagation_direction="forward"):
                        per_frame[int(res["frame_index"])] = _pack(res["outputs"], W, H)
        except Exception:
            _pred.close_session(sid)
            raise
        if not keep_session:
            _pred.close_session(sid)
            sid = None
    return per_frame, W, H, sid


def _append_frame_to_state(state, pil_img):
    """【增量核心·耦合上游内部结构】往活着的 inference_state 追加一帧，返回新帧下标。

    复刻 Sam3VideoInference 两处逻辑：
      · load_resource_as_video_frames 的 PIL 分支预处理（resize→/255→CHW→fp16→mean/std 归一化）；
      · _construct_initial_input_batch 的单帧容器（img_batch 拼接 + FindStage + 6 个逐帧列表追加）。
    上游若改字段名，这里会抛 AttributeError/KeyError → 调用方回退 replay。"""
    model = _pred.model
    size = int(model.image_size)
    img_np = np.array(pil_img.convert("RGB").resize((size, size)))
    img = torch.from_numpy(img_np / 255.0).permute(2, 0, 1).to(torch.float16)
    mean = torch.tensor(model.image_mean, dtype=torch.float16)[:, None, None]
    std = torch.tensor(model.image_std, dtype=torch.float16)[:, None, None]
    img = (img - mean) / std
    ib = state["input_batch"]
    dev = ib.img_batch.device
    ib.img_batch = torch.cat([ib.img_batch, img[None].to(dev)], dim=0)
    t = int(state["num_frames"])
    stage = FindStage(
        img_ids=[t], text_ids=[0],
        input_boxes=[torch.zeros(258)],
        input_boxes_mask=[torch.empty(0, dtype=torch.bool)],
        input_boxes_label=[torch.empty(0, dtype=torch.long)],
        input_points=[torch.empty(0, 257)],
        input_points_mask=[torch.empty(0)],
        object_ids=[],
    )
    stage = copy_data_to_device(convert_my_tensors(stage), dev, non_blocking=True)
    ib.find_inputs.append(stage)
    ib.find_targets.append(None)
    ib.find_metadatas.append(None)
    state["previous_stages_out"].append(None)
    state["per_frame_raw_point_input"].append(None)
    state["per_frame_raw_box_input"].append(None)
    state["per_frame_visual_prompt"].append(None)
    state["per_frame_geometric_prompt"].append(None)
    state["per_frame_cur_step"].append(0)
    state["num_frames"] = t + 1
    # SAM2 层子 state（每个对象桶一个）也各自持有定长 num_frames——不同步扩，tracker 对新帧号
    # 的 propagate 会得到空处理序列（实测 out_frame_idx 未赋值报错）。其余字段全是按帧号的字典，
    # 无需扩容；之后新建的子 state 用外层 num_frames，天然是新值。
    for ts_sub in state.get("tracker_inference_states", []):
        if isinstance(ts_sub, dict) and "num_frames" in ts_sub:
            ts_sub["num_frames"] = t + 1
    return t


def _prune_state_caches(state, keep_from):
    """修剪 state 里旧帧的重资产，防止代内显存线性膨胀：
      · feature_cache[帧号]（backbone 特征，大头）与 cached_frame_outputs[帧号]：直接删；
      · previous_stages_out[旧帧]：置 None（保列表位置，帧号索引不乱）。
    只动整型帧号键；"text"/"grounding_cache" 等特殊键保留。tracker 的 memory bank 不动。"""
    for name in ("feature_cache", "cached_frame_outputs"):
        c = state.get(name)
        if isinstance(c, dict):
            for k in [k for k in c if isinstance(k, int) and k < keep_from]:
                del c[k]
    outs = state.get("previous_stages_out")
    if isinstance(outs, list):
        for k in range(min(keep_from, len(outs))):
            outs[k] = None


def _close_live(s):
    """关闭 session 里活着的 sam3 会话（容错：predictor 侧不存在也不报错）。"""
    sid = s.pop("live_sid", None)
    if sid:
        try:
            with _LOCK:
                _pred.close_session(sid)
        except Exception:
            pass

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
    """回收空闲超时的流式 session（daemon 线程，60s 一轮），连带关闭其活着的 sam3 会话。"""
    while True:
        time.sleep(60)
        now = time.time()
        with _streams_lock:
            dead = [(sid, s) for sid, s in _streams.items() if now - s["last_ts"] > STREAM_TTL_SEC]
            for sid, _s in dead:
                del _streams[sid]
        for sid, s in dead:
            _close_live(s)
        if dead:
            logger.info("回收空闲流式 session：%s", [sid for sid, _ in dead])

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
    evicted = None
    with _streams_lock:
        if len(_streams) >= STREAM_MAX_SESSIONS:
            # 满了先回收最久未用的，保证新 session 能建（demo 场景可接受）
            oldest = min(_streams, key=lambda k: _streams[k]["last_ts"])
            evicted = _streams.pop(oldest)
            logger.info("流式 session 数达上限，回收最久未用：%s", oldest)
        _streams[sid] = {
            "text": req.text, "window": window, "forget_frames": forget,
            "ring": [],              # 最近 window 帧（PIL，CPU 内存；开代种子/replay 回退用）
            "next_global": 0,        # 全局帧计数（session 生命周期内单调递增）
            "registry": {},          # 公共对象注册表：pub_id -> {rle,last_global,score}
            "next_pub": 1,
            # 增量代（generation）状态：live_sid=活着的 sam3 会话；id_map=本代内部id→公共id
            "live_sid": None, "gen_frames": 0, "gen_base_global": 0, "id_map": {},
            "impl": "incremental" if (STREAM_INCREMENTAL and _INCR_IMPORTS_OK) else "replay",
            "last_ts": time.time(),
            "step_lock": threading.Lock(),   # 同一 session 的步进串行
        }
    if evicted is not None:
        _close_live(evicted)
    return {"session_id": sid, "text": req.text, "window": window, "forget_frames": forget,
            "impl": _streams[sid]["impl"]}

def _step_replay(s):
    """v2 路径：滚动窗口全量重放 + 跨窗口缝合（增量不可用时的兜底）。返回最新帧实例列表。"""
    newest_idx = len(s["ring"]) - 1
    base_global = s["next_global"] - newest_idx          # 窗口第 0 帧的全局帧号
    per_frame, _W, _H, _sid = _run_window(s["ring"], s["text"])
    id_map = _stitch_ids(s, per_frame, newest_idx, base_global)
    inst = []
    for it in per_frame.get(newest_idx, []):
        out = dict(it)
        out["obj_id"] = id_map.get(it["obj_id"], it["obj_id"])
        inst.append(out)
    return inst


def _step_incremental(s, img, g):
    """v3 路径：活 session 逐帧增量。返回最新帧实例列表（公共 obj_id）。

    · 无活代 → 用 ring（≤window 帧）开一代：整窗跑一遍 + 注册表跨代缝合出 id_map；
    · 有活代 → append 新帧 → 只 propagate 该帧 → 代内内部 id 稳定，经 id_map 换公共 id，
      新内部 id 先试与注册表近期对象 IoU 缝合（同物短暂消失后 tracker 发新 id 的情形），配不上发新 id；
    · 每步修剪旧帧重资产；代长到 STREAM_REBUILD_EVERY 关代（下步用 ring 重开，显存兜底）。"""
    if s["live_sid"] is None:
        newest_idx = len(s["ring"]) - 1
        per_frame, _W, _H, live = _run_window(s["ring"], s["text"], keep_session=True)
        s["live_sid"] = live
        s["gen_frames"] = len(s["ring"])
        s["gen_base_global"] = g - newest_idx
        s["id_map"] = _stitch_ids(s, per_frame, newest_idx, s["gen_base_global"])
        raw = per_frame.get(newest_idx, [])
    else:
        with _LOCK:
            state = _pred._get_session(s["live_sid"])["state"]
            t_new = _append_frame_to_state(state, img)
            # 清空 action_history：上游按"交互 demo"语义解析它，第二次 propagate 起会判成
            # propagation_fetch（只取缓存、不跑模型）→ 新帧无缓存输出恒空。清空则每步都走
            # propagation_full（真检测+跟踪），处理范围仍被 start/max 钳在新帧这一帧。
            state["action_history"].clear()
            outs = None
            with _infer_ctx():
                for res in _pred.propagate_in_video(
                        session_id=s["live_sid"], propagation_direction="forward",
                        start_frame_idx=t_new, max_frame_num_to_track=1):
                    if int(res["frame_index"]) == t_new:
                        outs = res["outputs"]
            W, H = img.size
            raw = _pack(outs, W, H) if outs is not None else []
            _prune_state_caches(state, keep_from=t_new - s["window"])
        s["gen_frames"] += 1
        # 内部 id → 公共 id：代内已见的直接查表；新内部 id 先试与注册表近期对象缝合，配不上发新 id
        id_map, registry = s["id_map"], s["registry"]
        for it in raw:
            iid = it["obj_id"]
            if iid not in id_map:
                best_iou, best_pub = 0.0, None
                used = set(id_map.values())
                for pub_id, ent in registry.items():
                    if pub_id in used or g - ent["last_global"] > s["forget_frames"]:
                        continue
                    iou = _rle_iou(it["mask_rle"], ent["rle"])
                    if iou >= STREAM_MATCH_IOU and iou > best_iou:
                        best_iou, best_pub = iou, pub_id
                if best_pub is None:
                    best_pub = s["next_pub"]
                    s["next_pub"] += 1
                id_map[iid] = best_pub
        # 刷新注册表（最新帧现身的对象）+ 遗忘离场过久的
        for it in raw:
            registry[id_map[it["obj_id"]]] = {"rle": it["mask_rle"], "last_global": g,
                                              "score": it["score"]}
        for pub_id in [p for p, e in registry.items()
                       if g - e["last_global"] > s["forget_frames"]]:
            del registry[pub_id]
    # 代长兜底：到期关代，下一步用 ring 种子重开（注册表缝合保证公共 id 连续）
    if s["gen_frames"] >= STREAM_REBUILD_EVERY:
        _close_live(s)
        s["live_sid"] = None
    inst = []
    for it in raw:
        out = dict(it)
        out["obj_id"] = s["id_map"].get(it["obj_id"], it["obj_id"])
        inst.append(out)
    return inst


@app.post("/v1/stream/frame")
def stream_frame(req: StreamFrameReq):
    """流式步进：追加 1 帧 → 增量 propagate（或回退全窗重放）→ 返回最新帧实例（obj_id 跨请求稳定）。"""
    if _pred is None:
        raise HTTPException(503, _err or "model not loaded")
    with _streams_lock:
        s = _streams.get(req.session_id)
    if s is None:
        raise HTTPException(404, "session 不存在或已被回收，请重新 /v1/stream/start")
    img = _decode(req.image_b64)
    with s["step_lock"]:
        s["last_ts"] = time.time()
        s["ring"].append(img)
        if len(s["ring"]) > s["window"]:
            del s["ring"][:len(s["ring"]) - s["window"]]
        g = s["next_global"]
        t0 = time.time()
        if s["impl"] == "incremental":
            try:
                inst = _step_incremental(s, img, g)
            except Exception as e:
                # 上游内部结构不符/运行时异常：本 session 永久降级 replay，服务不断
                logger.exception("增量路径失败，session %s 降级 replay：%s", req.session_id, e)
                _close_live(s)
                s["live_sid"] = None
                s["impl"] = "replay"
                inst = _step_replay(s)
        else:
            inst = _step_replay(s)
        run_ms = (time.time() - t0) * 1000.0
        s["next_global"] = g + 1
        n_reg = len(s["registry"])
        impl, gen_frames = s["impl"], s["gen_frames"]
    W, H = img.size
    gpu_mb = int(torch.cuda.memory_allocated() // (1024 * 1024)) if torch.cuda.is_available() else 0
    return {"session_id": req.session_id, "global_index": g,
            "width": W, "height": H, "window_frames": min(g + 1, s["window"]),
            "num_instances": len(inst), "instances": inst,
            "active_objects": n_reg, "run_ms": round(run_ms, 1),
            "impl": impl, "gen_frames": gen_frames, "gpu_mb": gpu_mb}

@app.get("/v1/stream")
def stream_list():
    """列出存活的流式 session（排障/看板用）。"""
    now = time.time()
    with _streams_lock:
        return {"sessions": [
            {"session_id": sid, "text": s["text"], "window": s["window"],
             "frames_seen": s["next_global"], "active_objects": len(s["registry"]),
             "impl": s["impl"], "gen_frames": s["gen_frames"],
             "idle_sec": round(now - s["last_ts"], 1)} for sid, s in _streams.items()]}

@app.delete("/v1/stream/{session_id}")
def stream_close(session_id: str):
    with _streams_lock:
        s = _streams.pop(session_id, None)
    if s is not None:
        _close_live(s)
    return {"closed": s is not None}

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, workers=1)
