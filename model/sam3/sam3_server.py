# -*- coding: utf-8 -*-
# SAM3 推理 HTTP 服务（FastAPI）。图像概念分割 + 视频短 tracking + 流式长记忆 session。
# 依据实测 API：build_sam3_predictor -> Sam3VideoPredictorMultiGPU，session 式调用，use_fa3=False 走 torch SDPA。
# 推理统一包在 torch.autocast(bf16)+inference_mode，规避 bf16/fp32 conv bias 不一致。
#
# 并发模型（v4）：_LOCK 从互斥锁改为有界信号量（SAM3_MAX_CONCURRENCY，默认 2）——语义是
#   「限流」而非「互斥」，生产两个词（food/drink）各一路流式 session 可并行推理（同一 predictor
#   双线程并发已前置实验验证：零异常零死锁、输出与串行逐位一致；每线程各自进 _infer_ctx）。
#   各共享可变状态自持锁：_streams 增删查由 _streams_lock；同一 session 的步进由 per-session
#   step_lock（回收/驱逐/关闭路径在关活会话前也先取它，防关到推理中段的会话）；debug 捕获是
#   threading.local；backbone 特征缓存自带小锁；predictor 内部 session 注册表是 uuid 键 +
#   GIL 原子的 dict 存取（实核对过上游源码），无需加锁。SAM3_MAX_CONCURRENCY=1 即回退旧串行。
# backbone 特征缓存（SAM3_EMB_CACHE，默认开）：视觉编码与文本 prompt 无关，两个词对同一帧
#   背靠背步进时第二个词必命中，省一次 backbone 前向（LRU=2，详见 _install_backbone_cache）。
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
#   上游升级若变动，运行时会抛错——自动回退到 v2 的"滚动窗口全量重放"路径（replay），功能不断；
#   降级非终身（v5）：replay 每满 SAM3_STREAM_RECOVER_EVERY 步自动尝试重建增量 session 切回增量，
#   失败则继续 replay 重新计步（详见 _step_with_fallback 的降级/恢复状态机）。
# triton autotune 并发竞态（v5 根治）：上游 connected_components 的 _local_prop_kernel 用
#   @triton.autotune(restore_value=["labels_ptr"]) 装饰，Autotuner 单例的 restore_copies/nargs
#   是无锁共享实例态；MAX_CONCURRENCY=2 时两路 session 并发进同一 kernel（重启后 autotune
#   冷缓存的 benchmark 窗口内必撞），后完成方抛 KeyError('labels_ptr') / TypeError(nargs=None)，
#   过去被误判成「上游 state 结构不符」而永久降级。_install_triton_autotune_lock 给
#   Autotuner.run 包进程级互斥锁消除（锁只罩 Python 端调度，GPU kernel 异步下发不受影响）。
import io, os, base64, shutil, tempfile, threading, time, uuid, logging, contextlib
import hashlib, collections
import numpy as np, torch
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from pycocotools import mask as mask_utils
import sam3
from prometheus_fastapi_instrumentator import Instrumentator

# 增量路径依赖的 sam3 内部结构（上游若重构导致 import 失败，整体回退 replay，不影响服务可用性）
try:
    from sam3.model.data_misc import convert_my_tensors, FindStage
    from sam3.model.utils.misc import copy_data_to_device
    _INCR_IMPORTS_OK = True
except Exception:  # noqa: BLE001
    _INCR_IMPORTS_OK = False

# force=True：import sam3 时上游已给 root logger 挂过 handler，若不强制接管，本配置会
# 变成 no-op、root 停在 WARNING——本文件所有 INFO 日志（含「加载完成」「缓存统计」）会被吞
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    force=True)
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
# 降级恢复周期：从增量降级到 replay 的 session，每满这么多步自动尝试重建增量 session 恢复
STREAM_RECOVER_EVERY = int(os.environ.get("SAM3_STREAM_RECOVER_EVERY", "20"))

# 推理有界并发限流（默认 2=两路流式 session 并行；=1 回退旧的全串行行为）。
# 注意 _LOCK 语义是「限流」而非「互斥」：临界区内不得依赖它保护共享可变状态（见文件头注释）。
MAX_CONCURRENCY = int(os.environ.get("SAM3_MAX_CONCURRENCY", "2"))
_LOCK = threading.BoundedSemaphore(MAX_CONCURRENCY)
_pred = None
_err = None

# backbone 视觉特征内容寻址缓存："0" 时不安装钩子；容量固定 LRU=2（两个词的生产场景恰好覆盖）
EMB_CACHE_ON = os.environ.get("SAM3_EMB_CACHE", "1") not in ("0", "false", "False")
EMB_CACHE_SIZE = 2

_streams = {}                     # session_id -> 流式 session 状态字典
_streams_lock = threading.Lock()  # 保护 _streams 的增删查（推理由 _LOCK 限流、session 内步进由各自 step_lock 串行）

@contextlib.contextmanager
def _infer_ctx():
    # 统一推理上下文：bf16 自动混合精度 + 关闭梯度
    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            yield

# ── presence / top-K query 原始分调试捕获（供 /v1/segment 的 debug 返回）────────
# 发布版 SAM3（has_presence_token=True → supervise_joint_box_scores=True）的检测头输出里：
#   · presence_logit_dec：全局 presence token 的 logit（"概念是否存在于图中"，全图共享）；
#   · pred_logits：每个 object query 的「联合分」logit = inverse_sigmoid(query条件分 × presence分)，
#     clamp ±10（见 sam3.model.sam3_image._update_scores_and_boxes）——被 clamp 后除法反推会骗人；
#   · 真·条件原始分：在 dot_prod_scoring 头（DotProductScoring，clamp ±12）上挂 forward hook，
#     直取未经 presence 加权的 pre-sigmoid logit（负样本是 -8/-12 这种明确负数，一眼可辨），
#     其 query 维顺序与 pred_logits 逐元素对齐（联合变换是逐元素的，不重排）。
# 库层 API 只回吐过完阈值/NMS 的最终分，这里在 detector.forward_grounding 上包一层，
# 在 NMS 抹分（logit -1e4）之前捕获原始输出，用于归因"漏检是 presence 门控杀的还是阈值砍的"。
# rescore_alpha 非 None 时启用「换阈值口径」：把 pred_logits 原地改写为
# logit(presence^α × cond)（presence 指数软化，α=1≈原始行为、α=0=完全忽略 presence），
# 下游 NMS 与 keep 阈值即卡在新分上（det_thresh>0 时经 property 覆盖 score_threshold_detection）。
# 状态是 threading.local：推理在请求线程内同步执行（_LOCK 只限流不换线程，捕获始终
# 发生在发起请求的线程），各请求线程互不可见——流式（锁在步进函数内部）与单图并发
# 设置参数也不会串扰。
class _DbgState(threading.local):
    def __init__(self):
        self.on = False
        self.topk = 10
        self.calls = []
        self.raw_q = None
        self.rescore_alpha = None
        self.det_thresh = 0.0

_dbg = _DbgState()


def _install_debug_hook():
    """在 detector.forward_grounding + dot_prod_scoring 上挂捕获钩子（加载完成后调用一次）。
    另把 model.score_threshold_detection 换成读 thread-local 的 property：请求线程设了
    det_thresh 就用它，否则用模型默认——keep 与 NMS 阈值随请求生效、请求间天然隔离。"""
    det = _pred.model.detector
    orig = det.forward_grounding

    model = _pred.model
    default_thresh = float(model.__dict__.pop("score_threshold_detection", 0.5))
    type(model).score_threshold_detection = property(
        lambda self: _dbg.det_thresh if _dbg.det_thresh > 0.0 else default_thresh)

    def _raw_hook(_mod, _inp, output):
        # DotProductScoring 输出 (num_layer, bs, num_query, 1)：取最后一层 = 未经 presence 加权的原始 logit
        if _dbg.on:
            try:
                _dbg.raw_q = output[-1].detach().float().flatten()
            except Exception as e:  # noqa: BLE001
                logger.warning("原始 query logit 捕获失败（不影响推理）：%r", e)

    det.dot_prod_scoring.register_forward_hook(_raw_hook)

    def _wrapped(*args, **kwargs):
        if _dbg.on:
            _dbg.raw_q = None   # 清上一次残留，防 instance 头路径下错配
        out = orig(*args, **kwargs)
        if _dbg.on:
            try:
                entry = {}
                pl = out.get("presence_logit_dec")
                if pl is not None:
                    entry["presence_logit"] = float(pl.detach().float().flatten()[0])
                lg = out.get("pred_logits")
                if lg is not None:
                    logits = lg.detach().float().flatten()
                    k = max(1, min(int(_dbg.topk), logits.numel()))
                    vals, idxs = logits.topk(k)
                    entry["topk_joint_logit"] = [float(x) for x in vals]
                    entry["topk_query_idx"] = [int(x) for x in idxs]
                    entry["num_queries"] = int(logits.numel())
                    raw = _dbg.raw_q
                    if raw is not None and raw.numel() == logits.numel():
                        entry["topk_cond_logit"] = [float(raw[i]) for i in idxs]
                # 换阈值口径：pred_logits 原地改写为 logit(presence^α × cond)，
                # 下游 NMS/keep 即卡在新分上（entry 里记录的是改写前的原始值）
                alpha = _dbg.rescore_alpha
                entry["rescore_applied"] = False
                if alpha is not None and pl is not None and lg is not None:
                    raw = _dbg.raw_q
                    if raw is not None and raw.numel() == lg.numel():
                        pres = torch.sigmoid(pl.detach().float().flatten()[0])
                        score = torch.sigmoid(raw) * pres.pow(float(alpha))
                        new_logit = torch.logit(score.clamp(1e-6, 1 - 1e-6))
                        out["pred_logits"] = new_logit.view_as(lg).to(lg.dtype)
                        entry["rescore_applied"] = True
                _dbg.calls.append(entry)
            except Exception as e:  # noqa: BLE001
                logger.warning("debug 捕获失败（不影响推理）：%r", e)
        return out

    det.forward_grounding = _wrapped
    logger.info("presence/top-K 调试钩子已安装（forward_grounding + dot_prod_scoring + 阈值 property）")


def _install_backbone_cache():
    """给 detector.backbone.forward_image 包一层内容寻址 LRU 缓存（加载完成后调用一次）。

    依据前置实验：视觉编码入口只吃图像、与文本 prompt 无关，输出 dict（backbone_fpn /
    vision_pos_enc / sam2_backbone_out）同一输入两次逐位相同；生产负载是两个词（food/drink）
    各一路流式 session 对同一帧背靠背步进，每帧每 session 恰好各调一次 forward_image——
    容量 LRU=2 即可让第二个词必命中，省掉一次 backbone 前向（单帧缓存实测 ~218MB 显存）。
      · key = 张量 shape + fp16 内容 md5（bf16 无法直转 numpy，统一降 fp16 取字节；hash 开销
        ~6MB 拷贝，远小于一次 backbone 前向）+ 额外 kwargs 的 repr（兼容未来 backbone 变体
        传参；出现未知位置参数时不猜 key，本次直接绕过缓存）；
      · miss → 调原函数，输出里所有张量 clone 后入缓存、并返回缓存值（防未来 compile/
        cudagraph 复用输出 buffer）；hit → 直接返回缓存引用（前置实验已证下游不原地改）；
      · 线程安全（与 _LOCK 有界并发共存）：缓存自持小锁；同 key 并发 miss 允许重复计算、
        后写覆盖，不做等待逻辑；任一环节异常 → 中文警告并回退直调原函数，服务不断。"""
    backbone = _pred.model.detector.backbone
    orig = backbone.forward_image
    cache = collections.OrderedDict()   # key -> 输出 dict（张量已 clone）
    cache_lock = threading.Lock()
    stats = {"hit": 0, "miss": 0}

    def _clone_tensors(obj):
        # 递归 clone 容器里的所有张量（结构不变；非张量成员原样引用）
        if torch.is_tensor(obj):
            return obj.clone()
        if isinstance(obj, dict):
            return {k: _clone_tensors(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(_clone_tensors(v) for v in obj)
        return obj

    def _wrapped(samples, *args, **kwargs):
        try:
            if args:                    # 未知位置参数：不猜 key，本次绕过缓存
                return orig(samples, *args, **kwargs)
            t = samples.detach().to(torch.float16)
            key = (tuple(t.shape),
                   hashlib.md5(t.cpu().numpy().tobytes()).hexdigest(),
                   repr(sorted(kwargs.items())))
            with cache_lock:
                hit = cache.get(key)
                if hit is not None:
                    cache.move_to_end(key)
                    stats["hit"] += 1
                else:
                    stats["miss"] += 1
                n = stats["hit"] + stats["miss"]
            if n % 500 == 0:            # 低频打点，便于线上确认缓存生效
                logger.info("backbone 缓存统计：hit=%d miss=%d", stats["hit"], stats["miss"])
            if hit is not None:
                return hit
        except Exception as e:  # noqa: BLE001
            logger.warning("backbone 缓存查询失败（回退直调原函数，不影响推理）：%r", e)
            return orig(samples, **kwargs)
        out = orig(samples, **kwargs)
        try:
            cached = _clone_tensors(out)
            with cache_lock:
                cache[key] = cached     # 同 key 并发 miss：后写覆盖
                cache.move_to_end(key)
                while len(cache) > EMB_CACHE_SIZE:
                    cache.popitem(last=False)
            return cached
        except Exception as e:  # noqa: BLE001
            logger.warning("backbone 缓存写入失败（本次结果不入缓存，不影响推理）：%r", e)
            return out

    backbone.forward_image = _wrapped
    logger.info("backbone 特征缓存已安装（LRU=%d，SAM3_EMB_CACHE=0 可关闭）", EMB_CACHE_SIZE)


def _dbg_begin(debug, topk, alpha, det_thresh):
    """按请求参数布置本线程的捕获/口径状态。返回 (rescore 是否启用, 规整后的 alpha/det_thresh)。"""
    alpha = min(max(float(alpha), 0.0), 1.0)
    det_thresh = min(max(float(det_thresh), 0.0), 0.95)
    rescore = det_thresh > 0.0 or abs(alpha - 1.0) > 1e-6
    if debug or rescore:
        _dbg.topk = max(1, min(int(topk), 50))
        _dbg.calls = []
        _dbg.on = True
        _dbg.rescore_alpha = alpha if rescore else None
        _dbg.det_thresh = det_thresh
    return rescore, alpha, det_thresh


def _dbg_end():
    """清空本线程的捕获/口径状态（calls 保留给响应组装读取）。"""
    _dbg.on = False
    _dbg.rescore_alpha = None
    _dbg.det_thresh = 0.0


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-float(x)))


def _build_debug_payload(calls):
    """把捕获到的第一帧原始输出整理成 /v1/segment 的 debug 字段。
    条件原始分直取 dot_prod_scoring 的 pre-sigmoid logit（不用联合分÷presence 反推——
    联合 logit 被 ±10 clamp 后除法会得出假分）；clamped 标记联合 logit 触到 clamp 限幅的行。"""
    if not calls:
        return None
    c = calls[0]
    presence_logit = c.get("presence_logit")
    presence = _sigmoid(presence_logit) if presence_logit is not None else None
    cond_logits = c.get("topk_cond_logit")
    queries = []
    for rank, (jl, qi) in enumerate(zip(c.get("topk_joint_logit", []),
                                        c.get("topk_query_idx", [])), start=1):
        cl = cond_logits[rank - 1] if cond_logits and rank - 1 < len(cond_logits) else None
        queries.append({"rank": rank, "query_idx": qi,
                        "joint_score": round(_sigmoid(jl), 6), "joint_logit": round(float(jl), 4),
                        "cond_score": round(_sigmoid(cl), 6) if cl is not None else None,
                        "cond_logit": round(float(cl), 4) if cl is not None else None,
                        "clamped": bool(jl <= -9.99 or jl >= 9.99)})
    return {"presence_logit": (round(float(presence_logit), 4)
                               if presence_logit is not None else None),
            "presence_score": round(presence, 6) if presence is not None else None,
            "num_queries": c.get("num_queries"), "topk": queries,
            "note": "joint=联合分(=最终检测置信度, NMS/阈值前, logit clamp ±10); "
                    "cond=真·条件原始分(dot_prod_scoring pre-sigmoid logit 直取, 未经 presence 加权, "
                    "clamp ±12); clamped=联合 logit 触 clamp 限幅"}


def _install_triton_autotune_lock():
    """给 triton Autotuner.run 包一把进程级可重入锁（labels_ptr 竞态的根因修复）。

    实锤根因（5090 journalctl + 上游源码核对）：sam3 上游 connected_components 的
    _local_prop_kernel（sam3/perflib/triton/connected_components.py，
    @triton.autotune(restore_value=["labels_ptr"])）由模块级单例 Autotuner 调度，
    其 restore_copies 与 nargs 都是无锁共享实例态（triton/runtime/autotuner.py 的
    _post_hook 先 copy_ 恢复再把 restore_copies 置空、run 里整体覆写 nargs）。
    本服务 MAX_CONCURRENCY=2 时 food/drink 两线程并发进同一 kernel：服务重启后
    autotune 冷缓存的 benchmark 窗口（跑几十次基准、窗口大）内两线程 pre/post hook
    交叉，后完成的一方必抛 KeyError('labels_ptr')；warmup 后 nargs 覆写竞态则偶发
    TypeError('NoneType' object is not a mapping)。崩点在 add_prompt 的
    fill_holes_in_mask_scores 深处，先前被误判成「上游 state 结构不符」而永久降级。

    修法：把 Autotuner.run 全局串行化。锁只罩 Python 端调度与一次性的 autotune
    benchmark（GPU kernel 本身异步下发，不在锁内等待执行），正常路径每次多持锁
    几十微秒，代价可忽略；用 RLock 防未来嵌套调用死锁；幂等可重复调用。"""
    try:
        from triton.runtime.autotuner import Autotuner
    except Exception as e:  # noqa: BLE001
        logger.warning("triton Autotuner 不可用，跳过并发锁安装（若并发>1 竞态风险仍在）：%r", e)
        return False
    if getattr(Autotuner.run, "_sam3_autotune_locked", False):   # 幂等：已装过不叠包
        return True
    lock = threading.RLock()
    orig_run = Autotuner.run

    def _locked_run(self, *args, **kwargs):
        # 串行化 autotune 调度，消除 restore_copies/nargs 的双线程互踩
        with lock:
            return orig_run(self, *args, **kwargs)

    _locked_run._sam3_autotune_locked = True
    Autotuner.run = _locked_run
    logger.info("triton Autotuner 并发锁已安装（根治 labels_ptr/nargs 竞态）")
    return True


def _load():
    global _pred, _err
    try:
        # 必须在首次推理前安装：重启后第一波并发步进就会触发 autotune benchmark 竞态
        _install_triton_autotune_lock()
        _mf = float(os.environ.get("SAM3_MEM_FRACTION", "0"))
        if _mf > 0 and torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(_mf, 0)  # 固定显存上限（9G≈0.28，与 5090 其他服务隔离）
            logger.info("SAM3 显存上限 fraction=%.3f（本进程最多用这么多）", _mf)
        logger.info("正在加载 SAM3 predictor: %s (version=%s, use_fa3=False)", CKPT, VERSION)
        _pred = sam3.build_sam3_predictor(checkpoint_path=CKPT, version=VERSION, use_fa3=False)
        _install_debug_hook()
        # 特征缓存安装失败只降级不致命（不能让它把整次加载标成失败）
        try:
            if EMB_CACHE_ON:
                _install_backbone_cache()
            else:
                logger.info("backbone 特征缓存未启用（SAM3_EMB_CACHE=0）")
        except Exception as e:  # noqa: BLE001
            logger.warning("backbone 特征缓存安装失败（跳过，不影响推理）：%r", e)
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
            # 已从 _streams 摘除；限流（非互斥）下须先取 step_lock 再关活会话，
            # 防把正在步进中段的 sam3 会话关到半截（锁序 step_lock→_LOCK 与步进一致，无死锁）
            with s["step_lock"]:
                _close_live(s)
        if dead:
            logger.info("回收空闲流式 session：%s", [sid for sid, _ in dead])

threading.Thread(target=_sweep_streams, daemon=True).start()

app = FastAPI(title="SAM3 Inference Server", version="2.0.0")
# Prometheus 埋点：暴露 /metrics，含每端点 QPS/延时直方图/in-flight/错误率
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

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
    debug: bool = False        # true=附带 presence 分 + top-K query 原始分（NMS/阈值前）
    topk: int = 10             # debug 时返回的 top-K query 数量（1~50）
    alpha: float = 1.0         # presence 指数软化：score=presence^α×cond；1≈原始行为，0=完全忽略 presence
    det_thresh: float = 0.0    # >0 时把检测保留/NMS 阈值临时改为该值（卡在上式 score 上）；0=模型默认 0.5

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
    debug: bool = False        # true=附带最新帧的 presence 分 + top-K query 原始分
    topk: int = 10             # debug 时返回的 top-K query 数量（1~50）
    alpha: float = 1.0         # presence 指数软化（同 /v1/segment），作用于本步整个检测窗口
    det_thresh: float = 0.0    # >0 时本步检测保留/NMS 阈值卡在 presence^α×cond 上；0=模型默认

@app.post("/v1/segment")
def segment(req: SegReq):
    if _pred is None:
        raise HTTPException(503, _err or "model not loaded")
    img = _decode(req.image_b64)
    W, H = img.size
    d = tempfile.mkdtemp(prefix="sam3_seg_")
    try:
        img.save(os.path.join(d, "00000.jpg"), quality=95)
        # 捕获/口径状态是 thread-local：推理在本请求线程内同步执行，无需与别的请求互斥
        rescore, alpha, det_thresh = _dbg_begin(req.debug, req.topk, req.alpha, req.det_thresh)
        with _LOCK:
            sid = _pred.start_session(resource_path=d)["session_id"]
            try:
                with _infer_ctx():
                    r = _pred.add_prompt(session_id=sid, frame_idx=0, text=req.text)
                inst = _pack(r["outputs"], W, H)
            finally:
                _dbg_end()
                _pred.close_session(sid)
        resp = {"width": W, "height": H, "num_instances": len(inst), "instances": inst}
        if rescore:
            resp["rescore"] = {"alpha": alpha, "det_thresh": det_thresh,
                               "applied": bool(_dbg.calls and _dbg.calls[0].get("rescore_applied"))}
        if req.debug:
            resp["debug"] = _build_debug_payload(_dbg.calls)
        return resp
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
            # 降级/恢复状态机（_step_with_fallback）：降级次数 + replay 模式下的恢复计步
            "degrade_count": 0, "replay_steps": 0,
            "last_ts": time.time(),
            "step_lock": threading.Lock(),   # 同一 session 的步进串行
        }
    if evicted is not None:
        # 先取被驱逐 session 的 step_lock 再关（同 _sweep_streams：防关到步进中段的会话）
        with evicted["step_lock"]:
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


def _degrade_to_replay(s, log_sid, err, scene):
    """统一降级动作：记完整轨迹日志 → 关活会话 → 切 replay → 降级计数 +1、恢复计步清零。
    只能在 except 块内调用（logger.exception 依赖当前异常上下文输出堆栈）。"""
    logger.exception("%s，session %s 降级 replay（累计第 %d 次；replay 每满 %d 步自动尝试恢复增量）：%s",
                     scene, log_sid, s["degrade_count"] + 1, STREAM_RECOVER_EVERY, err)
    _close_live(s)
    s["live_sid"] = None
    s["impl"] = "replay"
    s["degrade_count"] += 1
    s["replay_steps"] = 0


def _step_with_fallback(s, img, g, log_sid):
    """步进 + 降级/恢复状态机（降级不再终身）：
      · incremental：正常走增量；抛错 → 降级 replay，本步立即用 replay 补齐结果；
      · replay 且是「从增量降级下来的」session（degrade_count>0）：每满
        STREAM_RECOVER_EVERY 步尝试切回增量（live_sid 为空，_step_incremental 会用
        ring 种子重开一代，注册表缝合保证公共 obj_id 连续）；成功即恢复增量，
        失败记日志、继续 replay 并重新计步；
      · 一开始就是 replay 的 session（增量被关闭 / 上游 import 失败）不做恢复尝试。"""
    if s["impl"] == "incremental":
        try:
            return _step_incremental(s, img, g)
        except Exception as e:  # noqa: BLE001
            _degrade_to_replay(s, log_sid, e, "增量路径失败")
            return _step_replay(s)
    if s["degrade_count"] > 0 and STREAM_INCREMENTAL and _INCR_IMPORTS_OK:
        s["replay_steps"] += 1
        if s["replay_steps"] >= STREAM_RECOVER_EVERY:
            s["replay_steps"] = 0
            logger.info("session %s replay 已满 %d 步，尝试重建增量 session 恢复",
                        log_sid, STREAM_RECOVER_EVERY)
            s["impl"] = "incremental"
            try:
                inst = _step_incremental(s, img, g)
            except Exception as e:  # noqa: BLE001
                _degrade_to_replay(s, log_sid, e, "增量恢复尝试失败")
                return _step_replay(s)
            logger.info("session %s 增量路径恢复成功（此前累计降级 %d 次）",
                        log_sid, s["degrade_count"])
            return inst
    return _step_replay(s)


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
        # 捕获/口径状态是 thread-local（推理在本线程同步执行）：α/阈值作用于本步整个检测
        # 窗口（增量=新帧一帧；replay/开代=全窗口，口径一致）；debug 取最新帧那次前向
        rescore, alpha, det_thresh = _dbg_begin(req.debug, req.topk, req.alpha, req.det_thresh)
        try:
            # 降级/恢复状态机：增量抛错降 replay（非终身），replay 满周期自动尝试恢复增量
            inst = _step_with_fallback(s, img, g, req.session_id)
        finally:
            _dbg_end()
        run_ms = (time.time() - t0) * 1000.0
        s["next_global"] = g + 1
        n_reg = len(s["registry"])
        impl, gen_frames = s["impl"], s["gen_frames"]
        degrade_count = s["degrade_count"]
    W, H = img.size
    gpu_mb = int(torch.cuda.memory_allocated() // (1024 * 1024)) if torch.cuda.is_available() else 0
    resp = {"session_id": req.session_id, "global_index": g,
            "width": W, "height": H, "window_frames": min(g + 1, s["window"]),
            "num_instances": len(inst), "instances": inst,
            "active_objects": n_reg, "run_ms": round(run_ms, 1),
            "impl": impl, "gen_frames": gen_frames, "degrade_count": degrade_count,
            "gpu_mb": gpu_mb}
    if rescore:
        resp["rescore"] = {"alpha": alpha, "det_thresh": det_thresh,
                           "applied": bool(_dbg.calls and _dbg.calls[-1].get("rescore_applied"))}
    if req.debug:
        # 多次前向（replay/开代）时最后一次 = 最新帧；增量路径本就只有一次
        resp["debug"] = _build_debug_payload(_dbg.calls[-1:])
    return resp

@app.get("/v1/stream")
def stream_list():
    """列出存活的流式 session（排障/看板用）。"""
    now = time.time()
    with _streams_lock:
        return {"sessions": [
            {"session_id": sid, "text": s["text"], "window": s["window"],
             "frames_seen": s["next_global"], "active_objects": len(s["registry"]),
             "impl": s["impl"], "gen_frames": s["gen_frames"],
             "degrade_count": s["degrade_count"], "replay_steps": s["replay_steps"],
             "idle_sec": round(now - s["last_ts"], 1)} for sid, s in _streams.items()]}

@app.delete("/v1/stream/{session_id}")
def stream_close(session_id: str):
    with _streams_lock:
        s = _streams.pop(session_id, None)
    if s is not None:
        # 先取 step_lock 再关（同 _sweep_streams：防关到步进中段的会话）
        with s["step_lock"]:
            _close_live(s)
    return {"closed": s is not None}

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, workers=1)
