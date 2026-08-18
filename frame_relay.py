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
import re
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from fastapi import APIRouter, Body, File, Form, Query, UploadFile
from fastapi.responses import JSONResponse, Response

router = APIRouter()

DEVICE_TTL = 60.0        # 设备条目过期时间（秒）：超过没有新帧即视为下线并清理
UNKNOWN_DEVICE = "unknown"   # camera_info 缺失/无 device_id 时的兜底桶

# 流水线契约：处理回调返回 DEFERRED 表示「GPU 级已完成、产物由构建级稍后经
# set_product 异步写回」。worker 收到即取下一帧，不写产物槽——
# 这样 GPU 推理与 CPU 构建/网络调用重叠流水，链路吞吐从 sum(阶段) 变 max(阶段)。
DEFERRED = "__deferred__"

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

# per-device 帧率类配置（device_id -> {push_fps, product_interval}）。与 _config 分开存，
# 原因有二：(1) /api/frame/config 是全量覆盖语义，/panel 与 /experience 都会整体 POST，
# 塞进去会被互相抹掉；(2) 设备桶超时会被清理，而帧率配置要在设备掉线重连后仍然生效。
# 写入走 POST /api/frame/device-config（merge-patch 语义），推流端经 /api/frame/status
# 的 devices[].config 轮询生效；后续 18091 控制面（superadmin 经 /da3-api 反代）走同一对接口。
_dev_config: dict = {}
# 可写键分三类：数值（按范围钳制）、枚举（白名单）、颜色（#rrggbb）。
# 数值键：push_fps=RGB 推帧频率（fps）；product_interval=点云产物上传间隔（秒）；
# depth_*=硬件深度伪彩渲染参数——本服务只存储与下发，渲染在推流端（mac mini
# cam-pusher）应用：推流端每 2s 经 /api/frame/status 的 devices[].config 轮询取走，
# 约 2~4s 生效。默认值全部留在推流端（未下发的键=「默认」配置预设口径）。
DEV_CONFIG_LIMITS = {
    "push_fps": (0.1, 30.0), "product_interval": (0.5, 30.0),
    "depth_min_m": (0.05, 10.0),      # 固定量程近端（米）
    "depth_max_m": (0.1, 20.0),       # 固定量程远端（米）
    "depth_autorange": (0.0, 1.0),    # 1=逐帧分位自适应量程（0=固定量程）
    "depth_auto_lo": (0.0, 49.0),     # 自适应量程低分位（%）
    "depth_auto_hi": (51.0, 100.0),   # 自适应量程高分位（%）
    "depth_invert": (0.0, 1.0),       # 1=近暗远亮（默认近亮远暗）
    "depth_gamma": (0.2, 5.0),        # 归一化后 gamma（>1 提亮中间调）
    "depth_eq_clip": (0.5, 10.0),     # CLAHE clipLimit（depth_eq=clahe 时生效）
    "depth_fill_px": (1.0, 15.0),     # 孔洞填充核半径（像素）
    "depth_ema": (0.0, 0.95),         # 时域平滑 EMA 系数（0=关）
    "depth_edge": (0.0, 1.0),         # 深度边缘描边强度（0=关）
    "depth_contour_m": (0.0, 2.0),    # 等值线间隔（米，0=关）
    "depth_jpeg_q": (30.0, 95.0),     # 深度图 JPEG 编码质量
    "depth_fps": (0.0, 30.0),         # 深度独立推帧率（0=跟随 RGB 主帧节拍）
    # pc_*=设备点云（原始深度+RGB → /api/devpc/frame）推流参数：推流端仅在
    # devices[].pc_want=true（/experience 选中「设备点云」来源）时才推，
    # 未下发时用推流端默认值（1.5fps / stride 自适应到 ~424 列）
    "pc_fps": (0.2, 10.0),            # 设备点云推帧率
    "pc_stride": (1.0, 8.0),          # 深度降采样步长（0/未下发=自适应）
}
# 枚举键白名单（值统一小写存储）
DEV_CONFIG_ENUMS = {
    "depth_colormap": {"turbo", "jet", "viridis", "plasma", "inferno", "magma",
                       "hot", "bone", "ocean", "hsv", "parula", "cividis",
                       "twilight_shifted", "deepgreen", "gray", "lidar",
                       "radar", "indigo", "moss", "lavender"},
    "depth_eq": {"off", "global", "clahe"},          # 直方图均衡模式
    "depth_fill": {"off", "close", "inpaint"},       # 无效点孔洞填充模式
    "depth_smooth": {"off", "median", "bilateral"},  # 空域滤波模式
}
# 颜色键（#rrggbb）：无效点（深度 0 值）着色
DEV_CONFIG_COLORS = {"depth_invalid_color"}
_HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")

# per-device 配置落盘持久化（gitignored 本地态，学 sam3_score_cfg.json 的做法）：
# 服务重启不再回落默认帧率——此前配置是纯内存态，每次重启都得重新下发
_DEV_CONFIG_PATH = Path(__file__).resolve().parent / "dev_config.json"
try:
    _dev_config.update({str(d): dict(c) for d, c in
                        json.loads(_DEV_CONFIG_PATH.read_text()).items()
                        if isinstance(c, dict)})
    print(f"[frame-relay] 已回读 per-device 配置：{_dev_config}", flush=True)
except FileNotFoundError:
    pass
except Exception as e:
    print(f"[frame-relay] per-device 配置回读失败（忽略）：{type(e).__name__}: {e}",
          flush=True)


def _save_dev_config_locked() -> None:
    """持锁快照后落盘（写临时文件再原子替换，防写一半被杀留坏文件）。"""
    snap = json.dumps(_dev_config, ensure_ascii=False, indent=1)
    try:
        tmp = _DEV_CONFIG_PATH.with_suffix(".json.tmp")
        tmp.write_text(snap)
        tmp.replace(_DEV_CONFIG_PATH)
    except Exception as e:
        print(f"[frame-relay] per-device 配置落盘失败（忽略）：{type(e).__name__}: {e}",
              flush=True)


def _normalize_dev_patch(patch: dict):
    """校验并钳制一份 device-config patch，返回 (norm, 错误文案)。

    norm 里键值为 None 表示「删除该键恢复兜底」（原值传 null/空串）；出错时 norm=None。
    供 /api/frame/device-config 与深度配置预设保存两处复用，保证同一套白名单与钳制规则。
    """
    unknown = sorted(k for k in patch if k not in DEV_CONFIG_LIMITS
                     and k not in DEV_CONFIG_ENUMS and k not in DEV_CONFIG_COLORS)
    if unknown:
        return None, (f"不支持的键：{unknown}；可用："
                      f"{sorted([*DEV_CONFIG_LIMITS, *DEV_CONFIG_ENUMS, *DEV_CONFIG_COLORS])}")
    norm: dict = {}
    for k, v in patch.items():
        if v is None or v == "":
            norm[k] = None
        elif k in DEV_CONFIG_LIMITS:
            try:
                lo, hi = DEV_CONFIG_LIMITS[k]
                norm[k] = min(hi, max(lo, float(v)))
            except (TypeError, ValueError):
                return None, f"{k} 不是数字：{v!r}"
        elif k in DEV_CONFIG_ENUMS:
            sv = str(v).strip().lower()
            if sv not in DEV_CONFIG_ENUMS[k]:
                return None, f"{k} 取值非法：{v!r}；可用：{sorted(DEV_CONFIG_ENUMS[k])}"
            norm[k] = sv
        else:  # 颜色键
            sv = str(v).strip()
            if not _HEX_COLOR_RE.match(sv):
                return None, f"{k} 不是 #rrggbb 颜色：{v!r}"
            norm[k] = sv.lower()
    return norm, None


# ── 深度视图配置预设（/experience 深度图抽屉「配置预设」区，命名多预设、数量不限）──
# name -> {"saved_at": 秒级时间戳, "config": 深度渲染 per-device 键（白名单校验后存储）,
#          "display": 页面「深度显示」层参数, "dot": 点云化参数}
# display/dot 对本模块不透明：它们是页面本地（localStorage）参数，恢复时由页面自行应用；
# config 恢复时由页面 POST /api/frame/device-config 写回当前选中设备。
# 落盘持久化（gitignored 本地态，与 dev_config.json 同款原子写），服务重启不丢。
_depth_presets: dict = {}
_depth_presets_lock = threading.Lock()
_DEPTH_PRESETS_PATH = Path(__file__).resolve().parent / "depth_presets.json"
_DEPTH_PRESET_NAME_MAX = 40          # 预设名长度上限（字符）
_DEPTH_PRESET_BODY_MAX = 32 * 1024   # 单预设 JSON 体积上限（防异常客户端灌爆落盘文件）
try:
    _depth_presets.update({str(n): dict(p) for n, p in
                           json.loads(_DEPTH_PRESETS_PATH.read_text()).items()
                           if isinstance(p, dict)})
    print(f"[frame-relay] 已回读深度配置预设 {len(_depth_presets)} 个", flush=True)
except FileNotFoundError:
    pass
except Exception as e:
    print(f"[frame-relay] 深度配置预设回读失败（忽略）：{type(e).__name__}: {e}",
          flush=True)


def _save_depth_presets_locked() -> None:
    """持锁快照后落盘（写临时文件再原子替换，防写一半被杀留坏文件）。"""
    snap = json.dumps(_depth_presets, ensure_ascii=False, indent=1)
    try:
        tmp = _DEPTH_PRESETS_PATH.with_suffix(".json.tmp")
        tmp.write_text(snap)
        tmp.replace(_DEPTH_PRESETS_PATH)
    except Exception as e:
        print(f"[frame-relay] 深度配置预设落盘失败（忽略）：{type(e).__name__}: {e}",
              flush=True)

# DA3 处理回调：fn(image_bytes, config, device_id, seq, gen) -> 产物描述字典，
# 或返回 DEFERRED（流水线模式：产物由构建级稍后经 set_product(device_id, seq, gen, …)
# 异步写回）；由 app.py 在有模型时注入
#   {"kind":"image","bytes":b"...","content_type":"image/jpeg","meta":{...}}  # 深度图
#   {"kind":"model","url":"/glb/<token>/scene.glb","meta":{...}}              # 点云/网格 GLB
_processor: Optional[Callable[[bytes, dict, str], dict]] = None
_worker_started = False

# （硬件深度伪彩随主帧 /api/frame 的可选 depth 字段上报，见 ingest_frame）

# 设备点云按需推流标志的提供方：fn(device_id) -> bool（True=页面正选中「设备点云」
# 来源，推流端应推原始深度）。由 app.py 注入（demand 状态由 /api/devpc/status 的
# 轮询续期）；未注入时 /api/frame/status 恒报 False，推流端不推，链路零开销
_pc_want_provider: Optional[Callable[[str], bool]] = None


def set_pc_want_provider(fn: Callable[[str], bool]) -> None:
    """注入设备点云按需推流标志（见上方注释）。"""
    global _pc_want_provider
    _pc_want_provider = fn


def _pc_want(device_id: str) -> bool:
    """该设备当前是否需要推原始深度（provider 未注入/异常一律 False，不影响状态接口）。"""
    if _pc_want_provider is None:
        return False
    try:
        return bool(_pc_want_provider(device_id))
    except Exception:
        return False


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


def get_latest_frame(device_id: str) -> Optional[bytes]:
    """取指定设备最新一帧原始图片字节（无设备/无帧返回 None）。
    供旁路链取 RGB 用——如双目链触发 VLM 识别时，识别图要用彩色帧而非 IR。"""
    with _cv:
        st = _devices.get(device_id)
        return st["image"] if st else None


def get_latest_frame_seq(device_id: str):
    """取最新一帧的 (字节, seq, received_at)，无设备/无帧返回 (None, 0, 0.0)。
    seq 供旁路链去重用——直传 VLM 识别的触发间隔可能快过 RGB 推帧，
    seq 没变说明还是同一张图，重复送 VLM 纯属烧算力。
    received_at 是本服务收到该帧的时刻（epoch 秒，本机时钟）：识别链路拿它算
    「帧龄」与端到端延时，跨机器时钟的 capture/upload 时刻另见 _timing_locked。"""
    with _cv:
        st = _devices.get(device_id)
        if not st or st["image"] is None:
            return None, 0, 0.0
        return st["image"], int(st["seq"]), float(st["received_at"] or 0.0)


def _parse_ms(val) -> Optional[int]:
    """把 13 位毫秒级 epoch 时间戳（str/int/float）解析成 int 毫秒；无效/明显不是毫秒级返回 None。"""
    try:
        ms = int(float(val))
    except (TypeError, ValueError):
        return None
    # 粗校验量级：毫秒级 epoch 应在 1e12 ~ 1e14 之间（2001~5138 年），秒级/相对时间戳一律拒收
    return ms if 10 ** 12 <= ms < 10 ** 14 else None


def _timing_locked(st: dict) -> dict:
    """从设备桶抽取延时链路的三个时刻（均为 epoch 毫秒，缺失为 None）：
      capture_ts_ms  拍摄时间：multipart `timestamp` 字段（App 从设备媒体文件名解析的 13 位 ms）；
      upload_ts_ms   发起上传时间：camera_info JSON 里的 `upload_at_ms`（镜像链路专属，老版本 App 没有）；
      server_ts_ms   服务器收到时间：本服务收到该帧的时刻。
    注意三者分属设备/手机/服务器三个时钟，前端展示的差值含时钟偏差。"""
    upload = None
    if st["camera_info"]:
        try:
            upload = _parse_ms((json.loads(st["camera_info"]) or {}).get("upload_at_ms"))
        except (ValueError, TypeError, AttributeError):
            pass
    return {
        "capture_ts_ms": _parse_ms(st["timestamp"]),
        "upload_ts_ms": upload,
        "server_ts_ms": int(st["received_at"] * 1000) if st["received_at"] else None,
    }


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
        # 相机硬件深度图（帧源已伪彩渲染好的 JPEG，如 mac mini 推帧器）：独立计数，
        # 不参与 DA3 处理，仅供 /panel 左上角展示。depth_received_at 单独记时：
        # 深度独立帧率高于 RGB 时存在 depth-only 上报，须一并续设备 TTL
        "depth_image": None, "depth_seq": 0, "depth_content_type": "image/jpeg",
        "depth_received_at": 0.0,
        "received_at": 0.0, "prev_received_at": 0.0,
        "camera_info": None, "timestamp": None, "first_seen": now,
        "product": None, "product_seq": 0, "product_gen": 0, "product_error": None,
        # 外部产物（设备直传，如 Mac 端真深度点云）：字节存桶内经 /api/frame/product-model
        # 提供；ext_until 内该设备跳过 DA3 处理（产物槽位交给外部方），过期自动恢复 DA3
        "ext_model": None, "ext_until": 0.0, "ext_ver": 0,
    }


def _ext_active_locked(st: dict, now: float) -> bool:
    """该设备当前是否处于「外部产物接管」窗口内（是则 DA3 worker 跳过它）。"""
    return st.get("ext_until", 0.0) > now


def _prune_locked(now: float) -> None:
    """清理超过 DEVICE_TTL 没有新帧的设备条目（选中设备也会过期，随后自动回落）。
    RGB 帧与 depth-only 上报任一到达都算「有新帧」续 TTL。"""
    stale = [d for d, st in _devices.items()
             if now - max(st["received_at"], st.get("depth_received_at", 0.0)) > DEVICE_TTL]
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
                    # 键须含 received_at：设备桶过期重建后 seq 从 1 重新计数，若只看
                    # (dev, seq, 配置版本) 会与旧桶首帧撞键 → 新帧被误判"已处理"而永久漏处理
                    key = (dev, st["seq"], st["received_at"], _config_gen)
                    if key != last_key:
                        break
                _cv.wait()
            seq, gen, img, proc = st["seq"], _config_gen, st["image"], _processor
            config = dict(_config)
        try:
            product = proc(img, config, dev, seq, gen)
            err = None
        except Exception as e:  # 处理失败不影响原图展示，仅记录错误
            product, err = None, f"{type(e).__name__}: {e}"
        if product == DEFERRED:
            # 流水线模式：GPU 级已完成，产物由构建级经 set_product 异步写回；
            # 立即取下一帧，让 GPU 与构建级重叠
            last_key = key
            continue
        with _cv:
            st2 = _devices.get(dev)
            # 设备可能在处理期间过期下线，桶没了就丢弃产物；外部产物接管窗口内 DA3
            # 照常跑（SAM3 映射/高亮/实时识别等链路靠其副作用驱动），但产物槽位归
            # 外部方（如 Astra 真深度点云），DA3 的处理结果不写回
            if st2 is not None and not _ext_active_locked(st2, time.time()):
                st2["product"] = product if product is not None else st2["product"]
                st2["product_seq"] = seq
                st2["product_gen"] = gen
                st2["product_error"] = err
                _cv.notify_all()
        last_key = key


def set_product(device_id: str, seq: int, gen: int, product, error=None) -> None:
    """流水线构建级的异步产物写回口（线程安全）。

    语义与 worker 同步写回一致：设备下线丢弃、外部产物接管窗口内不写、
    product=None 时保留旧产物只更新 seq/gen/error。seq 落后于当前槽位（构建级
    单线程本不该发生，防御桶重建）时丢弃。"""
    with _cv:
        st = _devices.get(device_id)
        if st is None or _ext_active_locked(st, time.time()):
            return
        if seq < st["product_seq"]:
            return
        st["product"] = product if product is not None else st["product"]
        st["product_seq"] = seq
        st["product_gen"] = gen
        st["product_error"] = error
        _cv.notify_all()


def _target_device_locked(device: Optional[str]):
    """按 query 参数解析目标设备桶：不传=当前选中设备。返回 (device_id, 桶或None)。"""
    dev = (device or "").strip() or _effective_device_locked()
    if dev is None:
        return None, None
    return dev, _devices.get(dev)


@router.post("/api/frame")
async def ingest_frame(
    image: Optional[UploadFile] = File(None),
    depth: Optional[UploadFile] = File(None),
    camera_info: Optional[str] = Form(None),
    timestamp: Optional[str] = Form(None),
):
    """接收设备实时帧（与现有 /device/media/upload/image 同款 multipart 字段）。

    字段：image（二进制图片）、depth（可选，相机硬件深度图的伪彩 JPEG，随彩色帧
    一并上报）、camera_info（可选 JSON 字符串，其中 device_id 用于分桶）、
    timestamp（可选）。仅更新该设备桶的最新帧并唤醒处理线程，立即返回，不阻塞在 GPU 推理上。

    depth-only 模式：深度独立帧率（depth_fps）高于 RGB 推帧率时，推流端在 RGB
    节拍之间只带 depth 不带 image——此时仅更新深度槽与设备存活时间，不推进 RGB
    seq/received_at（fps 统计与单目 DA3 触发键不受影响）。image 与 depth 至少给一个。
    """
    data = await image.read() if image is not None else None
    depth_data = await depth.read() if depth is not None else None
    if not data and not depth_data:
        return JSONResponse({"ok": False, "error": "image 与 depth 至少一个非空"},
                            status_code=400)
    dev = _parse_device_id(camera_info)
    now = time.time()
    with _cv:
        st = _devices.get(dev)
        if st is None:
            st = _devices[dev] = _new_bucket(now)
        seq = st["seq"]
        if data:
            st["seq"] += 1
            seq = st["seq"]
            st["image"] = data
            st["content_type"] = image.content_type or "image/jpeg"
            st["prev_received_at"] = st["received_at"]
            st["received_at"] = now
            st["camera_info"] = camera_info
            st["timestamp"] = timestamp
        if depth_data:
            st["depth_seq"] += 1
            st["depth_image"] = depth_data
            st["depth_content_type"] = depth.content_type or "image/jpeg"
            st["depth_received_at"] = now
            # depth-only 先于任何 RGB 帧到达（罕见时序）：camera_info 也补上，
            # 设备能力字段不留空窗
            if not data and st["camera_info"] is None:
                st["camera_info"] = camera_info
        _prune_locked(now)
        _effective_device_locked()    # 首台设备出现即自动粘性选中
        _cv.notify_all()
    return JSONResponse({"ok": True, "seq": seq, "bytes": len(data or b""),
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


@router.post("/api/frame/product")
async def ingest_product(
    model: UploadFile = File(...),
    camera_info: Optional[str] = Form(None),
    meta: Optional[str] = Form(None),
    hold: Optional[str] = Form(None),
):
    """接收设备直传的模型类产物（GLB，如 Mac 端 Astra 真深度点云）。

    字段：model（GLB 二进制）、camera_info（JSON，含 device_id，与 /api/frame 同款）、
    meta（可选 JSON：label/fov_deg/shape 等，透传给面板展示）、hold（可选秒数，默认 10：
    接管窗口时长，窗口内该设备跳过 DA3、产物槽位归外部方；停止上传后过期自动恢复 DA3）。
    设备桶必须已存在（先经 /api/frame 发帧再传产物），否则 404。"""
    data = await model.read()
    if not data:
        return JSONResponse({"ok": False, "error": "空产物"}, status_code=400)
    dev = _parse_device_id(camera_info)
    try:
        meta_obj = dict(json.loads(meta)) if meta else {}
    except (ValueError, TypeError):
        meta_obj = {}
    try:
        hold_s = min(60.0, max(1.0, float(hold))) if hold else 10.0
    except (ValueError, TypeError):
        hold_s = 10.0
    now = time.time()
    with _cv:
        st = _devices.get(dev)
        if st is None:
            return JSONResponse({"ok": False, "error": f"设备不存在（先向 /api/frame 发帧）：{dev}"},
                                status_code=404)
        st["ext_model"] = data
        st["ext_until"] = now + hold_s
        st["ext_ver"] += 1
        ver = st["ext_ver"]
        # url 带版本号保证每次更新都变（面板以 url+product_seq 判断是否换图）
        st["product"] = {
            "kind": "model",
            "url": f"/api/frame/product-model?device={dev}&v={ver}",
            "meta": meta_obj,
        }
        st["product_seq"] = st["seq"]
        st["product_error"] = None
        _cv.notify_all()
    return JSONResponse({"ok": True, "device": dev, "ver": ver, "bytes": len(data),
                         "hold": hold_s})


@router.get("/api/frame/product-model")
def latest_product_model(device: Optional[str] = Query(None)):
    """返回指定设备（缺省=选中设备）最新的外部模型类产物（GLB）字节，供 model-viewer 加载。"""
    with _cv:
        _dev, st = _target_device_locked(device)
        data = st["ext_model"] if st else None
    if not data:
        return JSONResponse({"error": "暂无外部模型产物"}, status_code=404)
    return Response(
        content=data,
        media_type="model/gltf-binary",
        headers={"Cache-Control": "no-store"},
    )


@router.post("/api/frame/device-config")
async def set_device_config(body: dict = Body(...)):
    """按设备写帧率类配置（merge-patch：只改传入的键；键值传 null/空串=删除该键恢复兜底）。

    body 形如 {"device_id": "macmini-astra", "config": {"push_fps": 5, "product_interval": 1.5}}。
    与 /api/frame/config（全局一份、全量覆盖）不同：本配置按设备分桶、只接受
    三张白名单表里的键（数值按范围钳制、枚举按白名单、颜色按 #rrggbb 校验）；
    设备不必在线（先下发配置再起推流端同样生效）。
    """
    dev = str((body or {}).get("device_id") or "").strip()
    if not dev:
        return JSONResponse({"ok": False, "error": "缺少 device_id"}, status_code=400)
    patch = (body or {}).get("config")
    if not isinstance(patch, dict) or not patch:
        return JSONResponse({"ok": False, "error": "缺少 config（应为非空对象）"}, status_code=400)
    norm, err = _normalize_dev_patch(patch)
    if err:
        return JSONResponse({"ok": False, "error": err}, status_code=400)
    with _cv:
        cfg = dict(_dev_config.get(dev) or {})
        for k, v in norm.items():
            if v is None:
                cfg.pop(k, None)
            else:
                cfg[k] = v
        if cfg:
            _dev_config[dev] = cfg
        else:
            _dev_config.pop(dev, None)
        _save_dev_config_locked()
        _cv.notify_all()
    return JSONResponse({"ok": True, "device": dev, "config": cfg})


@router.get("/api/frame/depth-presets")
async def list_depth_presets():
    """列出全部深度视图配置预设（含完整参数，页面恢复时直接取用，最新保存在前）。"""
    with _depth_presets_lock:
        items = [{"name": n, **p} for n, p in _depth_presets.items()]
    items.sort(key=lambda x: x.get("saved_at") or 0, reverse=True)
    return JSONResponse({"ok": True, "presets": items})


@router.post("/api/frame/depth-presets")
async def save_depth_preset(body: dict = Body(...)):
    """保存一份深度视图配置预设（重名覆盖，数量不设上限）。

    body 形如 {"name": "展会白天", "config": {深度渲染键…}, "display": {…}, "dot": {…}}。
    config 走 device-config 同一套白名单校验/钳制（预设是完整快照，不含删除语义，
    传 null/空串的键直接剔除）；display/dot 为页面本地参数，按不透明对象存储。
    """
    name = str((body or {}).get("name") or "").strip()
    if not name:
        return JSONResponse({"ok": False, "error": "缺少预设名"}, status_code=400)
    if len(name) > _DEPTH_PRESET_NAME_MAX:
        return JSONResponse({"ok": False, "error": f"预设名过长（≤{_DEPTH_PRESET_NAME_MAX} 字符）"},
                            status_code=400)
    config = (body or {}).get("config") or {}
    display = (body or {}).get("display") or {}
    dot = (body or {}).get("dot") or {}
    if not all(isinstance(x, dict) for x in (config, display, dot)):
        return JSONResponse({"ok": False, "error": "config/display/dot 应为对象"},
                            status_code=400)
    norm, err = _normalize_dev_patch(config)
    if err:
        return JSONResponse({"ok": False, "error": err}, status_code=400)
    norm = {k: v for k, v in norm.items() if v is not None}
    preset = {"saved_at": int(time.time()), "config": norm,
              "display": display, "dot": dot}
    try:
        if len(json.dumps(preset, ensure_ascii=False)) > _DEPTH_PRESET_BODY_MAX:
            return JSONResponse({"ok": False, "error": "预设内容过大"}, status_code=400)
    except (TypeError, ValueError):
        return JSONResponse({"ok": False, "error": "预设内容不可序列化"}, status_code=400)
    with _depth_presets_lock:
        _depth_presets[name] = preset
        _save_depth_presets_locked()
        count = len(_depth_presets)
    return JSONResponse({"ok": True, "name": name, "count": count})


@router.post("/api/frame/depth-presets/delete")
async def delete_depth_preset(body: dict = Body(...)):
    """删除一份深度视图配置预设。走 POST+body 而非 DELETE+路径参数：
    预设名允许中文/空格/斜杠等任意字符，塞 URL 路径会被编码规则坑到。"""
    name = str((body or {}).get("name") or "").strip()
    if not name:
        return JSONResponse({"ok": False, "error": "缺少预设名"}, status_code=400)
    with _depth_presets_lock:
        if name not in _depth_presets:
            return JSONResponse({"ok": False, "error": f"预设不存在：{name}"}, status_code=404)
        _depth_presets.pop(name)
        _save_depth_presets_locked()
        count = len(_depth_presets)
    return JSONResponse({"ok": True, "name": name, "count": count})


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


@router.get("/api/frame/latest-depth")
def latest_depth(device: Optional[str] = Query(None)):
    """返回指定设备（缺省=选中设备）最新的相机硬件深度图字节（帧源已伪彩渲染，
    供 /panel 左上角 <img> 直接展示）。没有深度能力的帧源（如手机 App）恒 404。"""
    with _cv:
        _dev, st = _target_device_locked(device)
        img = st["depth_image"] if st else None
        ct = st["depth_content_type"] if st else "image/jpeg"
        seq = st["depth_seq"] if st else 0
    if img is None:
        return JSONResponse({"error": "暂无深度帧"}, status_code=404)
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
                # 该设备已下发的帧率类配置（/api/frame/device-config），推流端与控制面都从这读
                "config": dict(_dev_config.get(d) or {}),
                # 设备点云按需推流标志（app.py 注入；True=推流端应推原始深度）
                "pc_want": _pc_want(d),
            })
        if st is None:
            return JSONResponse({
                "seq": 0, "has_frame": False, "has_depth": False, "depth_seq": 0,
                "device": dev, "selected": selected,
                "devices": devices, "processor": _processor is not None,
                "config": _config, "config_gen": _config_gen,
                "received_at": None, "age": None, "interval": None, "fps": None,
                "content_type": None, "camera_info": None, "timestamp": None,
                "capture_ts_ms": None, "upload_ts_ms": None, "server_ts_ms": None,
                "product_kind": None, "product_url": None, "product_meta": None,
                "product_seq": 0, "product_gen": 0, "product_error": None,
            })
        interval = (st["received_at"] - st["prev_received_at"]) if st["prev_received_at"] else 0.0
        product = st["product"] or {}
        return JSONResponse({
            "seq": st["seq"],
            "has_frame": st["image"] is not None,
            "has_depth": st["depth_image"] is not None,
            "depth_seq": st["depth_seq"],
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
            **_timing_locked(st),
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
