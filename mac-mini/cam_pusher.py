# -*- coding: utf-8 -*-
"""mac mini 摄像头推帧器（浅体验区展会拓扑的帧源之一）。

用 pyorbbecsdk 抓取 mini 上插着的 Orbbec 相机，按 8060 的入帧契约推给 5090 的
da3-web 服务。每台相机独立成一个 device_id 桶，/panel 下拉可切换。

每个推帧节拍推两路（同一个节拍 → 同设备两路 fps 天然一致）：
  1. RGB 彩色帧 → POST /api/frame（契约与手机 App 一致，8060 单目链路零改动）；
     带硬件深度的相机同请求附可选字段 depth——深度帧在 mini 端做伪彩渲染
     （默认固定量程 TURBO：近亮暖/远暗冷/无效点黑，默认 0.2~2m 防手/物进出画面
     时整图颜色跳变），供 /panel「原设备深度图」格与 /experience「设备深度图」
     来源展示，不参与 DA3 处理。渲染参数（色彩映射/量程/gamma/均衡/填洞/滤波/
     描边/等值线等 depth_* 键）可由服务端 per-device 配置逐项覆盖，经
     device-config 轮询热生效（约 2~4s），未下发时全走默认=历史行为；
  2. 辅助帧    → POST /api/frame/aux（仅双目相机）：左 IR + 右 IR 灰度 JPEG，
     camera_info 带 stereo_supported / baseline_mm / laser_mode，
     供 8060 的双目 DA3 点云链路。

另有一路按需推流（独立节拍，默认关）：
  3. 设备点云原料 → POST /api/devpc/frame：D2C 对齐后的原始 uint16 深度（PNG
     无损、按 stride 降采样）+ 同帧 RGB JPEG + 彩色内参。仅当 /experience 选中
     「设备点云」来源（服务端 devices[].pc_want 亮起，10s TTL 按需续期）才推，
     供 8060 反投影成硬件真深度彩色点云 GLB；不推时链路零带宽开销。

要点：
  · 帧全部来自同一 pipeline 的同一 frameset，硬件同步，无配对逻辑；
  · G335 激光策略（LASER_MODE）：喂 DA3 的双目 IR 要"无散斑"，硬件深度要"有散斑"，
    两者冲突。interleave 模式用 OB_PROP_LASER_ON_OFF_PATTERN_INT 让投射器逐帧
    交替开关，按帧元数据 LASER_STATUS 分拣：无光帧取 IR、有光帧取深度；
    该属性设置失败则自动退回 on（散斑 IR 直喂 DA3。2026-08-13 实测 1.3.2 + G335
    固件不支持该属性，实际运行即此退路）；
  · 任一辅助流开不出来只降级该流（日志告警），RGB 主链路永不受影响；
  · G335 彩色流原生 MJPG（即 JPEG 字节），直接透传不转码；其他格式用 cv2 转 JPEG；
  · 相机线程各自独立：断线/异常自动重连重开，互不影响；
  · 推帧节流：相机 ~30fps 出帧，只按当前生效 fps 取最新帧上传，不积压；
  · 必须 root 运行（libusb 直读 USB），由 LaunchDaemon 承载。

配置经环境变量（LaunchDaemon 经 run-pusher.sh source 同目录 .env）：
  RELAY_URL     推帧目标，默认 http://192.168.0.50:8060
  PUSH_FPS      推帧频率兜底值，默认 3——仅在拿不到服务端配置时生效
  JPEG_QUALITY  非 MJPG 相机转码 JPEG 质量，默认 80
  LASER_MODE    G335 激光策略 interleave|on|off，默认 interleave（失败自动退 on）
  DEPTH_MIN_M / DEPTH_MAX_M  深度伪彩固定量程（米），默认 0.2~2.0

推帧频率的权威来源是 8060 服务端配置，**按设备分桶、两台相机各调各的**：
/panel 设备栏滑杆与 /experience「调节」抽屉都 POST /api/frame/device-config
（per-device merge-patch），后台线程每 2s 轮询 /api/frame/status 的
devices[].config.push_fps 热生效；取值优先级 per-device ＞ 全局 config.push_fps
（旧口径兜底）＞ PUSH_FPS。服务端不可达时沿用最后一次的值。
"""
import json
import logging
import os
import signal
import threading
import time

import cv2
import numpy as np
import requests
from pyorbbecsdk import (AlignFilter, Config, Context, OBFormat,
                         OBFrameMetadataType, OBFrameType, OBPropertyID,
                         OBSensorType, OBStreamType, Pipeline)

RELAY_URL = os.environ.get("RELAY_URL", "http://192.168.0.50:8060").rstrip("/")
PUSH_FPS = float(os.environ.get("PUSH_FPS", "3"))
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "80"))
LASER_MODE = os.environ.get("LASER_MODE", "interleave").strip().lower()
# 硬件深度图伪彩的固定量程（米）：固定而非逐帧 min/max 自适应——展台上手/物进出画面
# 时整图颜色才不会跳变闪烁。桌面演示场景默认 0.2~2m，按展台纵深经 .env 调整。
DEPTH_MIN_M = float(os.environ.get("DEPTH_MIN_M", "0.15"))
DEPTH_MAX_M = float(os.environ.get("DEPTH_MAX_M", "1.6"))

# 已知相机 PID → 能力描述（新相机接入时补这张表即可）
#   tag                 device_id 尾缀（macmini-<tag>）
#   stereo              是否有双目 IR（决定是否推 /api/frame/aux）
#   nominal_baseline_mm SDK 读不到基线时的规格兜底值
KNOWN_CAMERAS = {
    0x0800: {"tag": "g335", "stereo": True, "nominal_baseline_mm": 50.0},   # Orbbec Gemini 335
    0x060F: {"tag": "astra", "stereo": False, "nominal_baseline_mm": None},  # Astra Pro Plus
}

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("cam-pusher")

_stop = threading.Event()
# SDK 设备枚举/开流不保证并发安全，统一串行化
_sdk_lock = threading.Lock()

# ── 取帧僵死看门狗（2026-08-13 事故：USB 掉线后 G335 线程卡死在 SDK 取帧调用里，
# 不进重连循环、插拔无效，只能重启进程。展台 USB 被碰掉是常态，必须自愈）──
# 三级防线：
#   1. 线程自检：wait_for_frames 正常返回但连续 FRAME_STALL_S 无帧（拔线后 SDK
#      往往安静返回 None）→ 线程自己 raise 走既有 5s 重连；
#   2. watchdog 强拆：线程连自检都没动静（卡死在 SDK 调用内）→ watchdog 线程
#      代为 stop pipeline（尽力唤醒阻塞调用）、置代数废弃标记、起新线程重连；
#      卡死旧线程即使醒不来也只是泄漏一条线程，好过整进程僵死；
#   3. 进程重生：连续 WATCHDOG_REBUILD_MAX 次线程级重建仍无帧（如 SDK 全局
#      mutex 烂掉，新线程也卡在 _sdk_lock/开流上）→ os._exit 让 LaunchDaemon
#      （KeepAlive）拉起整个进程，代价是两台相机同断几秒。
FRAME_STALL_S = 10.0      # 线程自检阈值：超过即主动重建取帧管线
WATCHDOG_STALL_S = 20.0   # watchdog 判卡死阈值：自检都没触发=线程卡死在 SDK 里
WATCHDOG_REBUILD_MAX = 3  # 连续线程级重建无效上限，超过进程自杀重生

_watch_lock = threading.Lock()
# device_id -> {"ts": 最近活跃时刻, "gen": 当前线程代数, "pipe": 当前 pipeline,
#               "rebuilds": 连续未见帧的重建次数, "pid": USB PID, "spec": 能力描述}
# ts 的语义是「线程最近一次证明自己活着」：取到帧 / 开流成功 / 重连循环报错都算；
# 只有取到帧才清零 rebuilds。设备长期不在线（如 astra 未插）时线程每 5s 报
# 「设备未找到」并 touch，不会误触发 watchdog。
_watch: dict = {}


def _watch_register(device_id: str, pid: int, spec: dict) -> None:
    with _watch_lock:
        _watch[device_id] = {"ts": time.time(), "gen": 0, "pipe": None,
                             "rebuilds": 0, "pid": pid, "spec": spec}


def _watch_alive(device_id: str, gen: int) -> bool:
    """本线程是否仍是该设备的当前代（watchdog 重建后旧代应尽快退出）。"""
    with _watch_lock:
        return _watch[device_id]["gen"] == gen


def _watch_touch(device_id: str, gen: int, pipe=None, frame: bool = False) -> None:
    """线程报活。frame=True 表示真取到了帧（清零重建计数）；开流成功时传 pipe
    供 watchdog 强拆用。过期代的报活直接忽略。"""
    with _watch_lock:
        w = _watch[device_id]
        if w["gen"] != gen:
            return
        w["ts"] = time.time()
        if pipe is not None:
            w["pipe"] = pipe
        if frame:
            w["rebuilds"] = 0


def _safe_stop(pipe, device_id: str) -> None:
    """watchdog 线程里尽力 stop 卡死线程持有的 pipeline（stop 本身也可能阻塞，
    所以由独立 daemon 线程执行，卡住就随它去）。"""
    try:
        pipe.stop()
        log.info("[%s] watchdog 已停掉疑似僵死的 pipeline", device_id)
    except Exception as e:
        log.warning("[%s] watchdog 停 pipeline 失败（不影响新线程重建）: %s",
                    device_id, e)


def _watchdog_loop():
    """每 3s 巡检各相机线程活跃时间戳，按三级防线处置（见模块头注释）。"""
    while not _stop.wait(3):
        now = time.time()
        with _watch_lock:
            snapshot = [(d, dict(w)) for d, w in _watch.items()]
        for dev, w in snapshot:
            if now - w["ts"] < WATCHDOG_STALL_S:
                continue
            if w["rebuilds"] >= WATCHDOG_REBUILD_MAX:
                log.error("[%s] 线程级重建 %d 次仍无帧，进程自杀重生"
                          "（由 LaunchDaemon KeepAlive 拉起）", dev, w["rebuilds"])
                logging.shutdown()
                os._exit(1)
            with _watch_lock:
                w2 = _watch[dev]
                if w2["gen"] != w["gen"] or now - w2["ts"] < WATCHDOG_STALL_S:
                    continue   # 巡检期间已被处理/已恢复
                w2["gen"] += 1
                w2["rebuilds"] += 1
                w2["ts"] = now          # 给新线程完整的观察窗口
                gen, n = w2["gen"], w2["rebuilds"]
                pipe, w2["pipe"] = w2["pipe"], None
            log.warning("[%s] %.0f 秒无任何线程活动（疑似卡死在 SDK 取帧调用），"
                        "触发线程级重建（第 %d/%d 次）",
                        dev, WATCHDOG_STALL_S, n, WATCHDOG_REBUILD_MAX)
            if pipe is not None:
                threading.Thread(target=_safe_stop, args=(pipe, dev),
                                 name=f"{dev}-force-stop", daemon=True).start()
            threading.Thread(target=_camera_worker,
                             args=(w["pid"], w["spec"], dev, gen),
                             name=f"{dev}-gen{gen}", daemon=True).start()

# 当前生效的推帧频率：以服务端配置为准（per-device 优先、全局旧口径兜底），
# 都没有时用 PUSH_FPS（见模块头注释）。dict 按 device_id 分桶，两台相机各调各的
_fps_lock = threading.Lock()
_dev_fps: dict = {}                   # device_id -> per-device push_fps（服务端下发）
_global_fps: float = 0.0              # 全局 config.push_fps（旧口径兜底；0=未下发）
_dev_depth: dict = {}                 # device_id -> 深度伪彩渲染配置（depth_* 键，服务端下发）
# device_id -> 设备点云推流状态：{"want": /experience 是否正选中该来源（服务端
# 按 10s TTL 续期的按需标志）, 以及可选的 pc_fps / pc_stride 覆盖值}
_dev_pc: dict = {}


def _valid_fps(raw) -> float:
    """把服务端字段解析成合法 fps；无效/未下发返回 0。"""
    try:
        fps = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return fps if fps > 0 else 0.0


def get_push_interval(device_id: str) -> float:
    """该设备当前推帧间隔（秒）。频率钳制在 0.2~30fps：防误配 0 值忙等或打爆链路。"""
    with _fps_lock:
        fps = _dev_fps.get(device_id) or _global_fps or PUSH_FPS
    return 1.0 / min(max(fps, 0.2), 30.0)


def get_depth_cfg(device_id: str) -> dict:
    """该设备当前的深度伪彩渲染配置（服务端 device-config 下发的 depth_* 键；
    未下发时空 dict=全部走默认值，即历史固定量程 TURBO 行为）。"""
    with _fps_lock:
        return dict(_dev_depth.get(device_id) or {})


def get_pc_cfg(device_id: str) -> dict:
    """该设备当前的设备点云推流状态（want 按需标志 + 可选 pc_* 覆盖值；
    空 dict=不推）。"""
    with _fps_lock:
        return dict(_dev_pc.get(device_id) or {})


def _config_poller():
    """每 2s 从 8060 读一次配置并热生效：devices[].config 里的 push_fps（per-device
    帧率）、depth_*（深度伪彩渲染参数），及全局 config.push_fps（旧口径兜底）。
    服务端不可达时沿用当前值。"""
    global _global_fps
    session = requests.Session()
    last_desc = None
    while not _stop.is_set():
        try:
            resp = session.get(f"{RELAY_URL}/api/frame/status", timeout=(3.05, 5))
            resp.raise_for_status()
            st = resp.json()
            g = _valid_fps((st.get("config") or {}).get("push_fps"))
            dev = {}
            ddep = {}
            dpc = {}
            for d in st.get("devices") or []:
                did = str(d.get("device_id") or "")
                cfg = d.get("config") or {}
                fps = _valid_fps(cfg.get("push_fps"))
                if fps:
                    dev[did] = fps
                dcfg = {k: v for k, v in cfg.items() if str(k).startswith("depth_")}
                if dcfg:
                    ddep[did] = dcfg
                pcc = {k: v for k, v in cfg.items() if str(k).startswith("pc_")}
                if d.get("pc_want"):
                    pcc["want"] = True
                if pcc:
                    dpc[did] = pcc
            with _fps_lock:
                _global_fps = g
                _dev_fps.clear()
                _dev_fps.update(dev)
                _dev_depth.clear()
                _dev_depth.update(ddep)
                _dev_pc.clear()
                _dev_pc.update(dpc)
            # 设备点云的按需标志只按「哪些设备在推」进 desc：want 翻转要打日志，
            # 但 pc_* 数值抖动不至于刷屏
            pc_on = tuple(sorted(d for d, c in dpc.items() if c.get("want")))
            desc = (g, tuple(sorted(dev.items())),
                    tuple(sorted((k, tuple(sorted(v.items()))) for k, v in ddep.items())),
                    pc_on)
            if desc != last_desc:
                if last_desc is not None or g or dev or ddep or pc_on:   # 启动即空配置不打日志
                    log.info("推帧配置随服务端调整: per-device帧率=%s 全局兜底=%s 深度渲染=%s"
                             " 设备点云=%s",
                             dev or "无", g or "无", ddep or "默认",
                             list(pc_on) or "关")
                last_desc = desc
        except (requests.RequestException, ValueError, TypeError):
            pass  # 服务端不可达/字段异常时沿用当前值，推帧线程自身会报连不上
        _stop.wait(2)


def _find_device(ctx: Context, pid: int):
    """按 PID 重新枚举并返回设备对象；找不到返回 None（相机被拔/未就绪）。"""
    devs = ctx.query_devices()
    for i in range(devs.get_count()):
        d = devs.get_device_by_index(i)
        if d.get_device_info().get_pid() == pid:
            return d
    return None


def _to_jpeg(frame) -> bytes:
    """彩色帧 → JPEG 字节。MJPG 直接透传；常见 raw 格式经 cv2 转码。"""
    fmt = frame.get_format()
    data = frame.get_data()
    if fmt == OBFormat.MJPG:
        return bytes(data)
    w, h = frame.get_width(), frame.get_height()
    buf = np.asarray(data, dtype=np.uint8)
    if fmt == OBFormat.RGB:
        img = cv2.cvtColor(buf.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
    elif fmt == OBFormat.BGR:
        img = buf.reshape(h, w, 3)
    elif fmt in (OBFormat.YUYV, OBFormat.YUY2):
        img = cv2.cvtColor(buf.reshape(h, w, 2), cv2.COLOR_YUV2BGR_YUY2)
    elif fmt == OBFormat.UYVY:
        img = cv2.cvtColor(buf.reshape(h, w, 2), cv2.COLOR_YUV2BGR_UYVY)
    elif fmt == OBFormat.NV12:
        img = cv2.cvtColor(buf.reshape(h * 3 // 2, w), cv2.COLOR_YUV2BGR_NV12)
    elif fmt == OBFormat.I420:
        img = cv2.cvtColor(buf.reshape(h * 3 // 2, w), cv2.COLOR_YUV2BGR_I420)
    else:
        raise ValueError(f"暂不支持的彩色格式: {fmt}")
    ok, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ValueError("JPEG 编码失败")
    return jpeg.tobytes()


# 深度伪彩可选色彩映射（gray 特殊处理不走 applyColorMap）；个别老版本 OpenCV
# 缺少的映射回落 TURBO
_DEPTH_CMAPS = {name: getattr(cv2, "COLORMAP_" + name.upper(), cv2.COLORMAP_TURBO)
                for name in ("turbo", "jet", "viridis", "plasma", "inferno", "magma",
                             "hot", "bone", "ocean", "hsv", "parula", "cividis",
                             "twilight_shifted", "deepgreen")}


# 自建色表档位表：{名字: (索引档位, RGB 档位)}，索引 0=远、255=近。
# 对齐用户给的参考图观感，单/双色系靠明度与色温拉开前后空间：
#   lidar    激光雷达蓝青——远深藏青、中蔚蓝、近亮青白
#   radar    雷达蓝绿·暖尖——蓝青为底，最近处绿→黄→橙红点睛（经典扫描仪配色）
#   indigo   靛蓝冰晶——黑→暗靛→长春花蓝→近处白（巨石阵扫描风）
#   moss     苔原绿——墨绿黑→苔绿→草绿→近处浅卡其（森林苔原风）
#   lavender 薰衣草雾——暗灰绿→灰蓝→长春花→近处雾白（柔和低对比）
_CUSTOM_CMAP_STOPS = {
    "lidar": ([0, 90, 170, 225, 255],
              [(3, 10, 40), (0, 60, 180), (0, 140, 255),
               (0, 220, 220), (170, 255, 225)]),
    "radar": ([0, 110, 170, 210, 235, 255],
              [(2, 8, 45), (0, 90, 220), (60, 190, 235),
               (80, 220, 120), (235, 220, 70), (250, 120, 40)]),
    "indigo": ([0, 90, 170, 225, 255],
               [(8, 8, 25), (60, 60, 140), (125, 125, 225),
                (200, 200, 245), (255, 255, 255)]),
    "moss": ([0, 90, 170, 220, 255],
             [(5, 12, 8), (30, 70, 40), (90, 150, 80),
              (150, 190, 110), (225, 220, 170)]),
    "lavender": ([0, 80, 150, 210, 255],
                 [(40, 50, 45), (90, 100, 120), (130, 140, 200),
                  (170, 175, 225), (245, 245, 250)]),
}


def _build_custom_luts() -> dict:
    """按档位表插值生成 256x3 BGR 查表（numpy 查表着色，不依赖 OpenCV 自定义色表接口）。"""
    idx = np.arange(256, dtype=np.float32)
    luts = {}
    for name, (stops_i, stops_rgb) in _CUSTOM_CMAP_STOPS.items():
        si = np.array(stops_i, dtype=np.float32)
        sc = np.array(stops_rgb, dtype=np.float32)
        rgb = np.stack([np.interp(idx, si, sc[:, c]) for c in range(3)], axis=1)
        luts[name] = np.ascontiguousarray(rgb[:, ::-1]).astype(np.uint8)   # RGB→BGR
    return luts


_CUSTOM_LUTS = _build_custom_luts()


def _cfg_num(cfg: dict, key: str, default: float) -> float:
    """读数值配置项；缺失/非法一律回默认值（服务端已钳制过范围，这里只做兜底）。"""
    try:
        return float(cfg[key])
    except (KeyError, TypeError, ValueError):
        return float(default)


def _depth_to_jpeg(frame, cfg: dict, state: dict) -> bytes:
    """硬件深度帧（uint16，单位由 depth_scale 折算毫米）→ 伪彩 JPEG。

    默认（无下发配置）= 展会现场调定口径（2026-08-13）：自动分位量程（8~89%）、
    gamma 0.65、全局直方图均衡、形态学填洞（半径 5px）、时域平滑 0.15、JPEG 95、
    深度独立 1.5fps；色彩仍为 TURBO 近亮远暗、无效点黑。服务端可经 device-config 下发 depth_* 键逐项
    覆盖：色彩映射/方向/量程（固定或分位自适应）/gamma/直方图均衡/孔洞填充/
    时空域滤波/边缘描边/等值线/无效点颜色/JPEG 质量（/experience「调节」抽屉可视化
    调节）。state 为该相机线程私有的渲染状态（时域 EMA 上一帧等），重连后重置。

    时域稳频（2026-08-14，治高帧率闪烁/低帧率换帧整图变色）：空洞用上一帧有效值
    垫底（一帧记忆）、自适应量程 lo/hi 跨帧 EMA、global 均衡只统计有效像素且
    查表跨帧 EMA——三处逐帧统计量全部带时域惯性，色彩映射不再逐帧漂移。"""
    w, h = frame.get_width(), frame.get_height()
    scale = float(getattr(frame, "get_depth_scale", lambda: 1.0)() or 1.0)
    d = np.frombuffer(bytes(frame.get_data()), dtype=np.uint16).reshape(h, w)
    d_mm = d.astype(np.float32) * scale
    invalid = d_mm <= 0

    # ── 时域空洞垫底（一帧记忆）：散斑空洞逐帧闪变（像素有效↔无效来回翻转）是
    #    高帧率下画面闪烁的主源之一——用上一帧真实有效的值垫住本帧空洞。只记
    #    一帧、只存真实观测值（不存垫底后的结果）；且仅在帧间隔 ≤0.6s 时生效——
    #    低帧率下上一帧太陈旧，拿来填运动物体轮廓洞会出彩色镶边（重影） ──
    fill = str(cfg.get("depth_fill", "close"))
    now_ts = time.time()
    hole_src = np.where(invalid, np.float32(0.0), d_mm)
    prev_hole = state.get("hole_prev")
    prev_ts = state.get("hole_ts", 0.0)
    if (fill != "off" and prev_hole is not None
            and prev_hole.shape == d_mm.shape and now_ts - prev_ts <= 0.6):
        m = invalid & (prev_hole > 0)
        d_mm[m] = prev_hole[m]
        invalid = d_mm <= 0
    state["hole_prev"] = hole_src
    state["hole_ts"] = now_ts

    # ── 时域平滑（EMA）：只混合两帧都有效的像素；分辨率变化即重置 ──
    ema = min(max(_cfg_num(cfg, "depth_ema", 0.15), 0.0), 0.95)
    prev = state.get("ema_prev")
    if ema > 0 and prev is not None and prev.shape == d_mm.shape:
        m = (~invalid) & (prev > 0)
        d_mm[m] = ema * prev[m] + (1.0 - ema) * d_mm[m]
    state["ema_prev"] = d_mm.copy() if ema > 0 else None

    # ── 孔洞填充（close 模式，作用于深度值本身）：形态学闭运算用邻域深度补 0 值空洞 ──
    fill_px = max(1, int(_cfg_num(cfg, "depth_fill_px", 5)))
    if fill == "close" and invalid.any():
        ker = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * fill_px + 1, 2 * fill_px + 1))
        closed = cv2.morphologyEx(d_mm, cv2.MORPH_CLOSE, ker)
        d_mm = np.where(invalid, closed, d_mm)
        invalid = d_mm <= 0

    # ── 量程归一化：固定量程（默认，防手/物进出画面时整图颜色跳变）或逐帧分位自适应 ──
    valid_vals = d_mm[~invalid]
    if _cfg_num(cfg, "depth_autorange", 1.0) >= 0.5 and valid_vals.size > 256:
        p_hi = _cfg_num(cfg, "depth_auto_hi", 89.0)
        p_lo = min(_cfg_num(cfg, "depth_auto_lo", 8.0), p_hi - 1.0)
        lo, hi = np.percentile(valid_vals, [p_lo, p_hi])
        # 量程时域平滑：分位数随手部进出/空洞波动逐帧小幅漂移，直接用会让整图
        # 颜色映射逐帧平移（高帧率=闪、低帧率=换帧整图变色），lo/hi 跨帧 EMA 压住
        prev_rng = state.get("rng_prev")
        if prev_rng is not None:
            lo = 0.85 * prev_rng[0] + 0.15 * lo
            hi = 0.85 * prev_rng[1] + 0.15 * hi
        state["rng_prev"] = (float(lo), float(hi))
    else:
        lo = _cfg_num(cfg, "depth_min_m", DEPTH_MIN_M) * 1000.0
        hi = _cfg_num(cfg, "depth_max_m", DEPTH_MAX_M) * 1000.0
    t = np.clip((hi - d_mm) / max(hi - lo, 1.0), 0.0, 1.0)   # 近→1（近亮远暗）
    if _cfg_num(cfg, "depth_invert", 0.0) >= 0.5:
        t = 1.0 - t
    gamma = min(max(_cfg_num(cfg, "depth_gamma", 0.65), 0.2), 5.0)
    if abs(gamma - 1.0) > 1e-3:
        t = np.power(t, 1.0 / gamma)
    t8 = (t * 255.0).astype(np.uint8)

    # ── 孔洞填充（inpaint 模式，作用于归一化灰度图）：修补后不再视为无效点 ──
    if fill == "inpaint" and invalid.any():
        t8 = cv2.inpaint(t8, invalid.astype(np.uint8),
                         float(min(fill_px, 10)), cv2.INPAINT_TELEA)
        invalid = np.zeros_like(invalid)

    # ── 直方图均衡：global 模式只统计有效像素并对均衡查表做跨帧 EMA——无效散斑
    #    占比逐帧波动会推移整条 CDF，是逐帧统计操作里最大的闪烁贡献者 ──
    eq = str(cfg.get("depth_eq", "global"))
    if eq == "global":
        vals = t8[~invalid]
        if vals.size:
            cdf = np.bincount(vals.ravel(), minlength=256).astype(np.float64).cumsum()
            lut = np.clip(cdf * (255.0 / cdf[-1]), 0.0, 255.0)
        else:
            lut = np.arange(256, dtype=np.float64)
        prev_lut = state.get("eq_lut")
        if prev_lut is not None:
            lut = 0.85 * prev_lut + 0.15 * lut
        state["eq_lut"] = lut
        t8 = lut.astype(np.uint8)[t8]
    elif eq == "clahe":
        clip = min(max(_cfg_num(cfg, "depth_eq_clip", 2.5), 0.5), 10.0)
        t8 = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(t8)

    # ── 空域滤波 ──
    smooth = str(cfg.get("depth_smooth", "off"))
    if smooth == "median":
        t8 = cv2.medianBlur(t8, 5)
    elif smooth == "bilateral":
        t8 = cv2.bilateralFilter(t8, 7, 50, 50)

    # ── 伪彩映射 ──
    cmap = str(cfg.get("depth_colormap", "turbo"))
    if cmap == "gray":
        img = cv2.cvtColor(t8, cv2.COLOR_GRAY2BGR)
    elif cmap in _CUSTOM_LUTS:
        img = _CUSTOM_LUTS[cmap][t8]   # numpy 查表着色
    else:
        img = cv2.applyColorMap(t8, _DEPTH_CMAPS.get(cmap, cv2.COLORMAP_TURBO))

    # ── 等值线：按真实距离取模的提亮细带，只画有效像素 ──
    contour_m = _cfg_num(cfg, "depth_contour_m", 0.0)
    if contour_m > 0.01:
        band = (np.mod(d_mm / (contour_m * 1000.0), 1.0) < 0.06) & (~invalid)
        img[band] = (img[band].astype(np.float32) * 0.35 + 255.0 * 0.65).astype(np.uint8)

    # ── 深度边缘描边：Sobel 梯度大处向白色混合，强度可调 ──
    edge = min(max(_cfg_num(cfg, "depth_edge", 0.0), 0.0), 1.0)
    if edge > 0.01:
        gx = cv2.convertScaleAbs(cv2.Sobel(t8, cv2.CV_16S, 1, 0, ksize=3))
        gy = cv2.convertScaleAbs(cv2.Sobel(t8, cv2.CV_16S, 0, 1, ksize=3))
        em = cv2.addWeighted(gx, 0.5, gy, 0.5, 0) > 48
        img[em] = (img[em].astype(np.float32) * (1.0 - edge)
                   + 255.0 * edge).astype(np.uint8)

    # ── 无效点上色（默认黑，与历史行为一致） ──
    color = str(cfg.get("depth_invalid_color", "#000000"))
    try:
        bgr = (int(color[5:7], 16), int(color[3:5], 16), int(color[1:3], 16))
    except (ValueError, IndexError):
        bgr = (0, 0, 0)
    img[invalid] = bgr

    q = int(min(max(_cfg_num(cfg, "depth_jpeg_q", 95), 30), 95))
    ok, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    if not ok:
        raise ValueError("深度图 JPEG 编码失败")
    return jpeg.tobytes()


def _ir_to_gray(frame) -> np.ndarray:
    """IR 帧 → uint8 灰度矩阵。G335 左右 IR 默认 Y8；Y16 高位截取兼容。"""
    fmt = frame.get_format()
    w, h = frame.get_width(), frame.get_height()
    if fmt == OBFormat.Y8:
        return np.asarray(frame.get_data(), dtype=np.uint8).reshape(h, w)
    if fmt == OBFormat.Y16:
        raw = np.frombuffer(bytes(frame.get_data()), dtype=np.uint16).reshape(h, w)
        return (raw >> 8).astype(np.uint8)
    raise ValueError(f"暂不支持的 IR 格式: {fmt}")


def _gray_to_jpeg(gray: np.ndarray) -> bytes:
    ok, jpeg = cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ValueError("IR JPEG 编码失败")
    return jpeg.tobytes()


def _laser_status(frame):
    """帧元数据里的激光状态：1=有散斑 0=无散斑 None=读不到（固件/绑定不支持）。"""
    try:
        if frame.has_metadata(OBFrameMetadataType.LASER_STATUS):
            return int(frame.get_metadata_value(OBFrameMetadataType.LASER_STATUS))
    except Exception:
        pass
    return None


def _setup_laser(dev, device_id: str) -> str:
    """按 LASER_MODE 配置 G335 投射器，返回实际生效模式（interleave 失败退 on）。"""
    mode = LASER_MODE if LASER_MODE in ("interleave", "on", "off") else "interleave"
    try:
        if mode == "off":
            dev.set_bool_property(OBPropertyID.OB_PROP_LASER_BOOL, False)
        elif mode == "on":
            dev.set_bool_property(OBPropertyID.OB_PROP_LASER_BOOL, True)
        else:
            # 逐帧交替开关投射器：奇偶帧一开一关，配合帧元数据 LASER_STATUS 分拣
            dev.set_bool_property(OBPropertyID.OB_PROP_LASER_BOOL, True)
            dev.set_int_property(
                OBPropertyID.OB_PROP_LASER_ON_OFF_PATTERN_INT, 1)
        log.info("[%s] 激光策略生效: %s", device_id, mode)
        return mode
    except Exception as e:
        if mode == "interleave":
            log.warning("[%s] 激光交错不可用（%s），退回常开散斑模式", device_id, e)
            try:
                dev.set_bool_property(OBPropertyID.OB_PROP_LASER_BOOL, True)
            except Exception:
                pass
            return "on"
        log.warning("[%s] 激光模式 %s 设置失败: %s", device_id, mode, e)
        return mode


def _read_baseline_mm(dev, spec) -> float:
    """从 SDK 标定读双目基线（毫米）；读不到用规格兜底值。"""
    try:
        b = dev.get_baseline()
        for attr in ("baseline", "get_baseline"):
            v = getattr(b, attr, None)
            v = v() if callable(v) else v
            if v:
                return abs(float(v))
    except Exception:
        pass
    return spec.get("nominal_baseline_mm")


def _open_pipeline(dev, spec, device_id: str):
    """开流：彩色必开；深度尽力开；stereo 相机加开左右 IR。

    显式取 default profile——简写 enable_stream(sensor) 在 1.3.2 绑定上会段错误，
    勿改回。辅助流开不出来只降级该流并告警，彩色主链路不受影响。
    返回 (已 start 的 Pipeline, 实际开出的流集合)。
    """
    pipe = Pipeline(dev)
    cfg = Config()
    profiles = pipe.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    cfg.enable_stream(profiles.get_default_video_stream_profile())
    enabled = {"color"}
    wanted = [("depth", OBSensorType.DEPTH_SENSOR)]
    if spec["stereo"]:
        wanted = [("left", OBSensorType.LEFT_IR_SENSOR),
                  ("right", OBSensorType.RIGHT_IR_SENSOR)] + wanted
    for name, sensor in wanted:
        try:
            pl = pipe.get_stream_profile_list(sensor)
            cfg.enable_stream(pl.get_default_video_stream_profile())
            enabled.add(name)
        except Exception as e:
            log.warning("[%s] %s 流开启失败，降级跳过: %s", device_id, name, e)
    pipe.start(cfg)
    try:
        pipe.enable_frame_sync()   # 多流硬件同步：同一 frameset 内各帧对齐（Astra 不支持，降级）
    except Exception as e:
        log.warning("[%s] frame_sync 开启失败（不影响推帧）: %s", device_id, e)
    return pipe, enabled


def _get_typed_frame(fs, frame_type):
    """从 frameset 取指定类型帧并转 video frame；无则返回 None。"""
    try:
        f = fs.get_frame(frame_type)
        return f.as_video_frame() if f is not None else None
    except Exception:
        return None


class _IRCache:
    """双目 IR 滚动缓存：留最新的"干净 IR 对"，推送节拍取用编码。

    interleave 模式下无光帧才可用（按 LASER_STATUS 分拣），与深度帧交替出现，必须
    缓存；on/off 模式退化为"总是最新帧"。元数据读不到时不做分拣（None 视同可用）。"""

    def __init__(self):
        self.left = None      # (uint8 灰度矩阵, 单调时间)
        self.right = None

    def offer(self, left_f, right_f, laser_mode: str):
        if left_f is None or right_f is None:
            return
        if laser_mode == "interleave" and _laser_status(left_f) == 1:
            return               # 有散斑的 IR 不喂 DA3
        now = time.time()
        self.left = (_ir_to_gray(left_f), now)
        self.right = (_ir_to_gray(right_f), now)

    def snapshot(self, max_age: float):
        """取当前可用的 IR 对（超龄的丢弃，防拔线后推陈旧帧）。"""
        now = time.time()
        if self.left and self.right and now - self.left[1] <= max_age:
            return self.left[0], self.right[0]
        return None, None


def _push_aux(session, device_id: str, spec, cache: _IRCache,
              baseline_mm, laser_mode: str, now: float) -> bool:
    """推双目辅助帧到 /api/frame/aux。当前没有可用 IR 对时跳过（返回 False）。"""
    # 超龄阈值 = 3 个推帧周期：既容忍 interleave 半率出帧，又不至于推太陈旧的帧
    left, right = cache.snapshot(max_age=3 * get_push_interval(device_id))
    if left is None or right is None:
        return False
    info = {"device_id": device_id, "source": "mac-mini",
            "stereo_supported": True, "laser_mode": laser_mode}
    if baseline_mm:
        info["baseline_mm"] = float(baseline_mm)
    resp = session.post(
        f"{RELAY_URL}/api/frame/aux",
        files={"left": ("left.jpg", _gray_to_jpeg(left), "image/jpeg"),
               "right": ("right.jpg", _gray_to_jpeg(right), "image/jpeg")},
        data={"camera_info": json.dumps(info), "timestamp": str(int(now * 1000))},
        timeout=(3.05, 8),
    )
    resp.raise_for_status()
    return True


def _open_pc_context(pipe, streams, device_id: str):
    """创建设备点云推流上下文：D2C 对齐滤镜 + 彩色相机内参。

    仅同时开出彩色与深度流的相机可用；标定/滤镜拿不到时返回 None（该设备
    不推点云，其余链路不受影响）。对齐用软件 AlignFilter 而非硬件 D2C 模式：
    硬件模式会改动整条深度流的几何，影响现有伪彩深度链路，软件滤镜只在
    点云节拍上对单个帧组做一次对齐，原始深度帧原样保留。"""
    if not ("color" in streams and "depth" in streams):
        return None
    try:
        intr = pipe.get_camera_param().rgb_intrinsic
        if not (intr.fx > 0 and intr.fy > 0):
            raise RuntimeError(f"彩色内参非法：fx={intr.fx} fy={intr.fy}")
        align = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
        return {"align": align, "intr": intr, "err": 0, "logged": False}
    except Exception as e:
        log.warning("[%s] 设备点云上下文创建失败（该设备不推点云）: %s", device_id, e)
        return None


def _push_devpc(session, device_id: str, pc_ctx: dict, fs, cf, pcc: dict,
                now: float) -> None:
    """推一帧设备点云原料到 /api/devpc/frame：帧组过 D2C 对齐取「彩色相机坐标系
    的 uint16 深度」，按 stride 降采样后无损 PNG 上报，同帧 RGB JPEG + 内参随行。
    5090 端（app.py）反投影成彩色点云 GLB 供 /experience「设备点云」背景。"""
    out = pc_ctx["align"].process(fs)
    if out is None:
        raise RuntimeError("D2C 对齐输出为空")
    # SDK 版本差异：process 可能返回 Frame（需转 frameset）或直接返回 FrameSet
    afs = out.as_frame_set() if hasattr(out, "as_frame_set") else out
    adf = afs.get_depth_frame()
    if adf is None:
        raise RuntimeError("对齐后帧组无深度帧")
    w, h = adf.get_width(), adf.get_height()
    scale = float(getattr(adf, "get_depth_scale", lambda: 1.0)() or 1.0)
    d16 = np.frombuffer(bytes(adf.get_data()), dtype=np.uint16).reshape(h, w)
    stride = int(_cfg_num(pcc, "pc_stride", 0))
    if stride < 1:
        # 自适应步长：降到 ~424 列（G335 1280 宽 → stride 3 → 427x240 ≈ 10 万点），
        # 点数够背景观感、PNG 体积与 5090 端构建耗时都可控
        stride = max(1, round(w / 424))
    ok, png = cv2.imencode(".png", np.ascontiguousarray(d16[::stride, ::stride]))
    if not ok:
        raise RuntimeError("深度 PNG 编码失败")
    intr = pc_ctx["intr"]
    # 内参标定分辨率与实际流分辨率不一致时按比例缩放（对齐后深度=彩色流分辨率）
    sx = (w / intr.width) if intr.width else 1.0
    sy = (h / intr.height) if intr.height else 1.0
    meta = {"device_id": device_id, "width": w, "height": h, "stride": stride,
            "depth_scale": scale, "fx": intr.fx * sx, "fy": intr.fy * sy,
            "cx": intr.cx * sx, "cy": intr.cy * sy}
    resp = session.post(
        f"{RELAY_URL}/api/devpc/frame",
        files={"depth": ("depth.png", png.tobytes(), "image/png"),
               "rgb": ("rgb.jpg", _to_jpeg(cf), "image/jpeg")},
        data={"meta": json.dumps(meta)},
        timeout=(3.05, 8),
    )
    resp.raise_for_status()
    if not pc_ctx["logged"]:
        log.info("[%s] 设备点云开始推流：对齐深度 %dx%d stride=%d fx=%.1f fy=%.1f",
                 device_id, w, h, stride, meta["fx"], meta["fy"])
        pc_ctx["logged"] = True


def _camera_worker(pid: int, spec: dict, device_id: str, gen: int = 0):
    """单相机工作线程：开流 → 节流推帧（RGB+深度 / 双目 aux）→ 出错重连，直到
    进程退出或本线程代数被 watchdog 废弃（gen 落后即静默退出，由新代接管）。"""
    ctx = None
    session = requests.Session()
    push_err = 0
    conn_err = 0   # 连续开流失败计数（成功开流即清零），触发 Context 级重建
    while not _stop.is_set() and _watch_alive(device_id, gen):
        pipe = None
        try:
            with _sdk_lock:
                if ctx is None:
                    ctx = Context()
                dev = _find_device(ctx, pid)
                if dev is None:
                    raise RuntimeError("设备未找到（未插 / 被占用 / 尚未就绪）")
                pipe, streams = _open_pipeline(dev, spec, device_id)
                laser_mode = _setup_laser(dev, device_id) if spec["stereo"] else "off"
                baseline_mm = _read_baseline_mm(dev, spec) if spec["stereo"] else None
                # 设备点云上下文（D2C 对齐 + 彩色内参）：拿不到只关本路，不影响其余链路
                pc_ctx = _open_pc_context(pipe, streams, device_id)
            log.info("[%s] 已启动，流=%s 基线=%s", device_id, sorted(streams), baseline_mm)
            conn_err = 0
            # 开流成功即报活并登记 pipe（watchdog 强拆用）；给出帧留满一个自检窗口
            _watch_touch(device_id, gen, pipe=pipe)
            last_fs = time.time()
            last_cf = time.time()   # 最近一次拿到彩色帧的时刻（color 流僵死自检用）
            first = True
            first_depth = True
            depth_err = 0
            laser_seen_meta = False   # 本次连接内是否成功读到过 LASER_STATUS 元数据
            laser_skip = 0            # 垃圾深度帧（无光/元数据缺失）整帧跳过累计
            aux_err_logged = False
            last_push = 0.0
            cache = _IRCache()
            # 深度渲染线程私有状态（EMA 上一帧 / 上次深度推送时刻），重连即重置
            depth_state: dict = {}
            # 设备点云线程私有状态（上次推送时刻），重连即重置
            pc_state: dict = {}
            while not _stop.is_set():
                fs = pipe.wait_for_frames(500)
                if not _watch_alive(device_id, gen):
                    return   # 本代已被 watchdog 废弃：新线程已接管，静默退出
                if fs is None:
                    # 线程自检（第一级防线）：SDK 活着但长时间不出帧——拔线后
                    # wait_for_frames 常安静返回 None、pipeline 已失效，原地等待
                    # 插回也收不到帧，必须重建管线
                    if time.time() - last_fs > FRAME_STALL_S:
                        raise RuntimeError(
                            f"连续 {FRAME_STALL_S:.0f} 秒无帧，主动重建取帧管线")
                    continue
                last_fs = time.time()
                _watch_touch(device_id, gen, frame=True)
                cf = fs.get_color_frame()
                # 双目 IR 每个 frameset 都收进缓存（interleave 下 IR/深度交替出现，
                # 只在推送节拍看当前帧组会漏掉一半帧源）
                if "left" in streams and "right" in streams:
                    try:
                        cache.offer(_get_typed_frame(fs, OBFrameType.LEFT_IR_FRAME),
                                    _get_typed_frame(fs, OBFrameType.RIGHT_IR_FRAME),
                                    laser_mode)
                    except Exception as e:
                        if not aux_err_logged:
                            log.warning("[%s] 双目帧提取异常（只降级不中断）: %s",
                                        device_id, e)
                            aux_err_logged = True
                if cf is None:
                    # 看门狗盲区自检：帧组持续到达但一直没有彩色帧（如开流时
                    # uvc_stream_open_ctrl 失败、深度/IR 仍在流动——帧组会把
                    # 两级看门狗全喂活，线程却什么都不推、什么都不报）。彩色流
                    # 属于预期流时超时即重建管线（interleave 偶发缺帧不受影响）
                    if "color" in streams and time.time() - last_cf >= FRAME_STALL_S:
                        raise RuntimeError(
                            f"连续 {FRAME_STALL_S:.0f} 秒帧组无彩色帧"
                            "（color 流疑似僵死），主动重建取帧管线")
                    continue
                last_cf = time.time()
                if first:
                    log.info("[%s] 首帧 %dx%d 格式=%s", device_id,
                             cf.get_width(), cf.get_height(), cf.get_format())
                    first = False
                now = time.time()
                # ── RGB 与深度两路独立节拍：RGB 按 push_fps；深度 depth_fps>0 时按
                # 自己的节拍（可高于 RGB，RGB 节拍之间发 depth-only 上报），=0 跟随
                # RGB（历史行为）。两路都未到节拍则跳过本帧组 ──
                dcfg = get_depth_cfg(device_id)
                dfps = _cfg_num(dcfg, "depth_fps", 1.5)
                dep_itv = (1.0 / min(max(dfps, 0.2), 30.0)) if dfps > 0 else None
                rgb_due = now - last_push >= get_push_interval(device_id)
                dep_due = (now - depth_state.get("last_push", 0.0) >= dep_itv
                           ) if dep_itv else rgb_due
                # 设备点云独立节拍：仅在服务端按需标志（want）亮着时推
                pcc = get_pc_cfg(device_id)
                pc_itv = 1.0 / min(max(_cfg_num(pcc, "pc_fps", 1.5), 0.2), 10.0)
                pc_due = (pc_ctx is not None and bool(pcc.get("want"))
                          and now - pc_state.get("last_push", 0.0) >= pc_itv)
                if not rgb_due and not dep_due and not pc_due:
                    continue
                # 深度帧可用性：同帧组里带硬件深度帧则伪彩上报（帧组偶尔缺深度帧属
                # 正常，面板端保留上一张；伪彩失败只丢深度不影响彩色帧）。interleave
                # 下只用有散斑帧的深度（无光帧深度质量差）
                files = {}
                if rgb_due:
                    files["image"] = ("frame.jpg", _to_jpeg(cf), "image/jpeg")
                df = fs.get_depth_frame() if "depth" in streams else None
                # interleave 垃圾帧整帧跳过：只收明确有散斑（LASER_STATUS==1）的
                # 深度帧；无光帧（0）与元数据缺失帧（-1）一律丢弃——宁可僵住上一帧
                # 也不上垃圾帧（全屏暗闪嫌疑）。例外：本次连接从未读到过该元数据
                # （固件不支持，读永远是 -1）时放行，避免深度整路断流
                # 激光门控结论对伪彩深度与设备点云两路共用：interleave 下只认
                # 明确有散斑（LASER_STATUS==1）的深度帧
                laser_ok = df is not None
                if laser_ok and spec["stereo"] and laser_mode == "interleave":
                    ls = _laser_status(df)
                    if ls >= 0:
                        laser_seen_meta = True
                    if ls == 0:
                        laser_ok = False   # 无光帧：interleave 正常另一半，静默跳过
                    elif ls != 1 and laser_seen_meta:
                        # 元数据缺失(-1)但该固件明明支持——旧口径会把这种帧当好帧
                        # 放行（疑似全屏暗闪来源），现整帧跳过（僵住上一帧）并留证据。
                        # 固件完全不支持该元数据（从未读到过）时仍放行，避免深度断流
                        laser_ok = False
                        laser_skip += 1
                        if laser_skip == 1 or laser_skip % 20 == 0:
                            log.warning("[%s] 深度垃圾帧整帧跳过（激光状态=%d）累计 %d",
                                        device_id, ls, laser_skip)
                depth_use = dep_due and laser_ok
                if depth_use:
                    try:
                        files["depth"] = ("depth.jpg",
                                          _depth_to_jpeg(df, dcfg, depth_state),
                                          "image/jpeg")
                        depth_state["last_push"] = now
                        if first_depth:
                            log.info("[%s] 首帧深度 %dx%d 默认量程 %.1f~%.1fm"
                                     "（渲染参数按服务端 device-config 覆盖）",
                                     device_id, df.get_width(), df.get_height(),
                                     DEPTH_MIN_M, DEPTH_MAX_M)
                            first_depth = False
                    except Exception as e:
                        depth_err += 1
                        if depth_err == 1 or depth_err % 100 == 0:
                            log.warning("[%s] 深度帧伪彩失败 %d 次: %s",
                                        device_id, depth_err, e)
                # ── 设备点云按需推流（独立请求，失败只降级本路不影响主链路）：
                # 需散斑深度帧；无光/垃圾帧不更新节拍时刻，下一帧组即重试 ──
                if pc_due and laser_ok:
                    try:
                        _push_devpc(session, device_id, pc_ctx, fs, cf, pcc, now)
                        pc_ctx["err"] = 0
                        pc_state["last_push"] = now
                    except Exception as e:
                        pc_ctx["err"] += 1
                        pc_state["last_push"] = now  # 失败也按节奏走，避免忙等打爆链路
                        if pc_ctx["err"] == 1 or pc_ctx["err"] % 30 == 0:
                            log.warning("[%s] 设备点云推送失败 %d 次: %s",
                                        device_id, pc_ctx["err"], e)
                if not files:
                    # 深度节拍到了但本帧组无可用深度帧（interleave 无光帧等）：
                    # 等下一帧组，不空跑请求
                    continue
                try:
                    resp = session.post(
                        f"{RELAY_URL}/api/frame",
                        files=files,
                        data={
                            "camera_info": json.dumps(
                                {"device_id": device_id, "source": "mac-mini",
                                 "stereo_supported": bool(spec["stereo"])}),
                            "timestamp": str(int(now * 1000)),
                        },
                        timeout=(3.05, 5),
                    )
                    resp.raise_for_status()
                    # 双目 aux 跟着 RGB 主帧同一节拍推：失败只告警，不影响主链路
                    if rgb_due and spec["stereo"]:
                        try:
                            _push_aux(session, device_id, spec, cache,
                                      baseline_mm, laser_mode, now)
                        except Exception as e:
                            log.warning("[%s] aux 推送失败（主链路不受影响）: %s",
                                        device_id, e)
                    if push_err:
                        log.info("[%s] 推帧恢复正常", device_id)
                        push_err = 0
                    if rgb_due:
                        last_push = now
                except requests.RequestException as e:
                    push_err += 1
                    # 目标不可达时不刷屏：首次与每 30 次记一条
                    if push_err == 1 or push_err % 30 == 0:
                        log.warning("[%s] 推帧失败 %d 次: %s", device_id, push_err, e)
                    if rgb_due:
                        last_push = now  # 失败也按节奏走，避免忙等打爆目标
        except Exception as e:
            conn_err += 1
            log.warning("[%s] 相机链路异常，5 秒后重连: %s", device_id, e)
            if conn_err % 6 == 0:
                # 连续 ~30s 连不上：丢弃旧 Context 整个重建。重插 USB 后设备以
                # 新身份重新枚举，旧 Context 认不到新设备、会永远报
                # 「Create vendor command failed」（2026-08-14 真实发生），
                # 只有换全新 Context 才能接上重插后的相机
                log.warning("[%s] 连续 %d 次链路异常，重建 SDK Context",
                            device_id, conn_err)
                ctx = None
            # 重连等待也算线程活着（设备长期不在线属正常等待，不触发 watchdog）
            _watch_touch(device_id, gen)
            _stop.wait(5)
        finally:
            if pipe is not None:
                try:
                    pipe.stop()
                except Exception:
                    pass


def main():
    signal.signal(signal.SIGTERM, lambda *_: _stop.set())
    signal.signal(signal.SIGINT, lambda *_: _stop.set())
    log.info("cam-pusher 启动：目标=%s 兜底频率=%.1ffps 质量=%d 激光=%s"
             "（实际频率按设备跟随服务端 device-config，两台相机各调各的）",
             RELAY_URL, PUSH_FPS, JPEG_QUALITY, LASER_MODE)
    threading.Thread(target=_config_poller, name="config-poller",
                     daemon=True).start()

    # 启动时枚举一次已知相机；没插的也起线程（线程内会持续等它出现）
    with _sdk_lock:
        ctx = Context()
        present = set()
        devs = ctx.query_devices()
        for i in range(devs.get_count()):
            info = devs.get_device_by_index(i).get_device_info()
            log.info("发现设备: %s pid=%#06x sn=%s", info.get_name(),
                     info.get_pid(), info.get_serial_number())
            present.add(info.get_pid())
        del ctx  # 枚举用完即弃，工作线程各自持有独立 Context

    threads = []
    for pid, spec in KNOWN_CAMERAS.items():
        device_id = f"macmini-{spec['tag']}"
        if pid not in present:
            log.warning("[%s] 启动时未发现（pid=%#06x），线程将持续等待接入",
                        device_id, pid)
        _watch_register(device_id, pid, spec)
        t = threading.Thread(target=_camera_worker, args=(pid, spec, device_id, 0),
                             name=device_id, daemon=True)
        t.start()
        threads.append(t)
    # 取帧僵死看门狗：巡检各相机线程活跃度，卡死即线程级重建、屡建无效即进程重生
    threading.Thread(target=_watchdog_loop, name="frame-watchdog",
                     daemon=True).start()

    while not _stop.is_set():
        _stop.wait(1)
    log.info("收到退出信号，停止推帧")


if __name__ == "__main__":
    main()
