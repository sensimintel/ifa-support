# -*- coding: utf-8 -*-
"""mac mini 摄像头推帧器（浅体验区展会拓扑的帧源之一）。

用 pyorbbecsdk 抓取 mini 上插着的 Orbbec 相机，按 8060 的入帧契约推给 5090 的
da3-web 服务。每台相机独立成一个 device_id 桶，/panel 下拉可切换。

每个推帧节拍推两路（同一个节拍 → 同设备两路 fps 天然一致）：
  1. RGB 彩色帧 → POST /api/frame（契约与手机 App 一致，8060 单目链路零改动）；
     带硬件深度的相机同请求附可选字段 depth——深度帧在 mini 端做固定量程伪彩
     （近亮暖/远暗冷/无效点黑，默认 0.2~2m 防手/物进出画面时整图颜色跳变），
     供 /panel「原设备深度图」格展示，不参与 DA3 处理；
  2. 辅助帧    → POST /api/frame/aux（仅双目相机）：左 IR + 右 IR 灰度 JPEG，
     camera_info 带 stereo_supported / baseline_mm / laser_mode，
     供 8060 的双目 DA3 点云链路。

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
from pyorbbecsdk import (Config, Context, OBFormat, OBFrameMetadataType,
                         OBFrameType, OBPropertyID, OBSensorType, Pipeline)

RELAY_URL = os.environ.get("RELAY_URL", "http://192.168.0.50:8060").rstrip("/")
PUSH_FPS = float(os.environ.get("PUSH_FPS", "3"))
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "80"))
LASER_MODE = os.environ.get("LASER_MODE", "interleave").strip().lower()
# 硬件深度图伪彩的固定量程（米）：固定而非逐帧 min/max 自适应——展台上手/物进出画面
# 时整图颜色才不会跳变闪烁。桌面演示场景默认 0.2~2m，按展台纵深经 .env 调整。
DEPTH_MIN_M = float(os.environ.get("DEPTH_MIN_M", "0.2"))
DEPTH_MAX_M = float(os.environ.get("DEPTH_MAX_M", "2.0"))

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

# 当前生效的推帧频率：以服务端配置为准（per-device 优先、全局旧口径兜底），
# 都没有时用 PUSH_FPS（见模块头注释）。dict 按 device_id 分桶，两台相机各调各的
_fps_lock = threading.Lock()
_dev_fps: dict = {}                   # device_id -> per-device push_fps（服务端下发）
_global_fps: float = 0.0              # 全局 config.push_fps（旧口径兜底；0=未下发）


def _valid_fps(raw) -> float:
    """把服务端字段解析成合法 fps；无效/未下发返回 0。"""
    try:
        fps = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return fps if fps > 0 else 0.0


def get_push_interval(device_id: str) -> float:
    """该设备当前推帧间隔（秒）。频率钳制在 0.2~15fps：防误配 0 值忙等或打爆链路。"""
    with _fps_lock:
        fps = _dev_fps.get(device_id) or _global_fps or PUSH_FPS
    return 1.0 / min(max(fps, 0.2), 15.0)


def _config_poller():
    """每 2s 从 8060 读一次帧率配置并热生效：devices[].config.push_fps（per-device）
    与全局 config.push_fps（旧口径兜底）。服务端不可达时沿用当前值。"""
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
            for d in st.get("devices") or []:
                fps = _valid_fps((d.get("config") or {}).get("push_fps"))
                if fps:
                    dev[str(d.get("device_id") or "")] = fps
            with _fps_lock:
                _global_fps = g
                _dev_fps.clear()
                _dev_fps.update(dev)
            desc = (g, tuple(sorted(dev.items())))
            if desc != last_desc:
                if last_desc is not None or g or dev:   # 启动即空配置不打日志
                    log.info("推帧频率随服务端配置调整: per-device=%s 全局兜底=%s",
                             dev or "无", g or "无")
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


def _depth_to_jpeg(frame) -> bytes:
    """硬件深度帧（uint16，单位由 depth_scale 折算毫米）→ 伪彩 JPEG。
    固定量程归一化：近 → 亮/暖（TURBO 红端），远 → 暗/冷，无效点（0 值）→ 黑。"""
    w, h = frame.get_width(), frame.get_height()
    scale = float(getattr(frame, "get_depth_scale", lambda: 1.0)() or 1.0)
    d = np.frombuffer(bytes(frame.get_data()), dtype=np.uint16).reshape(h, w)
    d_mm = d.astype(np.float32) * scale
    lo, hi = DEPTH_MIN_M * 1000.0, DEPTH_MAX_M * 1000.0
    t = np.clip((hi - d_mm) / max(hi - lo, 1.0), 0.0, 1.0)
    img = cv2.applyColorMap((t * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[d == 0] = 0
    ok, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
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


def _camera_worker(pid: int, spec: dict, device_id: str):
    """单相机工作线程：开流 → 节流推帧（RGB+深度 / 双目 aux）→ 出错重连，直到进程退出。"""
    ctx = None
    session = requests.Session()
    push_err = 0
    while not _stop.is_set():
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
            log.info("[%s] 已启动，流=%s 基线=%s", device_id, sorted(streams), baseline_mm)
            first = True
            first_depth = True
            depth_err = 0
            aux_err_logged = False
            last_push = 0.0
            cache = _IRCache()
            while not _stop.is_set():
                fs = pipe.wait_for_frames(500)
                if fs is None:
                    continue
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
                    continue
                if first:
                    log.info("[%s] 首帧 %dx%d 格式=%s", device_id,
                             cf.get_width(), cf.get_height(), cf.get_format())
                    first = False
                now = time.time()
                if now - last_push < get_push_interval(device_id):
                    continue
                jpeg = _to_jpeg(cf)
                # 同帧组里带硬件深度帧则伪彩后随同上报（帧组偶尔缺深度帧属正常，
                # 面板端保留上一张；伪彩失败只丢深度不影响彩色帧）。interleave 下
                # 只用有散斑帧的深度（无光帧深度质量差）
                files = {"image": ("frame.jpg", jpeg, "image/jpeg")}
                df = fs.get_depth_frame() if "depth" in streams else None
                if df is not None and not (spec["stereo"] and laser_mode == "interleave"
                                           and _laser_status(df) == 0):
                    try:
                        files["depth"] = ("depth.jpg", _depth_to_jpeg(df), "image/jpeg")
                        if first_depth:
                            log.info("[%s] 首帧深度 %dx%d 量程 %.1f~%.1fm", device_id,
                                     df.get_width(), df.get_height(),
                                     DEPTH_MIN_M, DEPTH_MAX_M)
                            first_depth = False
                    except Exception as e:
                        depth_err += 1
                        if depth_err == 1 or depth_err % 100 == 0:
                            log.warning("[%s] 深度帧伪彩失败 %d 次: %s",
                                        device_id, depth_err, e)
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
                    # 双目 aux 跟着主帧同一节拍推：失败只告警，不影响主链路
                    if spec["stereo"]:
                        try:
                            _push_aux(session, device_id, spec, cache,
                                      baseline_mm, laser_mode, now)
                        except Exception as e:
                            log.warning("[%s] aux 推送失败（主链路不受影响）: %s",
                                        device_id, e)
                    if push_err:
                        log.info("[%s] 推帧恢复正常", device_id)
                        push_err = 0
                    last_push = now
                except requests.RequestException as e:
                    push_err += 1
                    # 目标不可达时不刷屏：首次与每 30 次记一条
                    if push_err == 1 or push_err % 30 == 0:
                        log.warning("[%s] 推帧失败 %d 次: %s", device_id, push_err, e)
                    last_push = now  # 失败也按节奏走，避免忙等打爆目标
        except Exception as e:
            log.warning("[%s] 相机链路异常，5 秒后重连: %s", device_id, e)
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
        t = threading.Thread(target=_camera_worker, args=(pid, spec, device_id),
                             name=device_id, daemon=True)
        t.start()
        threads.append(t)

    while not _stop.is_set():
        _stop.wait(1)
    log.info("收到退出信号，停止推帧")


if __name__ == "__main__":
    main()
