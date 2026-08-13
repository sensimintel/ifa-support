# -*- coding: utf-8 -*-
"""mac mini 摄像头推帧器（浅体验区展会拓扑的帧源之一）。

用 pyorbbecsdk 抓取 mini 上插着的 Orbbec 相机彩色流，按 8060 现有入帧契约
（multipart POST /api/frame，字段 image + camera_info JSON 里的 device_id）推给
5090 的 da3-web 服务。每台相机独立成一个 device_id 桶，/panel 下拉可切换，
8060 侧零改动。

要点：
  · 只开彩色流，不开深度（8060 只吃 RGB，深度由 DA3 自己算），省 USB 带宽；
  · G335 彩色流原生 MJPG（即 JPEG 字节），直接透传不转码；其他格式用 cv2 转 JPEG；
  · 相机线程各自独立：断线/异常自动重连重开，互不影响；
  · 推帧节流：相机 ~30fps 出帧，只按 PUSH_FPS 取最新帧上传，不积压；
  · 必须 root 运行（libusb 直读 USB），由 LaunchDaemon 承载。

配置经环境变量（LaunchDaemon 经 run-pusher.sh source 同目录 .env）：
  RELAY_URL     推帧目标，默认 http://192.168.0.50:8060
  PUSH_FPS      推帧频率兜底值，默认 3——仅在拿不到服务端配置时生效
  JPEG_QUALITY  非 MJPG 相机转码 JPEG 质量，默认 80

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
from pyorbbecsdk import Config, Context, OBFormat, OBSensorType, Pipeline

RELAY_URL = os.environ.get("RELAY_URL", "http://192.168.0.50:8060").rstrip("/")
PUSH_FPS = float(os.environ.get("PUSH_FPS", "3"))
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "80"))
# 硬件深度图伪彩的固定量程（米）：固定而非逐帧 min/max 自适应——展台上手/物进出画面
# 时整图颜色才不会跳变闪烁。桌面演示场景默认 0.2~2m，按展台纵深经 .env 调整。
DEPTH_MIN_M = float(os.environ.get("DEPTH_MIN_M", "0.2"))
DEPTH_MAX_M = float(os.environ.get("DEPTH_MAX_M", "2.0"))

# 已知相机 PID → 稳定可读的 device_id 尾缀（新相机接入时补这张表即可）
KNOWN_CAMERAS = {
    0x0800: "g335",   # Orbbec Gemini 335
    0x060F: "astra",  # Astra Pro Plus
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


def _open_pipeline(dev):
    """对设备开彩色+深度流（显式取 default profile——简写 enable_stream(sensor) 在
    1.3.2 绑定上会段错误，勿改回）。深度流开不出来时降级为只推彩色。
    返回 (已 start 的 Pipeline, 是否含深度流)。"""
    pipe = Pipeline(dev)
    cfg = Config()
    profiles = pipe.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    cfg.enable_stream(profiles.get_default_video_stream_profile())
    has_depth = False
    try:
        dprofiles = pipe.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        cfg.enable_stream(dprofiles.get_default_video_stream_profile())
        has_depth = True
    except Exception as e:
        log.warning("深度流不可用，降级只推彩色: %s", e)
    pipe.start(cfg)
    return pipe, has_depth


def _camera_worker(pid: int, device_id: str):
    """单相机工作线程：开流 → 节流推帧 → 出错重连，直到进程退出。"""
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
                pipe, has_depth = _open_pipeline(dev)
            log.info("[%s] 彩色%s流已启动", device_id, "+深度" if has_depth else "")
            first = True
            first_depth = True
            depth_err = 0
            last_push = 0.0
            while not _stop.is_set():
                fs = pipe.wait_for_frames(500)
                if fs is None:
                    continue
                cf = fs.get_color_frame()
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
                # 面板端保留上一张；伪彩失败只丢深度不影响彩色帧）
                files = {"image": ("frame.jpg", jpeg, "image/jpeg")}
                df = fs.get_depth_frame() if has_depth else None
                if df is not None:
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
                                {"device_id": device_id, "source": "mac-mini"}),
                            "timestamp": str(int(now * 1000)),
                        },
                        timeout=(3.05, 5),
                    )
                    resp.raise_for_status()
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
    log.info("cam-pusher 启动：目标=%s 兜底频率=%.1ffps 质量=%d"
             "（实际频率按设备跟随服务端 device-config，两台相机各调各的）",
             RELAY_URL, PUSH_FPS, JPEG_QUALITY)
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
    for pid, tag in KNOWN_CAMERAS.items():
        device_id = f"macmini-{tag}"
        if pid not in present:
            log.warning("[%s] 启动时未发现（pid=%#06x），线程将持续等待接入",
                        device_id, pid)
        t = threading.Thread(target=_camera_worker, args=(pid, device_id),
                             name=device_id, daemon=True)
        t.start()
        threads.append(t)

    while not _stop.is_set():
        _stop.wait(1)
    log.info("收到退出信号，停止推帧")


if __name__ == "__main__":
    main()
