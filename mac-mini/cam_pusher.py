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
  PUSH_FPS      每台相机的推帧频率，默认 3
  JPEG_QUALITY  非 MJPG 相机转码 JPEG 质量，默认 80
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


def _open_color_pipeline(dev):
    """对设备开彩色流（显式取 default profile——简写 enable_stream(sensor) 在
    1.3.2 绑定上会段错误，勿改回）。返回已 start 的 Pipeline。"""
    pipe = Pipeline(dev)
    cfg = Config()
    profiles = pipe.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    cfg.enable_stream(profiles.get_default_video_stream_profile())
    pipe.start(cfg)
    return pipe


def _camera_worker(pid: int, device_id: str):
    """单相机工作线程：开流 → 节流推帧 → 出错重连，直到进程退出。"""
    ctx = None
    session = requests.Session()
    interval = 1.0 / max(PUSH_FPS, 0.1)
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
                pipe = _open_color_pipeline(dev)
            log.info("[%s] 彩色流已启动", device_id)
            first = True
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
                if now - last_push < interval:
                    continue
                jpeg = _to_jpeg(cf)
                try:
                    resp = session.post(
                        f"{RELAY_URL}/api/frame",
                        files={"image": ("frame.jpg", jpeg, "image/jpeg")},
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
    log.info("cam-pusher 启动：目标=%s 频率=%.1ffps 质量=%d",
             RELAY_URL, PUSH_FPS, JPEG_QUALITY)

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
