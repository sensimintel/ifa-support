# -*- coding: utf-8 -*-
"""Mac 相机 RGB 推流器：抓 AVFoundation 相机帧，按 8060 面板配置的 push_fps 推到 /api/frame。

与手机 App 同款 multipart 契约（image + camera_info{device_id, upload_at_ms} + timestamp），
在 /panel 设备下拉里表现为一台虚拟设备。fps 由服务端 /api/frame/status 的 config.push_fps
下发（面板「推帧 fps」滑条），脚本周期轮询生效；服务端不可达时指数退避重试。
"""
import argparse
import json
import sys
import time

import cv2
import requests


def avf_index_by_name(name_sub: str):
    """按名称子串在 AVFoundation 视频设备列表里找索引（列表顺序与 OpenCV CAP_AVFOUNDATION 一致）。"""
    try:
        import AVFoundation as AVF
    except ImportError:
        return None, "缺少 pyobjc-framework-AVFoundation（跑 setup.sh 安装），或改用 --index"
    devs = AVF.AVCaptureDevice.devicesWithMediaType_(AVF.AVMediaTypeVideo)
    names = [str(d.localizedName()) for d in devs]
    for i, n in enumerate(names):
        if name_sub.lower() in n.lower():
            return i, None
    return None, f"没找到名称含「{name_sub}」的相机；可用相机：{names}"


class ConfigPoller:
    """轮询 /api/frame/status 拿面板下发的 push_fps（顺带确认服务端可达）。"""

    def __init__(self, server: str, session: requests.Session, fallback_fps: float):
        self.server, self.session = server, session
        self.fps = fallback_fps
        self._next_poll = 0.0

    def poll(self):
        if time.time() < self._next_poll:
            return
        self._next_poll = time.time() + 2.0
        try:
            cfg = self.session.get(f"{self.server}/api/frame/status", timeout=3).json().get("config") or {}
            fps = float(cfg.get("push_fps") or 0)
            if 0.1 <= fps <= 30:
                self.fps = fps
        except Exception:
            pass  # 服务端瞬断不影响本地节奏，下个周期再试


def push_loop(cap, server: str, device_id: str, fallback_fps: float, quality: int,
              on_tick=None):
    """主循环：按 push_fps 抓帧→JPEG→POST /api/frame。on_tick(now) 供扩展方挂周期任务。"""
    session = requests.Session()
    poller = ConfigPoller(server, session, fallback_fps)
    n_sent, n_err, last_log = 0, 0, time.time()
    while True:
        t0 = time.time()
        poller.poll()
        ok, frame = cap.read()
        if not ok:
            print("读帧失败，重试…", flush=True)
            time.sleep(1.0)
            continue
        if frame.shape[1] > 1280:  # 限宽 1280：识别/DA3 用不到更大，省带宽
            h = int(frame.shape[0] * 1280 / frame.shape[1])
            frame = cv2.resize(frame, (1280, h))
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            continue
        now_ms = int(time.time() * 1000)
        try:
            r = session.post(
                f"{server}/api/frame",
                files={"image": ("frame.jpg", jpg.tobytes(), "image/jpeg")},
                data={
                    "camera_info": json.dumps({"device_id": device_id, "upload_at_ms": now_ms}),
                    "timestamp": str(now_ms),
                },
                timeout=5,
            )
            r.raise_for_status()
            n_sent += 1
        except Exception as e:
            n_err += 1
            if n_err % 10 == 1:
                print(f"推帧失败（累计 {n_err}）：{e}", flush=True)
            time.sleep(min(5.0, 0.5 * n_err))
        if on_tick is not None:
            on_tick(time.time())
        if time.time() - last_log > 10:
            print(f"[{time.strftime('%H:%M:%S')}] 已推 {n_sent} 帧 · 目标 {poller.fps:.1f} fps"
                  f"（面板可调）· 失败 {n_err}", flush=True)
            last_log = time.time()
        time.sleep(max(0.0, 1.0 / poller.fps - (time.time() - t0)))


def main():
    ap = argparse.ArgumentParser(description="Mac 相机 RGB 推流到 8060 /api/frame")
    ap.add_argument("--server", default="http://192.168.0.50:8060")
    ap.add_argument("--camera", default="Gemini 335", help="AVFoundation 相机名称子串")
    ap.add_argument("--index", type=int, default=None, help="直接指定相机索引（跳过名称匹配）")
    ap.add_argument("--device-id", default="mac-g335")
    ap.add_argument("--fps", type=float, default=2.0, help="服务端没下发 push_fps 时的兜底 fps")
    ap.add_argument("--quality", type=int, default=85)
    args = ap.parse_args()

    idx = args.index
    if idx is None:
        idx, err = avf_index_by_name(args.camera)
        if idx is None:
            print(err, file=sys.stderr)
            sys.exit(1)
    cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print(f"打不开相机 index={idx}", file=sys.stderr)
        sys.exit(1)
    print(f"相机 index={idx} → {args.server} 设备号 {args.device_id}", flush=True)
    try:
        push_loop(cap, args.server, args.device_id, args.fps, args.quality)
    finally:
        cap.release()


if __name__ == "__main__":
    main()
