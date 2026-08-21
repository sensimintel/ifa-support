# -*- coding: utf-8 -*-
"""Astra Pro Plus 推流器：RGB 推帧（同 push_rgb）+ 真深度点云 GLB 直传 /api/frame/product。

深度经 Orbbec SDK v1 取（macOS 免 root），反投影成点云；GLB 坐标与 8060 DA3 产物同约定
（相机在原点、点云在 -Z 前方、Y 向上、单位米），并带 1mm 三角面锚（model-viewer 的 load
事件依赖场景里存在 mesh）。点色为按深度的 Turbo 伪彩（RGB 相机与深度相机没有标定外参，
不做贴色）。选中该设备时面板「DA3 产物」区显示本产物，服务端在接管窗口内跳过 DA3。
"""
import argparse
import io
import json
import math
import sys
import threading
import time

import cv2
import numpy as np
import requests
import trimesh

import push_rgb

ASTRA_PID = 0x060F
DEPTH_MIN_M, DEPTH_MAX_M = 0.25, 4.0    # 有效深度窗（米）：Astra 标称工作距离内
STRIDE = 3                              # 深度图降采样步长：640x480 → 213x160 ≈ 3.4 万点
                                        # （步长 2 的 GLB ~0.7MB/次会挤占 Wi-Fi，拖慢面板拉取 GLB）


class AstraDepth:
    """后台线程持续收深度帧，永远保留最新一帧；主线程按需取用。"""

    def __init__(self):
        from pyorbbecsdk import Context, Pipeline, Config, OBSensorType

        ctx = Context()
        devs = ctx.query_devices()
        dev = None
        for i in range(devs.get_count()):
            try:
                d = devs.get_device_by_index(i)
                if d.get_device_info().get_pid() == ASTRA_PID:
                    dev = d
                    break
            except Exception:
                continue  # 其他设备（如 Gemini 335 无 root）打不开，跳过
        if dev is None:
            raise RuntimeError("没找到 Astra Pro Plus（检查 USB 连接）")
        self._pipe = Pipeline(dev)
        cfg = Config()
        profile = self._pipe.get_stream_profile_list(
            OBSensorType.DEPTH_SENSOR).get_default_video_stream_profile()
        cfg.enable_stream(profile)
        self._pipe.start(cfg)
        self._fx, self._fy, self._cx, self._cy = self._intrinsics(profile)
        self._lock = threading.Lock()
        self._latest = None       # (depth_mm float32 HxW, scale)
        self._scale = 1.0
        threading.Thread(target=self._loop, daemon=True, name="astra-depth").start()

    def _intrinsics(self, profile):
        """深度内参：优先取 SDK 标定值；拿不到按 Astra 标称 FOV(H58.4° V45.5°)推算。"""
        try:
            p = self._pipe.get_camera_param().depth_intrinsic
            if p.fx > 0 and p.fy > 0:
                return float(p.fx), float(p.fy), float(p.cx), float(p.cy)
        except Exception:
            pass
        w, h = profile.get_width(), profile.get_height()
        fx = w / (2 * math.tan(math.radians(58.4) / 2))
        fy = h / (2 * math.tan(math.radians(45.5) / 2))
        return fx, fy, w / 2.0, h / 2.0

    @property
    def fov_v_deg(self):
        h = self._latest[0].shape[0] if self._latest is not None else 480
        return math.degrees(2 * math.atan(h / (2 * self._fy)))

    def _loop(self):
        while True:
            try:
                fs = self._pipe.wait_for_frames(500)
                df = fs.get_depth_frame() if fs else None
                if df is None:
                    continue
                w, h = df.get_width(), df.get_height()
                depth = np.frombuffer(df.get_data(), dtype=np.uint16).reshape(h, w)
                depth_mm = depth.astype(np.float32) * float(df.get_depth_scale())
                with self._lock:
                    self._latest = depth_mm
            except Exception as e:
                print(f"深度收帧异常：{e}", flush=True)
                time.sleep(0.5)

    def build_glb(self):
        """最新深度帧 → 点云 GLB 字节；无有效帧/有效点太少返回 None。"""
        with self._lock:
            depth_mm = None if self._latest is None else self._latest.copy()
        if depth_mm is None:
            return None, 0
        d = depth_mm[::STRIDE, ::STRIDE]
        H, W = d.shape
        z = d / 1000.0
        valid = (z > DEPTH_MIN_M) & (z < DEPTH_MAX_M)
        if int(valid.sum()) < 500:
            return None, 0
        us, vs = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        # 反投影（降采样后的像素坐标 ×STRIDE 折回原图内参），再 flip Y/Z 对齐 glTF 约定
        x = (us * STRIDE - self._cx) / self._fx * z
        y = (vs * STRIDE - self._cy) / self._fy * z
        pts = np.stack([x, -y, -z], axis=-1).reshape(-1, 3)[valid.reshape(-1)].astype(np.float32)
        # 点色：按深度的 Turbo 伪彩（近红远蓝），加满 alpha
        zn = (np.clip(z, DEPTH_MIN_M, DEPTH_MAX_M) - DEPTH_MIN_M) / (DEPTH_MAX_M - DEPTH_MIN_M)
        cmap = cv2.applyColorMap((zn * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        rgb = cmap[..., ::-1].reshape(-1, 3)[valid.reshape(-1)]
        cols = np.concatenate([rgb, np.full((rgb.shape[0], 1), 255, np.uint8)], axis=1)

        scene = trimesh.Scene()
        scene.add_geometry(trimesh.points.PointCloud(vertices=pts, colors=cols))
        # 1mm 三角面锚：纯点云不触发 model-viewer 的 load 事件（同 app.py 的 GLB 构建约定）
        scene.add_geometry(trimesh.Trimesh(
            vertices=np.array([[0, 0, 0], [1e-3, 0, 0], [0, 1e-3, 0]], dtype=np.float32),
            faces=np.array([[0, 1, 2]]), process=False))
        buf = io.BytesIO()
        scene.export(buf, file_type="glb")
        return buf.getvalue(), int(pts.shape[0])


def main():
    ap = argparse.ArgumentParser(description="Astra RGB 推帧 + 真深度点云直传 8060")
    ap.add_argument("--server", default="http://192.168.100.50:8060")
    ap.add_argument("--camera", default="USB相机", help="Astra RGB 的 AVFoundation 名称子串")
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--device-id", default="mac-astra")
    ap.add_argument("--fps", type=float, default=2.0, help="RGB 推帧兜底 fps")
    ap.add_argument("--quality", type=int, default=75)
    ap.add_argument("--product-interval", type=float, default=2.5,
                    help="点云产物上传间隔兜底（秒）；面板/控制面下发 product_interval 后以下发值为准")
    args = ap.parse_args()

    depth = AstraDepth()
    print("Astra 深度流已起，等待 RGB 相机…", flush=True)

    idx = args.index
    if idx is None:
        idx, err = push_rgb.avf_index_by_name(args.camera)
        if idx is None:
            print(err, file=sys.stderr)
            sys.exit(1)
    cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print(f"打不开相机 index={idx}", file=sys.stderr)
        sys.exit(1)

    session = requests.Session()
    state = {"next": 0.0, "sent": 0, "err": 0}

    def push_product(now, poller):
        """随 RGB 主循环的周期钩子：到点就构建并上传一次点云产物。

        上传间隔优先用面板/控制面下发的 per-device product_interval（poller 随主循环
        轮询而来），没有下发时用 --product-interval 兜底。接管窗 hold 随间隔联动
        （3 倍且不小于 10s）：间隔调大时窗口不过期，DA3 单目点云不会中途抢回槽位。
        """
        try:
            itv = float(poller.dev_config.get("product_interval") or 0)
        except (TypeError, ValueError):
            itv = 0.0
        if not 0.5 <= itv <= 30:
            itv = args.product_interval
        if now < state["next"]:
            return
        state["next"] = now + itv
        glb, npts = depth.build_glb()
        if glb is None:
            return
        meta = {
            "ext": 1,
            "cap": "Astra 真深度点云",
            "label": "Astra Pro Plus 深度直传",
            "stat": f"{npts / 1000:.0f}k点 · {len(glb) / 1e6:.1f}MB",
            "fov_deg": depth.fov_v_deg,
        }
        try:
            r = session.post(
                f"{args.server}/api/frame/product",
                files={"model": ("scene.glb", glb, "model/gltf-binary")},
                data={"camera_info": json.dumps({"device_id": args.device_id}),
                      "meta": json.dumps(meta), "hold": str(max(10.0, 3.0 * itv))},
                timeout=8,
            )
            r.raise_for_status()
            state["sent"] += 1
            if state["sent"] % 20 == 1:
                print(f"[{time.strftime('%H:%M:%S')}] 点云已传 {state['sent']} 次"
                      f"（{npts} 点 · {len(glb) / 1e6:.1f}MB）", flush=True)
        except Exception as e:
            state["err"] += 1
            if state["err"] % 10 == 1:
                print(f"点云上传失败（累计 {state['err']}）：{e}", flush=True)

    print(f"相机 index={idx} → {args.server} 设备号 {args.device_id}", flush=True)
    try:
        push_rgb.push_loop(cap, args.server, args.device_id, args.fps, args.quality,
                           on_tick=push_product)
    finally:
        cap.release()


if __name__ == "__main__":
    main()
