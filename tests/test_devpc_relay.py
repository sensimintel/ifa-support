# -*- coding: utf-8 -*-
"""设备点云在 frame_relay 侧的接入面测试：pc_* 配置键白名单/钳制、
devices[].pc_want 按需标志（provider 注入/未注入/异常三态）。

同 test_depth_presets 的做法：裸 FastAPI 挂 frame_relay.router 测（零 DA3/torch
依赖），配置落盘重定向到临时目录，用例间清空内存态互不污染。"""
import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

import frame_relay


class DevpcRelayTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_cfg_path = frame_relay._DEV_CONFIG_PATH
        frame_relay._DEV_CONFIG_PATH = Path(self._tmp.name) / "dev_config.json"
        frame_relay._dev_config.clear()
        frame_relay._devices.clear()
        frame_relay._selected = None
        frame_relay.set_pc_want_provider(None)
        app = FastAPI()
        app.include_router(frame_relay.router)
        self.client = TestClient(app)

    def tearDown(self):
        frame_relay._DEV_CONFIG_PATH = self._orig_cfg_path
        frame_relay._dev_config.clear()
        frame_relay._devices.clear()
        frame_relay._selected = None
        frame_relay.set_pc_want_provider(None)
        self._tmp.cleanup()

    def _post_frame(self, dev):
        r = self.client.post(
            "/api/frame",
            files={"image": ("f.jpg", b"\xff\xd8fake", "image/jpeg")},
            data={"camera_info": '{"device_id": "%s"}' % dev})
        self.assertTrue(r.json()["ok"])

    def _device_row(self, dev):
        st = self.client.get("/api/frame/status").json()
        return next(d for d in st["devices"] if d["device_id"] == dev)

    def test_pc_config_keys_accepted_and_clamped(self):
        r = self.client.post("/api/frame/device-config", json={
            "device_id": "macmini-g335",
            "config": {"pc_fps": 99, "pc_stride": 0.5}})
        self.assertTrue(r.json()["ok"])
        cfg = r.json()["config"]
        self.assertEqual(cfg["pc_fps"], 10.0)     # 上限钳制
        self.assertEqual(cfg["pc_stride"], 1.0)   # 下限钳制
        # 状态接口把配置随设备行下发（推流端从这读）
        self._post_frame("macmini-g335")
        row = self._device_row("macmini-g335")
        self.assertEqual(row["config"]["pc_fps"], 10.0)

    def test_pc_want_defaults_false_without_provider(self):
        self._post_frame("dev-a")
        self.assertFalse(self._device_row("dev-a")["pc_want"])

    def test_pc_want_follows_injected_provider(self):
        frame_relay.set_pc_want_provider(lambda d: d == "dev-a")
        self._post_frame("dev-a")
        self._post_frame("dev-b")
        self.assertTrue(self._device_row("dev-a")["pc_want"])
        self.assertFalse(self._device_row("dev-b")["pc_want"])

    def test_pc_want_provider_exception_degrades_to_false(self):
        def boom(_dev):
            raise RuntimeError("炸")
        frame_relay.set_pc_want_provider(boom)
        self._post_frame("dev-a")
        self.assertFalse(self._device_row("dev-a")["pc_want"])


if __name__ == "__main__":
    unittest.main()
