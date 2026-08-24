# -*- coding: utf-8 -*-
"""传输时序 + 队列进度字段的测试（跨仓契约 v1 §2a/§2b）。

三层各测各的：
  1. frame_relay._queue_locked 纯解析——键名映射（去 queue_ 前缀）、单键缺失/非法
     为 null、一个队列键都没有或 camera_info 坏掉时整体 None（区分「没升级」与「队列空」）；
  2. /api/frame/status——devices[] 每项与顶层单设备视图都带三时刻 + queue
     （裸 FastAPI 挂 router，同 test_devpc_relay 的做法，零 DA3/torch 依赖）；
  3. dx_backend /api/necklaces/online——原样透传 8060 的新字段，8060 未升级
     没给这些键时一律 None、不得报错（把 urlopen 换成录音机，不起真 8060）。

导入 dx_backend 前先把 SCALE_HOST 指到 127.0.0.1 并关后台线程：
不改的话在部署机上跑测试会去连真的秤模块，跟正在服务的那条常连接抢连接。
"""
import json
import os
import unittest
from unittest import mock

os.environ.setdefault("SCALE_HOST", "127.0.0.1")
os.environ.setdefault("DX_BACKGROUND_THREADS", "0")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import dx_backend  # noqa: E402
import frame_relay  # noqa: E402

# 13 位毫秒级 epoch 样本（2026 年前后），过 _parse_ms 的 1e12~1e14 量级校验
MS = 1_770_000_000_000


def _queue_of(camera_info):
    """按设备桶最小形状调用被测函数（桶里其余键与解析无关）。"""
    return frame_relay._queue_locked({"camera_info": camera_info})


class QueueParseTest(unittest.TestCase):
    """_queue_locked 纯函数：camera_info JSON → 队列进度对象。"""

    def test_全键齐全时逐键映射(self):
        q = _queue_of(json.dumps({
            "device_id": "odyss-0F0B",
            "enqueued_at_ms": MS,
            "attempt": 3,
            "queue_pending_count": 17,
            "queue_oldest_pending_ts_ms": MS - 60_000,
            "is_newest_slot": True,
        }))
        self.assertEqual(q, {
            "enqueued_at_ms": MS,
            "attempt": 3,
            "pending_count": 17,
            "oldest_pending_ts_ms": MS - 60_000,
            "is_newest_slot": True,
        })

    def test_部分键缺失时缺的为_null_不影响其余键(self):
        q = _queue_of(json.dumps({"attempt": "2", "queue_pending_count": 5.0}))
        self.assertEqual(q["attempt"], 2)            # 字符串数字也收
        self.assertEqual(q["pending_count"], 5)      # 浮点收成 int
        self.assertIsNone(q["enqueued_at_ms"])
        self.assertIsNone(q["oldest_pending_ts_ms"])
        self.assertIsNone(q["is_newest_slot"])

    def test_camera_info_为_None_整体为_None(self):
        self.assertIsNone(_queue_of(None))

    def test_camera_info_坏_JSON_整体为_None(self):
        self.assertIsNone(_queue_of("{queue_pending_count:"))

    def test_JSON_合法但不是对象_整体为_None(self):
        self.assertIsNone(_queue_of("[1, 2, 3]"))

    def test_一个队列键都没有时整体为_None(self):
        # 老版本 App 的 camera_info：只有身份与 upload_at_ms，没升级 ≠ 队列空
        self.assertIsNone(_queue_of(json.dumps(
            {"device_id": "odyss-0F0B", "upload_at_ms": MS})))

    def test_时间戳量级非法为_null(self):
        # 秒级时间戳过不了 _parse_ms 的毫秒级量级校验（1e12~1e14）
        q = _queue_of(json.dumps({
            "enqueued_at_ms": MS // 1000,
            "queue_oldest_pending_ts_ms": "不是数字",
        }))
        self.assertIsNotNone(q)   # 键在场，队列对象整体不为 None
        self.assertIsNone(q["enqueued_at_ms"])
        self.assertIsNone(q["oldest_pending_ts_ms"])

    def test_计数键非法为_null(self):
        q = _queue_of(json.dumps({"attempt": "abc", "queue_pending_count": None}))
        self.assertIsNotNone(q)
        self.assertIsNone(q["attempt"])
        self.assertIsNone(q["pending_count"])

    def test_is_newest_slot_三态(self):
        self.assertTrue(_queue_of(json.dumps({"is_newest_slot": True}))["is_newest_slot"])
        self.assertFalse(_queue_of(json.dumps({"is_newest_slot": False}))["is_newest_slot"])
        # 缺失（其它队列键在场撑起对象）为 null，而不是 False
        self.assertIsNone(_queue_of(json.dumps({"attempt": 1}))["is_newest_slot"])


class FrameStatusTimingTest(unittest.TestCase):
    """/api/frame/status：devices[] 每项与顶层单设备视图都带三时刻 + queue。"""

    def setUp(self):
        frame_relay._devices.clear()
        frame_relay._selected = None
        app = FastAPI()
        app.include_router(frame_relay.router)
        self.client = TestClient(app)
        self.addCleanup(frame_relay._devices.clear)
        self.addCleanup(setattr, frame_relay, "_selected", None)

    def _post_frame(self, dev, camera_extra=None, timestamp=None):
        info = {"device_id": dev, **(camera_extra or {})}
        data = {"camera_info": json.dumps(info)}
        if timestamp is not None:
            data["timestamp"] = str(timestamp)
        r = self.client.post(
            "/api/frame",
            files={"image": ("f.jpg", b"\xff\xd8fake", "image/jpeg")},
            data=data)
        self.assertTrue(r.json()["ok"])

    def test_devices_每项带三时刻与_queue(self):
        self._post_frame(
            "odyss-0F0B", timestamp=MS,
            camera_extra={"upload_at_ms": MS + 500, "enqueued_at_ms": MS + 100,
                          "attempt": 1, "queue_pending_count": 4,
                          "queue_oldest_pending_ts_ms": MS - 30_000,
                          "is_newest_slot": False})
        st = self.client.get("/api/frame/status").json()
        row = next(d for d in st["devices"] if d["device_id"] == "odyss-0F0B")
        self.assertEqual(row["capture_ts_ms"], MS)
        self.assertEqual(row["upload_ts_ms"], MS + 500)
        self.assertIsInstance(row["server_ts_ms"], int)
        self.assertEqual(row["queue"], {
            "enqueued_at_ms": MS + 100, "attempt": 1, "pending_count": 4,
            "oldest_pending_ts_ms": MS - 30_000, "is_newest_slot": False})
        # 顶层单设备视图同一解析口径
        self.assertEqual(st["queue"], row["queue"])
        self.assertEqual(st["capture_ts_ms"], MS)

    def test_老版本_App_没有队列键时_queue_为_null_三时刻照常(self):
        self._post_frame("odyss-1234", timestamp=MS,
                         camera_extra={"upload_at_ms": MS + 200})
        st = self.client.get("/api/frame/status").json()
        row = next(d for d in st["devices"] if d["device_id"] == "odyss-1234")
        self.assertIsNone(row["queue"])
        self.assertEqual(row["upload_ts_ms"], MS + 200)

    def test_无任何设备时顶层_queue_为_null(self):
        st = self.client.get("/api/frame/status").json()
        self.assertIsNone(st["queue"])
        self.assertFalse(st["has_frame"])


class _FakeResp:
    """urlopen 的最小替身：只支持 with 语境 + read()。"""

    def __init__(self, payload):
        self._data = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class NecklacesOnlinePassthroughTest(unittest.TestCase):
    """dx-backend /api/necklaces/online：新字段原样透传，8060 没给时为 None。"""

    def setUp(self):
        # 干净的内存状态且掐掉落盘——测试不该写仓目录里的 dx_data.json
        self._origin = dx_backend._state
        dx_backend._state = {
            "groups": [dx_backend._default_group(e) for e in dx_backend.EDGES],
            "phones": [],
            "tare_raw": {str(ch): 0 for ch in dx_backend.EDGES},
            "scale_connected": {str(ch): True for ch in dx_backend.EDGES},
        }
        self.addCleanup(setattr, dx_backend, "_state", self._origin)
        self._save = dx_backend._save_state
        dx_backend._save_state = lambda state: None
        self.addCleanup(setattr, dx_backend, "_save_state", self._save)
        self.client = TestClient(dx_backend.app)

    def _online(self, devices):
        with mock.patch("dx_backend.urllib.request.urlopen",
                        return_value=_FakeResp({"devices": devices})):
            r = self.client.get("/api/necklaces/online")
        self.assertEqual(r.status_code, 200)
        return r.json()

    def test_升级后的_8060_字段原样透传(self):
        queue = {"enqueued_at_ms": MS, "attempt": 2, "pending_count": 9,
                 "oldest_pending_ts_ms": MS - 10_000, "is_newest_slot": True}
        body = self._online([{
            "device_id": "odyss-0F0B", "age": 1.2, "seq": 42, "fps": 2.0,
            "capture_ts_ms": MS, "upload_ts_ms": MS + 500,
            "server_ts_ms": MS + 900, "queue": queue}])
        item = body["necklaces"][0]
        self.assertEqual(item["capture_ts_ms"], MS)
        self.assertEqual(item["upload_ts_ms"], MS + 500)
        self.assertEqual(item["server_ts_ms"], MS + 900)
        self.assertEqual(item["queue"], queue)

    def test_未升级的_8060_没给新键一律_None_不报错(self):
        body = self._online([{"device_id": "odyss-1234", "age": 0.5,
                              "seq": 7, "fps": 1.0}])
        item = body["necklaces"][0]
        self.assertIsNone(item["capture_ts_ms"])
        self.assertIsNone(item["upload_ts_ms"])
        self.assertIsNone(item["server_ts_ms"])
        self.assertIsNone(item["queue"])
        self.assertEqual(body["error"], "")


if __name__ == "__main__":
    unittest.main()
