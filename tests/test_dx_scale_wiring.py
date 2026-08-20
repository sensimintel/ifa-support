# -*- coding: utf-8 -*-
"""dx-backend 里秤事件的接线测试：喂样本 → 补归属 → 进上报队列 → 出接口。

检测算法本身在 test_dx_scale_events.py 测；这里只测 dx_backend 这一层容易悄悄坏掉的
四件事：按通道反查项链、没绑项链不上报、没接秤的通道不喂、掉线恢复记 resync。

导入前把 SCALE_HOST 指到 127.0.0.1：dx_backend 一导入就拉起 Modbus 轮询线程，
不改的话在部署机上跑测试会去连真的秤模块，跟正在服务的那条常连接抢连接。
"""
import os
import unittest

os.environ.setdefault("SCALE_HOST", "127.0.0.1")
# 后台轮询与上报线程会异步改状态（搬空事件队列、给检测器打 gap），断言必须按住它们
os.environ.setdefault("DX_BACKGROUND_THREADS", "0")

from fastapi.testclient import TestClient  # noqa: E402

import dx_backend  # noqa: E402
import dx_scale_events as sev  # noqa: E402

NECKLACE = "odyss-0F28"


def plateau_raws(gross_g, channel=1):
    """把毛重克数换算回 Modbus raw（分度 0.1），拼成一轮四通道读数。"""
    return {ch: int(round(gross_g / dx_backend.SCALE_DIVISION)) if ch == channel else 0
            for ch in dx_backend.SCALE_CHANNELS}


class ScaleWiringTestBase(unittest.TestCase):
    def setUp(self):
        # 原始采样落盘在测试里必须关掉，否则会往仓目录里写 jsonl
        self._raw_log = dx_backend.SCALE_RAW_LOG
        dx_backend.SCALE_RAW_LOG = False
        self.addCleanup(setattr, dx_backend, "SCALE_RAW_LOG", self._raw_log)

        # 检测器、内存环、上报队列都是模块级的，用例之间必须互不污染
        dx_backend._scale_detectors = {
            ch: sev.ScaleEventDetector(ch, sample_interval_s=dx_backend.SCALE_POLL_INTERVAL)
            for ch in dx_backend.SCALE_CHANNELS
        }
        dx_backend._scale_events_recent.clear()
        dx_backend._scale_upload_queue.clear()
        dx_backend._scale_online = True

        self._groups = dx_backend._state["groups"]
        self._connected = dict(dx_backend._state["scale_connected"])
        self.addCleanup(dx_backend._state.__setitem__, "groups", self._groups)
        self.addCleanup(dx_backend._state.__setitem__, "scale_connected", self._connected)
        self.client = TestClient(dx_backend.app)

    def bind(self, channel, device_id):
        dx_backend._state["groups"] = [
            {"edge": channel, "scale_channel": channel, "necklace_device_id": device_id}
        ]

    def feed(self, plateaus, channel=1, t0=0.0):
        """按 [(毛重 g, 保持秒数)] 走一遍轮询喂数，返回结束时刻。"""
        t = t0
        for gross, hold in plateaus:
            for _ in range(int(round(hold / dx_backend.SCALE_POLL_INTERVAL))):
                dx_backend._feed_detectors(round(t, 3), plateau_raws(gross, channel))
                t = round(t + dx_backend.SCALE_POLL_INTERVAL, 3)
        return t


class AttributionTest(ScaleWiringTestBase):
    def test_事件按通道补上项链并进上报队列(self):
        self.bind(1, NECKLACE)
        self.feed([(280.3, 3), (278.5, 3)])
        self.assertEqual(len(dx_backend._scale_upload_queue), 1)
        record = dx_backend._scale_upload_queue[0]
        self.assertEqual(record["device_id"], NECKLACE)
        self.assertEqual(record["kind"], sev.KIND_STEP)
        self.assertEqual(record["delta_g"], -1.8)

    def test_没绑项链的通道留痕但不上报(self):
        self.bind(1, "")
        self.feed([(280.3, 3), (278.5, 3)])
        # 事实还在（现场能看出「秤动了但没人认领」），只是没有归属可上报
        self.assertEqual(len(dx_backend._scale_events_recent), 1)
        self.assertEqual(len(dx_backend._scale_upload_queue), 0)

    def test_没接秤的通道不喂检测器(self):
        self.bind(1, NECKLACE)
        dx_backend._state["scale_connected"] = {"1": False, "2": False, "3": False, "4": False}
        self.feed([(280.3, 3), (278.5, 3)])
        self.assertEqual(len(dx_backend._scale_events_recent), 0)


class GapTest(ScaleWiringTestBase):
    def test_掉线恢复后记一条_resync_而不是一次取食(self):
        self.bind(1, NECKLACE)
        t = self.feed([(280.3, 3)])
        dx_backend._mark_detector_gap()
        self.feed([(240.0, 3)], t0=t)
        kinds = [r["kind"] for r in dx_backend._scale_events_recent]
        self.assertEqual(kinds, [sev.KIND_RESYNC])


class UploadPayloadTest(ScaleWiringTestBase):
    def test_上报体只带_services_认的八个字段(self):
        self.bind(1, NECKLACE)
        self.feed([(280.3, 3), (278.5, 3)])
        payload = dx_backend._scale_upload_payload(list(dx_backend._scale_upload_queue))
        self.assertEqual(len(payload["events"]), 1)
        self.assertEqual(set(payload["events"][0]), {
            "device_id", "scale_channel", "kind",
            "started_at", "occurred_at", "before_g", "after_g", "delta_g"})
        self.assertTrue(payload["events"][0]["occurred_at"].endswith("Z"))
        self.assertEqual(payload["events"][0]["scale_channel"], 1)


class EventsEndpointTest(ScaleWiringTestBase):
    def test_接口倒序返回并能只看可计入的事件(self):
        self.bind(1, NECKLACE)
        # 一次取食 + 一次端碗离台（lift 不可计入）
        self.feed([(280.3, 3), (278.5, 3), (11.4, 3)])
        body = self.client.get("/api/scale-events").json()
        self.assertEqual([e["kind"] for e in body["events"]],
                         [sev.KIND_LIFT, sev.KIND_STEP])
        self.assertEqual(body["detectors"][0]["state"], "absent")
        self.assertEqual(body["upload"]["pending"], 2)

        countable = self.client.get("/api/scale-events?countable_only=true").json()
        self.assertEqual([e["kind"] for e in countable["events"]], [sev.KIND_STEP])

    def test_接口带出生效中的检测参数(self):
        body = self.client.get("/api/scale-events").json()
        self.assertEqual(body["params"]["lift_threshold_g"], dx_backend.SCALE_LIFT_THRESHOLD_G)
        self.assertEqual(body["params"]["stable_epsilon_g"], dx_backend.SCALE_STABLE_EPSILON_G)


if __name__ == "__main__":
    unittest.main()
