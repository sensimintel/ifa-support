# -*- coding: utf-8 -*-
"""dx-backend 的 ifa 演示控制代理面测试：路径/方法拼装、请求体透传、状态码映射，
以及「services 响应整体透出、加字段不用改代理」这条契约。

dx_backend 只是控制面到 services 的透传代理，不做状态判断，所以这里把
_ifa_services_request 换成录音机——只断言「打给 services 的是什么」与
「services 的响应怎么回给控制面」，不起真的 services。

导入前先把 SCALE_HOST 指到 127.0.0.1：dx_backend 一导入就拉起 Modbus 轮询线程，
不改的话在部署机上跑测试会去连真的秤模块，跟正在服务的那条常连接抢连接。
"""
import os
import unittest

os.environ.setdefault("SCALE_HOST", "127.0.0.1")

from fastapi.testclient import TestClient  # noqa: E402

import dx_backend  # noqa: E402

DEVICE = "odyss-0F0B"


class IfaProxyTestBase(unittest.TestCase):
    """把 _ifa_services_request 换成录音机：记下调用参数，回放预置响应。"""

    def setUp(self):
        self.calls = []
        self.reply = ({}, 200)
        self._origin = dx_backend._ifa_services_request

        def fake(method, path, body=None):
            self.calls.append({"method": method, "path": path, "body": body})
            return self.reply

        dx_backend._ifa_services_request = fake
        self.addCleanup(setattr, dx_backend, "_ifa_services_request", self._origin)
        self.client = TestClient(dx_backend.app)

    @property
    def last(self):
        return self.calls[-1]


class MealStatePassthroughTest(IfaProxyTestBase):
    """五态 + cycle 字段：services 加字段，代理不挑字段、原样透出。"""

    def test_get_透出_cycle_与新状态(self):
        self.reply = ({"device_id": DEVICE, "tracked": True, "state": "report_published",
                       "cycle_id": "ifa-cycle:%s:1" % DEVICE, "cycle_started_at": 1.5}, 200)
        resp = self.client.get("/api/necklaces/%s/meal-state" % DEVICE)
        self.assertEqual(self.last["method"], "GET")
        self.assertEqual(self.last["path"], "/api/v1/ifa/devices/%s/meal-state" % DEVICE)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {
            "ok": True, "device_id": DEVICE, "tracked": True, "state": "report_published",
            "cycle_id": "ifa-cycle:%s:1" % DEVICE, "cycle_started_at": 1.5})

    def test_put_透传_state_并回放响应(self):
        self.reply = ({"device_id": DEVICE, "state": "ready", "cycle_id": ""}, 200)
        resp = self.client.put("/api/necklaces/%s/meal-state" % DEVICE,
                               json={"state": "ready"})
        self.assertEqual(self.last["method"], "PUT")
        self.assertEqual(self.last["body"], {"state": "ready"})
        self.assertEqual(resp.json()["cycle_id"], "")

    def test_put_带_persona_id_一并透传(self):
        """开一轮新演示时可以直接把这一轮绑给某个角色（现场先开轮、后选人的顺序）。"""
        self.reply = ({"device_id": DEVICE, "state": "ready", "persona_id": "leo"}, 200)
        resp = self.client.put("/api/necklaces/%s/meal-state" % DEVICE,
                               json={"state": "ready", "persona_id": "leo"})
        self.assertEqual(self.last["body"], {"state": "ready", "persona_id": "leo"})
        self.assertEqual(resp.json()["persona_id"], "leo")

    def test_put_不带_persona_id_时不塞空值(self):
        """留空是常态：主路径是访客在 App 上选完角色自己补绑，别把空串写进这一轮。"""
        self.reply = ({"device_id": DEVICE, "state": "ready"}, 200)
        self.client.put("/api/necklaces/%s/meal-state" % DEVICE,
                        json={"state": "ready", "persona_id": "  "})
        self.assertEqual(self.last["body"], {"state": "ready"})

    def test_缺_state_不打给_services(self):
        resp = self.client.put("/api/necklaces/%s/meal-state" % DEVICE, json={})
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(self.calls, [])


class MealStateDeleteTest(IfaProxyTestBase):
    """退出演示：与「本轮失败」是两回事，控制面必须能表达「这台不参与」。"""

    def test_delete_代理到_services(self):
        self.reply = ({"device_id": DEVICE, "removed": True}, 200)
        resp = self.client.delete("/api/necklaces/%s/meal-state" % DEVICE)
        self.assertEqual(self.last["method"], "DELETE")
        self.assertEqual(self.last["path"], "/api/v1/ifa/devices/%s/meal-state" % DEVICE)
        self.assertIsNone(self.last["body"])
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"ok": True, "device_id": DEVICE, "removed": True})

    def test_services_失败时透传状态码(self):
        self.reply = ({"error": "boom"}, 502)
        resp = self.client.delete("/api/necklaces/%s/meal-state" % DEVICE)
        self.assertEqual(resp.status_code, 502)
        self.assertFalse(resp.json()["ok"])


class MealSegmentsPassthroughTest(IfaProxyTestBase):
    """餐段项里的 cycle_id / window_truncated / report / params 必须完整到达控制面。"""

    def test_列表整体透传(self):
        segment = {"segment_id": "ifa-seg:%s:1712" % DEVICE,
                   "cycle_id": "ifa-cycle:%s:1" % DEVICE,
                   "window_truncated": True,
                   "report": {"status": "published", "journal_meal_id": "j-1",
                              "meal_record_id": "m-1", "published_at": 9.0, "error": ""},
                   "params": {"window_max_minutes": 20, "max_frames": 32,
                              "image_max_edge": 640, "image_jpeg_quality": 85,
                              "model": "gemini-3.1-pro-preview",
                              "prompt": "meal_analysis_ifa_v1"}}
        self.reply = ({"segments": [segment]}, 200)
        resp = self.client.get("/api/necklaces/%s/meal-segments?limit=3" % DEVICE)
        self.assertEqual(self.last["path"],
                         "/api/v1/ifa/devices/%s/meal-segments?limit=3" % DEVICE)
        self.assertEqual(resp.json()["segments"], [segment])

    def test_analyze_的_segment_id_冒号不被编码(self):
        # 回归用例：segment_id 形如 ifa-seg:<device>:<millis>，冒号被编成 %3A 就查不到段
        segment_id = "ifa-seg:%s:1712" % DEVICE
        self.reply = ({"accepted": True, "segment_id": segment_id}, 202)
        resp = self.client.post(
            "/api/necklaces/%s/meal-segments/%s/analyze" % (DEVICE, segment_id))
        self.assertEqual(
            self.last["path"],
            "/api/v1/ifa/devices/%s/meal-segments/%s/analyze" % (DEVICE, segment_id))
        self.assertEqual(resp.status_code, 202)
        self.assertTrue(resp.json()["ok"])


class CloseMealTest(IfaProxyTestBase):
    """强制关餐：无请求体的 POST 代理，成功 200、services 拒绝则原样透出状态码。"""

    def test_成功关餐(self):
        self.reply = ({"device_id": DEVICE, "closed": True}, 200)
        resp = self.client.post("/api/necklaces/%s/close-meal" % DEVICE)
        self.assertEqual(self.last["method"], "POST")
        self.assertEqual(self.last["path"], "/api/v1/ifa/devices/%s/close-meal" % DEVICE)
        self.assertIsNone(self.last["body"])
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"ok": True, "device_id": DEVICE, "closed": True})

    def test_services_拒绝时透传状态码(self):
        # 当前没有在进行中的餐之类的判断全在 services，代理只负责把它的结论带回来
        self.reply = ({"error": "当前没有进行中的用餐"}, 409)
        resp = self.client.post("/api/necklaces/%s/close-meal" % DEVICE)
        self.assertEqual(resp.status_code, 409)
        self.assertEqual(resp.json(), {"ok": False, "error": "当前没有进行中的用餐"})


class ParamsTest(IfaProxyTestBase):
    """深区分析参数：GET 带 scope 透出，PUT 部分更新原样转发、不在代理层校验取值。"""

    PARAMS = {"window_max_minutes": 20, "max_frames": 32, "image_max_edge": 640,
              "image_jpeg_quality": 85, "model": "gemini-3.1-pro-preview",
              "prompt": "meal_analysis_ifa_v1"}

    def test_get_透出_scope_与全量参数(self):
        self.reply = ({"device_id": DEVICE, "scope": "global", "params": self.PARAMS}, 200)
        resp = self.client.get("/api/necklaces/%s/params" % DEVICE)
        self.assertEqual(self.last["method"], "GET")
        self.assertEqual(self.last["path"], "/api/v1/ifa/devices/%s/params" % DEVICE)
        self.assertEqual(resp.json(), {"ok": True, "device_id": DEVICE,
                                       "scope": "global", "params": self.PARAMS})

    def test_put_部分更新原样透传(self):
        self.reply = ({"device_id": DEVICE, "scope": "device",
                       "params": dict(self.PARAMS, max_frames=48)}, 200)
        resp = self.client.put("/api/necklaces/%s/params" % DEVICE,
                               json={"params": {"max_frames": 48}})
        self.assertEqual(self.last["method"], "PUT")
        self.assertEqual(self.last["body"], {"params": {"max_frames": 48}})
        self.assertEqual(resp.json()["params"]["max_frames"], 48)

    def test_put_不认识的参数名也照样转发(self):
        # 合法值域只有 services 知道；代理再抄一份校验必然与它漂移，故一律转发
        self.reply = ({"error": "未知参数 foo"}, 400)
        resp = self.client.put("/api/necklaces/%s/params" % DEVICE,
                               json={"params": {"foo": 1}})
        self.assertEqual(self.last["body"], {"params": {"foo": 1}})
        self.assertEqual(resp.status_code, 400)

    def test_params_不是对象时本地就拒掉(self):
        for bad in ({}, {"params": "x"}, {"params": None}):
            resp = self.client.put("/api/necklaces/%s/params" % DEVICE, json=bad)
            self.assertEqual(resp.status_code, 400, bad)
        self.assertEqual(self.calls, [])


class FallbackReportPassthroughTest(IfaProxyTestBase):
    """推送报告兜底：食物表与推送请求都只做透传，状态闸门留在 services。"""

    def test_食物表透出(self):
        self.reply = ({"foods": [{"key": "blueberry", "name": "Blueberries",
                                  "calories_per_100g": 57}]}, 200)
        resp = self.client.get("/api/fallback-report/foods")
        self.assertEqual(self.last["method"], "GET")
        self.assertEqual(self.last["path"], "/api/v1/ifa/fallback-report/foods")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["foods"][0]["key"], "blueberry")

    def test_推送请求体原样透传(self):
        self.reply = ({"device_id": DEVICE, "journal_meal_id": 901,
                       "state": "report_published"}, 200)
        body = {"items": [{"food_key": "blueberry", "grams": 120},
                          {"food_key": "popcorn", "grams": 30}]}
        resp = self.client.post("/api/necklaces/%s/fallback-report" % DEVICE, json=body)
        self.assertEqual(self.last["method"], "POST")
        self.assertEqual(self.last["path"],
                         "/api/v1/ifa/devices/%s/fallback-report" % DEVICE)
        self.assertEqual(self.last["body"], body)
        self.assertEqual(resp.json()["journal_meal_id"], 901)

    def test_services_拒绝时透传状态码(self):
        self.reply = ({"error": "该设备当前没有演示周期"}, 404)
        resp = self.client.post("/api/necklaces/%s/fallback-report" % DEVICE,
                                json={"items": [{"food_key": "popcorn", "grams": 30}]})
        self.assertEqual(resp.status_code, 404)
        self.assertFalse(resp.json()["ok"])

    def test_items_不是数组时本地就拒掉(self):
        for bad in ({}, {"items": "x"}, {"items": None}):
            resp = self.client.post("/api/necklaces/%s/fallback-report" % DEVICE, json=bad)
            self.assertEqual(resp.status_code, 400, bad)
        self.assertEqual(self.calls, [])


class ServicesUnreachableTest(unittest.TestCase):
    """services 不可达时，新老 ifa 端点走同一条错误通道：502 + ok=false。"""

    def setUp(self):
        self._origin = dx_backend.IFA_SERVICES_BASE_URL
        # 端口 1 上不会有服务，走的是 _ifa_services_request 里真实的 URLError 分支
        dx_backend.IFA_SERVICES_BASE_URL = "http://127.0.0.1:1"
        self.addCleanup(setattr, dx_backend, "IFA_SERVICES_BASE_URL", self._origin)
        self.client = TestClient(dx_backend.app)

    def test_全部端点统一_502(self):
        cases = [
            ("GET", "/api/necklaces/%s/meal-state" % DEVICE, None),
            ("POST", "/api/necklaces/%s/close-meal" % DEVICE, None),
            ("GET", "/api/necklaces/%s/params" % DEVICE, None),
            ("PUT", "/api/necklaces/%s/params" % DEVICE, {"params": {"max_frames": 8}}),
            ("GET", "/api/fallback-report/foods", None),
            ("POST", "/api/necklaces/%s/fallback-report" % DEVICE,
             {"items": [{"food_key": "popcorn", "grams": 30}]}),
        ]
        for method, path, body in cases:
            resp = self.client.request(method, path, json=body)
            self.assertEqual(resp.status_code, 502, path)
            self.assertFalse(resp.json()["ok"], path)


if __name__ == "__main__":
    unittest.main()
