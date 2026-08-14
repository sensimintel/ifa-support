"""深度视图配置预设接口测试（/api/frame/depth-presets 保存/列表/删除）。

直接把 frame_relay.router 挂进一个裸 FastAPI 应用测（零 DA3/torch 依赖），
落盘路径重定向到临时目录，不碰仓内运行时文件。
"""
import json
import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

import frame_relay


class DepthPresetsTest(unittest.TestCase):
    def setUp(self):
        # 落盘路径指到临时目录 + 清空内存态，用例间互不污染
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_path = frame_relay._DEPTH_PRESETS_PATH
        frame_relay._DEPTH_PRESETS_PATH = Path(self._tmp.name) / "depth_presets.json"
        frame_relay._depth_presets.clear()
        app = FastAPI()
        app.include_router(frame_relay.router)
        self.client = TestClient(app)

    def tearDown(self):
        frame_relay._DEPTH_PRESETS_PATH = self._orig_path
        frame_relay._depth_presets.clear()
        self._tmp.cleanup()

    def _save(self, name="展会白天", **kw):
        body = {"name": name,
                "config": {"depth_colormap": "lidar", "depth_gamma": 0.65},
                "display": {"br": 0.8, "fit": "cover"},
                "dot": {"on": 1, "mode": 0, "pitch": 7}}
        body.update(kw)
        return self.client.post("/api/frame/depth-presets", json=body)

    def test_save_list_delete_roundtrip(self):
        # 保存两份 → 列表齐全且带完整参数 → 删除一份 → 只剩一份
        self.assertTrue(self._save("展会白天").json()["ok"])
        self.assertTrue(self._save("展会夜场").json()["ok"])
        presets = self.client.get("/api/frame/depth-presets").json()["presets"]
        self.assertEqual({p["name"] for p in presets}, {"展会白天", "展会夜场"})
        p = next(x for x in presets if x["name"] == "展会白天")
        self.assertEqual(p["config"]["depth_colormap"], "lidar")
        self.assertEqual(p["display"]["fit"], "cover")
        self.assertEqual(p["dot"]["pitch"], 7)
        self.assertIn("saved_at", p)
        r = self.client.post("/api/frame/depth-presets/delete", json={"name": "展会白天"})
        self.assertTrue(r.json()["ok"])
        left = self.client.get("/api/frame/depth-presets").json()["presets"]
        self.assertEqual([p["name"] for p in left], ["展会夜场"])

    def test_same_name_overwrites_not_duplicates(self):
        self._save("A", config={"depth_gamma": 0.5})
        self._save("A", config={"depth_gamma": 1.5})
        presets = self.client.get("/api/frame/depth-presets").json()["presets"]
        self.assertEqual(len(presets), 1)
        self.assertEqual(presets[0]["config"]["depth_gamma"], 1.5)

    def test_config_validated_and_clamped_like_device_config(self):
        # 未知键 → 400；越界数值按 device-config 同款范围钳制；null 键从快照剔除
        r = self._save(config={"bogus_key": 1})
        self.assertEqual(r.status_code, 400)
        r = self._save(config={"depth_gamma": 99, "depth_ema": None})
        self.assertTrue(r.json()["ok"])
        p = self.client.get("/api/frame/depth-presets").json()["presets"][0]
        self.assertEqual(p["config"]["depth_gamma"], 5.0)
        self.assertNotIn("depth_ema", p["config"])

    def test_bad_requests_rejected(self):
        self.assertEqual(self._save(name="").status_code, 400)          # 缺预设名
        self.assertEqual(self._save(name="长" * 41).status_code, 400)   # 预设名过长
        self.assertEqual(self._save(config=[1, 2]).status_code, 400)    # config 非对象
        r = self.client.post("/api/frame/depth-presets/delete", json={"name": "不存在"})
        self.assertEqual(r.status_code, 404)

    def test_persisted_to_disk(self):
        # 保存即落盘（原子写），文件内容可独立回读
        self._save("落盘验证")
        on_disk = json.loads(frame_relay._DEPTH_PRESETS_PATH.read_text())
        self.assertIn("落盘验证", on_disk)
        self.assertEqual(on_disk["落盘验证"]["config"]["depth_colormap"], "lidar")


if __name__ == "__main__":
    unittest.main()
