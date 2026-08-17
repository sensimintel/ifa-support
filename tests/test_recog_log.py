# -*- coding: utf-8 -*-
"""VLM 识别观测日志：环形缓冲 / 响应截断 / 列表与详情投影 + app.py 侧接线断言。

缓冲与投影是纯逻辑模块（recog_log.py，零 cv2/torch 依赖）直接测；
app.py 与页面那侧只做接线断言（正则抽源码），与 test_recog_direct 同款做法。
"""
import re
import unittest
from pathlib import Path

import recog_log

ROOT = Path(__file__).resolve().parents[1]


def _entry(log, name="可乐", raw="RAW", items=None, device="dev-a", trigger="direct"):
    e = log.begin(device, trigger,
                  [{"id": 7, "name": "拿铁", "type": "液体", "ref_img": "data:image/jpeg;base64,AAA"}],
                  n_food=1, n_drink=2, img_orig="data:orig", img_boxed=None)
    e["ts"] = 1786900000.0
    e["req"] = {"label": "Qwen", "model": "m", "endpoint": "http://x/v1/chat",
                "direct": True, "n_images": 2, "prompt": "P" * 50,
                "max_tokens": 1536, "temperature": 0,
                "img_full": "data:image/jpeg;base64,FULL", "img_boxed_full": None,
                "img_full_px": "1280x720"}
    log.set_response(e, True, raw=raw, items=items if items is not None else [{"name": name}])
    e["outcome"] = [{"name": name, "action": "new", "card_id": 3, "gate": ""}]
    log.commit(e)
    return e


class BufferTest(unittest.TestCase):
    def test_ids_increment_and_newest_first(self):
        log = recog_log.RecogLog(max_items=5)
        _entry(log, "A")
        _entry(log, "B")
        items, total = log.list()
        self.assertEqual(total, 2)
        self.assertEqual([it["id"] for it in items], [2, 1])   # 最新在前

    def test_ring_drops_oldest(self):
        log = recog_log.RecogLog(max_items=3)
        for i in range(6):
            _entry(log, f"食物{i}")
        self.assertEqual(len(log), 3)
        items, total = log.list(limit=10)
        self.assertEqual([it["id"] for it in items], [6, 5, 4])
        self.assertEqual(total, 3)

    def test_filter_by_device_and_limit_clamped(self):
        log = recog_log.RecogLog(max_items=10)
        _entry(log, device="dev-a")
        _entry(log, device="dev-b")
        _entry(log, device="dev-a")
        items, total = log.list(device="dev-a")
        self.assertEqual(total, 2)
        self.assertTrue(all(it["device"] == "dev-a" for it in items))
        # limit 钳制在 [1, max_items]，非法值退回默认
        self.assertEqual(len(log.list(limit=0)[0]), 1)
        self.assertEqual(len(log.list(limit=999)[0]), 3)
        self.assertEqual(len(log.list(limit="x")[0]), 3)

    def test_clear(self):
        log = recog_log.RecogLog()
        _entry(log)
        log.clear()
        self.assertEqual(log.list()[1], 0)


class ResponseTest(unittest.TestCase):
    def test_raw_truncated_with_flag(self):
        log = recog_log.RecogLog(raw_max=10)
        e = _entry(log, raw="x" * 50)
        self.assertEqual(len(e["resp"]["raw"]), 10)
        self.assertTrue(e["resp"]["truncated"])

    def test_short_raw_not_flagged(self):
        log = recog_log.RecogLog(raw_max=100)
        e = _entry(log, raw="x" * 50)
        self.assertFalse(e["resp"]["truncated"])

    def test_failed_round_is_logged(self):
        """失败轮同样进日志——"这一轮压根没调通"正是最该被看见的。"""
        log = recog_log.RecogLog()
        e = log.begin("dev", "direct", [], img_orig="data:orig")
        log.set_response(e, False, error="URLError: timed out")
        log.commit(e)
        got = log.list()[0][0]
        self.assertFalse(got["resp"]["ok"])
        self.assertIn("timed out", got["resp"]["error"])
        self.assertEqual(got["resp"]["n_items"], 0)

    def test_set_response_on_none_entry_is_noop(self):
        recog_log.RecogLog().set_response(None, True, raw="x")   # 不抛即可


class ProjectionTest(unittest.TestCase):
    def test_list_strips_heavy_fields(self):
        log = recog_log.RecogLog()
        _entry(log, raw="MODEL RAW OUTPUT")
        got = log.list()[0][0]
        self.assertNotIn("prompt", got["req"])
        self.assertNotIn("raw", got["resp"])
        self.assertNotIn("items", got["resp"])
        self.assertEqual(got["req"]["prompt_len"], 50)      # 只给长度
        self.assertEqual(got["n_candidates"], 1)
        self.assertNotIn("ref_img", got["candidates"][0])   # 参考图留给详情
        self.assertEqual(got["candidates"][0]["name"], "拿铁")
        self.assertEqual(got["img_orig"], "data:orig")      # 请求图列表里就要能看

    def test_full_image_served_by_endpoint_not_inlined(self):
        """原尺寸图不塞进 JSON（几百 KB），只标「有没有」+ 尺寸，图走 full_image()。"""
        log = recog_log.RecogLog()
        e = _entry(log)
        for got in (log.list()[0][0], log.get(e["id"])):
            self.assertNotIn("img_full", got["req"])
            self.assertTrue(got["has_full"])
            self.assertFalse(got["has_boxed_full"])          # 本例直传口径没有图2
            self.assertEqual(got["req"]["img_full_px"], "1280x720")
        self.assertEqual(log.full_image(e["id"]), "data:image/jpeg;base64,FULL")
        self.assertIsNone(log.full_image(e["id"], "boxed"))
        self.assertIsNone(log.full_image(999))

    def test_full_images_dropped_outside_window(self):
        """原尺寸图只留最近 full_keep 条，老条目降级为「只有缩略图与判定」。"""
        log = recog_log.RecogLog(max_items=10, full_keep=2)
        first = _entry(log, "A")
        for name in ("B", "C", "D"):
            _entry(log, name)
        self.assertIsNone(log.full_image(first["id"]))
        self.assertFalse(log.get(first["id"])["has_full"])
        newest = log.list()[0][0]["id"]
        self.assertEqual(log.full_image(newest), "data:image/jpeg;base64,FULL")

    def test_detail_carries_full_text(self):
        log = recog_log.RecogLog()
        e = _entry(log, raw="MODEL RAW OUTPUT")
        got = log.get(e["id"])
        self.assertEqual(got["req"]["prompt"], "P" * 50)
        self.assertEqual(got["resp"]["raw"], "MODEL RAW OUTPUT")
        self.assertEqual(got["resp"]["items"], [{"name": "可乐"}])
        self.assertTrue(got["candidates"][0]["ref_img"].startswith("data:image/jpeg"))

    def test_detail_missing_returns_none(self):
        self.assertIsNone(recog_log.RecogLog().get(404))

    def test_outcome_survives_projection(self):
        log = recog_log.RecogLog()
        _entry(log, "薯条")
        for got in (log.list()[0][0], log.get(1)):
            self.assertEqual(got["outcome"][0]["name"], "薯条")
            self.assertEqual(got["outcome"][0]["action"], "new")


class AppWiringTest(unittest.TestCase):
    """app.py 侧接线：日志真的被写、SAM3 门控真的写观测、接口真的挂上。"""

    @classmethod
    def setUpClass(cls):
        cls.src = (ROOT / "app.py").read_text(encoding="utf-8")

    def test_worker_opens_and_commits_log(self):
        self.assertIn("vlog = _vlmlog_begin(dev, \"direct\" if boxed is None else \"sam3\"",
                      self.src)
        self.assertIn("_vlmlog.commit(vlog)", self.src)
        self.assertIn("_recognize_dedup(orig, boxed, candidates, n_food, n_drink, tgt, log=vlog)",
                      self.src)

    def test_every_gate_writes_reason(self):
        # 五道闸门都要把拒合并原因写进日志，否则控制面只能看到"又建了一张新卡"
        for needle in ("回显不一致：", "类型不一致：", "证据矛盾：", "低置信：", "名称零重叠："):
            self.assertRegex(self.src, r'gate = "%s' % needle)

    def test_routes_registered(self):
        for route in ('@app.get("/api/recoglog/list")',
                      '@app.get("/api/recoglog/{entry_id}")',
                      '@app.post("/api/recoglog/clear")'):
            self.assertIn(route, self.src)

    def test_gate_records_sam3_observation(self):
        # 识别门控是现网唯一在跑的 SAM3 生产链路——不写回控制面日志就是空的
        m = re.search(r"def _sam3_gate_dets\(rgb, device=None\):(.*?)\n# ──", self.src, re.S)
        self.assertIsNotNone(m)
        self.assertIn('src="gate", device=device, n_prod=n_prod', m.group(1))
        # 门控要把 debug 一路带出来（流式步进本就捕获，不多跑推理）
        self.assertIn("insts, _gidx, impl, dbg = _sam3_stream_frame(word, rgb)", m.group(1))

    def test_full_image_is_the_request_body_image(self):
        # 日志里的原图必须是送进请求体的那两张（同一变量），不是另编码一份
        self.assertIn('"img_full": u1, "img_boxed_full": u2', self.src)
        self.assertIn('THUMB_W = int(os.environ.get("OBS_THUMB_W"', self.src)

    def test_image_endpoints_registered(self):
        # 两条链路的原图都要能点开看（控制面上所有图都可点开看原图）
        self.assertIn('@app.get("/api/recoglog/{entry_id}/image/{kind}")', self.src)
        self.assertIn('@app.get("/api/sam3tune/image/{entry_id}/{kind}")', self.src)

    def test_tune_entries_strip_internal_frames(self):
        # 观测条目持有 ndarray（懒编码用），必须在投影时剥掉，否则 JSON 序列化直接炸
        self.assertIn('out = {k: v for k, v in entry.items() if not k.startswith("_")}', self.src)
        self.assertIn('"live": live', self.src)
        self.assertIn('_tune_public(it) for it in items', self.src)

    def test_food_word_not_rerun_when_already_in_wordlist(self):
        # 词面已覆盖就不补跑：现网词表 food/drink 都标 label=drink，按 label 判会白跑一次
        self.assertIn('if not any(w == SAM3_TEXT_DEFAULT for (w, _lbl) in targets):', self.src)

    def test_gate_observation_is_switchable(self):
        # 观测写回跑在识别触发线程里，展台上要能一键关
        self.assertIn('SAM3_OBS_LOG', self.src)
        self.assertIn('if not SAM3TUNE_OBS_ON:', self.src)

    def test_history_sampling_and_filter(self):
        # 门控每轮都写历史会把 30 条缓冲在十几秒内冲掉；控制面还要能按设备/条数拉
        self.assertIn("SAM3TUNE_HIST_MIN_GAP", self.src)
        self.assertIn("def sam3tune_history(device: Optional[str] = None, limit: int = 0):",
                      self.src)


if __name__ == "__main__":
    unittest.main()
