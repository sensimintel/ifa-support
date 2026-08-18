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


class TimingsTest(unittest.TestCase):
    """链路分段：识别调用内部的三段并进 entry["timings"]，投影时原样带出。"""

    def test_set_response_merges_timings(self):
        log = recog_log.RecogLog()
        e = log.begin("dev", "direct", [], img_orig="data:orig")
        log.set_response(e, True, raw="{}", items=[],
                         timings={"encode_ms": 12.3, "http_ms": 3400.0, "parse_ms": 0.4})
        e["timings"]["total_ms"] = 4200.0          # worker 落卡后补的端到端
        log.commit(e)
        got = log.list()[0][0]["timings"]
        self.assertEqual(got["http_ms"], 3400.0)
        self.assertEqual(got["encode_ms"], 12.3)
        self.assertEqual(got["total_ms"], 4200.0)

    def test_failed_round_still_reports_encode_and_http(self):
        """调用失败也要留分段——「卡在网络还是卡在编码」正是这时候最该知道的。"""
        log = recog_log.RecogLog()
        e = log.begin("dev", "sam3", [], img_orig="data:orig")
        log.set_response(e, False, error="timed out",
                         timings={"encode_ms": 10.0, "http_ms": 30000.0})
        log.commit(e)
        got = log.list()[0][0]
        self.assertFalse(got["resp"]["ok"])
        self.assertEqual(got["timings"]["http_ms"], 30000.0)


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
        # （断言失败只报 needle：assertRegex 会把整份 app.py 打进报告，噪音淹没结论）
        for needle in ("回显不一致：", "类型不一致：", "证据不通过（", "低置信：", "名称零重叠："):
            self.assertTrue('gate = "%s' % needle in self.src,
                            "闸门没把拒合并原因写进日志：%s" % needle)

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

    def test_end_to_end_timing_starts_at_frame_arrival(self):
        """端到端从「帧到达 8060」起算，不是从触发起算——否则帧在缓存里等的时间被藏掉。"""
        self.assertIn('"frame_age_ms": (round((stage["pick_at"] - stage["frame_recv_at"]) * 1000.0, 1)',
                      self.src)
        self.assertIn('vlog["timings"]["total_ms"] = (round((now2 - stage["frame_recv_at"])',
                      self.src)
        # 拿不到帧时刻时留空，不用别的起点冒充
        self.assertIn('if stage.get("frame_recv_at") else None', self.src)

    def test_frame_age_excludes_decode_and_gate(self):
        """帧龄只能是「帧在缓存里等触发线程」那段：用入队时刻减帧到达时刻的话，
        解码与 SAM3 门控（都发生在入队之前）会被算两遍，各段之和就大于端到端。"""
        self.assertIn('"pick_at": _t', self.src)
        loop = self.src[self.src.index("def _recog_direct_loop():"):]
        self.assertNotIn('enq_ts - stage["frame_recv_at"]', self.src)
        # 触发线程里门控跑完到入队的零头单独留一格，不并进别的段
        self.assertIn('"enqueue_ms":', self.src)
        self.assertIn('_t = time.time()          # 触发线程拿起这一帧、开始本轮处理的时刻', loop)

    def test_vlm_call_split_into_local_and_network(self):
        """VLM 那段拆成 本地编码 / HTTP 往返 / 解析——只报一个 llm_ms 归不了因。"""
        for needle in ('"encode_ms": round(encode_ms, 1)',
                       '"http_ms": round(http_ms, 1)',
                       '"parse_ms": round((time.time() - _t_parse) * 1000.0, 1)',
                       '"req_bytes": len(body)'):
            self.assertIn(needle, self.src, needle)

    def test_list_endpoint_refreshes_the_baseline(self):
        """基线要有人去刷新，否则 HTTP 段永远拆不开：放在列表接口里（有人看才探，
        自带 2s 缓存），而不是识别 worker 里。"""
        fn = self.src[self.src.index("def recoglog_list("):]
        fn = fn[:fn.index("@app.get(\"/api/recoglog/{entry_id}\")")]
        self.assertIn("_tunnel_probe()", fn)

    def test_tunnel_rtt_is_read_only_baseline(self):
        """网络基线取自隧道探测的缓存值，worker 里绝不主动再发一枪（会搅乱被测对象）。"""
        fn = self.src[self.src.index("def _tunnel_rtt_ms():"):]
        fn = fn[:fn.index("def _tunnel_probe")]
        self.assertIn('return _tunnel["rtt_ms"]', fn)
        self.assertNotIn("urlopen", fn)

    def test_word_label_is_explicit_then_sticky(self):
        """label 决定命中算 food 还是 drink（框颜色 + prompt 的软接地信息）：
        显式给了就用，只发词面则沿用旧值，最后才落词面兜底——绝不再无脑 drink。"""
        cfg = self.src[self.src.index("def sam3tune_config_set"):]
        cfg = cfg[:cfg.index("@app.get(\"/api/sam3tune/state\")")]
        self.assertIn('known = {w["word"]: w.get("label") for w in', cfg)
        self.assertIn('label = known.get(word) or ("food" if word == SAM3_TEXT_DEFAULT else "drink")',
                      cfg)
        self.assertNotIn('else None) or "drink"', cfg)   # 老的无脑兜底不能再出现

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
