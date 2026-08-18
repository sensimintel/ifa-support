# -*- coding: utf-8 -*-
"""识别输出契约改造：去 box、描述改英/德、调用改流式。

SSE 拼装是纯逻辑模块（recog_sse.py，零 cv2/torch 依赖）直接测；
prompt / 解析 / worker / 页面那侧只做接线断言（正则抽源码），
与 test_recog_log、test_recog_direct 同款做法。
"""
import unittest
from pathlib import Path

import recog_match
import recog_sse

ROOT = Path(__file__).resolve().parents[1]


def _sse(*chunks):
    """把若干 JSON 块包成 SSE 行（收尾补 [DONE]）。"""
    return [("data: " + c).encode() for c in chunks] + [b"data: [DONE]"]


def _delta(text):
    return '{"choices":[{"delta":{"content":"%s"}}]}' % text


class SseTest(unittest.TestCase):
    """流式拼装：内容顺序、ttft 归属、usage 捡拾、脏块容错、总时长兜底。"""

    def test_concat_in_order(self):
        out, usage, ttft = recog_sse.read_sse_completion(
            _sse(_delta("Ban"), _delta("ana")), t_start=0.0, timeout=30, now=lambda: 1.0)
        self.assertEqual(out, "Banana")
        self.assertEqual(usage, {})
        self.assertIsNotNone(ttft)

    def test_ttft_marks_first_content_not_first_chunk(self):
        """空 delta 的开场块（角色块/心跳）不能算首字——否则 ttft 会被低估。"""
        clock = iter([0.5, 0.5, 2.0, 2.0, 3.0, 3.0, 3.0])
        lines = _sse('{"choices":[{"delta":{"role":"assistant"}}]}', _delta("A"))
        out, _u, ttft = recog_sse.read_sse_completion(
            lines, t_start=0.0, timeout=30, now=lambda: next(clock))
        self.assertEqual(out, "A")
        self.assertAlmostEqual(ttft, 2000.0, places=1)

    def test_usage_only_tail_chunk(self):
        out, usage, _t = recog_sse.read_sse_completion(
            _sse(_delta("hi"), '{"choices":[],"usage":{"prompt_tokens":11,"completion_tokens":2}}'),
            t_start=0.0, timeout=30, now=lambda: 1.0)
        self.assertEqual(out, "hi")
        self.assertEqual(usage["prompt_tokens"], 11)
        self.assertEqual(usage["completion_tokens"], 2)

    def test_garbage_and_comment_lines_survive(self):
        """半行 JSON、注释心跳、非 data 行都只能被跳过，不能毁掉整轮。"""
        lines = [b": keep-alive", b"", b"data: {half", b'data: ' + _delta("ok").encode(),
                 b"data: [DONE]"]
        out, _u, _t = recog_sse.read_sse_completion(
            lines, t_start=0.0, timeout=30, now=lambda: 1.0)
        self.assertEqual(out, "ok")

    def test_no_content_gives_none_ttft(self):
        out, _u, ttft = recog_sse.read_sse_completion(
            _sse('{"choices":[{"delta":{}}]}'), t_start=0.0, timeout=30, now=lambda: 1.0)
        self.assertEqual(out, "")
        self.assertIsNone(ttft)

    def test_total_timeout_raises(self):
        """urlopen 的 timeout 只管单次 read，总时长必须由这里兜住。"""
        with self.assertRaises(TimeoutError):
            recog_sse.read_sse_completion(
                _sse(_delta("a"), _delta("b")), t_start=0.0, timeout=5, now=lambda: 99.0)


class WiringTest(unittest.TestCase):
    """app.py 侧接线：输出契约真的改了，而不是只改了 prompt 或只改了解析。"""

    @classmethod
    def setUpClass(cls):
        cls.src = (ROOT / "app.py").read_text(encoding="utf-8")

    def assertHas(self, needle):
        """app.py 里必须有这段。失败只报 needle——assertIn 会把整份源码打进报告。"""
        self.assertTrue(needle in self.src, "app.py 里找不到：%s" % needle)

    def assertLacks(self, needle):
        self.assertTrue(needle not in self.src, "app.py 里仍残留：%s" % needle)

    def test_prompt_asks_en_and_de_description(self):
        self.assertHas("description_en：一句话英文描述")
        self.assertHas("description_de：一句话德文描述")
        self.assertLacks("一句话中文描述")

    def test_german_description_gets_a_longer_budget(self):
        """德语句子比英语长，两者共用 60 字符预算会把德文结尾削掉半个词。"""
        self.assertHas("RECOG_DESC_DE_MAX = 90")
        self.assertHas("[:RECOG_DESC_DE_MAX]")

    def test_prompt_no_longer_asks_for_box(self):
        self.assertLacks("box：该物品在图1中的包围框")
        self.assertLacks("412,530,668,845")

    def test_parse_emits_de_and_drops_box(self):
        self.assertHas('"description_en": desc_en, "description_de": desc_de')
        self.assertLacks('raw_box = it.get("box")')

    def test_card_carries_de_not_chinese_desc(self):
        self.assertHas('"description_de": it.get("description_de", "")')
        self.assertLacks('"description": it["description"]')

    def test_thumbnail_renders_without_boxes(self):
        """模型不再给 box → 缩略图只能渲无框点云，不许再拼 shot_dets。"""
        self.assertHas("shot_url = _save_cloud_shot(pred, [], conf)")
        self.assertLacks("shot_dets = [")

    def test_ref_crop_helper_removed(self):
        """按 box 裁特写的函数没有调用方了，必须一起删掉，不留死代码。"""
        self.assertLacks("_make_ref_crop")
        self.assertLacks("REF_CROP_MARGIN")

    def test_stream_enabled_and_switchable(self):
        self.assertHas('RECOG_STREAM = os.environ.get("RECOG_STREAM", "1")')
        self.assertHas('payload["stream"] = True')
        self.assertHas('payload["stream_options"] = {"include_usage": True}')
        self.assertHas("recog_sse.read_sse_completion(")

    def test_stream_flag_survives_log_projection(self):
        """req 字段有白名单，新加的 stream 不进白名单就到不了控制面（实测踩过）。"""
        import recog_log
        self.assertIn("stream", recog_log._REQ_KEYS)

    def test_ttft_recorded_in_timings(self):
        self.assertHas('"ttft_ms": (round(ttft_ms, 1) if ttft_ms is not None else None)')

    def test_recog_page_shows_en_and_de(self):
        self.assertHas("c.description_en")
        self.assertHas("c.description_de")
        self.assertLacks("el.querySelector('.desc').textContent=c.description;")


if __name__ == "__main__":
    unittest.main()


class EvidenceCodeTest(unittest.TestCase):
    """证据码：合法性校验 + 解码。旧版闸门判的是「文本含『不一致』」，
    模型换个措辞就绕过去了；码可校验，非法本身就是拒合并的理由。"""

    def test_all_consistent_passes(self):
        verdict, text = recog_match.check_evidence("B1C1S1V1")
        self.assertEqual(verdict, "ok")
        self.assertEqual(text, "品牌与包装一致；颜色与外观一致；形状与份量一致；容器与摆放一致")

    def test_any_zero_blocks_merge(self):
        verdict, text = recog_match.check_evidence("B1C0S1V?")
        self.assertEqual(verdict, "mismatch")
        self.assertIn("颜色与外观不一致", text)
        self.assertIn("容器与摆放看不清", text)

    def test_unclear_alone_still_ok(self):
        """? 不是矛盾，只该压低 confidence，别在闸门三就拦——那是闸门四的活。"""
        self.assertEqual(recog_match.check_evidence("B1C?S1V1")[0], "ok")

    def test_none_with_match_is_contradiction(self):
        """自称无相似候选却仍给了 match：矛盾，拒合并。"""
        verdict, text = recog_match.check_evidence("NONE")
        self.assertEqual(verdict, "none")
        self.assertEqual(text, "无相似候选")

    def test_case_insensitive(self):
        self.assertEqual(recog_match.check_evidence("b1c1s1v1")[0], "ok")
        self.assertEqual(recog_match.check_evidence("none")[0], "none")

    def test_malformed_codes_rejected(self):
        for bad in ("", None, "B1C1S1", "B1C1S1V2", "品牌一致；颜色一致", "B1C1S1V1X"):
            self.assertEqual(recog_match.check_evidence(bad)[0], "malformed", bad)

    def test_malformed_text_carries_the_bad_code(self):
        """非法码要原样进日志，否则排障时看不出模型到底吐了什么。"""
        self.assertIn("品牌一致", recog_match.check_evidence("品牌一致")[1])


class MatchFieldWiringTest(WiringTest):
    """match 三字段：evidence 换码、reason 删除、matched_name 保持不动。"""

    def test_prompt_asks_for_evidence_code(self):
        self.assertHas("结果写成 8 字符证据码 BxCxSxVx")
        self.assertHas("无相似候选时写 NONE")

    def test_match_reason_gone_everywhere(self):
        self.assertLacks("match_reason")

    def test_matched_name_gate_intact(self):
        """闸门一是抓编号幻觉的唯一防线，不能跟着一起删。"""
        self.assertHas('it.get("matched_name") or "").strip().casefold()')

    def test_gate_three_uses_code_verdict(self):
        self.assertHas("ev_verdict, ev_text = recog_match.check_evidence")
        self.assertHas('elif ev_verdict != "ok":')
        self.assertLacks('"不一致" in (it.get("match_evidence")')

    def test_log_stores_decoded_text_not_raw_code(self):
        """控制面给人看，存解码后的中文——superadmin 那侧因此不用改。"""
        self.assertHas('"evidence": ev_text')


def _load_pick_rule():
    """从 app.py 抽出 _pick_rule 源码执行——它只依赖内置，不用扛 cv2/torch。"""
    import re as _re
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    m = _re.search(r"^def _pick_rule\(last_pick, candidates\):.*?\n(?=\n\ndef )", src, _re.M | _re.S)
    if not m:
        raise AssertionError("app.py 里找不到 _pick_rule —— 单选粘性规则被改名或删了")
    ns = {}
    exec(m.group(0), ns)
    return ns["_pick_rule"]


class PickRuleTest(unittest.TestCase):
    """一轮只出一个 item 时的挑选规则：上次选中的还在就别换，否则挑最有把握的。"""

    @classmethod
    def setUpClass(cls):
        cls.rule = staticmethod(_load_pick_rule())

    def test_no_history_degrades_to_most_confident(self):
        for empty in (None, {}, {"name": ""}):
            t = self.rule(empty, [])
            self.assertIn("挑你最有把握", t)
            self.assertNotIn("上一轮", t)

    def test_history_in_candidates_carries_the_index(self):
        """模型对编号的指称比对名称稳，上次那个还在候选里就把编号一起给它。"""
        cands = [{"id": 7, "name": "Water"}, {"id": 9, "name": "Wine"}]
        t = self.rule({"name": "Wine", "card_id": 9}, cands)
        self.assertIn("「Wine」", t)
        self.assertIn("清单[2]", t)
        self.assertIn("仍在当前画面里", t)

    def test_history_not_in_candidates_falls_back_to_name(self):
        """上次那张卡掉出活跃窗口后不在候选里了，只提名称，别编个不存在的编号。"""
        t = self.rule({"name": "Persimmon", "card_id": 99}, [{"id": 7, "name": "Water"}])
        self.assertIn("「Persimmon」", t)
        self.assertNotIn("清单[", t)

    def test_history_survives_empty_candidate_list(self):
        t = self.rule({"name": "Water", "card_id": 1}, None)
        self.assertIn("「Water」", t)


class SingleItemWiringTest(WiringTest):
    """每轮只更新一张卡：prompt 明令单选 + worker 侧门禁兜底。"""

    def test_prompt_demands_exactly_one_item(self):
        self.assertHas("**画面里就算有好几样，也只输出一个 item**")
        self.assertHas("items 里**只放一个对象**")
        self.assertLacks("逐一分别输出")

    def test_prompt_carries_last_pick_rule(self):
        self.assertHas("+ _pick_rule(last_pick, candidates)")
        # 参考食物库上线后这行还多带了 refs=refs，只断言粘性锚确实传了下去
        self.assertHas("last_pick=lp,")

    def test_gate_truncates_to_one(self):
        self.assertHas('gate_dropped = [str(i.get("name") or "") for i in items[1:]]')
        self.assertHas("items = items[:1]")

    def test_gate_runs_before_cards_are_touched(self):
        """门禁必须卡在「拿到结果」与「更新卡片」之间，落到循环里就晚了。"""
        gate = self.src.index("gate_dropped = [str(i.get")
        touch = self.src.index("cards = _recog_cards.setdefault(dev, [])")
        self.assertLess(gate, touch)

    def test_dropped_items_are_observable(self):
        """被丢弃的项要能在控制面看到，否则「模型到底守不守规矩」无从判断。"""
        self.assertHas('vlog["resp"]["gate_dropped"] = gate_dropped')
        import recog_log
        self.assertIn("gate_dropped", recog_log._RESP_KEYS)

    def test_last_pick_written_back_and_expires(self):
        self.assertHas("_recog_last_pick[dev] = {")
        self.assertHas('if lp and time.time() - lp.get("ts", 0) > RECOG_ACTIVE_WINDOW:')

    def test_clear_drops_last_pick(self):
        """卡都清空了还留着上次命中，下一轮 prompt 会指着一张不存在的卡说话。"""
        self.assertHas("_recog_last_pick.pop(dev, None)")


class TunnelProbeWiringTest(WiringTest):
    """隧道探测三态：拥塞不得被当成断线，否则守护会拆掉正在干活的隧道。"""

    def test_probe_returns_three_states(self):
        self.assertHas('state = "busy"')
        self.assertHas('state = "down"')
        self.assertHas('isinstance(e.reason, (socket.timeout, TimeoutError))')

    def test_busy_keeps_previous_up_flag(self):
        """busy 时状态灯不许翻红——链路没断，翻红会让人以为故障。"""
        self.assertHas('_tunnel["up"] = _tunnel["up"] if state == "busy" else (state == "up")')

    def test_probe_timeout_widened(self):
        self.assertHas("TUNNEL_PROBE_TIMEOUT = 8.0")
        self.assertLacks("timeout=3.0")

    def test_keeper_endpoint_exposes_state(self):
        self.assertHas('"rebuild": req, "up": up, "state": state')


class KeeperRebuildPolicyTest(unittest.TestCase):
    """守护的重建判据：只认「进程退出」和「硬失败」，拥塞另算。"""

    @classmethod
    def setUpClass(cls):
        cls.src = (ROOT / "tools" / "qwen_tunnel_keeper.py").read_text(encoding="utf-8")

    def assertHas(self, needle):
        self.assertTrue(needle in self.src, "keeper 里找不到：%s" % needle)

    def test_busy_does_not_count_as_down(self):
        self.assertHas('down_n = down_n + 1 if server_state == "down" else 0')
        self.assertHas('busy_n = busy_n + 1 if server_state == "busy" else 0')

    def test_rebuild_needs_dead_or_hard_failure(self):
        self.assertHas("elif (dead or down_n >= 3 or busy_n >= BUSY_LIMIT)")

    def test_busy_backstop_is_far_from_trigger_happy(self):
        """兜底阈值必须远大于硬失败阈值，否则满负载下会退化成定时重建。"""
        import re as _re
        limit = int(_re.search(r"^BUSY_LIMIT = (\d+)", self.src, _re.M).group(1))
        self.assertGreaterEqual(limit, 30, "拥塞兜底阈值太小，满负载会被误触发")

    def test_heartbeat_timeout_widened(self):
        self.assertHas("HEARTBEAT_TIMEOUT = 15.0")
        self.assertHas("timeout=HEARTBEAT_TIMEOUT")

    def test_falls_back_when_server_has_no_state_field(self):
        """守护可能先于 5090 升级，老服务端没有 state 字段时不能乱重建。"""
        self.assertHas('if state not in ("up", "busy", "down"):')


class OrderingWiringTest(WiringTest):
    """时序保护：更旧的结果不许覆盖更新的，也不许把待机态顶回卡片态。"""

    def test_watermark_uses_wall_clock_not_seq(self):
        """seq 在设备桶闲置 60s 被清后会从 1 重计，拿它当水位线会永久卡死。"""
        self.assertHas("_recog_applied_at = {}")
        self.assertHas('frame_at < _recog_applied_at.get(dev, 0.0)')
        self.assertLacks('_recog_applied_at[dev] = stage["seq"]')

    def test_check_and_advance_share_one_lock(self):
        """检查与推进必须在同一把锁内，否则并发多路会同时通过检查。"""
        seg = self.src[self.src.index("stale = (frame_at is not None"):][:700]
        self.assertIn("_recog_applied_at[dev] = frame_at", seg)
        self.assertIn("dropped_out_of_order", seg)

    def test_stale_result_does_not_land(self):
        self.assertHas("items = []       # 后面的落卡循环自然空转")

    def test_gate_runs_before_cards_are_touched(self):
        gate = self.src.index("stale = (frame_at is not None")
        touch = self.src.index("cards = _recog_cards.setdefault(dev, [])")
        self.assertLess(gate, touch)

    def test_card_timestamp_is_frame_time_not_land_time(self):
        """飞了 20 秒才回来的结果带的是 20 秒前的画面，不该盖一个"此刻"的时间戳。"""
        self.assertHas('target["last_ts"] = frame_at or now')
        self.assertHas('"last_ts": frame_at or now')
        self.assertLacks('target["last_ts"] = now\n')

    def test_frontend_ignores_stale_head_entirely(self):
        """不只是不上屏——curCard/lastCardKey/cardShownAt 都不能动，
        否则下次更新会先闪一下旧内容。口径是**帧时刻**（now-last_ts=端到端延时），
        阈值走独立的 FRESH_TTL_MS，不再与驻留共用一个数。"""
        self.assertHas("const ts=head?(head.last_ts?head.last_ts*1000:Date.now()):0;")
        self.assertHas("if(head&&Date.now()-ts<=FRESH_TTL_MS){")
        # 过期分支里不许有任何状态写入：整个赋值都在 if 内
        body = self.src[self.src.index("function applyRecog(r){"):]
        body = body[:body.index("\n}")]
        for name in ("curCard=", "lastCardKey=", "cardShownAt="):
            self.assertLess(body.index("if(head&&Date.now()-ts<=FRESH_TTL_MS){"),
                            body.index(name), "%s 必须在新鲜度闸之内" % name)

    def test_drops_are_observable(self):
        self.assertHas('"dropped_out_of_order": 0')
        self.assertHas('vlog["resp"]["dropped_reason"]')
        import recog_log
        self.assertIn("dropped_reason", recog_log._RESP_KEYS)
