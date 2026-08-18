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
