# -*- coding: utf-8 -*-
"""识别输出契约改造：去 box、描述改英/德、调用改流式。

SSE 拼装是纯逻辑模块（recog_sse.py，零 cv2/torch 依赖）直接测；
prompt / 解析 / worker / 页面那侧只做接线断言（正则抽源码），
与 test_recog_log、test_recog_direct 同款做法。
"""
import unittest
from pathlib import Path

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

    def test_ttft_recorded_in_timings(self):
        self.assertHas('"ttft_ms": (round(ttft_ms, 1) if ttft_ms is not None else None)')

    def test_recog_page_shows_en_and_de(self):
        self.assertHas("c.description_en")
        self.assertHas("c.description_de")
        self.assertLacks("el.querySelector('.desc').textContent=c.description;")


if __name__ == "__main__":
    unittest.main()
