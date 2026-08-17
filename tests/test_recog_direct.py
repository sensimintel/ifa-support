# -*- coding: utf-8 -*-
"""主链路直传 VLM 识别：配置规范化/钳制/落盘 + 触发闸门 + /experience 抽屉接线。

配置与闸门是纯逻辑模块（recog_direct.py，零 torch 依赖）直接测；
app.py / 页面那侧只做接线断言（正则抽源码），与 test_experience_defaults 同款做法。
"""
import json
import re
import tempfile
import unittest
from pathlib import Path

import recog_direct

ROOT = Path(__file__).resolve().parents[1]


class NormalizeTest(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual(recog_direct.normalize({}), recog_direct.DEFAULTS)
        # 默认口径：开启、0.5s、串行
        self.assertEqual(recog_direct.DEFAULTS,
                         {"on": True, "interval_s": 0.5, "concurrency": 1})

    def test_merge_patch_keeps_untouched_keys(self):
        base = {"on": True, "interval_s": 2.0, "concurrency": 3}
        got = recog_direct.normalize({"interval_s": 1.0}, base)
        self.assertEqual(got, {"on": True, "interval_s": 1.0, "concurrency": 3})

    def test_clamp_out_of_range(self):
        got = recog_direct.normalize({"interval_s": 999, "concurrency": 99})
        self.assertEqual(got["interval_s"], recog_direct.LIMITS["interval_s"][1])
        self.assertEqual(got["concurrency"], recog_direct.LIMITS["concurrency"][1])
        got = recog_direct.normalize({"interval_s": 0, "concurrency": 0})
        self.assertEqual(got["interval_s"], recog_direct.LIMITS["interval_s"][0])
        self.assertEqual(got["concurrency"], recog_direct.LIMITS["concurrency"][0])

    def test_bad_values_ignored(self):
        base = {"on": True, "interval_s": 1.5, "concurrency": 2}
        for patch in ({"interval_s": "abc"}, {"concurrency": None},
                      {"interval_s": float("nan")}, {"unknown": 1}, "not a dict", None):
            self.assertEqual(recog_direct.normalize(patch, base), base, patch)

    def test_on_accepts_string_and_bool(self):
        for val, want in (("0", False), ("false", False), ("off", False),
                          ("1", True), ("true", True), (0, False), (1, True)):
            self.assertIs(recog_direct.normalize({"on": val})["on"], want, val)

    def test_concurrency_is_int(self):
        self.assertIsInstance(recog_direct.normalize({"concurrency": 3.7})["concurrency"], int)


class GateTest(unittest.TestCase):
    """触发闸门：直传恒放行；SAM3 口径要有证据、且直传关着。"""

    def test_direct_always_passes(self):
        self.assertTrue(recog_direct.should_trigger(True, False, True))
        self.assertTrue(recog_direct.should_trigger(True, False, False))

    def test_sam3_blocked_when_direct_on(self):
        self.assertFalse(recog_direct.should_trigger(False, True, True))

    def test_sam3_passes_when_direct_off_with_evidence(self):
        self.assertTrue(recog_direct.should_trigger(False, True, False))
        self.assertFalse(recog_direct.should_trigger(False, False, False))


class ConfigPersistTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "recog_direct_cfg.json"

    def tearDown(self):
        self._tmp.cleanup()

    def test_update_persists_and_reloads(self):
        cfg = recog_direct.DirectConfig(self.path)
        cfg.update({"on": False, "interval_s": 3.0, "concurrency": 4})
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8"))["interval_s"], 3.0)
        again = recog_direct.DirectConfig(self.path)      # 重启后回读，不回落默认
        self.assertEqual(again.snapshot(),
                         {"on": False, "interval_s": 3.0, "concurrency": 4})
        self.assertFalse(again.enabled())
        self.assertEqual(again.interval_s(), 3.0)
        self.assertEqual(again.concurrency(), 4)

    def test_broken_file_falls_back_to_defaults(self):
        self.path.write_text("{ 不是 json", encoding="utf-8")
        self.assertEqual(recog_direct.DirectConfig(self.path).snapshot(),
                         recog_direct.DEFAULTS)


class WiringTest(unittest.TestCase):
    """接线断言：后端闸门与前端抽屉控件都在，防日后改动悄悄改掉触发口径。"""

    def setUp(self):
        self.src = (ROOT / "app.py").read_text(encoding="utf-8")

    def test_gate_is_single_source(self):
        # _maybe_recognize 只认 recog_direct.should_trigger 这一处规则
        self.assertIn("recog_direct.should_trigger(direct, bool(detections)", self.src)

    def test_direct_loop_uses_rgb_frame_and_seq_dedup(self):
        self.assertIn("get_latest_frame_seq(dev)", self.src)
        self.assertIn("direct=True", self.src)

    def test_gate_mode_runs_sam3_on_rgb_and_still_recognizes(self):
        """直传关 = SAM3 先筛 + 命中照样送 VLM（不是"只跑 SAM3 不识别"）。"""
        self.assertIn("def _sam3_gate_dets(rgb, device=None):", self.src)
        loop = self.src[self.src.index("def _recog_direct_loop():"):]
        loop = loop[:loop.index("threading.Thread(target=_recog_direct_loop")]
        self.assertIn("_sam3_gate_dets(arr, device=dev)", loop)
        # 门控命中后必须走 direct=False 的识别提交（带框 + 检测器口径 prompt）
        self.assertIn("direct=False", loop)
        # food 也算证据（历史单目口径只收 drink，食物因此从不触发）
        gate = self.src[self.src.index("def _sam3_gate_dets(rgb, device=None):"):]
        self.assertIn('lbl == "food"', gate[:gate.index("def _recog_direct_loop")])

    def test_direct_prompt_variant_exists(self):
        # 直传口径没有图2带框图，prompt 必须换成「只有图1」的说法
        self.assertIn("图1=当前画面原图（没有检测框", self.src)

    def test_experience_drawer_has_direct_controls(self):
        for ident in ("r_rd_itv", "r_rd_conc", 'name="rdon"', "/api/recog/direct/config"):
            self.assertIn(ident, self.src, ident)

    def test_direct_slider_range_matches_backend_limits(self):
        m = re.search(r'id="r_rd_itv" min="([\d.]+)" max="([\d.]+)"', self.src)
        self.assertIsNotNone(m)
        self.assertEqual((float(m.group(1)), float(m.group(2))),
                         recog_direct.LIMITS["interval_s"])
        m = re.search(r'id="r_rd_conc" min="(\d+)" max="(\d+)"', self.src)
        self.assertIsNotNone(m)
        self.assertEqual((int(m.group(1)), int(m.group(2))),
                         recog_direct.LIMITS["concurrency"])


class TrimmedSurfaceTest(unittest.TestCase):
    """去冗余(2026-08-17)后的接线断言：删掉的链路不要悄悄长回来。

    每个「环节」只留一份样本：DA3+SAM3 能力样本=单目链；点云构建=单目 + devpc；
    识别实现=产线一份；调试页=/panel + /recog + /sam3tune。"""

    def setUp(self):
        self.app = (ROOT / "app.py").read_text(encoding="utf-8")
        self.relay = (ROOT / "frame_relay.py").read_text(encoding="utf-8")
        self.pusher = (ROOT / "mac-mini" / "cam_pusher.py").read_text(encoding="utf-8")

    def test_stereo_chain_gone(self):
        for token in ("_stereo_sam3_overlays", "build_stereo_pointcloud_glb",
                      "/api/stereohl/status", "_sam3_ir_stream_frame"):
            self.assertNotIn(token, self.app, token)
        for token in ("/api/frame/aux", "stereo_product", "set_stereo_processor"):
            self.assertNotIn(token, self.relay, token)
        for token in ("_push_aux", "_IRCache", "LEFT_IR_SENSOR"):
            self.assertNotIn(token, self.pusher, token)

    def test_experience_sources_are_hardware_depth_only(self):
        i = self.app.index('<select id="selStyle">')
        block = self.app[i:self.app.index("</select>", i)]
        opts = re.findall(r'<option value="(\w+)"', block)
        self.assertEqual(opts, ["devdepth", "devpc"], "背景来源应只剩两条硬件深度链路")

    def test_experiment_bench_and_gradio_gone(self):
        self.assertFalse((ROOT / "exp_app.py").exists())
        self.assertFalse((ROOT / "run-exp.sh").exists())
        for token in ("import gradio", "mount_gradio_app", "SPLIT_PAGE", "SAM3_PAGE"):
            self.assertNotIn(token, self.app, token)

    def test_mono_chain_kept_as_sample(self):
        # 单目链是保留的唯一 DA3+SAM3 能力样本（现网 DISABLE_MONO_PIPELINE=1 停用中）
        for token in ("_maybe_sam3cloud", "_sam3cloud_refresh", "_sam3_recent_drinks",
                      "DISABLE_MONO_PIPELINE", "build_pointcloud_boxes_glb"):
            self.assertIn(token, self.app, token)


if __name__ == "__main__":
    unittest.main()
