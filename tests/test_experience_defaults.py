# -*- coding: utf-8 -*-
"""/experience 内置默认值 == 服务器「默认」配置预设（2026-08-17 调定）。

三层各自的默认落点：
  · display（深度显示）→ 页内 JS `let ddCss={...}`
  · dot（点云化）      → 页内 JS `let dotCfg={...}`
  · config（深度渲染） → mini 端 cam_pusher 的各处 fallback 值
用正则从源码抽取 JS 对象字面量逐键比对；深度渲染只抽查与旧默认不同的键
（radar/EMA 0.2/JPEG 80/30fps/量程 0.05~1.4），其余键历史默认本就与预设一致。"""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 服务器「默认」预设的 display / dot 快照（saved_at 1786947200）
PRESET_DISPLAY = {"br": 2.5, "ct": 1, "sa": 1.05, "hu": 0, "inv": 0, "bl": 0,
                  "op": 1, "fit": "cover", "mir": 1, "rot": 0, "pix": 0}
PRESET_DOT = {"on": 1, "mode": 1, "pitch": 5, "r": 34, "jitter": 0, "motion": 0,
              "speed": 3, "bg": "#000000", "gshape": 1, "pn": 50000, "psize": 3.5,
              "pcontrast": 0.5, "pmono": 0, "pcolor": "#fffdf7", "pdrift": 3,
              "prespawn": 0, "pfloat": 0, "pshape": 1, "pdepth": 100}


def _parse_js_object(src: str, var: str) -> dict:
    """从源码抽取 `let <var>={...};` 字面量并解析成 dict（数值/单引号字符串两类值）。"""
    m = re.search(r"let %s=\{(.*?)\};" % var, src, re.S)
    assert m, f"未找到 let {var}={{...}}"
    out = {}
    for k, v in re.findall(r"(\w+):('[^']*'|[-\d.]+)", m.group(1)):
        out[k] = v.strip("'") if v.startswith("'") else (float(v) if "." in v else int(v))
    return out


class ExperienceDefaultsTest(unittest.TestCase):
    def test_display_defaults_match_preset(self):
        got = _parse_js_object((ROOT / "app.py").read_text(encoding="utf-8"), "ddCss")
        self.assertEqual(got, PRESET_DISPLAY)

    def test_dot_defaults_match_preset(self):
        got = _parse_js_object((ROOT / "app.py").read_text(encoding="utf-8"), "dotCfg")
        self.assertEqual(got, PRESET_DOT)

    def test_default_bg_source_is_g335_device_depth(self):
        # 打开默认=g335 的设备深度图：来源兜底 devdepth + 加载时按需选中 g335 一次
        src = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn("localStorage.getItem('exp_bg')||'devdepth'", src)
        self.assertIn("const PREF_DEVICE='macmini-g335'", src)

    def test_pusher_depth_render_defaults_match_preset(self):
        src = (ROOT / "mac-mini" / "cam_pusher.py").read_text(encoding="utf-8")
        for needle in ('os.environ.get("DEPTH_MIN_M", "0.05")',
                       'os.environ.get("DEPTH_MAX_M", "1.4")',
                       '_cfg_num(cfg, "depth_ema", 0.2)',
                       '_cfg_num(cfg, "depth_jpeg_q", 80)',
                       'cfg.get("depth_colormap", "radar")',
                       '_cfg_num(dcfg, "depth_fps", 30.0)'):
            self.assertIn(needle, src, needle)


if __name__ == "__main__":
    unittest.main()
