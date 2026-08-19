import ast
import json
import tempfile
import unittest
from pathlib import Path


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def load_pointcloud_preset():
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == "_sam3hl_cfg" for target in node.targets):
                return ast.literal_eval(node.value)
    raise AssertionError("_sam3hl_cfg was not found")


class PointcloudPresetTest(unittest.TestCase):
    def test_default_preset_matches_saved_experience_style(self):
        preset = load_pointcloud_preset()

        expected = {
            "style": "solid",
            "strength": 65,
            "dim": 52,
            "color_mode": "custom",
            "color": "#fffdf7",
            "view_tilt": 0.0,
            "view_zoom": 1.40,
            "eye_lift": 0.0,
            "eye_back": 0.05,
            "sat": 0.0,
            "val": 1.35,
            "conf": 18,
            "hue": 0.0,
            "outlier_mad": 10.0,
            "pt_size": 0.65,
            "pt_shape": 2,
            "pt_atten": 0,
            "pt_opacity": 0.82,
            "pt_blend": 1,
            "pt_density": 100,
            "pt_hue": 0.0,
            "pt_sat": 0.0,
            "pt_val": 1.18,
            "pt_contrast": 1.35,
            "pt_exposure": 1.65,
            "pt_colormode": 0,
            "pt_fog": 0.0,
            "pt_pulse": 1,
            "pt_pulse_speed": 0.35,
            "pt_sparkle": 0.24,
            "pt_bg": "#000000",
        }

        for key, value in expected.items():
            self.assertEqual(preset[key], value, key)

    def test_highlight_config_is_persisted_to_a_local_preset_file(self):
        source = APP_PATH.read_text(encoding="utf-8")

        self.assertIn('_SAM3HL_PRESET_PATH = Path(__file__).resolve().parent / "sam3hl_preset.json"', source)
        self.assertIn("def _load_sam3hl_preset()", source)
        self.assertIn("def _save_sam3hl_preset()", source)
        self.assertIn("_save_sam3hl_preset()", source)

    def test_saved_preset_json_round_trips_the_current_style(self):
        preset = load_pointcloud_preset()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sam3hl_preset.json"
            path.write_text(json.dumps(preset, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
            restored = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(restored, preset)

    def test_experience_overlay_matches_figma_logo_and_idle_layout(self):
        source = APP_PATH.read_text(encoding="utf-8")

        # logo：Figma 653-25299，左边距 26、210×60、垂直居中
        self.assertIn("#logo{position:absolute;left:.26rem;top:50%;transform:translateY(-50%);width:2.1rem", source)
        # UI 定位画布：设计稿 2240×1260 内接屏幕并水平居中，UI 不贴视口两端
        self.assertIn("width:22.4rem;height:12.6rem", source)
        # 面板只负责贴右与垂直居中，宽度/右边距由各态自己给
        self.assertIn("#panel{position:absolute;right:0;top:50%;transform:translateY(-50%);display:grid;justify-items:end}", source)
        # 待机态 Figma 653-25306：515 宽、右边距 43
        self.assertIn("#idle{width:5.15rem;margin-right:.43rem}", source)
        # 识别成功卡 Figma 748-1145：532 宽、右边距 26、40px 背景模糊
        self.assertIn("#card{position:relative;width:5.32rem;margin-right:.26rem;", source)
        self.assertIn("backdrop-filter:blur(.4rem)", source)
        # 出入场动效：玻璃底独立成层做缩放、每行套遮罩从下往上推入
        self.assertIn("#cardbg{position:absolute;inset:0", source)
        self.assertIn("#card.bgin #cardbg{transform:scaleY(1)}", source)
        self.assertIn("#card .rvw>*{transform:translateY(105%);opacity:0;", source)
        self.assertIn("#card .rvw.in>*{transform:none;opacity:1}", source)
        # 标题行不裁切：h1 的 g/y 降部伸出行盒，遮罩会齐根切掉
        self.assertIn("#card .rvw.nom{overflow:visible}", source)
        self.assertIn('<div class="rvw nom"><h1 id="cname"></h1></div>', source)
        # 分割线不占逐行节拍，等内容行全部落地后一起出现
        self.assertIn("const rows=cardRows(),isRule=w=>!!w.querySelector('hr');", source)
        self.assertIn("rules.forEach(w=>cardTimers.push(setTimeout(()=>w.classList.add('in'),tail)))", source)
        self.assertIn("h1{font-family:'ABC Arizona Serif',Georgia,serif;font-weight:400;font-size:.5rem", source)


if __name__ == "__main__":
    unittest.main()
