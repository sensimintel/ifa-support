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
            "dim": 30,
            "color_mode": "auto",
            "view_tilt": 0.0,
            "view_zoom": 1.40,
            "eye_lift": 0.0,
            "eye_back": 0.05,
            "sat": 2.0,
            "val": 1.2,
            "conf": 30,
            "hue": 40.0,
            "outlier_mad": 10.0,
            "pt_size": 0.9,
            "pt_shape": 1,
            "pt_atten": 1,
            "pt_opacity": 1.0,
            "pt_blend": 0,
            "pt_density": 88,
            "pt_hue": 0.0,
            "pt_sat": 0.9,
            "pt_val": 1.1,
            "pt_contrast": 1.0,
            "pt_exposure": 2.2,
            "pt_colormode": 0,
            "pt_fog": 0.0,
            "pt_pulse": 1,
            "pt_pulse_speed": 0.45,
            "pt_sparkle": 0.18,
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

        self.assertIn("#logo{position:absolute;left:.26rem;top:50%;transform:translateY(-50%);width:2.1rem", source)
        self.assertIn("#panel{position:absolute;right:.43rem;top:50%;transform:translateY(-50%);width:5.15rem", source)
        self.assertIn("h1{font-family:'ABC Arizona Serif',Georgia,serif;font-weight:400;font-size:.5rem", source)


if __name__ == "__main__":
    unittest.main()
