# -*- coding: utf-8 -*-
"""/experience 展示批次：同一个物体散成多张卡时，屏幕内容锚定最早那张。

这段逻辑是 app.py 里 EXPERIENCE_PAGE 内嵌的 JS，正则抽出来交给 node 真跑——
接线断言只能证明"代码长这样"，证明不了 min(id) 选对没有。node 不在就跳过
（与 test_devpc_geometry 缺 numpy 时的做法一致）。
"""
import json
import re
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")


def _extract_js():
    """从 app.py 里抽出 grpKey / grpAnchor 两个函数源码。"""
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    m = re.search(r"^function grpKey\(c\)\{.*?^\}", src, re.M | re.S)
    if not m:
        raise AssertionError("app.py 里找不到 grpKey/grpAnchor —— 展示批次逻辑被改名或删了")
    return m.group(0)


@unittest.skipIf(NODE is None, "node 不可用（精简测试环境），跳过 experience JS 单测")
class GroupAnchorTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.js = _extract_js()

    def run_js(self, body):
        script = self.js + "\n" + body
        out = subprocess.run([NODE, "-e", script], capture_output=True, text=True, timeout=30)
        self.assertEqual(out.returncode, 0, out.stderr)
        return json.loads(out.stdout.strip())

    def test_anchor_is_earliest_card_not_the_head(self):
        """cards[0] 是后端最后碰过的那条，可能是这批里第 14 张——内容必须取 id 最小的。"""
        cards = [{"id": 24, "name": "Wine", "type": "液体", "calories_kcal": 200},
                 {"id": 2, "name": "Wine", "type": "液体", "calories_kcal": 120},
                 {"id": 11, "name": "Wine", "type": "液体", "calories_kcal": 150}]
        got = self.run_js(
            "const cards=%s;"
            "console.log(JSON.stringify(grpAnchor(cards, grpKey(cards[0]))));" % json.dumps(cards))
        self.assertEqual(got["id"], 2)
        self.assertEqual(got["calories_kcal"], 120)

    def test_group_key_ignores_case_and_padding(self):
        """模型对同一个东西大小写/空格常不稳，不该因此分成两批。"""
        got = self.run_js(
            "console.log(JSON.stringify(["
            "grpKey({name:' Wine ',type:'液体'})===grpKey({name:'wine',type:'液体'}),"
            "grpKey({name:'Wine',type:'液体'})===grpKey({name:'Wine',type:'食物'})]));")
        self.assertEqual(got, [True, False], "大小写/空格应归一，类型不同必须分批")

    def test_different_names_stay_separate(self):
        """Orange 和 Tangerine 是两批——同义词归并是另一档方案，这一档不做。"""
        cards = [{"id": 5, "name": "Tangerine", "type": "食物"},
                 {"id": 4, "name": "Orange", "type": "食物"}]
        got = self.run_js(
            "const cards=%s;"
            "console.log(JSON.stringify(grpAnchor(cards, grpKey(cards[0])).id));" % json.dumps(cards))
        self.assertEqual(got, 5, "不该把 Tangerine 锚到 Orange 上")

    def test_anchor_falls_back_when_group_has_one_card(self):
        cards = [{"id": 9, "name": "Persimmon", "type": "食物"}]
        got = self.run_js(
            "const cards=%s;"
            "console.log(JSON.stringify(grpAnchor(cards, grpKey(cards[0])).id));" % json.dumps(cards))
        self.assertEqual(got, 9)

    def test_missing_fields_do_not_throw(self):
        """老卡/异常推送里 name 或 type 可能缺失，展示层不能整页炸掉。"""
        got = self.run_js(
            "const cards=[{id:3},{id:1,name:null,type:undefined}];"
            "console.log(JSON.stringify(grpAnchor(cards, grpKey(cards[0])).id));")
        self.assertEqual(got, 1)


class ExperienceWiringTest(unittest.TestCase):
    """接线：内容取锚点、两个时钟口径不混、驻留只续期不重画、流水同口径去重。"""

    @classmethod
    def setUpClass(cls):
        cls.src = (ROOT / "app.py").read_text(encoding="utf-8")

    def assertHas(self, needle):
        self.assertTrue(needle in self.src, "app.py 里找不到：%s" % needle)

    def assertLacks(self, needle):
        self.assertNotIn(needle, self.src, "app.py 里不该再有：%s" % needle)

    def applyRecogBody(self):
        m = re.search(r"function applyRecog\(r\)\{(.*?)\n\}", self.src, re.S)
        self.assertIsNotNone(m, "找不到 applyRecog")
        return m.group(1)

    def test_card_renders_anchor_not_head(self):
        self.assertHas("if(key!==lastCardKey){lastCardKey=key;"
                       "curCard=grpAnchor(cards,key)||head;renderCard(curCard);}")

    def test_two_clocks_are_separate_knobs(self):
        """驻留与新鲜度是两个口径不同的量，合成一个就会出现
        「屏上驻留 = 驻留 − 端到端延时」——延时把驻留吃掉。"""
        self.assertHas("let DWELL_MS=Math.min(30,Math.max(1,"
                       "+(localStorage.getItem('exp_card_dwell_s')||2)))*1000;")
        self.assertHas("let FRESH_TTL_MS=Math.min(20,Math.max(2,"
                       "+(localStorage.getItem('exp_card_ttl_s')||5)))*1000;")
        self.assertLacks("FRESH_MS=")        # 合成语义的旧变量必须彻底消失
        # 两个滑杆都在抽屉里
        self.assertHas('id="r_card_s" min="1" max="30" step="0.5" value="2"')
        self.assertHas('id="r_card_ttl" min="2" max="20" step="0.5" value="5"')

    def test_dwell_clock_counts_from_render_not_frame_time(self):
        """驻留起点必须是**上屏时刻**：取帧时刻会让端到端延时直接从驻留里扣，
        实测延时中位 2.1s，4s 驻留只剩 1.9s，延时 ≥ 驻留时更是完全不上屏。"""
        body = self.applyRecogBody()
        self.assertIn("cardShownAt=Date.now();", body)
        self.assertNotIn("cardShownAt=Math.min", body)   # 旧的帧时刻起点
        self.assertNotIn("Date.now()-FRESH", body)       # 旧的钳制口径

    def test_dwell_renews_on_every_fresh_push_without_repaint(self):
        """同一批的后续识别只续期、不重画：续期若跟着 key 走，批次身份稳定后
        key 不再变，明明在持续识别却会到期回落待机。"""
        body = self.applyRecogBody()
        branch = next(l for l in body.split("\n") if "if(key!==lastCardKey)" in l)
        self.assertNotIn("cardShownAt", branch, "续期不能只在换批次时发生")
        self.assertLess(body.index("if(key!==lastCardKey)"),
                        body.index("cardShownAt=Date.now();"),
                        "先重画（若换批）再统一续期")

    def test_dwell_expiry_ticks_finer_than_the_shortest_dwell(self):
        """驻留可短到 1s，1s 的 tick 会把它拖成最多 2s（量化误差与驻留同量级）。"""
        self.assertHas("Date.now()-cardShownAt<DWELL_MS?'card':'idle'),250);")

    def test_timeline_dedupes_by_group(self):
        self.assertHas("const seen=new Set(),uniq=[];")
        self.assertHas("const rows=uniq.slice(0,10).reverse();")


if __name__ == "__main__":
    unittest.main()
