# -*- coding: utf-8 -*-
"""识别 prompt 的拼装：固定段/可变段的切分、排布顺序、新的判同契约。

这套测试的头号任务是守住一条肉眼守不住的纪律：**固定段里不许混进任何每轮会变的量**
——混进一个，vLLM 的 prefix cache 整段作废，而线上只会表现为"ttft 好像变慢了"，
没人会想到是 prompt 里多了一句「食物框×2」。

排布本身也有测试：2026-08-18 那次重排（当前画面挪到所有参考图之后、每张图前面加
方括号标签）是为了治「参考图污染当前画面」，排回去就等于把那个 bug 放回来。
"""
import re
import unittest
from pathlib import Path

import foodref
import recog_match
import recog_prompt

ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "app.py").read_text(encoding="utf-8")


def _cands(n=2):
    return [{"id": i + 1, "name": "食物%d" % i, "type": "食物",
             "desc": "d%d" % i, "ref_img": "data:image/jpeg;base64,IMG%d" % i}
            for i in range(n)]


def _refs(n=2):
    out = []
    for i in range(n):
        it = foodref.normalize_item({"name": "参考%d" % i, "calories_kcal": 100})
        it["id"] = i + 1
        out.append(it)
    return out


class FixedHeadTest(unittest.TestCase):
    """固定段：只认配置态，不认画面态。"""

    def test_identical_across_rounds(self):
        # 同样的配置，不管这一轮画面里有几个框、上一轮选了什么、有几个候选，
        # 固定段必须逐字节相同——否则缓存前缀白算
        a = recog_prompt.fixed_head(False)
        b = recog_prompt.fixed_head(False)
        self.assertEqual(a, b)

    def test_carries_no_per_round_values(self):
        head = recog_prompt.fixed_head(False, _refs(), "medium")
        for leak in ("食物框×", "液体框×", "上一轮上屏的是", "候选[1] 食物0",
                     "最近 30 秒内已经记录过的物品如下"):
            self.assertNotIn(leak, head, "固定段混进了每轮变化的量：%s" % leak)

    def test_direct_variant_drops_boxed_line(self):
        self.assertNotIn("【当前画面 · 检测框版】", recog_prompt.fixed_head(True))
        self.assertIn("【当前画面 · 检测框版】", recog_prompt.fixed_head(False))

    def test_reference_catalog_section_only_when_present(self):
        with_refs = recog_prompt.fixed_head(False, _refs(), "high")
        self.assertIn("任务零", with_refs)
        self.assertIn("REFERENCE 角标", with_refs)
        self.assertIn("high", with_refs)
        self.assertNotIn("任务零", recog_prompt.fixed_head(False))

    def test_bans_ordinal_references(self):
        # 旧版靠「图1/图2/图3」指称，一条消息里塞十几张图时根本对不齐
        head = recog_prompt.fixed_head(False)
        self.assertIn("不要用「第几张图」来指称任何图片", head)
        self.assertNotIn("图1=当前画面原图", head)

    def test_grounding_contract_is_checkable(self):
        """「注意力放在当前画面」必须落成可检查的输出契约，而不是一句嘱咐。"""
        head = recog_prompt.fixed_head(False)
        self.assertIn("写进 seen 字段", head)
        self.assertIn("name 必须能被 seen 里写下的事实支持", head)
        # 现场那个 case：金黄包装 → 橙色 → Orange，必须被这条堵死
        self.assertIn("包装的颜色不是内容物的颜色", head)

    def test_evidence_code_requires_collateral(self):
        head = recog_prompt.fixed_head(False)
        self.assertIn("cur_text 不是 none", head)     # B=1 的抵押物
        self.assertIn("? 不丢人，瞎填 1 才是错", head)
        for blank in recog_match.DIFF_BLANKS[:3]:      # diff 的逃逸话术要点名
            self.assertIn(blank, head)

    def test_examples_never_show_all_ones(self):
        """连"允许合并"的正面示例都带一个 ?——这是消灭清一色全 1 最省事的一招。"""
        head = recog_prompt.fixed_head(False)
        self.assertIn("B1C1S?V1", head)      # 包装零食允并：S 位判不了
        self.assertIn("B?C1S1V1", head)      # 天然食材允并：读不到文字 B 只能 ?
        self.assertIn('"match_evidence":"NONE"', head)   # 无相似候选的形状
        self.assertNotIn("B1C1S1V1", head)

    def test_examples_cover_orthogonal_branches_not_one_food(self):
        """2026-08-24 few-shot 换血：旧版 Werther's 在固定段出现 4 次，而 cur_text
        是无法 OCR 核验的 B 位抵押物——few-shot 等于供应了一个现成的合法值。
        新版每个示例一个正交分支、食物词汇彼此不同。"""
        head = recog_prompt.fixed_head(False)
        self.assertNotIn("Werther", head)
        for food in ("Pringles", "Banana", "Coffee"):
            self.assertIn(food, head)
        # 液体示例必须演示「按内容物命名」而不是容器名
        self.assertIn('"type":"液体"', head)

    def test_catalog_hit_example_only_when_refs(self):
        """命中是展台主路径，「写完 ref_evidence 就结束对象」这种反直觉形状必须有
        示例钉住；没开参考库时清单不存在，该示例整个跳过。"""
        with_refs = recog_prompt.fixed_head(False, _refs(), "medium")
        self.assertIn('"ref_id":2,"ref_confidence":"high"', with_refs)
        self.assertIn("示例五", with_refs)           # 命中示例排第一，共 5 个
        no_refs = recog_prompt.fixed_head(False)
        self.assertNotIn("Milka", no_refs)
        self.assertNotIn("示例五", no_refs)          # 无参考库只有 4 个
        # 命中示例的对象必须在 ref_evidence 后收束——不许把判同字段带进来
        hit_obj = with_refs[with_refs.index("Milka Alpine"):]
        hit_obj = hit_obj[:hit_obj.index("}]}")]
        for absent in ("cur_text", "match_evidence", "calories_kcal"):
            self.assertNotIn(absent, hit_obj, "命中示例不该带 %s" % absent)

    def test_match_confidence_removed_from_contract(self):
        """置信度改由服务端从证据码推导（recog_match.derive_confidence），
        prompt 不再要求模型自报 match_confidence 字段。"""
        for head in (recog_prompt.fixed_head(False),
                     recog_prompt.fixed_head(False, _refs(), "medium")):
            self.assertNotIn('"match_confidence"', head)
            self.assertNotIn("6. match_confidence", head)


class EdibleConstraintTest(unittest.TestCase):
    """只认能进嘴的东西（治 2026-08-21 识别迁到本机 Qwen3.6 后把手机/电脑/容器当 item）：
    硬性约束 4 + edible 字段 + 容器名禁令 + 空数组反例 + 收尾复读，一处都不许少。"""

    def test_fixed_head_bans_non_food_items(self):
        p = recog_prompt.fixed_head(True)
        self.assertIn("能进嘴的食物或饮料", p)
        self.assertIn("纯包装不算食物", p)

    def test_fixed_head_asks_edible_field(self):
        p = recog_prompt.fixed_head(True)
        self.assertIn("edible：布尔值 true/false", p)
        # 所有正例都必须带 edible:true——字段顺序契约靠示例钉死。
        # 无参考库 3 个正例（零食/天然食材/液体），开参考库多一个命中正例
        self.assertEqual(p.count('"edible":true'), 3)
        self.assertEqual(
            recog_prompt.fixed_head(True, _refs(), "medium").count('"edible":true'), 4)

    def test_fixed_head_bans_container_names(self):
        self.assertIn("Container、Cup、Bottle", recog_prompt.fixed_head(True))

    def test_fixed_head_has_empty_items_example(self):
        # 空数组反例必须垫底：旧版只有正例，弱模型见样学样必给一个 item。
        # 无参考库 4 个示例（空数组是示例四），开参考库 5 个（示例五）
        p = recog_prompt.fixed_head(True)
        self.assertIn("示例四", p)
        self.assertIn('{"items":[]}', p)
        self.assertIn('{"items":[]}', recog_prompt.fixed_head(True, _refs(), "medium"))

    def test_tail_repeats_the_edible_constraint(self):
        # 收尾紧贴生成位置，是模型印象最深的一段，硬性约束 4 必须在这里复读
        t = recog_prompt.tail(None, [])
        self.assertIn("能进嘴的食物或饮料", t)
        self.assertIn("items 必须是空数组", t)


class VariableBlocksTest(unittest.TestCase):
    """可变段：编号对得齐、标签把图片命名、收尾复读约束。"""

    def test_candidate_numbering_matches_labels(self):
        cands = _cands(3)
        intro = recog_prompt.candidates_intro(cands)
        for i, c in enumerate(cands):
            self.assertIn("[%d] %s" % (i + 1, c["name"]), intro)
            self.assertIn("候选[%d] %s" % (i + 1, c["name"]),
                          recog_prompt.candidate_label(i + 1, c))

    def test_intro_warns_names_may_be_wrong(self):
        # 候选名只是过去几轮的结论，不能被当成"当前画面里存在这些东西"的证据
        intro = recog_prompt.candidates_intro(_cands())
        self.assertIn("可能就有错的", intro)
        self.assertIn("不说明它现在还在桌上", intro)

    def test_empty_candidates_branch(self):
        intro = recog_prompt.candidates_intro([])
        self.assertIn("match 一律为 null", intro)
        self.assertIn("match_evidence 写 NONE", intro)

    def test_current_label_closes_the_reference_section(self):
        with_c = recog_prompt.current_label(True)
        self.assertIn("以上【历史参考图】到此结束", with_c)
        self.assertIn("【当前画面】", with_c)
        self.assertNotIn("以上", recog_prompt.current_label(False))

    def test_boxed_label_carries_the_counts(self):
        label = recog_prompt.boxed_label(2, 1)
        self.assertIn("食物框×2", label)
        self.assertIn("液体框×1", label)

    def test_tail_repeats_the_only_constraint_that_matters(self):
        tail = recog_prompt.tail(None, [])
        self.assertIn("只描述上面那张【当前画面】", tail)


class PickRuleTest(unittest.TestCase):
    """粘性规则：从「要求确认」改成「独立复核」。"""

    def test_no_history_degrades(self):
        self.assertIn("挑你最有把握", recog_prompt.pick_rule(None, []))

    def test_confirmation_bias_wording_gone(self):
        """旧文案「只要它仍在画面里就继续选它」把开放问题偷换成是非题，
        VLM 的 yes-bias 会让一次误报一路确认下去（线上出现过 rev=252）。"""
        t = recog_prompt.pick_rule({"name": "Orange", "card_id": 9},
                                   [{"id": 9, "name": "Orange"}])
        self.assertIn("「Orange」", t)
        self.assertIn("清单[1]", t)
        self.assertIn("不保证正确", t)
        self.assertIn("独立写完 seen 与 name", t)
        self.assertNotIn("就继续选它", t)
        self.assertNotIn("仍在当前画面里", t)

    def test_prefers_switching_over_staying_wrong(self):
        t = recog_prompt.pick_rule({"name": "Orange", "card_id": 1}, [])
        self.assertIn("一直报错的名字比换一次名字糟糕得多", t)


class RenderForLogTest(unittest.TestCase):
    """观测日志：控制面读 req.prompt 这一个字符串，图片位置必须看得见。"""

    def test_images_become_placeholders_in_order(self):
        content = [{"type": "text", "text": "A"},
                   {"type": "image_url", "image_url": {"url": "data:...."}},
                   {"type": "text", "text": "B"}]
        self.assertEqual(recog_prompt.render_for_log(content), "A\n〔图片〕\nB")

    def test_empty(self):
        self.assertEqual(recog_prompt.render_for_log([]), "")


class LayoutWiringTest(unittest.TestCase):
    """app.py 侧的排布接线：当前画面必须排在所有参考图之后。"""

    def _pos(self, needle):
        idx = APP.find(needle)
        self.assertNotEqual(idx, -1, "app.py 里找不到：%s" % needle)
        return idx

    def test_current_frame_comes_after_reference_images(self):
        head = self._pos("recog_prompt.fixed_head(")
        intro = self._pos("recog_prompt.candidates_intro(cands)")
        label = self._pos("recog_prompt.candidate_label(i + 1, c)")
        cur = self._pos("recog_prompt.current_label(bool(cands))")
        tail = self._pos("recog_prompt.tail(last_pick, cands)")
        self.assertLess(head, intro)
        self.assertLess(intro, label)
        self.assertLess(label, cur, "当前画面必须排在所有历史参考图之后")
        self.assertLess(cur, tail)

    def test_candidates_without_images_are_dropped_before_numbering(self):
        """无图的候选若进了清单文字，编号就会与实际图片错位。"""
        self.assertIn('cands = [c for c in (candidates or []) if c.get("ref_img")]', APP)

    def test_log_records_the_whole_block_sequence(self):
        self.assertIn("prompt = recog_prompt.render_for_log(content)", APP)

    def test_image_count_counts_image_blocks(self):
        # 文字块也进了 content，n_images 不能再用 len(content)-1
        self.assertIn('sum(1 for b in content if b.get("type") == "image_url")', APP)


class CropWiringTest(unittest.TestCase):
    """参考图：按检测框裁特写 + 降到 256（治「整帧对照必然全一致」）。"""

    def test_ref_image_is_cropped_from_the_raw_frame(self):
        # 从原图裁，不是从带框图裁——红蓝框会干扰判同
        self.assertIn("_make_ref_img(orig, recog_match.pick_ref_box(dets))", APP)

    def test_ref_image_edge_is_small(self):
        self.assertIn("REF_IMG_EDGE = 256", APP)

    def test_detections_reach_the_worker(self):
        self.assertIn('"dets": list(detections or [])', APP)
        self.assertIn('dets = extra.get("dets") or []', APP)


class CandidatePolicyWiringTest(unittest.TestCase):
    """候选池：数量、同名去重、绝对存活上限、强制复检。"""

    def test_candidate_cap_lowered(self):
        self.assertIn("RECOG_MAX_CANDIDATES = 3", APP)

    def test_absolute_lifetime_exists(self):
        self.assertIn("RECOG_CARD_MAX_LIFE = 120.0", APP)
        self.assertIn('now - c.get("first_ts", c.get("last_ts", 0)) <= RECOG_CARD_MAX_LIFE', APP)
        self.assertIn('"first_ts": frame_at or now', APP)

    def test_same_name_candidates_deduped(self):
        self.assertIn("key = foodref.name_key(c.get(\"name\"))", APP)

    def test_recheck_round_drops_candidate_and_sticky(self):
        self.assertIn("RECOG_RECHECK_EVERY = 15", APP)
        self.assertIn('c.get("rev", 0) % RECOG_RECHECK_EVERY == 0', APP)
        self.assertIn('lp.get("card_id") in (extra.get("recheck") or [])', APP)
        # 无提示下仍报同名 → 回收合并，否则复检轮会凭空多一张重复卡
        self.assertIn("复检回收", APP)

    def test_self_evidence_gate_wired_into_the_chain(self):
        self.assertIn("self_ok, self_reason = recog_match.check_self_evidence(it)", APP)
        self.assertIn("elif not self_ok:", APP)

    def test_catalog_hit_merge_has_name_guard(self):
        """ref_id 并卡绕过五道闸，一次错的命中会永久自锁——名称与存活上限是兜底。"""
        m = re.search(r'c\.get\("ref_hit_id"\) == ref_hit\s*\n\s*and foodref\.name_key', APP)
        self.assertTrue(m, "参考库并卡缺名称一致性校验")


if __name__ == "__main__":
    unittest.main()
