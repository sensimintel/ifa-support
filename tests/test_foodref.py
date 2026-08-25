# -*- coding: utf-8 -*-
"""参考食物库：配置/条目校验、目录落盘、参考区文案、命中判定与回填。

目录与判定是纯逻辑模块（foodref.py，零 cv2/torch 依赖）直接测；
app.py 那侧的图片规范化、三段式消息、命中覆盖只做接线断言（正则抽源码），
与 test_recog_direct、test_recog_output_schema 同款做法。
"""
import json
import re
import tempfile
import unittest
from pathlib import Path

import foodref

ROOT = Path(__file__).resolve().parents[1]
APP = ((ROOT / "app.py").read_text(encoding="utf-8")
       + (ROOT / "recog_prompt.py").read_text(encoding="utf-8"))


def _item(**kw):
    base = {"name": "Snickers", "type": "食物", "calories_kcal": 250}
    base.update(kw)
    return base


class ConfigTest(unittest.TestCase):
    def test_defaults(self):
        # 默认口径：开着、384px（320 实测偏糊）、q88、命中至少 medium
        self.assertEqual(foodref.DEFAULTS,
                         {"on": True, "edge": 384, "quality": 88,
                          "min_confidence": "medium"})
        self.assertEqual(foodref.normalize_config({}), foodref.DEFAULTS)

    def test_edge_must_be_a_choice(self):
        self.assertEqual(foodref.normalize_config({"edge": 448})["edge"], 448)
        # 不在档位里的值一律忽略（保持原值），不做四舍五入到最近档
        self.assertEqual(foodref.normalize_config({"edge": 400})["edge"], 384)
        self.assertEqual(foodref.normalize_config({"edge": "x"})["edge"], 384)

    def test_merge_patch_keeps_untouched_keys(self):
        base = foodref.normalize_config({"edge": 256, "min_confidence": "high"})
        got = foodref.normalize_config({"on": False}, base)
        self.assertEqual(got, dict(base, on=False))

    def test_bad_values_are_ignored(self):
        base = foodref.normalize_config({})
        self.assertEqual(foodref.normalize_config({"min_confidence": "很高"}, base), base)
        self.assertEqual(foodref.normalize_config("不是字典", base), base)
        self.assertEqual(foodref.normalize_config({"quality": 999}, base)["quality"], 95)

    def test_fit_size_keeps_aspect_and_32_grid(self):
        # 4:3 原图 → 总像素≈edge²，两边都是 32 的倍数（视觉 token 覆盖 32×32 像素）
        w, h = foodref.fit_size(4000, 3000, 384)
        self.assertEqual((w % 32, h % 32), (0, 0))
        # 32 网格在小尺寸下就是这个量化精度（384 档只有 12×11 个格子），容差按它给
        self.assertAlmostEqual(w / h, 4 / 3, delta=0.1)
        self.assertAlmostEqual(w * h / (384 * 384), 1.0, delta=0.12)

    def test_fit_size_never_below_min_pixels(self):
        # 极端长宽比也不能缩到 min_pixels 以下——预处理器会把它放大回来，白缩
        for wh in ((4000, 300), (300, 4000), (1280, 720), (1000, 1000)):
            w, h = foodref.fit_size(wh[0], wh[1], 256)
            self.assertGreaterEqual(w * h, foodref.MIN_PIXELS, str(wh))

    def test_fit_size_is_deterministic(self):
        self.assertEqual(foodref.fit_size(1280, 720, 384), foodref.fit_size(1280, 720, 384))

    def test_token_estimate(self):
        # 该模型 patch_size=16 + merge_size=2 → 32×32 像素 1 个视觉 token
        self.assertEqual(foodref.est_tokens(256), 64)
        self.assertEqual(foodref.est_tokens(384), 144)
        self.assertEqual(foodref.est_tokens(512), 256)


class ItemTest(unittest.TestCase):
    def test_name_required(self):
        with self.assertRaises(ValueError):
            foodref.normalize_item({"name": "   "})

    def test_defaults_and_clamps(self):
        got = foodref.normalize_item(_item(calories_kcal="9999", protein_g="abc",
                                           classification="bad"))
        self.assertEqual(got["calories_kcal"], 5000)     # 钳到上限
        self.assertIsNone(got["protein_g"])              # 非法 → None（前端隐藏该行）
        self.assertEqual(got["classification"], "Bad")   # 大小写不敏感的白名单
        self.assertEqual(got["name_en"], "Snickers")     # 未填英文名 → 回落展示名
        self.assertTrue(got["enabled"])

    def test_type_normalized(self):
        self.assertEqual(foodref.normalize_item(_item(type="drink"))["type"], "液体")
        self.assertEqual(foodref.normalize_item(_item(type="饮料"))["type"], "液体")
        self.assertEqual(foodref.normalize_item(_item(type="随便"))["type"], "食物")

    def test_aliases_split_dedup_and_cap(self):
        got = foodref.normalize_item(_item(aliases="士力架, snickers ,SNICKERS,,巧克力棒"))
        self.assertEqual(got["aliases"], ["士力架", "snickers", "巧克力棒"])
        many = foodref.normalize_item(_item(aliases=[str(i) for i in range(20)]))
        self.assertEqual(len(many["aliases"]), foodref.ALIAS_MAX)

    def test_merge_keeps_untouched_fields(self):
        base = foodref.normalize_item(_item(description_en="A bar.", fat_g=12))
        got = foodref.normalize_item({"name": "Snickers", "calories_kcal": 260}, base)
        self.assertEqual(got["description_en"], "A bar.")
        self.assertEqual(got["fat_g"], 12.0)
        self.assertEqual(got["calories_kcal"], 260)

    def test_name_key(self):
        self.assertEqual(foodref.name_key("Coca-Cola"), foodref.name_key("coca cola"))
        self.assertEqual(foodref.name_key(" 香蕉 "), "香蕉")


class SelectKeptTest(unittest.TestCase):
    """编辑时逐张删图的挑选逻辑：keep 决定留哪些旧图、顺序照 keep 给的来。"""

    IMAGES = [{"n": 0, "w": 100, "h": 80}, {"n": 1, "w": 200, "h": 160}]

    def test_keep_subset_in_given_order(self):
        got = foodref.select_kept(self.IMAGES, [1])
        self.assertEqual([im["n"] for im in got], [1])
        self.assertEqual(got[0]["w"], 200)
        # 顺序照 keep 的来，不按原序号排
        got = foodref.select_kept(self.IMAGES, [1, 0])
        self.assertEqual([im["n"] for im in got], [1, 0])

    def test_keep_empty_means_delete_all(self):
        self.assertEqual(foodref.select_kept(self.IMAGES, []), [])

    def test_unknown_dup_and_bad_values_ignored(self):
        got = foodref.select_kept(self.IMAGES, [9, 0, 0, "x", None, 1])
        self.assertEqual([im["n"] for im in got], [0, 1])

    def test_capped_and_copies_not_aliases(self):
        many = [{"n": i} for i in range(5)]
        got = foodref.select_kept(many, [0, 1, 2, 3, 4])
        self.assertEqual(len(got), foodref.MAX_IMAGES_PER_ITEM)
        got[0]["n"] = 99                     # 返回的是副本，改它不该污染目录态
        self.assertEqual(many[0]["n"], 0)


class CatalogTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "food_catalog.json"
        self.cat = foodref.Catalog(self.path)

    def tearDown(self):
        self.dir.cleanup()

    def test_upsert_persists_and_bumps_version(self):
        v0 = self.cat.version()
        item = self.cat.upsert(_item())
        self.assertEqual(item["id"], 1)
        self.assertGreater(self.cat.version(), v0)
        # 换个实例重新读盘：内容与版本都要还原
        again = foodref.Catalog(self.path)
        self.assertEqual(again.version(), self.cat.version())
        self.assertEqual(again.get(1)["name"], "Snickers")

    def test_update_keeps_images_and_id(self):
        self.cat.upsert(_item())
        self.cat.set_images(1, [{"n": 0, "w": 100, "h": 80}])
        got = self.cat.upsert({"name": "Snickers", "calories_kcal": 300}, item_id=1)
        self.assertEqual(got["id"], 1)
        self.assertEqual(len(got["images"]), 1)
        self.assertEqual(got["calories_kcal"], 300)

    def test_item_cap(self):
        for i in range(foodref.MAX_ITEMS):
            self.cat.upsert(_item(name="食物%d" % i))
        with self.assertRaises(ValueError):
            self.cat.upsert(_item(name="第 21 种"))

    def test_images_capped_per_item(self):
        self.cat.upsert(_item())
        got = self.cat.set_images(1, [{"n": 0}, {"n": 1}, {"n": 2}])
        self.assertEqual(len(got["images"]), foodref.MAX_IMAGES_PER_ITEM)

    def test_menu_items_order_is_stable_and_filtered(self):
        # 前缀能不能命中 KV 缓存，全看这个顺序稳不稳——恒按 id 升序
        for name in ("A", "B", "C"):
            self.cat.upsert(_item(name=name))
        self.cat.set_images(1, [{"n": 0}])
        self.cat.set_images(3, [{"n": 0}])
        self.cat.upsert({"name": "C", "enabled": False}, item_id=3)
        self.cat.set_images(2, [{"n": 0}])
        menu = self.cat.menu_items()
        self.assertEqual([it["id"] for it in menu], [1, 2])   # 3 被停用；顺序按 id
        self.cat.upsert({"name": "A2"}, item_id=1)            # 改名不改顺序
        self.assertEqual([it["id"] for it in self.cat.menu_items()], [1, 2])

    def test_menu_skips_items_without_images(self):
        self.cat.upsert(_item())
        self.assertEqual(self.cat.menu_items(), [])   # 有名无图不进清单
        self.cat.set_images(1, [{"n": 0}])
        self.assertEqual(len(self.cat.menu_items()), 1)

    def test_delete(self):
        self.cat.upsert(_item())
        self.assertTrue(self.cat.delete(1))
        self.assertFalse(self.cat.delete(1))
        self.assertEqual(self.cat.snapshot()["items"], [])

    def test_config_change_bumps_version_only_when_changed(self):
        v0 = self.cat.version()
        self.cat.set_config({"edge": 448})
        v1 = self.cat.version()
        self.assertGreater(v1, v0)
        self.cat.set_config({"edge": 448})     # 同值再写一次：不该换版（否则白丢缓存）
        self.assertEqual(self.cat.version(), v1)

    def test_broken_file_does_not_kill_catalog(self):
        self.path.write_text("{不是 JSON", encoding="utf-8")
        cat = foodref.Catalog(self.path)
        self.assertEqual(cat.snapshot()["items"], [])
        self.assertEqual(cat.config(), foodref.DEFAULTS)

    def test_bad_row_is_skipped_not_fatal(self):
        self.path.write_text(json.dumps(
            {"version": 5, "items": [{"id": 1, "name": ""}, {"id": 2, "name": "好的"}]}),
            encoding="utf-8")
        cat = foodref.Catalog(self.path)
        self.assertEqual([it["name"] for it in cat.snapshot()["items"]], ["好的"])


class MenuTextTest(unittest.TestCase):
    def test_intro_says_not_current_frame(self):
        intro = foodref.menu_intro(3, 5)
        self.assertIn("3", intro)
        self.assertIn("not the current frame", intro)

    def test_item_label_carries_index_and_look(self):
        item = foodref.normalize_item(_item(name="士力架", name_en="Snickers",
                                            look="深棕包装"))
        label = foodref.item_label(2, item)
        self.assertTrue(label.startswith("[2] 士力架"))
        self.assertIn("Snickers", label)
        self.assertIn("深棕包装", label)

    def test_item_label_skips_duplicate_english_name(self):
        item = foodref.normalize_item(_item(name="Banana", name_en="banana"))
        self.assertEqual(foodref.item_label(1, item).count("anana"), 1)

    def test_text_is_deterministic(self):
        # 前缀逐字节稳定是本方案的地基：同样的输入必须拼出同样的字符串
        items = [foodref.normalize_item(_item(name="A")),
                 foodref.normalize_item(_item(name="B"))]
        self.assertEqual(foodref.task_zero(items), foodref.task_zero(items))
        self.assertEqual(foodref.menu_ack(2), foodref.menu_ack(2))

    def test_task_zero_lists_every_index(self):
        items = [foodref.normalize_item(_item(name="A")),
                 foodref.normalize_item(_item(name="B"))]
        text = foodref.task_zero(items, "high")
        self.assertIn("[1] A", text)
        self.assertIn("[2] B", text)
        self.assertIn("high", text)
        self.assertIn("ref_evidence", text)
        # 命中时不许模型编营养，且直接省略后续字段（省 decode）——查库回填的前提
        self.assertIn("end the object right after ref_evidence", text)
        self.assertIn("**all omitted**", text)
        # 未命中路径必须显式说「才继续填完」，否则模型会学着在自由路径上也乱省
        self.assertIn("On a miss (ref_id=null)", text)


class BuildBlocksTest(unittest.TestCase):
    """参考区拼装：顺序、编号、跳过无图项，以及「同样的库拼出同样的字节」。"""

    def setUp(self):
        self.items = []
        for i, name in enumerate(("Snickers", "Banana", "Cola")):
            it = foodref.normalize_item(_item(name=name))
            it["id"] = i + 1
            it["images"] = [{"n": 0}] if name != "Banana" else [{"n": 0}, {"n": 1}]
            self.items.append(it)

    def uri_of(self, item_id, n):
        return "data:image/jpeg;base64,IMG-%d-%d" % (item_id, n)

    def test_blocks_order_is_label_then_images(self):
        kept, blocks, total = foodref.build_blocks(self.items, self.uri_of)
        self.assertEqual([it["id"] for it in kept], [1, 2, 3])
        self.assertEqual(total, 4)
        kinds = [b["type"] for b in blocks]
        # 开场白 → [标签 + 图...] × 3
        self.assertEqual(kinds, ["text", "text", "image_url", "text", "image_url",
                                 "image_url", "text", "image_url"])
        self.assertIn("[2] Banana", blocks[3]["text"])
        self.assertEqual(blocks[4]["image_url"]["url"], "data:image/jpeg;base64,IMG-2-0")

    def test_items_without_usable_images_are_skipped_and_renumbered(self):
        # 第 1 项的图取不到 → 整项不进清单，后面的编号顺延（Banana 变成 [1]）
        def spotty(item_id, n):
            return None if item_id == 1 else self.uri_of(item_id, n)
        kept, blocks, total = foodref.build_blocks(self.items, spotty)
        self.assertEqual([it["id"] for it in kept], [2, 3])
        self.assertEqual(total, 3)
        self.assertIn("[1] Banana", blocks[1]["text"])

    def test_empty_when_nothing_has_images(self):
        self.assertEqual(foodref.build_blocks(self.items, lambda *_: None), ([], [], 0))
        self.assertEqual(foodref.build_blocks([], self.uri_of), ([], [], 0))

    def test_byte_for_byte_stable(self):
        # 前缀能不能命中 KV 缓存全靠这条：同一份库连拼两次必须一模一样
        a = foodref.build_blocks(self.items, self.uri_of)[1]
        b = foodref.build_blocks(self.items, self.uri_of)[1]
        self.assertEqual(json.dumps(a, ensure_ascii=False),
                         json.dumps(b, ensure_ascii=False))

    def test_does_not_mutate_the_catalog_snapshot(self):
        before = json.dumps(self.items, ensure_ascii=False)
        foodref.build_blocks(self.items, self.uri_of)
        self.assertEqual(json.dumps(self.items, ensure_ascii=False), before)


class ResolveTest(unittest.TestCase):
    def setUp(self):
        self.items = [
            foodref.normalize_item(_item(name="Snickers", aliases="士力架",
                                         description_en="A chocolate bar.",
                                         calories_kcal=250, classification="Bad")),
            foodref.normalize_item(_item(name="Banana", type="食物",
                                         calories_kcal=89, classification="Good")),
        ]
        for i, it in enumerate(self.items):
            it["id"] = i + 1

    def test_conf_ok(self):
        self.assertTrue(foodref.conf_ok("high", "medium"))
        self.assertTrue(foodref.conf_ok("medium", "medium"))
        self.assertFalse(foodref.conf_ok("low", "medium"))
        self.assertFalse(foodref.conf_ok("", "medium"))      # 缺省按 low 处理
        self.assertTrue(foodref.conf_ok("low", "low"))

    def test_hit_by_ref_id(self):
        parsed = {"name": "Snickers", "ref_id": 1, "ref_confidence": "high",
                  "ref_evidence": "盘子中间，深棕包装"}
        item, src, _ = foodref.resolve_hit(parsed, self.items)
        self.assertEqual((item["id"], src), (1, "ref_id"))

    def test_low_confidence_rejected(self):
        parsed = {"name": "Snickers", "ref_id": 1, "ref_confidence": "low",
                  "ref_evidence": "有个棒状物"}
        item, src, reason = foodref.resolve_hit(parsed, self.items)
        self.assertIsNone(item)
        self.assertIn("置信度", reason)

    def test_evidence_required(self):
        # 说不出画面位置 = 照着参考图编，宁可不命中
        parsed = {"name": "Snickers", "ref_id": 1, "ref_confidence": "high",
                  "ref_evidence": "  "}
        item, _, reason = foodref.resolve_hit(parsed, self.items)
        self.assertIsNone(item)
        self.assertIn("证据", reason)

    def test_out_of_range_ref_id_falls_back_to_name(self):
        parsed = {"name": "Banana", "ref_id": 99, "ref_confidence": "high",
                  "ref_evidence": "桌子右侧"}
        item, src, _ = foodref.resolve_hit(parsed, self.items)
        self.assertEqual((item["id"], src), (2, "name"))

    def test_alias_match_without_ref_id(self):
        parsed = {"name": "士力架", "ref_id": None}
        item, src, _ = foodref.resolve_hit(parsed, self.items)
        self.assertEqual((item["id"], src), (1, "name"))

    def test_miss(self):
        parsed = {"name": "Kit Kat", "ref_id": None}
        item, src, reason = foodref.resolve_hit(parsed, self.items)
        self.assertIsNone(item)
        self.assertIsNone(src)
        self.assertIn("未命中", reason)

    def test_empty_catalog_never_hits(self):
        item, _, _ = foodref.resolve_hit({"name": "Snickers", "ref_id": 1}, [])
        self.assertIsNone(item)

    def test_apply_hit_overrides_every_content_field(self):
        parsed = {"name": "snickers bar", "type": "液体", "ref_id": 1,
                  "calories_kcal": 999, "protein_g": 1.0, "carbs_g": 2.0,
                  "fat_g": 3.0, "description_en": "模型编的", "classification": "Good"}
        got = foodref.apply_hit(parsed, self.items[0], "ref_id")
        self.assertEqual(got["name"], "Snickers")           # 名称照库
        self.assertEqual(got["type"], "食物")               # 类型照库
        self.assertEqual(got["calories_kcal"], 250)         # 营养照库，模型的 999 丢弃
        self.assertEqual(got["description_en"], "A chocolate bar.")
        self.assertEqual(got["classification"], "Bad")
        self.assertEqual((got["ref_hit_id"], got["source"]), (1, "catalog"))

    def test_budget(self):
        for it in self.items:
            it["images"] = [{"n": 0}, {"n": 1}]
        got = foodref.budget(self.items, 384)
        self.assertEqual((got["items"], got["images"]), (2, 4))
        self.assertEqual(got["tokens"], 4 * 144)


class AppWiringTest(unittest.TestCase):
    """app.py 侧接线：参考区必须独占第一条消息，且命中要走查库回填。"""

    def assertHas(self, needle, src=None):
        # 不用 assertIn：它失败时会把整个 app.py 打进报错信息，报告就没法读了
        self.assertTrue(needle in (APP if src is None else src),
                        "找不到：%s" % needle)

    def test_prefix_is_its_own_message_with_ack(self):
        self.assertHas('messages.append({"role": "user", "content": ref_blocks})')
        self.assertHas('foodref.menu_ack(len(ref_items))')
        # 当前画面那条消息必须排在参考区之后
        pos_ref = APP.index('"content": ref_blocks')
        pos_cur = APP.index('messages.append({"role": "user", "content": content})')
        self.assertLess(pos_ref, pos_cur)

    def test_menu_cached_by_version(self):
        # 版本没变就必须复用同一批字节，否则 prefix cache 每轮都 miss
        self.assertHas('if cached.get("version") == version:')
        self.assertHas("_foodref_uri_cache")

    def test_reupload_drops_stale_cache(self):
        # 缓存文件名不含内容哈希，换图不清缓存就会一直发老图
        self.assertIn("_foodref_drop_cache(item[\"id\"])", APP)

    def test_hit_overrides_and_dedups_by_ref(self):
        self.assertHas("foodref.apply_hit(it, hit, src)")
        self.assertHas('c.get("ref_hit_id") == ref_hit')

    def test_ref_image_normalization_targets_32px_grid(self):
        # 视觉 token 覆盖 32×32 像素，不对齐就等于把尺寸交给预处理器决定
        self.assertHas("foodref.fit_size(w, h, edge)")
        self.assertEqual(foodref.MIN_PIXELS, 65536)   # 模型 min_pixels，小于它会被放大回来

    def test_reference_banner_kept(self):
        self.assertHas("REFERENCE - NOT CURRENT FRAME")

    def test_endpoints_exist(self):
        for route in ("/api/foodref/list", "/api/foodref/config", "/api/foodref/item",
                      "/api/foodref/item/{item_id}", "/api/foodref/image/{item_id}/{n}"):
            self.assertHas(route)

    def test_log_whitelist_carries_ref_fields(self):
        """req/resp 的投影是白名单：漏了字段，控制面就看不见参考库这一段
        （踩过——stream 当年也是这么漏掉的）。"""
        import recog_log
        self.assertIn("ref", recog_log._REQ_KEYS)
        self.assertIn("ref_prompt", recog_log._REQ_DETAIL_KEYS)
        self.assertIn("ref", recog_log._RESP_KEYS)

    def test_gitignored(self):
        ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        # 5090 的 ~/da3-web 是只读 checkout，运行时产物不能污染工作区
        self.assertHas("food_catalog.json", ignore)
        self.assertHas("food_ref/", ignore)

    def test_prompt_declares_reference_section(self):
        self.assertHas("foodref.task_zero(refs, min_conf)")   # 在 recog_prompt.fixed_head 里
        self.assertTrue(re.search(r"ref_id\\\":null", APP))


if __name__ == "__main__":
    unittest.main()
