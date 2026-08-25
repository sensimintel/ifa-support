# -*- coding: utf-8 -*-
"""识别 prompt 的拼装（纯逻辑，无 cv2/torch 依赖，可单测）。

为什么单独成模块：这套 prompt 现在被拆成「固定段」与「可变段」两截，而**固定段里
混进任何一个每轮会变的量，整段 KV 缓存就作废**。这条纪律靠肉眼守不住，必须有测试
断言「同样的配置、不同的画面/候选/上一轮结论，固定段字节完全相同」。

语言（2026-08-25 起）：**prompt 正文与模型的 JSON 产出一律英文**。
展台面向的是德国观众与英文界面，seen / diff / ref_evidence 这些自由文本过去是中文，
控制面之外没人看得懂；德文描述字段同期整条链路删除——它只在旧调试面展示过，
运营也从来没有录入口。服务端的日志、注释、控制面文案仍是中文，只有「发给模型的
字符串」和「模型吐回来的字符串」改英文。

排布（2026-08-18 重排，治「参考图污染当前画面」）：
    [固定段] 图片指称约定 + 硬性约束 + 任务零(参考库) + 任务一 + 字段定义
             + 判同流程 + JSON 骨架与一组正交示例（开参考库时含命中正例）
    [可变段] 任务二开场 + 候选清单文字
             → [PAST REFERENCE · CANDIDATE[i]] 标签 + 该候选的裁剪图（逐个交错）
             → 「参考图到此结束，下面这一张是 [CURRENT FRAME]」+ 当前帧
             → [CURRENT FRAME · BOXED] + 带框图（仅门控口径）
             → 粘性规则 + 收尾复读

三处关键设计，都是照着线上那次「爆米花被反复报成 Orange/Clementine」开的方子：
  1. **每张图前面都有一行方括号标签**，并明令不许用「第几张图」指称——序数指称在
     一条消息里塞进十几张图时根本对不齐（旧版就是靠数序数）；
  2. **当前画面排在所有参考图之后**，成为离生成最近的那张图（旧版它在最前面，
     离任务描述最远，而 8 张历史参考图紧贴着指令）；
  3. **任务一强制「先描述、后命名」**：先写 seen（颜色/形状/包装上读到的文字/位置），
     name 必须被 seen 支持——把「注意力放在当前画面」从一句嘱咐变成可检查的输出契约。
"""
import foodref
import recog_match

# 描述与判同用到的字段上限（与 app.py 的 guardrail 对齐，超长截断而非报错）。
# 2026-08-25 改英文后同步放宽：同样一句话英文的字符数约是中文的 2~3 倍，
# 沿用中文口径会把 seen/diff 从中间切断，落到闸门六时看起来像「没写完」。
SEEN_MAX = 200
CUR_TEXT_MAX = 60
DIFF_MAX = 100



def fixed_head(direct: bool, refs=None, min_conf: str = "medium") -> str:
    """固定段：整段不含任何每轮变化的量，因此能进 vLLM 的 prefix cache。

    只有 direct（触发口径）、refs（参考食物库快照）、min_conf 三个配置态会改变它，
    而它们一变本来就该换版。**严禁**把 n_food/n_drink、上一轮选中项、候选数量
    写进这里——漏一个，整段缓存作废（tests/test_recog_prompt.py 有断言守着）。"""
    boxed_line = ("  · [CURRENT FRAME · BOXED] -- the very same current frame with detection "
                  "boxes drawn on it (red = suspected food, blue = suspected liquid/container);\n"
                  if not direct else "")
    ref_line = ("  · The images carrying a REFERENCE badge in the previous message are "
                "**registered reference photos of catalog food**. They are neither the table "
                "right now, nor proof that these things are on the table.\n" if refs else "")
    p = (
        "This message contains two kinds of images. Every image is preceded by a bracketed "
        "label line. Always refer to an image by that label and **never by ordinal position "
        "(\"the first image\", \"image 2\", ...)**:\n"
        "  · [CURRENT FRAME] -- the real table the camera sees right now. There is exactly one "
        "of it in this message and it comes after all reference images;\n"
        + boxed_line +
        "  · [PAST REFERENCE · CANDIDATE[i] Xxx] -- an object cropped out of a frame from an "
        "earlier round, used only for Task 2, **it is not the table right now**;\n"
        + ref_line +
        "\nHard constraints (breaking any of them voids the whole round):\n"
        "  1. Task 1 may only describe [CURRENT FRAME]. Anything that appears solely in a "
        "[PAST REFERENCE] image and cannot be seen in [CURRENT FRAME] must never go into items;\n"
        "  2. When [CURRENT FRAME] holds neither food nor liquid, items must be an empty array "
        "even if the reference images are full of food. Never make something up;\n"
        "  3. **Pixels first, lists second.** Every name must first be read by you out of the "
        "pixels of [CURRENT FRAME]. The names on the candidate list are only conclusions from "
        "earlier rounds and **they may well be wrong**; a name being on the list is never "
        "evidence that such a thing is on the table now.\n"
        "  4. items may only hold **food or drink that can go into a mouth**. Phones, laptops, "
        "keyboards, remote controls, cutlery, napkins, table decorations and every other "
        "non-food object must never be output as an item, however prominent they look. "
        "**Packaging alone is not food**: an empty bag, box, bottle or cup, i.e. any package or "
        "container whose edible content you can neither see nor read, is never output. A "
        "detection box only says \"there may be something there\"; a box around a non-food "
        "object or an empty container still produces no item.\n\n"
    )
    if refs:
        p += foodref.task_zero(refs, min_conf) + "\n"
    p += (
        "Task 1 · Recognition (look only at [CURRENT FRAME]):\n"
        "Step one, **before you look at any list and at any reference image**, describe in your "
        "own words the single most prominent edible thing in [CURRENT FRAME] and write it into "
        "the seen field:\n"
        "  seen = colour / shape / any text you read with your own eyes on the package or "
        "container / where it sits in the frame\n"
        "  e.g. \"dark blue square soft package, reads OREO on the front, centre-right of the table\"\n"
        "  e.g. \"orange spherical fruit, dimpled peel, no text at all, left half of a white plate\"\n"
        "  When not a single character can be read on the package, write none in the text slot, "
        "but colour, shape and position must all be filled in.\n"
        "Step two, only now name it (name). **name must be supported by the facts written in "
        "seen**:\n"
        "  · when the package text you read into seen conflicts with the name you were about to "
        "write, seen wins and the name changes;\n"
        "  · only when seen says \"unpackaged, spherical or lumpy natural produce\" may it be "
        "named a fruit or a vegetable; as soon as seen carries package text, or a description "
        "such as \"square soft package / bagged / boxed / canned\", it must **never** be named a "
        "fruit -- **the colour of the packaging is not the colour of the content**.\n"
        "**Even when several things are visible, output exactly one item** -- the display shows "
        "one at a time and anything extra is dropped.\n"
        "When there is neither food nor liquid in the frame, return an empty array; never make "
        "one up.\n\n"
        "Fields to output for this one item (**every field in English**, no other language):\n"
        "  name: the concrete name, short, in English. Food may use the brand name (e.g. Banana, "
        "Snickers); a liquid is always named after **its content** (e.g. Water, Coffee, Cola) -- "
        "text on the container may only be used as the name when it is the brand of the drink "
        "itself (e.g. Coca-Cola on a cola can), while decorative wording on a mug (e.g. Good "
        "morning) is not a name. When the content cannot be made out and no drink brand can be "
        "read on the container, this is not an outputtable liquid: **never** put a container word "
        "such as Container, Cup or Bottle in the name, simply do not output this item;\n"
        "  type: either \"food\" or \"drink\";\n"
        "  edible: boolean true/false -- is it, right now in this frame, visible food or drink "
        "that **can go straight into a mouth**? Non-food objects, empty packages, empty "
        "containers and fake food props are all false, and when in doubt write false. The server "
        "only shows entries whose edible is true;\n"
        "  description_en: one English sentence (at most 60 characters);\n"
        "  calories_kcal / protein_g / carbs_g / fat_g: integer kcal plus three gram values. "
        "These four numbers are **always estimated from the portion actually visible in "
        "[CURRENT FRAME]**, not the per-100g reference values: first judge size, volume and count "
        "by eye (against the hand, cutlery or container in the frame), then convert from that "
        "portion; when only a part is left, estimate the part that is left; a liquid is estimated "
        "from the container volume and the fill level;\n"
        "  classification: health grade, one of " + "/".join(foodref.CLASSIFICATIONS) +
        " (Good for nutrient-dense, natural, barely processed; Bad for high sugar, high salt, "
        "deep fried or heavily processed; Neutral in between).\n\n"
        + _judge_flow() + "\n" + _json_skeleton(bool(refs))
    )
    return p


def _judge_flow() -> str:
    """判同流程：让「填 1」变贵、强制先给否定证据。

    旧版的证据码退化成清一色 B1C1S1V1，根因有二：参考图是整帧（四项一致是客观事实），
    以及填 1 零成本。这里对第二条下手——B=1 必须以 cur_text 里照抄的包装文字作抵押，
    并把 ? 明确正当化（服务端另有两道纯格式闸门做硬校验）。"""
    return (
        "Task 2 · Same-object procedure (the fields must be output in the order below, evidence "
        "always before conclusion):\n"
        "  1. cur_text: copy out, character by character, the text you **read with your own eyes** "
        "on the package or container of that thing in [CURRENT FRAME]. Write none when not a "
        "single character can be read. **Never** copy text off a reference image into this slot.\n"
        "  2. diff: the **single most obvious difference** between it and the candidate you "
        "picked, one short English sentence of roughly 5 to 15 words. One must be written. When "
        "you truly cannot find any, spell out why (e.g. \"the reference only shows a corner of "
        "the wrapper, so the portion cannot be compared\"). Boilerplate such as \"" +
        "\" / \"".join(recog_match.DIFF_BLANKS[:3]) + "\" is **forbidden** -- writing it voids "
        "this judgement.\n"
        "  3. match_evidence: an 8-character evidence code BxCxSxVx, x ∈ 1 (agrees) / 0 (differs) "
        "/ ? (cannot tell):\n"
        "       B brand and package text: **1 is allowed only when cur_text is not none and it "
        "really agrees with the text read on the candidate reference image**; cur_text=none (no "
        "package, or unreadable) is always ?, never 1;\n"
        "       C colour and appearance; S shape and portion; V container, cutlery and placement.\n"
        "     Write ? whenever you cannot tell -- **? is nothing to be ashamed of, a made-up 1 "
        "is the mistake**. A reference image is a small crop out of an older frame, so S and V "
        "often simply cannot be judged and ? is the normal, expected answer there. Four 1s is an "
        "**extremely strong** claim, allowed only when you read concrete text into cur_text and "
        "can point at the matching visual fact for every single position.\n"
        "  4. match: the candidate number, or null. **Default null**; a number is allowed only "
        "when all of these hold:\n"
        "       · the code carries no 0;\n"
        "       · packaged product: B must be 1 (i.e. you read the package text and it agrees);\n"
        "       · unpackaged produce or drink: at least two of C, S, V are 1;\n"
        "       · the same name alone is never a reason -- two different bananas, two different "
        "cups of coffee must stay separate records;\n"
        "       · different products of one category are always null: chocolate bar vs cereal "
        "bar, cola vs orange juice, the same product with a new flavour or new packaging, "
        "**a packaged snack vs a fruit of the same colour**;\n"
        "       · the object cannot be located in the reference image, or the reference is too "
        "unclear → null.\n"
        "  5. matched_name: when match is not null, copy the name of that number from the list "
        "word for word; otherwise null.\n"
        "Wrongly merging two different things is far worse than creating a duplicate card: "
        "better one card too many than one wrong merge.\n"
    )
    # 旧契约还有第 6 项 match_confidence——它是证据码的确定性函数（B=1 或 CSV 全 1
    # 才 high），2026-08-24 起由服务端 recog_match.derive_confidence 推导，字段删除


def _json_skeleton(has_refs: bool = False) -> str:
    """JSON 骨架 + 一组正交示例（2026-08-24 换血）。

    旧版三个示例只覆盖「包装零食拒并/允并」一个分支的两面，且 Werther's 在固定段
    出现 4 次——cur_text 是 B 位抵押物又无法 OCR 核验，few-shot 恰好供应了一个
    「长得就很合法」的现成值。新版原则：**每个示例覆盖一个正交分支**、食物词汇
    彼此不同、避开展台常驻登记食物；允并示例的证据码都带 ?（消灭清一色全 1）。

    has_refs=True 时多一个「参考库命中」正例——命中是展台主路径，
    「写完 ref_evidence 就结束对象」这种反直觉输出形状必须有示例钉住；
    没开参考库时该示例讲不通（清单不存在），整个跳过。示例数量随之 4↔5，
    但 has_refs 本来就是换版级配置态，不破坏固定段的逐字节稳定。"""
    demos = []
    if has_refs:
        demos.append((
            " (the food in [CURRENT FRAME] hits the reference list -- name copies the list name "
            "word for word, the object ends right after ref_evidence, every other field is "
            "omitted)",
            "{\"items\":[{\"seen\":\"purple square chocolate bar package, reads Milka on the "
            "front, centre of the table\",\"name\":\"Milka Alpine Milk Chocolate\","
            "\"type\":\"food\",\"edible\":true,\"ref_id\":2,\"ref_confidence\":\"high\","
            "\"ref_evidence\":\"centre of the table, white Milka lettering on a purple wrapper, "
            "same as list entry [2]\"}]}"))
    demos.append((
        " (a packaged snack that misses the reference list and is the same can as candidate[1] "
        "-- merging allowed; the code carries a ?, matched_name copies the candidate name)",
        "{\"items\":[{\"seen\":\"green cylindrical can of crisps, reads Pringles on the tube, "
        "right side of the table\",\"name\":\"Pringles Sour Cream\",\"type\":\"food\","
        "\"edible\":true,\"ref_id\":null,\"ref_confidence\":null,\"ref_evidence\":null,"
        "\"description_en\":\"Stackable potato chips in a can.\","
        "\"calories_kcal\":530,\"protein_g\":4.0,\"carbs_g\":50.0,\"fat_g\":35.0,"
        "\"classification\":\"Bad\",\"cur_text\":\"Pringles\","
        "\"diff\":\"the reference only shows the upper half of the tube\","
        "\"match_evidence\":\"B1C1S?V1\",\"match\":1,"
        "\"matched_name\":\"Pringles Sour Cream\"}]}"))
    demos.append((
        " (unpackaged natural produce with no readable text -- B can only be ?, merged with "
        "candidate[1] on colour, shape and placement all agreeing)",
        "{\"items\":[{\"seen\":\"yellow crescent fruit with brown speckles, no text at all, "
        "centre of a white plate\",\"name\":\"Banana\",\"type\":\"food\",\"edible\":true,"
        "\"ref_id\":null,\"ref_confidence\":null,\"ref_evidence\":null,"
        "\"description_en\":\"A ripe banana with brown spots.\","
        "\"calories_kcal\":105,\"protein_g\":1.3,\"carbs_g\":27.0,\"fat_g\":0.4,"
        "\"classification\":\"Good\",\"cur_text\":\"none\","
        "\"diff\":\"the stem points left in the reference, right here\","
        "\"match_evidence\":\"B?C1S1V1\",\"match\":1,\"matched_name\":\"Banana\"}]}"))
    demos.append((
        " (a liquid is named after **its content**, wording on the mug is not a name; with no "
        "similar candidate in the frame match_evidence is NONE)",
        "{\"items\":[{\"seen\":\"white mug holding dark brown liquid, no drink brand on the mug, "
        "left side of the table\",\"name\":\"Coffee\",\"type\":\"drink\",\"edible\":true,"
        "\"ref_id\":null,\"ref_confidence\":null,\"ref_evidence\":null,"
        "\"description_en\":\"A mug of black coffee.\","
        "\"calories_kcal\":5,\"protein_g\":0.3,\"carbs_g\":0.0,\"fat_g\":0.0,"
        "\"classification\":\"Neutral\",\"cur_text\":\"none\","
        "\"diff\":\"no similar candidate to compare against\",\"match_evidence\":\"NONE\","
        "\"match\":null,\"matched_name\":null}]}"))
    demos.append((
        " ([CURRENT FRAME] holds only a phone and an empty cup, nothing edible -- the right "
        "answer is an empty array, never pad it with an electronic device or an empty container)",
        "{\"items\":[]}"))
    body = "".join("Example %d%s:\n%s\n" % (i + 1, title, js)
                   for i, (title, js) in enumerate(demos))
    return (
        "Output JSON only, no explanation, **all values in English**. The field order must be "
        "exactly the one below:\n" + body +
        "items holds **exactly one object**; when [CURRENT FRAME] has neither food nor liquid, "
        "items is the empty array [].\n"
    )


# ══════════════════════════════════════════════════════════════════════
# 可变段：每轮都变，本来就进不了缓存，怎么排都不影响命中
# ══════════════════════════════════════════════════════════════════════
def candidates_intro(candidates) -> str:
    """任务二开场 + 候选清单文字（紧挨着它们的参考图）。"""
    if not candidates:
        return ("Task 2 · Deduplication: nothing has been recorded yet, so everything you "
                "recognise is new. match is always null, match_evidence is NONE, matched_name is "
                "null, while cur_text and diff are still filled in as required above.\n")
    lines = "\n".join("  [%d] %s (%s) -- %s" % (
        i + 1, c.get("name", ""), foodref.type_en(c.get("type")),
        c.get("desc") or "no description")
        for i, c in enumerate(candidates))
    return ("Task 2 · Deduplication: the objects recorded within the last 30 seconds are listed "
            "below (number · name · type · description). Once more: **these names are conclusions "
            "from earlier rounds and some of them may simply be wrong**, and they are no evidence "
            "at all that these things are in the current frame.\n" + lines + "\n"
            "The [PAST REFERENCE] image of each entry follows below. Seeing something in a "
            "reference image only means it was in frame at some point, **not that it is still on "
            "the table**.\n")


def candidate_label(idx: int, cand) -> str:
    """一张历史参考图前面的标签行（idx 是 1-based 的候选编号）。"""
    return ("[PAST REFERENCE · CANDIDATE[%d] %s] -- cropped out of a frame from an earlier "
            "round, not the current frame" % (idx, cand.get("name", "")))


def current_label(has_candidates: bool) -> str:
    """当前画面前面的标签行：把参考图区段明确收口。"""
    prefix = "The [PAST REFERENCE] images end here. " if has_candidates else ""
    return (prefix + "This next image, and only this one, is [CURRENT FRAME] -- the real table "
            "the camera sees right now:")


def boxed_label(n_food: int, n_drink: int) -> str:
    """带框图前面的标签行（仅门控口径）。检测计数是每轮变量，只能待在可变段。"""
    return ("[CURRENT FRAME · BOXED] the same current frame, red boxes = suspected food, blue "
            "boxes = suspected liquid/container. The detector hit this round: %d food box(es), "
            "%d liquid box(es) (the detector can miss things, but never output something that "
            "plainly is not in the current frame)." % (n_food, n_drink))


def pick_rule(last_pick, candidates) -> str:
    """挑选规则：从「要求确认」改成「独立复核」。

    旧文案是「上一轮选中的是 X，只要它仍在画面里就继续选它」——这把开放问题偷换成
    了是非题，而 VLM 对是非题的 yes-bias 极强：错一次就会一路确认下去（线上出现过
    一张卡自我确认 252 次）。新文案强制先独立得出结论，再决定要不要沿用旧名字。"""
    if not last_pick or not last_pick.get("name"):
        return ("Picking rule: pick the one you are most confident about and that dominates "
                "[CURRENT FRAME].\n")
    idx = next((i + 1 for i, c in enumerate(candidates or [])
                if c.get("id") == last_pick.get("card_id")), None)
    who = "\"%s\"%s" % (last_pick["name"], (" (that is list entry [%d])" % idx) if idx else "")
    return ("Picking rule (run the steps in this order, the order may not be swapped):\n"
            "  · the previous round displayed %s. That is **only the previous conclusion and it "
            "is not guaranteed to be correct**;\n"
            "  · first write seen and name independently, exactly as Task 1 demands -- while "
            "writing, put the previous conclusion entirely aside: do not use it to \"confirm\" "
            "anything and do not let it colour how you read the frame;\n"
            "  · only afterwards compare: keep that wording only when the conclusion you reached "
            "independently really refers to the same thing as %s, so the display stays stable;\n"
            "  · as soon as one detail does not line up (packaging, form, colour, material, any "
            "single one of them), output your own conclusion and do not accommodate the previous "
            "round -- **a name that stays wrong is far worse than a name that changes once**.\n"
            % (who, who))


def tail(last_pick, candidates) -> str:
    """收尾：粘性规则 + 最后再钉一次「只描述当前画面」。

    这段紧贴生成位置，是模型印象最深的一段，所以把最关键的那条约束放在这里复读。"""
    return (pick_rule(last_pick, candidates) +
            "Start now: describe only what really exists in that [CURRENT FRAME] image above; "
            "the reference images and the list may be used for the same-object judgement of "
            "Task 2 and for nothing else. Repeating hard constraint 4 once more: output only "
            "food or drink that can go into a mouth; when the frame holds nothing but non-food "
            "such as phones, laptops, cutlery, empty packages or empty containers, items must be "
            "the empty array. Output JSON only, no explanation, all values in English.\n")


def render_for_log(content) -> str:
    """把 content 数组渲染成可读文本，供观测日志与控制面展示。

    控制面的「VLM 识别日志」读的是 req.prompt 这一个字符串——prompt 拆成 blocks 之后
    若只记文字块，图片排在哪里就看不见了，而这次改动的要害恰恰是排布。渲染成
    「文字原样 + 图片位置占位」后，控制面零改动就能看到真实顺序。"""
    out = []
    for block in content or []:
        if block.get("type") == "text":
            out.append(block.get("text", ""))
        else:
            out.append("〔图片〕")
    return "\n".join(out)
