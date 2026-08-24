# -*- coding: utf-8 -*-
"""参考食物库的配置与目录态（纯逻辑，无 cv2/torch 依赖，可单测）。

背景：浅体验区原本每一轮都让 VLM 从零识别，名称、描述、营养四个数字全由模型现编——
同一根香蕉两轮之间卡路里会从 89 跳到 105，展台上肉眼可见。参考食物库把这件事翻过来：
运营在控制面预先录入展台要摆的那几种食物（实物图 + 营养 + 描述），识别时把这批参考图
钉在请求最前面当"参考清单"，VLM 只回答"桌上这个是不是清单里的第几项"，命中就由服务端
查库回填内容，模型不再编数字。

本模块只管**数据与文本**：配置钳制、条目校验、目录落盘、参考区文案拼装、命中判定。
图片的解码/缩放/烧角标/JPEG 编码留在 app.py（那边才有 cv2），本模块只搬运文件名与字符串。

前缀稳定性是本模块的硬约束：参考区一旦变了字节，vLLM 的 prefix cache 就整段失效。
所以 `menu_items()` 恒按 id 升序返回，任何影响前缀的改动都必须 `bump()` 版本号，
让调用方知道要重建缓存。
"""
import json
import re
import threading
import time
from pathlib import Path

# ── 规模上限（控制面与服务端双重校验，前端不可信）────────────────────────────
MAX_ITEMS = 20              # 最多登记多少种食物
MAX_IMAGES_PER_ITEM = 2     # 每种最多几张实物图
MAX_REF_IMAGES = MAX_ITEMS * MAX_IMAGES_PER_ITEM

# 参考图规范化边长的可选档位（正方像素当量，实际按原图长宽比缩放、不加黑边）。
# 视觉 token ≈ (edge/32)²（该模型 patch_size=16 + merge_size=2，即 32×32 像素 1 token）；
# 256 是模型的有效下限（shortest_edge=65536 像素，再小会被放大回来，纯浪费）。
EDGE_CHOICES = [256, 320, 384, 448, 512]

# 默认口径：开着、384px（320 实测偏糊，包装小字读不出）、q88（参考图只编码一次，
# 不必为体积牺牲清晰度）、命中置信度至少 medium。
DEFAULTS = {"on": True, "edge": 384, "quality": 88, "min_confidence": "medium"}

CONFIDENCE_ORDER = ["low", "medium", "high"]     # 从松到严
TYPES = ["食物", "液体"]
CLASSIFICATIONS = ["Good", "Neutral", "Bad"]
_CLS_CANON = {c.lower(): c for c in CLASSIFICATIONS}

# 文本字段长度上限（与卡片/展示端口径一致，超长直接截断而不是报错）
NAME_MAX = 40
LOOK_MAX = 60
DESC_EN_MAX = 60
DESC_DE_MAX = 90
PORTION_MAX = 24
ALIAS_MAX = 8               # 每项最多几个别名


def est_tokens(edge: int) -> int:
    """单张参考图的视觉 token 估算：32×32 像素 1 token。"""
    side = max(1, int(edge) // 32)
    return side * side


# 模型的 min_pixels（preprocessor_config 的 shortest_edge）：像素数比它少的图会被
# 预处理器放大回来，缩得再小也白缩，反而多花一次重采样。
MIN_PIXELS = 65536


def fit_size(w: int, h: int, edge: int):
    """参考图规范化尺寸：按「总像素 ≈ edge²」等比缩放，两边向上取整到 32 的倍数。

    不做正方形 letterbox——补出来的黑边一样按 32×32 一个 token 计费，纯浪费。
    对齐 32 是因为视觉 token 正好覆盖 32×32 像素，不对齐的话尺寸就由预处理器说了算。"""
    edge = int(edge)
    if w <= 0 or h <= 0:
        return edge, edge
    scale = ((edge * edge) / float(w * h)) ** 0.5
    nw = max(32, -(-int(round(w * scale)) // 32) * 32)
    nh = max(32, -(-int(round(h * scale)) // 32) * 32)
    while nw * nh < MIN_PIXELS:
        nw += 32
        nh += 32
    return nw, nh


def _clamp_num(v, lo, hi, as_int=False):
    """数字字段钳制：非法/NaN 返回 None（前端隐藏该行，与识别链路的 guardrail 同款）。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:
        return None
    f = min(hi, max(lo, f))
    return int(round(f)) if as_int else round(f, 1)


def _text(v, limit):
    """文本字段：转字符串、去换行、截断。"""
    return str(v or "").strip().replace("\n", " ")[:limit]


def name_key(s: str) -> str:
    """名称归一化键：小写、去掉所有非字母数字与非中日韩字符。

    用于「模型没给编号但名字对得上」的兜底命中：Coca-Cola / coca cola / COCACOLA
    归一化后都是 cocacola。中文按字符保留（不做分词）。"""
    s = str(s or "").lower()
    return "".join(re.findall(r"[a-z0-9一-鿿]+", s))


def normalize_config(patch, base=None) -> dict:
    """merge-patch 语义的配置规范化：非法值忽略、越界回落默认，绝不抛异常。

    控制面来的是不可信输入，坏值不该打断识别链路（与 recog_direct.normalize 同款约定）。"""
    out = dict(DEFAULTS if base is None else base)
    for key in DEFAULTS:
        out.setdefault(key, DEFAULTS[key])
    if not isinstance(patch, dict):
        return out
    if "on" in patch:
        val = patch["on"]
        out["on"] = (val.strip().lower() in ("1", "true", "on", "yes")
                     if isinstance(val, str) else bool(val))
    if "edge" in patch:
        try:
            edge = int(float(patch["edge"]))
        except (TypeError, ValueError):
            edge = None
        if edge in EDGE_CHOICES:
            out["edge"] = edge
    if "quality" in patch:
        q = _clamp_num(patch["quality"], 60, 95, as_int=True)
        if q is not None:
            out["quality"] = q
    if "min_confidence" in patch:
        conf = str(patch["min_confidence"] or "").strip().lower()
        if conf in CONFIDENCE_ORDER:
            out["min_confidence"] = conf
    return out


def normalize_item(patch, base=None) -> dict:
    """条目字段校验：名称必填（空名直接 ValueError，其余字段一律容错）。

    营养四项按**画面上这一份的总量**录入（不是每 100 克）——展示端显示的就是这份的
    数字，库里存定值才能让命中项逐轮不抖。"""
    out = dict(base or {})
    src = patch if isinstance(patch, dict) else {}

    def take(key, default=None):
        return src[key] if key in src else out.get(key, default)

    name = _text(take("name"), NAME_MAX)
    if not name:
        raise ValueError("名称不能为空")
    out["name"] = name
    out["name_en"] = _text(take("name_en"), NAME_MAX) or name
    out["look"] = _text(take("look"), LOOK_MAX)
    out["portion"] = _text(take("portion"), PORTION_MAX)
    out["description_en"] = _text(take("description_en"), DESC_EN_MAX)
    out["description_de"] = _text(take("description_de"), DESC_DE_MAX)

    typ = str(take("type", "食物") or "").strip().lower()
    out["type"] = "液体" if ("液" in typ or "饮" in typ or "drink" in typ
                             or "liquid" in typ) else "食物"

    aliases = take("aliases", [])
    if isinstance(aliases, str):
        aliases = re.split(r"[,，;；\n]", aliases)
    if not isinstance(aliases, (list, tuple)):
        aliases = []
    seen, clean = set(), []
    for a in aliases:
        a = _text(a, NAME_MAX)
        if not a or name_key(a) in seen:
            continue
        seen.add(name_key(a))
        clean.append(a)
    out["aliases"] = clean[:ALIAS_MAX]

    out["calories_kcal"] = _clamp_num(take("calories_kcal"), 0, 5000, as_int=True)
    out["protein_g"] = _clamp_num(take("protein_g"), 0, 500)
    out["carbs_g"] = _clamp_num(take("carbs_g"), 0, 500)
    out["fat_g"] = _clamp_num(take("fat_g"), 0, 500)
    out["classification"] = _CLS_CANON.get(
        str(take("classification", "") or "").strip().lower(), "")

    enabled = take("enabled", True)
    out["enabled"] = (enabled.strip().lower() in ("1", "true", "on", "yes")
                      if isinstance(enabled, str) else bool(enabled))
    out.setdefault("images", [])
    return out


def select_kept(images, keep) -> list:
    """按 keep 序号列表挑出要保留的旧图元信息（编辑时逐张删图/换图用）。

    keep 是控制面传来的旧图序号数组（按展示顺序）；未知序号忽略、重复去重、
    非法值跳过，返回的元信息按 keep 顺序排列——序号重排（0..k-1）由调用方在
    落盘时完成，这里只做挑选，不碰文件。"""
    by_n = {}
    for im in images or []:
        if isinstance(im, dict) and im.get("n") is not None:
            try:
                by_n.setdefault(int(im["n"]), im)
            except (TypeError, ValueError):
                continue
    out, seen = [], set()
    for n in keep or []:
        try:
            n = int(n)
        except (TypeError, ValueError):
            continue
        if n in seen or n not in by_n:
            continue
        seen.add(n)
        out.append(dict(by_n[n]))
    return out[:MAX_IMAGES_PER_ITEM]


def item_public(item: dict) -> dict:
    """下发控制面的条目投影：去掉内部字段，图片只给数量与尺寸元信息。"""
    out = {k: v for k, v in item.items() if k != "images"}
    out["images"] = [{"n": im.get("n"), "w": im.get("w"), "h": im.get("h")}
                     for im in item.get("images", [])]
    return out


class Catalog:
    """参考食物目录：内存态 + JSON 落盘（原子写），线程安全。

    version 是**前缀版本号**：条目、图片、edge/quality 任一变化都 +1。调用方拿它当
    缓存键——版本没变就必须复用同一批图片字节，否则 prefix cache 每轮都 miss。"""

    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self.data = {"version": 1, "config": dict(DEFAULTS), "items": [], "next_id": 1}
        self._load()

    # ── 落盘 ──────────────────────────────────────────────────────────
    def _load(self):
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(raw, dict):
            return
        items = []
        for it in raw.get("items") or []:
            if not isinstance(it, dict):
                continue
            try:
                norm = normalize_item(it)
            except ValueError:
                continue          # 坏数据跳过，不让一条脏记录锁死整个库
            norm["id"] = int(it.get("id") or 0) or (len(items) + 1)
            norm["images"] = [im for im in (it.get("images") or []) if isinstance(im, dict)]
            items.append(norm)
        self.data = {
            "version": max(1, int(raw.get("version") or 1)),
            "config": normalize_config(raw.get("config")),
            "items": sorted(items, key=lambda x: x["id"]),
            "next_id": max([int(raw.get("next_id") or 1)]
                           + [it["id"] + 1 for it in items]),
        }

    def _save_locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(self.path)

    def _bump_locked(self):
        self.data["version"] += 1
        self._save_locked()

    # ── 读 ────────────────────────────────────────────────────────────
    def snapshot(self) -> dict:
        with self._lock:
            return json.loads(json.dumps(self.data, ensure_ascii=False))

    def config(self) -> dict:
        with self._lock:
            return dict(self.data["config"])

    def version(self) -> int:
        with self._lock:
            return int(self.data["version"])

    def enabled(self) -> bool:
        with self._lock:
            return bool(self.data["config"]["on"])

    def get(self, item_id):
        with self._lock:
            for it in self.data["items"]:
                if it["id"] == int(item_id):
                    return json.loads(json.dumps(it, ensure_ascii=False))
        return None

    def menu_items(self) -> list:
        """参与识别的条目：只取 enabled 且至少有一张图的，**恒按 id 升序**。

        顺序恒定是前缀能命中 KV 缓存的前提，不许按时间/命中频次重排。"""
        with self._lock:
            items = [json.loads(json.dumps(it, ensure_ascii=False))
                     for it in self.data["items"]
                     if it.get("enabled") and it.get("images")]
        return sorted(items, key=lambda x: x["id"])[:MAX_ITEMS]

    # ── 写 ────────────────────────────────────────────────────────────
    def set_config(self, patch) -> dict:
        with self._lock:
            new = normalize_config(patch, self.data["config"])
            changed = new != self.data["config"]
            self.data["config"] = new
            if changed:
                self._bump_locked()      # edge/quality 变了 = 参考图字节变了
            else:
                self._save_locked()
            return dict(new)

    def upsert(self, patch, item_id=None) -> dict:
        """新增或更新一条。新增时校验种类上限，更新时保留未提交的字段。"""
        with self._lock:
            items = self.data["items"]
            if item_id is None:
                if len(items) >= MAX_ITEMS:
                    raise ValueError("最多登记 %d 种食物，先删掉一些再加" % MAX_ITEMS)
                item = normalize_item(patch)
                item["id"] = self.data["next_id"]
                item["images"] = []
                item["created_at"] = time.time()
                item["updated_at"] = item["created_at"]
                self.data["next_id"] += 1
                items.append(item)
            else:
                cur = next((it for it in items if it["id"] == int(item_id)), None)
                if cur is None:
                    raise KeyError("条目不存在：%s" % item_id)
                merged = normalize_item(patch, cur)
                merged["id"] = cur["id"]
                merged["images"] = cur.get("images", [])
                merged["created_at"] = cur.get("created_at", time.time())
                merged["updated_at"] = time.time()
                items[items.index(cur)] = merged
                item = merged
            items.sort(key=lambda x: x["id"])
            self._bump_locked()
            return json.loads(json.dumps(item, ensure_ascii=False))

    def set_images(self, item_id, images) -> dict:
        """整组替换某条的图片元信息（图片文件本身由 app.py 写盘）。"""
        with self._lock:
            cur = next((it for it in self.data["items"] if it["id"] == int(item_id)), None)
            if cur is None:
                raise KeyError("条目不存在：%s" % item_id)
            cur["images"] = list(images or [])[:MAX_IMAGES_PER_ITEM]
            cur["updated_at"] = time.time()
            self._bump_locked()
            return json.loads(json.dumps(cur, ensure_ascii=False))

    def delete(self, item_id) -> bool:
        with self._lock:
            before = len(self.data["items"])
            self.data["items"] = [it for it in self.data["items"]
                                  if it["id"] != int(item_id)]
            if len(self.data["items"]) == before:
                return False
            self._bump_locked()
            return True


# ══════════════════════════════════════════════════════════════════════
# 参考区文案（进 prompt 最前面的那一段，必须逐字节稳定）
# ══════════════════════════════════════════════════════════════════════
def menu_intro(n_items: int, n_images: int) -> str:
    """参考区开场白。措辞要点：这批图**不是当前画面**，只是待会儿要比对的清单。

    历史教训：画面空无一物时模型会照着参考图报 Snickers，所以隔离要在文字与图片
    （角标横幅）两处同时说，且后面任务里还要再申明一次。"""
    return ("本展台已登记 %d 种参考食物，下面依次给出它们的编号、名称与实物照片"
            "（共 %d 张，每张左上角烧有 REF 角标）。\n"
            "这些照片**全部是参考清单**，不是当前画面、也不代表桌上现在有这些东西——"
            "它们只用于稍后判断「桌上那一样是不是其中某一项」。\n" % (n_items, n_images))


def item_label(idx: int, item: dict) -> str:
    """清单里一项的标签行（紧跟在它的实物图前面）。

    idx 是 1-based 的清单编号，也就是模型要回填的 ref_id。"""
    bits = ["[%d] %s" % (idx, item.get("name") or "")]
    name_en = item.get("name_en") or ""
    if name_en and name_key(name_en) != name_key(item.get("name") or ""):
        bits.append(name_en)
    bits.append(item.get("type") or "食物")
    if item.get("look"):
        bits.append(item["look"])
    return " · ".join(bits)


def build_blocks(items, uri_of):
    """拼参考区的 content 片段：开场白 → [标签行 + 该项的图]×N。

    uri_of(item_id, n) 返回那张图的 dataURI（app.py 那边从磁盘缓存取），返回 None
    表示图没了——**整项跳过**：给了名字却没有图，模型只能瞎对。
    返回 (进清单的条目, blocks, 图片总数)；条目下标 +1 就是模型要回填的 ref_id。
    顺序完全由 items 决定（调用方保证按 id 升序），本函数不重排、不去重。"""
    kept, uris = [], []
    for it in items:
        got = [u for u in (uri_of(it["id"], im.get("n", i))
                           for i, im in enumerate(it.get("images") or [])) if u]
        if not got:
            continue
        kept.append(it)
        uris.append(got)
    if not kept:
        return [], [], 0
    total = sum(len(u) for u in uris)
    blocks = [{"type": "text", "text": menu_intro(len(kept), total)}]
    for idx, (it, group) in enumerate(zip(kept, uris)):
        blocks.append({"type": "text", "text": item_label(idx + 1, it)})
        blocks.extend({"type": "image_url", "image_url": {"url": u}} for u in group)
    return kept, blocks, total


def menu_ack(n_items: int) -> str:
    """预填的 assistant 回执：把固定前缀的边界钉死在这句话上。"""
    return ("已登记 %d 项参考食物，我不会把它们当成当前画面里的东西。"
            "接下来请给我当前画面。" % n_items)


def task_zero(items: list, min_confidence: str = "medium") -> str:
    """任务零·命中判定的 prompt 段（拼在当前画面那条消息里，属于每轮可变部分）。

    判定口径刻意「宽进严出」：让模型放心给编号（判的是同一种商品而不是同一个个体），
    把关交给服务端的置信度闸门与证据校验——模型在这一步犹豫，展台上就是认不出来。"""
    n = len(items)
    lines = "\n".join("  " + item_label(i + 1, it) for i, it in enumerate(items))
    return (
        "任务零·参考清单命中判定（先做这个，它决定后面几项要不要填）：\n"
        "开头那 %d 项参考食物的编号与名称是：\n%s\n"
        "看当前画面里你要报的那一样东西，判断它是不是清单里某一项的**同一种商品/同一种食材**：\n"
        "  · 判的是「同一种」，不是「同一个」——同一款包装的另一根、同一种水果的另一颗，"
        "都算命中；角度、光线、摆放、已经吃掉一部分，都不影响命中；\n"
        "  · 品牌包装文字、颜色、形状、容器这四项里，只要有**两项对得上且没有一项明显冲突**，"
        "就给编号，不要因为拍摄条件不同而犹豫；\n"
        "  · 但同类目不同产品必须判不命中：不同品牌/口味的巧克力棒、可乐与橙汁、"
        "换了包装的同名产品，一律 ref_id=null；\n"
        "  · 清单里没有的东西就是没有，ref_id=null，照常按你自己的判断识别，不要硬套编号。\n"
        "输出这三个字段：\n"
        "  ref_id：命中的清单编号（整数 1~%d），没命中填 null；\n"
        "  ref_confidence：\"high\"=包装文字/品牌可辨认或形状颜色高度一致；"
        "\"medium\"=像，但拍得不够清楚；\"low\"=只是同类目。"
        "（低于 \"%s\" 的判定会被系统丢弃当作没命中）；\n"
        "  ref_evidence：一句话说明你在**当前画面的哪个位置**看到它、凭什么判成这一项"
        "（如 \"盘子中间偏左，深棕包装上有白色 Snickers 字样\"）。"
        "说不出画面位置就说明你在照参考图编，这时必须填 null。\n"
        "命中时（ref_id 不为 null）：name 一字不差照抄清单里的名称，"
        "然后**写完 ref_evidence 就直接结束这个对象**——description_en / description_de /"
        " calories_kcal / protein_g / carbs_g / fat_g / classification / cur_text / diff /"
        " match_evidence / match / matched_name 这些字段**一律省略不写**："
        "内容系统会查库回填，去重系统会按清单编号自动归并，你写了也会被整组丢弃。\n"
        "未命中（ref_id=null）时，才继续把上面这些字段全部填完。\n"
        % (n, lines, n, min_confidence)
    )


# ══════════════════════════════════════════════════════════════════════
# 命中判定与回填
# ══════════════════════════════════════════════════════════════════════
def conf_ok(conf: str, minimum: str) -> bool:
    """置信度是否达标（缺省/非法一律当 low 处理）。"""
    try:
        got = CONFIDENCE_ORDER.index(str(conf or "").strip().lower())
    except ValueError:
        got = 0
    try:
        need = CONFIDENCE_ORDER.index(str(minimum or "medium").strip().lower())
    except ValueError:
        need = 1
    return got >= need


def match_by_name(name: str, items: list):
    """名称/别名兜底匹配：模型忘了给编号但名字对得上时，照样算命中。"""
    key = name_key(name)
    if not key:
        return None
    for it in items:
        keys = {name_key(it.get("name")), name_key(it.get("name_en"))}
        keys |= {name_key(a) for a in it.get("aliases") or []}
        if key in keys - {""}:
            return it
    return None


def resolve_hit(parsed, items, min_confidence="medium"):
    """判定一条模型输出是否命中参考清单，返回 (item, source, reason)。

    source：'ref_id'=模型直接给的编号；'name'=编号为空但名称/别名对上；None=未命中。
    reason 是给观测日志看的判定理由（控制面排障时最想知道的就是"为什么没命中"）。"""
    if not items:
        return None, None, "参考库为空"
    ref_id = parsed.get("ref_id")
    if isinstance(ref_id, int) and 1 <= ref_id <= len(items):
        if not conf_ok(parsed.get("ref_confidence"), min_confidence):
            return None, None, "置信度 %s 低于阈值 %s" % (
                parsed.get("ref_confidence") or "空", min_confidence)
        if not str(parsed.get("ref_evidence") or "").strip():
            return None, None, "没给画面位置证据，按幻觉丢弃"
        return items[ref_id - 1], "ref_id", "编号 [%d] 命中" % ref_id
    hit = match_by_name(parsed.get("name"), items)
    if hit is not None:
        return hit, "name", "编号为空，名称「%s」对上清单" % parsed.get("name")
    return None, None, "未命中清单"


# 命中后由库覆盖的字段：名称、描述、营养四项、健康分级。
# 覆盖是**整组**的，不做逐字段回退——半查库半模型编出来的卡片没法解释。
OVERRIDE_FIELDS = ["description_en", "description_de", "calories_kcal",
                   "protein_g", "carbs_g", "fat_g", "classification"]


def apply_hit(parsed: dict, item: dict, source: str) -> dict:
    """把库里的内容整组覆盖到模型输出上，并打命中标记。"""
    out = dict(parsed)
    out["name"] = item.get("name_en") or item.get("name")
    out["type"] = item.get("type") or "食物"
    for key in OVERRIDE_FIELDS:
        out[key] = item.get(key)
    out["ref_hit_id"] = item["id"]
    out["ref_source"] = source
    out["source"] = "catalog"
    return out


def budget(items: list, edge: int) -> dict:
    """参考区规模：种类/图数/token 估算，供控制面常驻显示与保存前校验。"""
    n_imgs = sum(min(len(it.get("images") or []), MAX_IMAGES_PER_ITEM) for it in items)
    per = est_tokens(edge)
    return {"items": len(items), "images": n_imgs, "tokens": n_imgs * per,
            "tokens_per_image": per, "max_items": MAX_ITEMS,
            "max_images": MAX_REF_IMAGES}
