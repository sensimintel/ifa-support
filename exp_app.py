"""
感知链路实验台（8061）——与产线 8060 完全隔离的「完整识别链路」实验环境，UI 复刻 8060 主页。

页面形态（与 8060 主页同构）：
- `/` 分栏首页：左栏 = 设备原帧 + 实验检测标注帧 + 点云产物（DA3 产物经 8060 共享，浏览器
  直取 8060 接口，不重复跑 DA3）；右栏 = iframe `/recog` 实验识别卡片流；右上「调节」抽屉
  热调实验链路全部参数。
- `/recog` 实验识别卡片流：完整复刻 8060 /recog（打字机、扇形叠卡、去重合并置顶、
  lightbox、Qwen/Gemini 切换），卡片字段含 卡路里/宏量/健康分级。

链路（全部在本进程，与 8060 零共享状态）：
  8060 只读接口拉帧(采样率可调) → 实验检测级(SAM3 /v1/segment 可调阈值、LA 可选对照、
  embedding 匹配插槽) → 命中触发 VLM 识别(与产线同款 prompt/解析/五道合并闸门，可魔改) → 实验卡片流。

隔离与负载：
- 不加载任何大模型：SAM3/LA 是 5090 已常驻的 HTTP 服务，SAM3 走无状态 /v1/segment
  （不碰产线流式 session）；VLM 在 GCP，识别逻辑本进程独立一份（正是为了随便魔改）。
- DA3 不重复跑：点云产物直接展示 8060 产线产物（8060 CORS 全放开，浏览器跨端口直取）。
- 总开关默认关 = 不拉帧不推理，GPU 零增量。
"""

import base64
import json
import os
import re
import threading
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from fastapi import Body, FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

app = FastAPI(title="感知链路实验台")

# ── 上游地址（与 8060 共用仓根 .env，run-exp.sh 负责加载） ──
FRAME_SOURCE = os.environ.get("EXP_FRAME_SOURCE", "http://127.0.0.1:8060").rstrip("/")
SAM3_ENDPOINT = os.environ.get("SAM3_ENDPOINT", "http://127.0.0.1:8013").rstrip("/")
LOCATE_ENDPOINT = os.environ.get("EXP_LOCATE_ENDPOINT",
                                 "http://127.0.0.1:8000/v1/chat/completions")
LOCATE_MODEL = "nvidia/LocateAnything-3B"
SAM3_TIMEOUT = float(os.environ.get("EXP_SAM3_TIMEOUT", "30"))
LOCATE_TIMEOUT = float(os.environ.get("EXP_LOCATE_TIMEOUT", "20"))
UNKNOWN_DEVICE = "unknown"

# ══════════════════════════════════════════════════════════════════════
# 实验链路配置（「调节」抽屉热改；本进程私有状态，与 8060 无任何共享）
# 检测查询词分「食物 / 液体」两组：沿用产线语义（红框=食物、蓝框=液体），
# VLM 识别 prompt 的图2 带框图按此上色。
# ══════════════════════════════════════════════════════════════════════
_cfg = {
    "enabled": False,            # 总开关：关=不拉帧不推理，GPU 零增量
    "interval": 1.0,             # 检测采样间隔（秒）
    "sam3_on": True,             # SAM3 检测（无状态 /v1/segment）
    "sam3_food_queries": "food",
    "sam3_drink_queries": "drink",
    "sam3_thresh": 0.5,          # SAM3 score 阈值（客户端过滤；低于阈值画灰框便于调参）
    # SAM3 server 端打分口径（与 8060 /sam3tune 同款语义，随请求生效、请求间天然隔离）：
    # keep = presence^α × cond > det_thresh；α=1 且 det_thresh=0 时完全等价模型默认行为
    "sam3_alpha": 1.0,           # presence α（presence 指数软化：1=原始行为，0=完全忽略 presence）
    "sam3_det_thresh": 0.0,      # server 端检测阈值；0=恢复模型默认（0.5，联合分口径）
    "sam3_mask": True,           # 标注帧叠 mask 半透明高亮
    "la_on": False,              # LocateAnything 对照（默认关——实验主题就是不走 LA）
    "la_food_queries": "food",
    "la_drink_queries": "bottle; glass",
    "embed_on": False,           # embedding 匹配插槽（_embed_match 接入前无输出）
    "recog_on": True,            # 命中检测后是否触发 VLM 识别（关=只看检测级）
    "recog_interval": 4.0,       # 两次 VLM 识别最小间隔（秒），与产线同款节流语义
}
_cfg_lock = threading.Lock()

# ── 实验产物：最新标注帧 + 最近轮次记录 ──
_out = {"jpg": None, "seq": 0, "src_seq": -1}
_rounds = []
_out_lock = threading.Lock()
ROUNDS_KEEP = 60

_FOOD_COLOR = (222, 52, 52)    # RGB：食物=红（与产线一致）
_DRINK_COLOR = (46, 120, 235)  # 液体=蓝
_CUT_COLOR = (142, 142, 147)   # 低于阈值被裁掉的实例=灰


def _split_queries(s):
    """查询串 → 查询列表（分号/逗号分隔，去空白去空项）。"""
    return [q.strip() for q in str(s).replace("，", ",").replace("；", ";")
            .replace(",", ";").split(";") if q.strip()]


def _http_json(url, payload=None, timeout=10.0, headers=None):
    """极简 HTTP JSON 调用（与 8060 同款 urllib 风格，不引新依赖）。"""
    data = json.dumps(payload).encode() if payload is not None else None
    h = {"Content-Type": "application/json"} if data else {}
    h.update(headers or {})
    req = urllib.request.Request(url, data=data, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _b64_jpg(rgb):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf.tobytes()).decode() if ok else None


def _img_data_uri(rgb):
    """RGB ndarray → data URI（JPEG）。"""
    b = _b64_jpg(rgb)
    return ("data:image/jpeg;base64," + b) if b else None


# ══════════════════════════════════════════════════════════════════════
# 帧来源：拉 8060 选中设备的最新帧（只读接口，对产线零影响）
# ══════════════════════════════════════════════════════════════════════
def _fetch_frame():
    """返回 (rgb, 帧号, 设备id)；无帧/失败返回 (None, -1, unknown)。"""
    try:
        st = _http_json(FRAME_SOURCE + "/api/frame/status", timeout=5.0)
        if not st.get("has_frame"):
            return None, -1, UNKNOWN_DEVICE
        seq = int(st.get("seq", 0))
        dev = str(st.get("device") or UNKNOWN_DEVICE)
        req = urllib.request.Request(FRAME_SOURCE + "/api/frame/latest?t=%d" % seq)
        with urllib.request.urlopen(req, timeout=5.0) as r:
            raw = r.read()
        arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            return None, -1, UNKNOWN_DEVICE
        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB), seq, dev
    except Exception as e:
        print(f"[exp] 拉帧失败：{type(e).__name__}: {e}", flush=True)
        return None, -1, UNKNOWN_DEVICE


# ══════════════════════════════════════════════════════════════════════
# 检测级：SAM3（无状态）与 LocateAnything（对照）
# ══════════════════════════════════════════════════════════════════════
def _rle_decode(size, counts):
    """COCO 压缩 RLE → (H,W) uint8 0/1（与 8060 同款自带解码，5090 无 pycocotools）。"""
    h, w = int(size[0]), int(size[1])
    s = counts if isinstance(counts, str) else counts.decode()
    cnts, p = [], 0
    while p < len(s):
        x, k, more = 0, 0, 1
        while more:
            c = ord(s[p]) - 48
            x |= (c & 0x1f) << (5 * k)
            more = c & 0x20
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= (-1) << (5 * k)
        if len(cnts) > 2:
            x += cnts[-2]
        cnts.append(x)
    mask = np.zeros(h * w, np.uint8)
    idx, val = 0, 0
    for c in cnts:
        c = max(0, int(c))
        if idx >= h * w:
            break
        mask[idx:idx + c] = val
        idx += c
        val ^= 1
    return mask.reshape((h, w), order="F")


def _sam3_segment(rgb, query, alpha=1.0, det_thresh=0.0):
    """单查询 SAM3 分割：返回实例列表（归一化 box + score + mask RLE），失败返回 []。
    alpha/det_thresh=「换阈值口径」旋钮（server 随请求生效、请求间隔离，不影响产线）：
    keep = presence^α × cond > det_thresh；det_thresh=0 用模型默认（0.5 联合分口径）。"""
    b64 = _b64_jpg(rgb)
    if not b64:
        return []
    payload = {"image_b64": b64, "text": query,
               "alpha": float(alpha), "det_thresh": float(det_thresh)}
    try:
        r = _http_json(SAM3_ENDPOINT + "/v1/segment", payload, timeout=SAM3_TIMEOUT)
        return r.get("instances") or []
    except Exception as e:
        print(f"[exp] SAM3({query}) 调用失败：{type(e).__name__}: {e}", flush=True)
        return []


_LOCATE_RE = re.compile(r"<ref>(.*?)</ref>\s*<box>(.*?)</box>", re.S)
_LOCATE_INT_RE = re.compile(r"<(-?\d+)>")


def _locate_one(rgb, query):
    """单查询 LocateAnything：返回 [(x1,y1,x2,y2) 0-1 归一化]，失败返回 []。"""
    uri = _b64_jpg(rgb)
    if not uri:
        return []
    payload = {
        "model": LOCATE_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + uri}},
            {"type": "text",
             "text": f"Locate all the instances that matches the following description: {query}."},
        ]}],
        "max_tokens": 512,
    }
    try:
        r = _http_json(LOCATE_ENDPOINT, payload, timeout=LOCATE_TIMEOUT)
        content = r["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[exp] LocateAnything({query}) 调用失败：{type(e).__name__}: {e}", flush=True)
        return []
    boxes = []
    for _ref, box in _LOCATE_RE.findall(content):
        ints = [int(x) for x in _LOCATE_INT_RE.findall(box)]
        for i in range(0, len(ints) - 3, 4):
            x1, y1, x2, y2 = ints[i:i + 4]
            boxes.append((min(x1, x2) / 999.0, min(y1, y2) / 999.0,
                          max(x1, x2) / 999.0, max(y1, y2) / 999.0))
    return boxes


# ══════════════════════════════════════════════════════════════════════
# embedding 匹配插槽：接小模型（SigLIP 等）做特征匹配只需实现这一个函数。
# 输入=检测框裁剪图（RGB ndarray），输出={"name":..,"score":..} 或 None。
# 模型加载放模块级懒加载（首次调用才加载），避免服务启动就占显存。
# ══════════════════════════════════════════════════════════════════════
def _embed_match(crop_rgb):
    return None   # 未接入：占位返回 None


# ══════════════════════════════════════════════════════════════════════
# VLM 识别链路（移植自产线 app.py，供实验独立魔改——prompt/闸门/节流都可以随便改，
# 不影响 8060）：识别 prompt + 解析 guardrail + 去重合并五道闸门 + 卡片流。
# ══════════════════════════════════════════════════════════════════════
RECOG_ENDPOINT = os.environ.get("RECOG_ENDPOINT", "").strip()
RECOG_API_KEY = os.environ.get("RECOG_API_KEY", "").strip()
RECOG_MODEL = os.environ.get("RECOG_MODEL", "Qwen3.6-35B-A3B-FP8").strip()
RECOG_TARGETS = {
    "qwen": {"label": "Qwen", "endpoint": RECOG_ENDPOINT,
             "api_key": RECOG_API_KEY, "model": RECOG_MODEL},
    "gemini": {"label": "Gemini Pro",
               "endpoint": os.environ.get("RECOG_ENDPOINT_GEMINI", "").strip(),
               "api_key": os.environ.get("RECOG_API_KEY_GEMINI", "").strip(),
               "model": os.environ.get("RECOG_MODEL_GEMINI", "gemini-3.1-pro-preview").strip()},
}
_recog_target = os.environ.get("RECOG_TARGET", "qwen").strip()
if _recog_target not in RECOG_TARGETS or not RECOG_TARGETS[_recog_target]["endpoint"]:
    _recog_target = "qwen"
RECOG_TIMEOUT = 30.0
RECOG_MAX_CARDS = 200
RECOG_ACTIVE_WINDOW = 30.0   # 去重活跃窗口(秒)
RECOG_MAX_CANDIDATES = 8

_recog_lock = threading.Lock()
_recog_cv = threading.Condition(_recog_lock)
_recog_cards = {}            # device_id -> [card...]
_recog_id = 0
_recog_last_ts = 0.0
_recog_pending = []
_recog_worker_started = False

FOOD_CLASSIFICATIONS = ["Good", "Neutral", "Bad"]
_CLS_CANON = {c.lower(): c for c in FOOD_CLASSIFICATIONS}
RECOG_DESC_MAX = 20


def _recog_num(v, lo, hi, as_int=False):
    """营养数字字段 guardrail：转数字并夹进 [lo, hi]；非法/NaN 返回 None。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:
        return None
    f = min(hi, max(lo, f))
    return int(round(f)) if as_int else round(f, 1)


def _build_recog_prompt(candidates, n_food=0, n_drink=0):
    """「识别 + 去重」prompt（与产线同源；实验想改就直接改这里）。"""
    p = (
        "图1=当前画面原图；图2=当前画面带检测框版本（红框=疑似食物，蓝框=疑似液体/容器）。"
        "从图3起（若有）全部是带“HISTORY REF”横幅的历史参考图，只用于任务二对照，**不是当前画面**。\n"
        "检测器在当前画面的命中：食物框×" + str(n_food) + "、液体框×" + str(n_drink) + ""
        "（检测器可能漏检，但当前画面里明显不存在的东西绝不要输出）。\n"
        "任务一·识别：只识别图1/图2 当前画面里**真实存在**的食物、以及液体/饮料"
        "（咖啡/水/可乐/茶/杯装饮品等），逐一分别输出，食物和液体必须拆成不同 item；"
        "不要局限于框，画面里明显的都要识别；但**严禁**把只出现在历史参考图里的物品当成当前物品输出——"
        "当前画面没有食物/液体时，即使参考图里有，items 也必须为空数组。每个物品输出：\n"
        "  name：具体名称（简短，优先英文）。食物可用品牌名（如 Banana、Snickers）；"
        "液体一律按**内容物**命名（如 Water、Coffee、Cola、Orange juice）——"
        "容器上的文字只有当它是饮料产品本身的品牌（如可乐罐上的 Coca-Cola）才可用作名称，"
        "杯子/瓶子上的装饰文案（如 Good morning）不是名称；\n"
        "  type：只能是“食物”或“液体”；\n"
        "  description：一句话中文描述（不超过" + str(RECOG_DESC_MAX) + "字，如“快速补能的小食”）；\n"
        "  description_en：一句话英文描述（不超过 60 字符，如 \"A quick source of everyday energy.\"）；\n"
        "  calories_kcal：整数卡路里；protein_g / carbs_g / fat_g：蛋白质/碳水/脂肪克数（数字，最多 1 位小数）。"
        "这四个营养数字**一律按画面里这一份的实际可见份量估算**，绝不是每 100 克的标准值：\n"
        "    · 先目测这份食物的大小/体积/数量（对照画面里的手、餐具、容器等参照物），"
        "再由份量换算出总卡路里与总克数——一根大香蕉和一根小香蕉的数字必须不同，"
        "一整盘炒饭和小半碗炒饭的数字必须不同；\n"
        "    · 只剩一部分（吃剩一半、喝剩小半杯）就按剩下的量估；\n"
        "    · 液体按容器容量与液面高度估算内容物的量；\n"
        "  classification：食物健康分级，只能从这些里选一个：" + "、".join(FOOD_CLASSIFICATIONS)
        + "（营养密度高、天然少加工的选 Good；高糖/高盐/油炸/高度加工的选 Bad；介于两者之间选 Neutral）；\n"
        "  box：该物品在图1中的包围框 [x1,y1,x2,y2]，0-1000 归一化整数（左上、右下），"
        "框要紧贴物品本体。\n"
    )
    if candidates:
        lines = "\n".join("  [%d] %s（%s）—— %s" % (i + 1, c.get("name", ""),
                                                    c.get("type") or "食物", c.get("desc") or "无描述")
                          for i, c in enumerate(candidates))
        p += (
            "任务二·去重：以下是最近30秒已记录的物品清单（编号·名称·类型·描述），其参考图依次是图3、图4…"
            "（图3=[1]、图4=[2]，以此类推）：\n" + lines + "\n"
            "判断识别出的每个物品是否就是清单里某一项的**同一个具体物品**在持续出现。"
            "默认它是新物品（match=null）；只有走完下面的对照流程并全部通过，才允许 match。\n"
            "对每个物品，先在 match_evidence 字段里对最像的那个候选做逐项对照"
            "（对照该候选的参考图与当前画面里的这个物品），每一项只能填“一致/不一致/看不清”：\n"
            "  · 品牌与包装文字（无包装食物填“无包装”）；\n"
            "  · 颜色与外观；\n"
            "  · 形状与份量（明显被吃掉/喝掉一部分属于允许的变化）；\n"
            "  · 容器/餐具/摆放位置。\n"
            "然后按对照结果下结论：\n"
            "  · 任何一项为“不一致” → match=null，禁止合并；\n"
            "  · 包装食品：品牌与包装文字必须“一致”才可 match；\n"
            "  · 无包装食物/饮品：颜色外观、形状份量、容器摆放三项中至少两项“一致”且无一项“不一致”，"
            "才可 match；只是名称相同（如都叫 Banana、都叫 Coffee）绝不构成 match 的理由——"
            "不同的两根香蕉、两杯咖啡必须分开记录；\n"
            "  · 同类目但不同产品必须判新：巧克力棒 vs 谷物棒、可乐 vs 橙汁、品牌/口味/包装不同的同类零食，一律 match=null；\n"
            "  · 参考图看不清、被遮挡、或在参考图里无法定位该候选物品时 → match=null。\n"
            "match_confidence 的判定标准（先给证据再定档，不允许跳过对照直接定档）：\n"
            "  · high：对照项中至少两项明确“一致”、零项“不一致”，且参考图清晰可辨；\n"
            "  · low：其余一切情况（有“看不清”项、只有一项“一致”、或仅凭名称相似）。\n"
            "错误合并（把不同食物记成同一个）比重复建卡严重得多；宁可多一张卡，不可错并一次。\n"
            "每个物品额外输出这些字段（严格按此顺序，证据先于结论）：\n"
            "  match_evidence：上述逐项对照结果（如“品牌一致；颜色一致；形状一致；容器一致”，"
            "或“颜色不一致（红vs绿）”）；无相似候选时写“无相似候选”；\n"
            "  match_reason：一句简短中文，给出 match 或判新的结论依据；\n"
            "  match：候选编号，或 null；\n"
            "  matched_name：match≠null 时，一字不差照抄清单里该编号的名称；match=null 时为 null；\n"
            "  match_confidence：match≠null 时按上面的标准填“high”或“low”。\n"
        )
    else:
        p += ("任务二·去重：当前没有已记录的物品，识别到的都是新的，match 一律为 null，"
              "match_evidence 写“无相似候选”，match_reason 写“无已记录物品”，matched_name 为 null。\n")
    p += (
        "只输出 JSON，不要任何解释："
        "{\"items\":[{\"name\":\"Banana\",\"type\":\"食物\",\"description\":\"快速补能的小食\","
        "\"description_en\":\"A quick source of everyday energy.\","
        "\"calories_kcal\":89,\"protein_g\":1.1,\"carbs_g\":22.8,\"fat_g\":0.3,"
        "\"classification\":\"Good\",\"box\":[412,530,668,845],"
        "\"match_evidence\":\"无相似候选\",\"match_reason\":\"画面新出现的物品\","
        "\"match\":null,\"matched_name\":null,\"match_confidence\":null}]}。"
        "画面里没有食物也没有液体时，items 为空数组。"
    )
    return p


def _parse_recog(content):
    """从模型输出抽 items 并逐字段静态 guardrail（与产线同款）。"""
    try:
        obj = json.loads(content[content.index("{"): content.rindex("}") + 1])
    except Exception:
        return []
    items = obj.get("items") if isinstance(obj, dict) else None
    if not isinstance(items, list):
        return []
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        name = str(it.get("name", "")).strip()[:40]
        if not name:
            continue
        typ = str(it.get("type", "")).strip().lower()
        is_liquid = ("液" in typ or "饮" in typ or "drink" in typ or "liquid" in typ)
        desc = str(it.get("description", "")).strip().replace("\n", " ")[:RECOG_DESC_MAX]
        desc_en = str(it.get("description_en", "")).strip().replace("\n", " ")[:60]
        kcal = _recog_num(it.get("calories_kcal"), 0, 5000, as_int=True)
        protein = _recog_num(it.get("protein_g"), 0, 500)
        carbs = _recog_num(it.get("carbs_g"), 0, 500)
        fat = _recog_num(it.get("fat_g"), 0, 500)
        cls = _CLS_CANON.get(str(it.get("classification", "")).strip().lower(), "")
        try:
            match = int(it.get("match"))
        except (TypeError, ValueError):
            match = None
        evidence = str(it.get("match_evidence", "")).strip().replace("\n", " ")[:80]
        reason = str(it.get("match_reason", "")).strip().replace("\n", " ")[:60]
        mname = str(it.get("matched_name") or "").strip()[:40]
        conf = str(it.get("match_confidence") or "").strip().lower()
        conf = conf if conf in ("high", "low") else ""
        box = None
        raw_box = it.get("box")
        if isinstance(raw_box, (list, tuple)) and len(raw_box) == 4:
            try:
                x1, y1, x2, y2 = [min(1000, max(0, int(v))) for v in raw_box]
                if abs(x2 - x1) >= 10 and abs(y2 - y1) >= 10:
                    box = (min(x1, x2) / 1000.0, min(y1, y2) / 1000.0,
                           max(x1, x2) / 1000.0, max(y1, y2) / 1000.0)
            except (TypeError, ValueError):
                box = None
        out.append({"name": name, "type": "液体" if is_liquid else "食物",
                    "description": desc, "description_en": desc_en,
                    "calories_kcal": kcal, "protein_g": protein,
                    "carbs_g": carbs, "fat_g": fat, "classification": cls,
                    "match": match, "match_evidence": evidence, "match_reason": reason,
                    "matched_name": mname, "match_confidence": conf, "box": box})
    return out


def _recognize_dedup(orig_rgb, boxed_rgb, candidates, n_food=0, n_drink=0, target=None):
    """调多模态 VLM 识别 + 去重（一次多图请求；与产线同款）。"""
    cfg = target or RECOG_TARGETS[_recog_target]
    if not cfg["endpoint"]:
        return []
    u1, u2 = _img_data_uri(orig_rgb), _img_data_uri(boxed_rgb)
    if not u1 or not u2:
        return []
    content = [{"type": "image_url", "image_url": {"url": u1}},
               {"type": "image_url", "image_url": {"url": u2}}]
    for c in candidates:
        if c.get("ref_img"):
            content.append({"type": "image_url", "image_url": {"url": c["ref_img"]}})
    content.append({"type": "text", "text": _build_recog_prompt(candidates, n_food, n_drink)})
    payload = {"model": cfg["model"],
               "messages": [{"role": "user", "content": content}],
               "max_tokens": 1536, "temperature": 0}
    headers = {"Authorization": f"Bearer {cfg['api_key']}"} if cfg["api_key"] else {}
    try:
        r = _http_json(cfg["endpoint"], payload, timeout=RECOG_TIMEOUT, headers=headers)
        out = r["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[exp] 识别调用失败：{type(e).__name__}: {e}", flush=True)
        return []
    return _parse_recog(out)


def _name_tokens(s):
    """名称 → 词面 token 集（合并闸门五用，与产线同款）。"""
    s = (s or "").strip().lower()
    toks = set(re.findall(r"[a-z0-9]+", s))
    cjk = re.findall(r"[一-鿿]", s)
    toks |= {a + b for a, b in zip(cjk, cjk[1:])} or set(cjk)
    return toks


def _make_ref_img(rgb):
    """历史参考图打 HISTORY REF 横幅 + 灰边框（防参考图内容泄漏成当前画面物品）。"""
    out = rgb.copy()
    H, W = out.shape[:2]
    bh = max(28, H // 12)
    cv2.rectangle(out, (0, 0), (W, bh), (90, 90, 90), -1)
    cv2.putText(out, "HISTORY REF - NOT CURRENT FRAME", (10, int(bh * 0.72)),
                cv2.FONT_HERSHEY_SIMPLEX, bh / 44.0, (255, 255, 255),
                max(1, bh // 16), cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (W - 1, H - 1), (90, 90, 90), max(4, W // 100))
    return out


REF_CROP_MARGIN = 0.18
REF_CROP_MIN = 224


def _make_ref_crop(rgb, box):
    """新卡参考图：按物品 box 外扩裁剪特写 + HISTORY REF 横幅（治 over-merge，与产线同款）。"""
    if not box:
        return None
    H, W = rgb.shape[:2]
    x1, y1, x2, y2 = box[0] * W, box[1] * H, box[2] * W, box[3] * H
    bw, bh = x2 - x1, y2 - y1
    if bw * bh < 0.02 * W * H:
        return None
    mx, my = bw * REF_CROP_MARGIN, bh * REF_CROP_MARGIN
    x1, y1, x2, y2 = x1 - mx, y1 - my, x2 + mx, y2 + my
    if x2 - x1 < REF_CROP_MIN:
        cx = (x1 + x2) / 2
        x1, x2 = cx - REF_CROP_MIN / 2, cx + REF_CROP_MIN / 2
    if y2 - y1 < REF_CROP_MIN:
        cy = (y1 + y2) / 2
        y1, y2 = cy - REF_CROP_MIN / 2, cy + REF_CROP_MIN / 2
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(W, int(x2)), min(H, int(y2))
    if x2 - x1 < 32 or y2 - y1 < 32:
        return None
    return _make_ref_img(rgb[y1:y2, x1:x2])


def _draw_boxes(rgb, detections):
    """原图上画 food(红)/drink(蓝) 检测框（VLM 图2 用，与产线同款）。"""
    out = rgb.copy()
    H, W = out.shape[:2]
    for (label, nx1, ny1, nx2, ny2) in detections:
        color = _FOOD_COLOR if label == "food" else _DRINK_COLOR
        cv2.rectangle(out, (int(nx1 * W), int(ny1 * H)), (int(nx2 * W), int(ny2 * H)),
                      color, max(2, W // 300))
    return out


# ── 识别缩略图：实验版用「VLM box 裁自原图的 crop」存盘（产线是点云渲染图；本进程无 DA3） ──
SHOT_DIR = Path(__file__).resolve().parent / "exp_shots"
SHOT_KEEP = 160


def _save_crop_shot(orig_rgb, box):
    """按 VLM box 裁原图存缩略图，返回 /shotimg url；box 缺失时存整帧。"""
    H, W = orig_rgb.shape[:2]
    if box:
        x1, y1 = max(0, int(box[0] * W)), max(0, int(box[1] * H))
        x2, y2 = min(W, int(box[2] * W)), min(H, int(box[3] * H))
        crop = orig_rgb[y1:y2, x1:x2] if (x2 - x1 >= 24 and y2 - y1 >= 24) else orig_rgb
    else:
        crop = orig_rgb
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return None
    SHOT_DIR.mkdir(parents=True, exist_ok=True)
    name = uuid.uuid4().hex + ".jpg"
    (SHOT_DIR / name).write_bytes(buf.tobytes())
    files = sorted(SHOT_DIR.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in files[SHOT_KEEP:]:
        try:
            p.unlink()
        except OSError:
            pass
    return "/shotimg/" + name


def _recog_worker():
    """识别后台单线程：取最新任务丢弃积压，识别 + 五道合并闸门（与产线同款语义）。"""
    global _recog_id
    while True:
        with _recog_cv:
            while not _recog_pending:
                _recog_cv.wait()
            orig, boxed, frame, t, candidates, n_food, n_drink, enq_ts, dev = \
                _recog_pending.pop()
            _recog_pending.clear()
        wait_ms = (time.time() - enq_ts) * 1000.0
        boxed_uri = _img_data_uri(_make_ref_img(boxed))
        tgt = RECOG_TARGETS[_recog_target]
        _tq = time.time()
        items = _recognize_dedup(orig, boxed, candidates, n_food, n_drink, tgt)
        llm_ms = (time.time() - _tq) * 1000.0
        print("[exp] 识别一轮：排队%.0fms · %s %.0fms · 返回 %d 项（去重候选 %d 个：%s）" % (
            wait_ms, tgt["label"], llm_ms, len(items), len(candidates),
            "、".join(c["name"] for c in candidates) or "无"), flush=True)
        now = time.time()
        with _recog_lock:
            cards = _recog_cards.setdefault(dev, [])
            for it in items:
                target = None
                m = it.get("match")
                if isinstance(m, int) and 1 <= m <= len(candidates):
                    cand = candidates[m - 1]
                    if (it.get("matched_name") or "").strip().casefold() \
                            != cand["name"].strip().casefold():
                        print("[exp] 识别拒合并（回显不一致）：『%s』match=%s 回显『%s』≠候选『%s』" % (
                            it["name"], m, it.get("matched_name") or "空", cand["name"]), flush=True)
                    elif (cand.get("type") or "食物") != it["type"]:
                        print("[exp] 识别拒合并（类型不一致）：『%s』(%s) match=%s 候选『%s』(%s)" % (
                            it["name"], it["type"], m, cand["name"],
                            cand.get("type") or "食物"), flush=True)
                    elif "不一致" in (it.get("match_evidence") or ""):
                        print("[exp] 识别拒合并（证据矛盾）：『%s』match=%s 证据『%s』" % (
                            it["name"], m, it.get("match_evidence")), flush=True)
                    elif it.get("match_confidence") != "high":
                        print("[exp] 识别拒合并（低置信）：『%s』match=%s confidence=%s" % (
                            it["name"], m, it.get("match_confidence") or "缺省"), flush=True)
                    elif not (_name_tokens(it["name"]) & _name_tokens(cand["name"])):
                        print("[exp] 识别拒合并（名称零重叠）：『%s』→候选『%s』" % (
                            it["name"], cand["name"]), flush=True)
                    else:
                        target = next((c for c in cards if c["id"] == cand["id"]), None)
                if target is not None:
                    print("[exp] 识别去重：『%s』match=%s(high) → 合并到卡%s『%s』" % (
                        it["name"], m, target["id"], target["name"]), flush=True)
                else:
                    print("[exp] 识别新卡：『%s』(%s) match=%s｜理由：%s" % (
                        it["name"], it["type"], m,
                        it.get("match_reason") or "（模型未给）"), flush=True)
                shot_url = _save_crop_shot(orig, it.get("box"))
                if target is not None:
                    target.setdefault("merge_history", []).append({
                        "t": t, "name": it["name"],
                        "evidence": it.get("match_evidence", ""),
                        "reason": it.get("match_reason", ""),
                        "confidence": it.get("match_confidence", "")})
                    del target["merge_history"][:-20]
                    if shot_url:
                        target["shots"].append(shot_url)
                    del target["shots"][:-8]
                    if it.get("box"):
                        target["box"] = it["box"]
                    target["last_ts"] = now
                    target["t"] = t
                    target["frame"] = frame
                    target["latency_ms"] = int(llm_ms)
                    target["latency_model"] = tgt["label"]
                    target["rev"] = target.get("rev", 0) + 1
                    cards.remove(target)
                    cards.append(target)
                else:
                    _recog_id += 1
                    crop = _make_ref_crop(orig, it.get("box"))
                    ref_uri = (_img_data_uri(crop) if crop is not None else None) or boxed_uri
                    cards.append({
                        "id": _recog_id, "status": "done",
                        "name": it["name"], "type": it["type"], "description": it["description"],
                        "description_en": it.get("description_en", ""),
                        "calories_kcal": it.get("calories_kcal"),
                        "protein_g": it.get("protein_g"), "carbs_g": it.get("carbs_g"),
                        "fat_g": it.get("fat_g"), "classification": it.get("classification", ""),
                        "box": it.get("box"),
                        "shots": [shot_url] if shot_url else [], "ref_img": ref_uri,
                        "merge_history": [],
                        "latency_ms": int(llm_ms), "latency_model": tgt["label"],
                        "frame": frame, "t": t, "last_ts": now, "rev": 0})
            if len(cards) > RECOG_MAX_CARDS:
                del cards[:len(cards) - RECOG_MAX_CARDS]


def _maybe_recognize(orig_rgb, detections, frame, device, min_interval):
    """检测命中 + 节流通过 → 取活跃卡快照作去重候选、提交异步识别（与产线同款语义）。"""
    global _recog_last_ts, _recog_worker_started
    if not detections or not RECOG_TARGETS[_recog_target]["endpoint"]:
        return
    now = time.time()
    with _recog_lock:
        if now - _recog_last_ts < min_interval:
            return
        _recog_last_ts = now
        if not _recog_worker_started:
            _recog_worker_started = True
            threading.Thread(target=_recog_worker, daemon=True).start()
        active = [c for c in _recog_cards.get(device, [])
                  if c.get("status") == "done" and c.get("ref_img")
                  and now - c.get("last_ts", 0) <= RECOG_ACTIVE_WINDOW]
        active.sort(key=lambda c: c.get("last_ts", 0), reverse=True)
        candidates = [{"id": c["id"], "name": c.get("name", ""), "type": c.get("type", "食物"),
                       "desc": c.get("description", ""), "ref_img": c["ref_img"]}
                      for c in active[:RECOG_MAX_CANDIDATES]]
    t = time.strftime("%H:%M:%S")
    boxed = _draw_boxes(orig_rgb, detections)
    n_food = sum(1 for d in detections if d[0] == "food")
    n_drink = len(detections) - n_food
    with _recog_cv:
        _recog_pending.append((orig_rgb.copy(), boxed, frame, t, candidates,
                               n_food, n_drink, time.time(), device))
        _recog_cv.notify()


# ══════════════════════════════════════════════════════════════════════
# 实验检测主循环：采样帧 → 检测级（SAM3/LA/embed）→ 标注帧 + 逐轮记录 → 触发 VLM 识别
# ══════════════════════════════════════════════════════════════════════
def _draw_round(rgb, sam3_res, la_res, thresh, with_mask):
    """标注帧：SAM3 过阈值=红(食物)/蓝(液体)框+可选 mask 高亮，低于阈值=灰框；LA=同色系细框。"""
    out = rgb.copy()
    H, W = out.shape[:2]
    lw = max(2, W // 400)
    for (label, query, insts) in sam3_res:
        base = _FOOD_COLOR if label == "food" else _DRINK_COLOR
        for ins in insts:
            bx = ins.get("box_xywh_norm") or []
            if len(bx) != 4:
                continue
            x, y, w, h = bx
            p1, p2 = (int(x * W), int(y * H)), (int((x + w) * W), int((y + h) * H))
            kept = ins.get("score", 0) >= thresh
            c = base if kept else _CUT_COLOR
            cv2.rectangle(out, p1, p2, c, lw if kept else max(1, lw // 2))
            cv2.putText(out, "%s %.2f" % (query, ins.get("score", 0)),
                        (p1[0], max(14, p1[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2, cv2.LINE_AA)
            if kept and with_mask:
                rle = ins.get("mask_rle") or {}
                if rle.get("counts") and rle.get("size"):
                    try:
                        m = _rle_decode(rle["size"], rle["counts"])
                        if m.shape != (H, W):
                            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                        sel = m.astype(bool)
                        out[sel] = (out[sel] * 0.55 + np.array(base) * 0.45).astype(np.uint8)
                    except Exception:
                        pass
    for (label, query, boxes) in la_res:
        base = _FOOD_COLOR if label == "food" else _DRINK_COLOR
        for (x1, y1, x2, y2) in boxes:
            p1, p2 = (int(x1 * W), int(y1 * H)), (int(x2 * W), int(y2 * H))
            cv2.rectangle(out, p1, p2, base, max(1, lw // 2))
            cv2.putText(out, "LA:" + query, (p1[0], min(H - 6, p2[1] + 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, base, 1, cv2.LINE_AA)
    return out


def _run_round(rgb, src_seq, dev, cfg):
    """跑一轮实验链路：检测 → 标注/记录 → 命中触发 VLM 识别。"""
    rec = {"t": time.strftime("%H:%M:%S"), "src_seq": src_seq,
           "sam3": [], "la": [], "embed": [], "ms": {}}
    sam3_res, la_res, detections = [], [], []   # detections=[(food|drink,x1,y1,x2,y2)...] 0-1
    if cfg["sam3_on"]:
        t0 = time.time()
        groups = [("food", q) for q in _split_queries(cfg["sam3_food_queries"])] + \
                 [("drink", q) for q in _split_queries(cfg["sam3_drink_queries"])]
        for label, q in groups:   # SAM3 服务端有 GPU 锁，串行发送不排队
            insts = _sam3_segment(rgb, q, alpha=float(cfg["sam3_alpha"]),
                                  det_thresh=float(cfg["sam3_det_thresh"]))
            kept = [i for i in insts if i.get("score", 0) >= cfg["sam3_thresh"]]
            sam3_res.append((label, q, insts))
            for i in kept:
                bx = i.get("box_xywh_norm") or []
                if len(bx) == 4:
                    x, y, w, h = bx
                    detections.append((label, x, y, x + w, y + h))
            rec["sam3"].append({"label": label, "query": q, "total": len(insts),
                                "kept": len(kept),
                                "scores": [round(i.get("score", 0), 3) for i in insts]})
        rec["ms"]["sam3"] = round((time.time() - t0) * 1000)
    if cfg["la_on"]:
        t0 = time.time()
        groups = [("food", q) for q in _split_queries(cfg["la_food_queries"])] + \
                 [("drink", q) for q in _split_queries(cfg["la_drink_queries"])]
        with ThreadPoolExecutor(max_workers=max(1, len(groups))) as ex:   # LA 有 LB 可并发
            results = list(ex.map(lambda g: (g[0], g[1], _locate_one(rgb, g[1])), groups))
        la_res = results
        for label, q, boxes in results:
            for b in boxes:
                detections.append((label,) + tuple(b))
        rec["la"] = [{"label": l, "query": q, "boxes": len(bs)} for l, q, bs in results]
        rec["ms"]["la"] = round((time.time() - t0) * 1000)
    if cfg["embed_on"]:
        t0 = time.time()
        H, W = rgb.shape[:2]
        for (label, x1, y1, x2, y2) in detections:
            crop = rgb[max(0, int(y1 * H)):int(y2 * H), max(0, int(x1 * W)):int(x2 * W)]
            m = _embed_match(crop) if crop.size else None
            rec["embed"].append({"from": label, "match": m or "插槽未接入"})
        rec["ms"]["embed"] = round((time.time() - t0) * 1000)
    t0 = time.time()
    anno = _draw_round(rgb, sam3_res, la_res, cfg["sam3_thresh"], cfg["sam3_mask"])
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(anno, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 85])
    rec["ms"]["draw"] = round((time.time() - t0) * 1000)
    with _out_lock:
        if ok:
            _out["jpg"] = buf.tobytes()
            _out["seq"] += 1
            _out["src_seq"] = src_seq
        _rounds.append(rec)
        del _rounds[:-ROUNDS_KEEP]
    n_food = sum(1 for d in detections if d[0] == "food")
    n_drink = len(detections) - n_food
    print("[exp] 检测一轮：帧%d · 食物框%d 液体框%d · sam3 %dms la %dms" % (
        src_seq, n_food, n_drink, rec["ms"].get("sam3", 0), rec["ms"].get("la", 0)), flush=True)
    # 命中 → 触发实验 VLM 识别（异步、节流；与产线同款语义但完全独立一份）
    if cfg["recog_on"] and detections:
        _maybe_recognize(rgb, detections, "e%d" % src_seq, dev,
                         float(cfg["recog_interval"]))


def _worker():
    """检测后台主循环：按采样间隔消费 8060 新帧；总开关关 = 完全静默。"""
    last_seq, last_run = -1, 0.0
    while True:
        with _cfg_lock:
            cfg = dict(_cfg)
        if not cfg["enabled"]:
            time.sleep(0.3)
            continue
        if time.time() - last_run < max(0.2, float(cfg["interval"])):
            time.sleep(0.1)
            continue
        rgb, seq, dev = _fetch_frame()
        if rgb is None or seq == last_seq:
            time.sleep(0.3)
            continue
        last_seq, last_run = seq, time.time()
        try:
            _run_round(rgb, seq, dev, cfg)
        except Exception as e:
            print(f"[exp] 检测轮异常：{type(e).__name__}: {e}", flush=True)


threading.Thread(target=_worker, daemon=True).start()


# ══════════════════════════════════════════════════════════════════════
# API：实验配置/状态 + 识别卡片流（与 8060 /api/recog/* 同构，页面可整体复刻）
# ══════════════════════════════════════════════════════════════════════
@app.get("/api/exp/status")
def exp_status():
    with _cfg_lock:
        cfg = dict(_cfg)
    with _out_lock:
        seq, src_seq = _out["seq"], _out["src_seq"]
        rounds = list(_rounds[-20:])
    return JSONResponse({"config": cfg, "seq": seq, "src_seq": src_seq, "rounds": rounds})


@app.get("/api/exp/health")
def exp_health():
    """上游依赖健康：8060 帧源 / SAM3 / LA / VLM 配置（页面顶栏指示灯）。"""
    def probe(url):
        try:
            with urllib.request.urlopen(url, timeout=3.0) as r:
                return r.status < 500
        except Exception:
            return False
    return JSONResponse({
        "frame": probe(FRAME_SOURCE + "/api/frame/status"),
        "sam3": probe(SAM3_ENDPOINT + "/health"),
        "la": probe(LOCATE_ENDPOINT.rsplit("/v1/", 1)[0] + "/v1/models"),
        "vlm": bool(RECOG_TARGETS[_recog_target]["endpoint"]),
    })


@app.post("/api/exp/config")
def exp_config(body: dict = Body(default=None)):
    """更新实验配置（只认已知键，热生效）。"""
    body = body or {}
    with _cfg_lock:
        for k in _cfg:
            if k in body:
                if isinstance(_cfg[k], bool):
                    _cfg[k] = bool(body[k])
                elif isinstance(_cfg[k], float):
                    _cfg[k] = float(body[k])
                else:
                    _cfg[k] = body[k]
        cfg = dict(_cfg)
    print("[exp] 配置更新：" + json.dumps(cfg, ensure_ascii=False), flush=True)
    return JSONResponse({"ok": True, "config": cfg})


@app.get("/api/exp/latest")
def exp_latest():
    """最新实验标注帧 JPEG。"""
    with _out_lock:
        jpg = _out["jpg"]
    if not jpg:
        return JSONResponse({"error": "暂无标注帧（总开关未开或还没消费到帧）"}, status_code=404)
    return Response(jpg, media_type="image/jpeg")


def _current_device():
    """当前设备 = 8060 选中设备（卡片桶按它分）。"""
    try:
        st = _http_json(FRAME_SOURCE + "/api/frame/status", timeout=3.0)
        return str(st.get("device") or UNKNOWN_DEVICE)
    except Exception:
        return UNKNOWN_DEVICE


@app.get("/api/recog/list")
def recog_list(device: str = None):
    """实验识别卡片列表（与 8060 同构：/recog 复刻页直接复用）。"""
    dev = (device or "").strip() or _current_device()
    with _recog_lock:
        cards = [{k: v for k, v in c.items() if k != "ref_img"}
                 for c in _recog_cards.get(dev, []) if c.get("status") != "empty"]
    cards.reverse()
    tgt = RECOG_TARGETS[_recog_target]
    return JSONResponse({"enabled": bool(tgt["endpoint"]), "device": dev, "cards": cards,
                         "target": _recog_target, "model": tgt["model"],
                         "targets": {k: bool(v["endpoint"]) for k, v in RECOG_TARGETS.items()}})


@app.post("/api/recog/target")
def recog_target_set(body: dict = Body(default=None)):
    """切换识别目标（qwen|gemini）：免重启，下一轮识别即用新目标。"""
    global _recog_target
    want = str((body or {}).get("target", "")).strip()
    if want not in RECOG_TARGETS:
        return JSONResponse({"error": "target 只支持 " + "|".join(RECOG_TARGETS)}, status_code=400)
    if not RECOG_TARGETS[want]["endpoint"]:
        return JSONResponse({"error": f"目标 {want} 未配置 endpoint"}, status_code=400)
    _recog_target = want
    print(f"[exp] 识别目标切换 → {want}（{RECOG_TARGETS[want]['model']}）", flush=True)
    return JSONResponse({"ok": True, "target": want, "model": RECOG_TARGETS[want]["model"]})


@app.post("/api/recog/clear")
def recog_clear(device: str = None):
    """清空当前设备的实验识别卡片。"""
    dev = (device or "").strip() or _current_device()
    with _recog_lock:
        _recog_cards.pop(dev, None)
    return JSONResponse({"ok": True, "device": dev})


@app.get("/shotimg/{name}")
def shot_img(name: str):
    """识别缩略图（VLM box 裁自原图的 JPEG）。"""
    if not re.fullmatch(r"[0-9a-f]{32}\.jpg", name):
        return JSONResponse({"error": "非法文件名"}, status_code=400)
    p = SHOT_DIR / name
    if not p.exists():
        return JSONResponse({"error": "不存在"}, status_code=404)
    return FileResponse(str(p), media_type="image/jpeg")


# ══════════════════════════════════════════════════════════════════════
# 首页（复刻 8060 主页形态）：左=设备帧+实验标注帧+点云产物(经 8060)，右=iframe /recog，
# 右上「调节」抽屉热改实验链路参数 + 逐轮结果流水。
# ══════════════════════════════════════════════════════════════════════
EXP_PAGE = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>感知链路实验台 · 8061</title>
<script type="module" src="https://unpkg.com/@google/model-viewer@3.5.0/dist/model-viewer.min.js"></script>
<style>
 *{box-sizing:border-box}
 html,body{margin:0;height:100%;font-family:system-ui,-apple-system,'Segoe UI',sans-serif;background:#0d0d0f;color:#e5e5ea}
 .top{height:48px;display:flex;align-items:center;gap:14px;padding:0 18px;background:#1c1c1e;font-size:14px}
 .top b{font-size:15px}
 .top .tag{font-size:12px;color:#9a9aa0}
 .dot{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:5px;background:#8e8e93}
 .dot.ok{background:#34c759}.dot.bad{background:#ff453a}
 .hlth{margin-left:auto;font-size:12px;color:#9a9aa0;display:flex;gap:12px;align-items:center}
 .top button{background:#2c2c2e;border:1px solid #3a3a3c;color:#e5e5ea;font-size:12px;padding:6px 16px;border-radius:99px;cursor:pointer}
 .top button:hover{background:#3a3a3c}
 .wrap{display:flex;height:calc(100% - 48px)}
 .pane{min-width:0;display:flex;flex-direction:column}
 .pane.l{flex:1 1 54%;border-right:1px solid #2c2c2e}
 .pane.r{flex:1 1 46%}
 .pane.r iframe{flex:1;width:100%;border:0;background:#fff}
 .bar{height:34px;flex:none;display:flex;align-items:center;gap:8px;padding:0 14px;background:#141416;font-size:13px;font-weight:600;border-bottom:1px solid #2c2c2e}
 .bar .dim{font-weight:400;color:#9a9aa0;font-size:12px}
 .grid{flex:1;display:grid;grid-template-columns:1fr 1fr;grid-template-rows:auto 1fr;min-height:0}
 .cell{display:flex;flex-direction:column;min-width:0;min-height:0;border-bottom:1px solid #2c2c2e}
 .cell.c1{border-right:1px solid #2c2c2e}
 .cell.c3{grid-column:1/3;border-bottom:0}
 .imgbox{flex:1;display:flex;align-items:center;justify-content:center;background:#050505;overflow:hidden;min-height:120px}
 .imgbox img{max-width:100%;max-height:100%;display:none}
 .imgbox .wait{font-size:12px;color:#6e6e73;padding:14px;text-align:center}
 #mv{width:100%;height:100%;display:none;--poster-color:#050505;background:#050505}
 /* 调节抽屉（右滑出，物理像素） */
 #cfg{position:fixed;top:0;right:-380px;bottom:0;width:360px;z-index:20;overflow-y:auto;
   background:rgba(18,18,20,.97);border-left:1px solid #2c2c2e;transition:right .3s ease;
   font-size:13px;color:#c7c7cc;padding:14px 16px 20px}
 #cfg.on{right:0}
 #cfg .hd{display:flex;align-items:center;justify-content:space-between;font-size:14px;font-weight:600;color:#e5e5ea;margin-bottom:6px}
 #cfg .hd button{background:none;border:0;color:#8e8e93;font-size:16px;cursor:pointer}
 #cfg h3{font-size:13px;margin:14px 0 6px;color:#e5e5ea;border-top:1px solid #2c2c2e;padding-top:12px}
 #cfg h3:first-of-type{border-top:0;padding-top:0}
 .fld{margin:8px 0;display:flex;align-items:center;gap:8px;flex-wrap:wrap}
 .fld label{min-width:96px;color:#9a9aa0}
 .fld input[type=text]{flex:1;min-width:140px;background:#1c1c1e;border:1px solid #3a3a3c;color:#e5e5ea;border-radius:6px;padding:5px 8px;font-size:13px}
 .fld input[type=number]{width:70px;background:#1c1c1e;border:1px solid #3a3a3c;color:#e5e5ea;border-radius:6px;padding:5px 8px;font-size:13px}
 .fld input[type=range]{flex:1;min-width:110px}
 .fld b{color:#6ab7ff;font-variant-numeric:tabular-nums}
 .sw{position:relative;width:40px;height:22px;flex:none}
 .sw input{opacity:0;width:0;height:0}
 .sw i{position:absolute;inset:0;border-radius:99px;background:#3a3a3c;transition:.2s;cursor:pointer}
 .sw i:before{content:'';position:absolute;left:2px;top:2px;width:18px;height:18px;border-radius:50%;background:#fff;transition:.2s}
 .sw input:checked+i{background:#34c759}
 .sw input:checked+i:before{transform:translateX(18px)}
 .hint{font-size:11.5px;color:#6e6e73;line-height:1.5;margin:4px 0 0}
 .rd{font-size:11.5px;font-family:ui-monospace,Menlo,monospace;color:#c7c7cc;padding:5px 6px;border-bottom:1px solid #232325;line-height:1.5}
 .rd .t{color:#6e6e73}.rd .f{color:#ff6961}.rd .d{color:#6ab7ff}.rd .cut{color:#8e8e93}
</style></head><body>
<div class="top"><b>感知链路实验台</b>
 <span class="tag">8061 · 与产线 8060 / experience 完全隔离 · 帧/点云取自 8060 只读接口 · 检测/VLM 独立可魔改</span>
 <span class="hlth"><span><span class="dot" id="h_frame"></span>帧源</span>
  <span><span class="dot" id="h_sam3"></span>SAM3</span>
  <span><span class="dot" id="h_la"></span>LA</span>
  <span><span class="dot" id="h_vlm"></span>VLM</span></span>
 <button id="btnCfg">调节</button></div>
<div class="wrap">
 <div class="pane l">
  <div class="grid">
   <div class="cell c1"><div class="bar"><span class="dot" style="background:#0a84ff"></span>设备原帧
     <select id="selDev" style="display:none;background:#1c1c1e;border:1px solid #3a3a3c;color:#e5e5ea;border-radius:6px;padding:2px 6px;font-size:12px" title="选择设备（与 8060 同一选中设备）"></select>
     <span class="dim" id="devlab">–</span></div>
    <div class="imgbox"><img id="raw"><span class="wait" id="raww">等待设备帧…</span></div></div>
   <div class="cell c2"><div class="bar"><span class="dot" style="background:#34c759"></span>实验检测标注帧 <span class="dim" id="seqinfo">总开关未开</span></div>
    <div class="imgbox"><img id="anno"><span class="wait" id="annow">打开「调节」里的总开关开始实验</span></div></div>
   <div class="cell c3"><div class="bar"><span class="dot" style="background:#bf5af2"></span>点云产物 <span class="dim">共享 8060 产线 DA3 产物（框为产线检测，仅参考）· 可拖转视角</span></div>
    <div class="imgbox"><model-viewer id="mv" camera-controls touch-action="pan-y"
      interaction-prompt="none" shadow-intensity="0.3" exposure="1.35"
      min-camera-orbit="-Infinity 0deg 1%" max-camera-orbit="Infinity 180deg 2000%"></model-viewer>
     <img id="prod"><span class="wait" id="prodw">等待 8060 产物…</span></div></div>
  </div>
 </div>
 <div class="pane r">
  <div class="bar"><span class="dot" style="background:#34c759"></span>实验识别 · 检测命中 → VLM（独立卡片流，不影响产线）</div>
  <iframe src="/recog" title="实验识别卡片流"></iframe>
 </div>
</div>

<div id="cfg">
 <div class="hd">实验链路调节 <button id="cfgClose">✕</button></div>
 <h3>总控</h3>
 <div class="fld"><label>总开关</label><span class="sw"><input type="checkbox" id="c_enabled"><i></i></span>
  <span class="hint">关 = 不拉帧不推理，GPU 零增量</span></div>
 <div class="fld"><label>采样间隔</label><input type="number" id="c_interval" min="0.2" step="0.1"> 秒</div>
 <h3>SAM3 检测（无状态 /v1/segment，不碰产线流式）</h3>
 <div class="fld"><label>启用</label><span class="sw"><input type="checkbox" id="c_sam3_on"><i></i></span></div>
 <div class="fld"><label>食物查询词</label><input type="text" id="c_sam3_food_queries" placeholder="food"></div>
 <div class="fld"><label>液体查询词</label><input type="text" id="c_sam3_drink_queries" placeholder="drink; cup"></div>
 <div class="fld"><label>presence α</label><input type="range" id="c_sam3_alpha" min="0" max="1" step="0.05"><b id="v_alpha">1.00</b></div>
 <div class="fld"><label>检测阈值</label><input type="number" id="c_sam3_det_thresh" min="0" max="0.95" step="0.05"></div>
 <div class="hint">server 端口径（与 8060 /sam3tune 同款、随请求生效不碰产线）：keep = presence^α × cond > 检测阈值；
  α=1 且检测阈值=0 ⇒ 完全等价模型默认（0.5 联合分口径）。α 调低 = 弱化“概念是否在图中”的 presence 门控。</div>
 <div class="fld"><label>score 阈值</label><input type="range" id="c_sam3_thresh" min="0" max="1" step="0.05"><b id="v_thresh">0.50</b></div>
 <div class="fld"><label>mask 高亮</label><span class="sw"><input type="checkbox" id="c_sam3_mask"><i></i></span>
  <span class="hint">score 阈值是返回结果上的客户端二次过滤：低于阈值画灰框，直观看阈值卡掉了什么（设 0 = 全信 server 口径）</span></div>
 <h3>LocateAnything（对照，默认关）</h3>
 <div class="fld"><label>启用</label><span class="sw"><input type="checkbox" id="c_la_on"><i></i></span></div>
 <div class="fld"><label>食物查询词</label><input type="text" id="c_la_food_queries"></div>
 <div class="fld"><label>液体查询词</label><input type="text" id="c_la_drink_queries"></div>
 <h3>VLM 识别（实验独立一份，卡片见右栏）</h3>
 <div class="fld"><label>命中即识别</label><span class="sw"><input type="checkbox" id="c_recog_on"><i></i></span></div>
 <div class="fld"><label>识别节流</label><input type="number" id="c_recog_interval" min="1" step="0.5"> 秒/轮</div>
 <h3>embedding 匹配（插槽）</h3>
 <div class="fld"><label>启用</label><span class="sw"><input type="checkbox" id="c_embed_on"><i></i></span>
  <span class="hint">接入小模型前无输出；实现 exp_app.py 的 _embed_match() 即接通</span></div>
 <h3>逐轮检测结果（最新在上）</h3>
 <div id="rounds"></div>
</div>
<script>
const $=id=>document.getElementById(id);
const B8060=location.protocol+'//'+location.hostname+':8060';   // 8060 CORS 全放开，跨端口直取
const BOOLS=['enabled','sam3_on','sam3_mask','la_on','embed_on','recog_on'];
const TEXTS=['sam3_food_queries','sam3_drink_queries','la_food_queries','la_drink_queries'];
let inited=false,lastAnno=-1,lastRawSeq=-1,lastMvUrl='',lastProdSeq=-1,pushTimer=null;

function push(){
  clearTimeout(pushTimer);
  pushTimer=setTimeout(()=>{
    const body={};
    BOOLS.forEach(k=>body[k]=$('c_'+k).checked);
    TEXTS.forEach(k=>body[k]=$('c_'+k).value);
    body.interval=+$('c_interval').value||1.0;
    body.recog_interval=+$('c_recog_interval').value||4.0;
    body.sam3_thresh=+$('c_sam3_thresh').value;
    body.sam3_alpha=+$('c_sam3_alpha').value;
    body.sam3_det_thresh=+$('c_sam3_det_thresh').value||0;
    fetch('/api/exp/config',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify(body)}).catch(()=>{});
  },200);
}
BOOLS.concat(TEXTS).forEach(k=>$('c_'+k).addEventListener('change',push));
['c_interval','c_recog_interval','c_sam3_det_thresh'].forEach(id=>$(id).addEventListener('change',push));
$('c_sam3_thresh').addEventListener('input',()=>{$('v_thresh').textContent=(+$('c_sam3_thresh').value).toFixed(2);push();});
$('c_sam3_alpha').addEventListener('input',()=>{$('v_alpha').textContent=(+$('c_sam3_alpha').value).toFixed(2);push();});
$('btnCfg').onclick=()=>$('cfg').classList.toggle('on');
$('cfgClose').onclick=()=>$('cfg').classList.remove('on');

function fillCfg(c){   // 只首次回填，之后以页面为准（避免打字被轮询覆盖）
  if(inited)return;inited=true;
  BOOLS.forEach(k=>$('c_'+k).checked=!!c[k]);
  TEXTS.forEach(k=>$('c_'+k).value=c[k]||'');
  $('c_interval').value=c.interval;$('c_recog_interval').value=c.recog_interval;
  $('c_sam3_thresh').value=c.sam3_thresh;$('v_thresh').textContent=(+c.sam3_thresh).toFixed(2);
  $('c_sam3_alpha').value=c.sam3_alpha;$('v_alpha').textContent=(+c.sam3_alpha).toFixed(2);
  $('c_sam3_det_thresh').value=c.sam3_det_thresh;
}
function rdLine(r){
  let s='<span class="t">'+r.t+' 帧'+r.src_seq+'</span>';
  (r.sam3||[]).forEach(x=>{s+=' <span class="'+(x.label==='food'?'f':'d')+'">'+x.query+' '+x.kept+'/'+x.total+'</span>'
    +(x.scores&&x.scores.length?' <span class="cut">['+x.scores.join(',')+']</span>':'');});
  (r.la||[]).forEach(x=>{s+=' <span class="'+(x.label==='food'?'f':'d')+'">LA:'+x.query+'×'+x.boxes+'</span>';});
  const ms=r.ms||{};s+=' <span class="t">'+Object.entries(ms).map(([k,v])=>k+v+'ms').join(' ')+'</span>';
  return '<div class="rd">'+s+'</div>';
}
async function tickExp(){
  try{
    const s=await(await fetch('/api/exp/status',{cache:'no-store'})).json();
    fillCfg(s.config);
    if(s.seq){$('seqinfo').textContent='第 '+s.seq+' 轮 · 帧 '+s.src_seq;
      if(s.seq!==lastAnno){lastAnno=s.seq;
        $('anno').src='/api/exp/latest?t='+s.seq;
        $('anno').style.display='block';$('annow').style.display='none';}}
    else $('seqinfo').textContent=s.config.enabled?'等命中…':'总开关未开';
    $('rounds').innerHTML=(s.rounds||[]).slice().reverse().map(rdLine).join('');
  }catch(e){}
}
// 设备下拉：与 8060 /panel /experience 同一套「选中设备」（改这里=全局切换，8060 也跟着切）
let lastDevKey='';
function renderDevices(s){
  const devs=s.devices||[],sel=$('selDev');
  sel.style.display=devs.length?'':'none';
  if(document.activeElement!==sel){   // 下拉展开操作中不重建选项
    const key=devs.map(d=>d.device_id).join('|')+'#'+(s.selected||'');
    if(key!==lastDevKey){lastDevKey=key;
      sel.innerHTML=devs.map(d=>'<option value="'+d.device_id+'"'
        +(d.device_id===s.selected?' selected':'')+'>'+d.device_id+'</option>').join('');}
  }
}
$('selDev').onchange=()=>{
  fetch(B8060+'/api/frame/select',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({device_id:$('selDev').value})}).catch(()=>{});
};
async function tick8060(){   // 原帧 + 点云产物都来自 8060（只读；设备切换除外）
  try{
    const s=await(await fetch(B8060+'/api/frame/status',{cache:'no-store'})).json();
    renderDevices(s);
    $('devlab').textContent=s.device||'–';
    if(s.has_frame&&s.seq!==lastRawSeq){lastRawSeq=s.seq;
      $('raw').src=B8060+'/api/frame/latest?t='+s.seq;
      $('raw').style.display='block';$('raww').style.display='none';}
    if(s.product_kind==='model'&&s.product_url){
      const url=B8060+s.product_url;
      if(url!==lastMvUrl){lastMvUrl=url;
        const mv=$('mv');mv.src=url;mv.style.display='block';
        $('prod').style.display='none';$('prodw').style.display='none';}
    }else if(s.product_kind==='image'&&s.product_seq!==lastProdSeq){lastProdSeq=s.product_seq;
      $('prod').src=B8060+'/api/frame/latest-product?t='+s.product_seq;
      $('prod').style.display='block';$('mv').style.display='none';$('prodw').style.display='none';}
  }catch(e){}
}
async function hlth(){
  try{
    const h=await(await fetch('/api/exp/health',{cache:'no-store'})).json();
    [['h_frame',h.frame],['h_sam3',h.sam3],['h_la',h.la],['h_vlm',h.vlm]].forEach(([id,ok])=>{
      $(id).className='dot '+(ok?'ok':'bad');});
  }catch(e){}
}
setInterval(tickExp,800);tickExp();
setInterval(tick8060,600);tick8060();
setInterval(hlth,5000);hlth();
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def home():
    return EXP_PAGE


# ══ 实验识别卡片流页：整体复刻 8060 /recog（UI/交互逐位一致），仅改标题/导航/副标题；
#    /api/recog/* 与 8060 同构、由本进程提供（实验独立卡片桶）。 ══
EXP_RECOG_PAGE = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>实验识别 · 8061</title>
<style>
 :root{--bg:#f4f5f7;--panel:#fff;--panel2:#f6f7f9;--ink:#1b1e24;--muted:#69707b;--faint:#98a0ac;
   --line:#e5e8ec;--accent:#0071e3;--accent-soft:#e7f1fd;--food:#de3434;--food-soft:#fdeaea;
   --liquid:#2e78eb;--liquid-soft:#e9f0fd;--mono:ui-monospace,"SF Mono",Menlo,monospace;
   --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,"PingFang SC",sans-serif;}
 *{box-sizing:border-box}
 html,body{height:100%}
 body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.5;
   -webkit-font-smoothing:antialiased;display:flex;flex-direction:column}
 .nav{display:flex;gap:16px;align-items:center;padding:10px 16px;background:var(--panel);
   border-bottom:1px solid var(--line);font-size:13px}
 .nav a{color:var(--muted);text-decoration:none}.nav a.active{color:var(--accent);font-weight:600}
 .nav a.home{margin-left:auto;color:var(--faint)}
 .head{padding:14px 18px 12px;background:var(--panel);border-bottom:1px solid var(--line)}
 .head .l1{display:flex;align-items:center;gap:9px}
 .head h2{margin:0;font-size:16px;font-weight:650}
 .live{display:inline-flex;align-items:center;gap:6px;font-size:11px;font-weight:600;color:#1a9e5f;
   background:rgba(26,158,95,.12);padding:3px 9px;border-radius:999px;margin-left:auto}
 .live.off{color:var(--faint);background:rgba(120,130,145,.12)}
 .clr{margin-left:10px;font-size:12px;font-weight:600;color:var(--muted);background:var(--panel2);border:1px solid var(--line);padding:5px 13px;border-radius:8px;cursor:pointer}
 .clr:hover{color:var(--food);border-color:var(--food)}
 /* 识别目标切换（Qwen / Gemini Pro）：分段按钮，未配置 endpoint 的目标置灰 */
 .seg{display:inline-flex;border:1px solid var(--line);border-radius:9px;overflow:hidden;background:var(--panel2)}
 .seg button{font-size:12px;font-weight:600;padding:5px 13px;border:0;background:transparent;color:var(--muted);cursor:pointer}
 .seg button.sel{background:var(--accent);color:#fff}
 .seg button:disabled{opacity:.4;cursor:not-allowed}
 .live i{width:7px;height:7px;border-radius:50%;background:currentColor;animation:pulse 1.4s infinite}
 @keyframes pulse{0%,100%{opacity:1}50%{opacity:.25}}
 .head .sub{font-size:12px;color:var(--muted);margin-top:3px}
 .head .sub code{font-family:var(--mono);font-size:11px;color:var(--accent);background:var(--accent-soft);padding:1px 6px;border-radius:5px}
 .feed{flex:1;overflow-y:auto;padding:14px;display:flex;flex-direction:column;gap:12px}
 .empty{color:var(--faint);font-size:13px;text-align:center;margin:36px 10px}
 .rcard{position:relative;display:grid;grid-template-columns:118px 1fr auto;gap:15px;padding:12px;border:1px solid var(--line);
   border-radius:13px;background:var(--panel2);animation:rise .34s cubic-bezier(.2,.7,.3,1)}
 /* 卡片右侧：最近一次识别该食物的 VLM 延时 + 所用模型 */
 .rlat{display:flex;flex-direction:column;align-items:flex-end;justify-content:center;gap:3px;min-width:72px}
 .rlat .lv{font-size:15px;font-weight:650;font-family:var(--mono);font-variant-numeric:tabular-nums}
 .rlat .lm{font-size:10.5px;color:var(--faint)}
 @keyframes rise{from{opacity:0;transform:translateY(-10px)}to{opacity:1;transform:none}}
 .thumb{position:relative;width:104px;aspect-ratio:3/4;border-radius:9px;overflow:hidden;border:1px solid var(--line);
   background-color:#10141a;
   background-image:radial-gradient(58% 42% at 50% 60%,rgba(196,204,218,.5),rgba(120,130,150,.14) 55%,transparent 74%),
     radial-gradient(rgba(205,214,228,.5) .6px,transparent .8px);background-size:auto,4px 4px}
 .thumb img{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;opacity:0;transition:opacity .3s}
 .thumb .tag{position:absolute;left:6px;bottom:6px;font-size:9px;font-family:var(--mono);color:#fff;
   background:rgba(10,14,20,.62);padding:1px 6px;border-radius:4px;z-index:2}
 .rbody{min-width:0;display:flex;flex-direction:column;gap:11px}
 .fld{display:flex;flex-direction:column;gap:3px;min-width:0}
 .flab{font-size:10px;letter-spacing:.3px;color:var(--faint)}
 .name{font-size:16px;font-weight:650;line-height:1.25;display:flex;align-items:center;gap:7px;min-width:0}
 .name.pending .nm{color:var(--faint)}
 .tdot{width:8px;height:8px;border-radius:50%;background:var(--faint);flex:0 0 auto}
 .tdot.food{background:var(--food)}.tdot.liquid{background:var(--liquid)}
 .desc{font-size:13px;color:var(--muted);line-height:1.5}
 .tags{display:flex;flex-wrap:wrap;gap:6px}
 .chip{font-size:11.5px;font-weight:600;color:var(--accent);background:var(--accent-soft);padding:2px 9px;border-radius:999px}
 .sig{font-size:14px;font-weight:650;color:var(--ink)}
 .hide{display:none}
 .meta{margin-top:2px;font-size:10.5px;color:var(--faint);font-family:var(--mono);font-variant-numeric:tabular-nums;display:flex;gap:12px}
 .caret{display:inline-block;width:2px;height:1em;background:var(--accent);vertical-align:-1px;margin-left:1px;animation:blink 1s step-end infinite}
 @keyframes blink{50%{opacity:0}}
 /* 扇形叠卡缩略图（一食物 30 秒内去重合并的多帧点云截图） */
 .stack{position:relative;width:118px;height:150px;cursor:pointer;flex:0 0 auto}
 .stack .sh{position:absolute;left:9px;top:3px;width:100px;height:133px;border-radius:9px;border:2px solid var(--panel);
   background-color:#10141a;overflow:hidden;box-shadow:0 2px 8px rgba(15,22,34,.22);transition:transform .28s cubic-bezier(.2,.7,.3,1);
   background-image:radial-gradient(58% 42% at 50% 60%,rgba(196,204,218,.4),transparent 72%),radial-gradient(rgba(205,214,228,.4) .6px,transparent .8px);background-size:auto,4px 4px}
 .stack .sh img{width:100%;height:100%;object-fit:cover;display:block;opacity:0;transition:opacity .3s}
 .stack .cnt{position:absolute;right:-2px;bottom:-2px;z-index:30;font-size:11px;font-weight:700;font-family:var(--mono);color:#fff;background:var(--accent);padding:2px 8px;border-radius:999px;box-shadow:0 1px 4px rgba(0,0,0,.3)}
 .stack .shint{position:absolute;left:0;right:0;bottom:-16px;text-align:center;font-size:9.5px;color:var(--faint)}
 .stack:hover .sh{box-shadow:0 4px 14px rgba(15,22,34,.3)}
 /* 去重命中：卡片闪 + 右上角“更新”提示 */
 .updated{position:absolute;top:10px;right:10px;z-index:6;font-size:11px;font-weight:700;color:#fff;background:var(--accent);padding:3px 10px;border-radius:999px;opacity:0;transform:translateY(-4px);transition:opacity .25s,transform .25s;pointer-events:none}
 .updated.on{opacity:1;transform:none}
 .flash{animation:cardflash .85s ease}
 @keyframes cardflash{0%{box-shadow:0 0 0 0 var(--accent-soft)}35%{box-shadow:0 0 0 3px var(--accent)}100%{box-shadow:0 0 0 0 transparent}}
 /* lightbox 看图集 */
 .lb{position:fixed;inset:0;background:rgba(10,14,20,.82);display:none;flex-direction:column;align-items:center;justify-content:center;gap:13px;z-index:100;padding:24px}
 .lb.on{display:flex}
 .lb .big{width:min(88vw,600px);max-height:74vh;aspect-ratio:3/4;border-radius:14px;overflow:hidden;border:2px solid rgba(255,255,255,.15);background:#10141a}
 .lb .big img{width:100%;height:100%;object-fit:contain;display:block;cursor:zoom-in;transform-origin:center center;will-change:transform;transition:transform .12s ease-out}
 .lb .cap{color:#dfe4ea;font-size:13px;font-family:var(--mono)}
 .lb .strip{display:flex;gap:8px;flex-wrap:wrap;justify-content:center;max-width:min(82vw,420px)}
 .lb .th{width:50px;height:66px;border-radius:7px;overflow:hidden;border:2px solid transparent;cursor:pointer;opacity:.55;background:#10141a}
 .lb .th img{width:100%;height:100%;object-fit:cover;display:block}
 .lb .th.sel{border-color:var(--accent);opacity:1}
 .lb .x{position:absolute;top:16px;right:20px;color:#fff;font-size:26px;cursor:pointer;opacity:.85;line-height:1}
 /* 智能跳顶：下翻时不打断，顶部浮「↑ 有新卡」按钮 */
 .newpill{position:fixed;top:104px;left:50%;transform:translateX(-50%) translateY(-8px);z-index:60;
   font-size:12px;font-weight:700;color:#fff;background:var(--accent);padding:6px 16px;border-radius:999px;
   box-shadow:0 4px 14px rgba(0,0,0,.28);cursor:pointer;opacity:0;pointer-events:none;transition:opacity .25s,transform .25s}
 .newpill.on{opacity:1;transform:translateX(-50%);pointer-events:auto}
 @media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style></head><body>
<div class="nav"><a class="active" href="/recog">实验识别</a><a class="home" href="/" target="_top">↗ 实验台首页</a></div>
<div class="head">
  <div class="l1"><h2>实验识别 · Experimental Recognition</h2>
    <span class="seg" id="seg"><button data-t="qwen">Qwen</button><button data-t="gemini">Gemini Pro</button></span>
    <span class="live" id="live"><i></i>识别中</span><button class="clr" id="clr">清空</button></div>
  <div class="sub">实验检测命中 → <code id="model">Qwen3-VL</code> 识别（名称/描述/卡路里/宏量/分级）· 同一物 30 秒内去重合并 · 缩略图=VLM box 裁剪 · 当前设备 <code id="devlab">–</code></div>
</div>
<div class="feed" id="feed"><div class="empty" id="empty">等待 food/drink 命中…</div></div>
<div class="newpill" id="newpill">↑ 有新卡</div>
<div class="lb" id="lb"><span class="x" id="lbx">✕</span>
 <div class="big"><img id="lbBig" alt=""></div>
 <div class="cap" id="lbCap"></div>
 <div class="strip" id="lbStrip"></div>
</div>
<script>
const $=id=>document.getElementById(id);
const feed=$('feed');
const state=new Map();   // id -> {el, rev, shots:[img_url...]}
const typeCls=t=> t==='液体' ? 'liquid' : 'food';

// —— 打字机 ——
function typeInto(node,text){ let i=0; node.textContent='';
  const caret=document.createElement('span'); caret.className='caret'; node.appendChild(caret);
  const timer=setInterval(()=>{ i++; caret.remove(); node.textContent=text.slice(0,i); node.appendChild(caret);
    if(i>=text.length){clearInterval(timer); caret.remove();}}, 46);
}

// —— 扇形叠卡：每张 = Qwen box 裁剪图（/shotimg/*），扇形叠，>5 张右下角显示数字，点击开 lightbox ——
function renderStack(el, name, shots){
  const stack=el.querySelector('.stack');
  stack.querySelectorAll('.sh,.cnt,.shint').forEach(n=>n.remove());
  const show=shots.slice(-5);                 // 最多露 5 张边角
  show.forEach((url,i)=>{
    const isTop=i===show.length-1, off=show.length-1-i, dir=(off%2?-1:1);
    const d=document.createElement('div'); d.className='sh';
    d.style.transform='rotate('+(isTop?0:dir*(4+off*3))+'deg) translate('+(isTop?0:dir*(5+off*3))+'px,'+(off*2)+'px)';
    d.style.zIndex=String(i+1);
    const img=document.createElement('img'); d.appendChild(img);
    img.onload=()=>{img.style.opacity=1;}; img.src=url;
    stack.appendChild(d);
  });
  if(shots.length>5){ const b=document.createElement('span'); b.className='cnt'; b.textContent='×'+shots.length; stack.appendChild(b); }
  const h=document.createElement('span'); h.className='shint'; h.textContent='点击看 '+shots.length+' 张'; stack.appendChild(h);
  stack.onclick=()=>openLB(name, shots);
}

function cardEl(c){
  const el=document.createElement('div'); el.className='rcard';
  el.innerHTML='<div class="updated">更新</div><div class="stack"></div>'
    +'<div class="rbody">'
    +'  <div class="fld"><div class="flab">识别对象 / Detected Food</div>'
    +'    <div class="name"><span class="tdot '+typeCls(c.type)+'"></span><span class="nm"></span></div></div>'
    +'  <div class="fld f-desc hide"><div class="flab">一句话描述 / Description</div><div class="desc"></div></div>'
    +'  <div class="fld f-nutr hide"><div class="flab">卡路里与营养（按可见份量估算）/ Nutrition (est.)</div><div class="tags nutr"></div></div>'
    +'  <div class="fld f-cls hide"><div class="flab">健康分级 / Classification</div><div class="sig cls"></div></div>'
    +'  <div class="meta"><span>帧 '+(c.frame||'')+'</span><span>'+(c.t||'')+'</span></div>'
    +'</div>'
    +'<div class="rlat"><div class="flab">识别延时 / Latency</div><div class="lv"></div><div class="lm"></div></div>';
  latFill(el, c);
  typeInto(el.querySelector('.nm'), c.name||'');
  if(c.description){ el.querySelector('.f-desc').classList.remove('hide'); el.querySelector('.desc').textContent=c.description; }
  const nutr=[];   // 老卡（改版前识别的）没有这些字段 → 整段隐藏，不炸
  if(c.calories_kcal!=null)nutr.push(c.calories_kcal+' kcal');
  if(c.protein_g!=null)nutr.push('蛋白 '+c.protein_g+'g');
  if(c.carbs_g!=null)nutr.push('碳水 '+c.carbs_g+'g');
  if(c.fat_g!=null)nutr.push('脂肪 '+c.fat_g+'g');
  if(nutr.length){ el.querySelector('.f-nutr').classList.remove('hide');
    el.querySelector('.nutr').innerHTML=nutr.map(t=>'<span class="chip">'+t+'</span>').join(''); }
  if(c.classification){ el.querySelector('.f-cls').classList.remove('hide'); el.querySelector('.cls').textContent=c.classification; }
  return el;
}
// —— 卡片右侧延时：最近一次识别该食物的 VLM 耗时 + 所用模型（新建与去重合并都会刷新）——
function latFill(el,c){
  const v=el.querySelector('.rlat .lv'), m=el.querySelector('.rlat .lm');
  if(c.latency_ms!=null){ v.textContent=(c.latency_ms/1000).toFixed(1)+'s'; m.textContent=c.latency_model||''; }
  else { v.textContent='–'; m.textContent=''; }
}
function flashUpdate(el){
  el.classList.remove('flash'); void el.offsetWidth; el.classList.add('flash');
  const u=el.querySelector('.updated'); u.classList.add('on');
  clearTimeout(el._ut); el._ut=setTimeout(()=>u.classList.remove('on'), 1600);
}

// —— lightbox 看图集 ——
let lbShots=[], lbIdx=0, lbName='';
let lbScale=1, lbX=0, lbY=0, lbDrag=false, lbSx=0, lbSy=0;
function lbApply(){ const b=$('lbBig'); b.style.transform='translate('+lbX+'px,'+lbY+'px) scale('+lbScale+')';
  b.style.cursor= lbScale>1 ? (lbDrag?'grabbing':'grab') : 'zoom-in'; }
function lbReset(){ lbScale=1; lbX=0; lbY=0; lbApply(); }
function openLB(name, shots){ if(!shots.length)return;
  lbShots=shots.slice(); lbIdx=lbShots.length-1; lbName=name||'';
  const strip=$('lbStrip'); strip.innerHTML='';
  lbShots.forEach((url,i)=>{ const t=document.createElement('div'); t.className='th'+(i===lbIdx?' sel':'');
    const img=document.createElement('img'); img.src=url; t.appendChild(img);
    t.onclick=()=>{lbIdx=i; renderLB();}; strip.appendChild(t); });
  $('lb').classList.add('on'); renderLB();
}
function renderLB(){ const url=lbShots[lbIdx], big=$('lbBig');
  big.src=url;
  $('lbCap').textContent=lbName+' · '+(lbIdx+1)+' / '+lbShots.length+' 张 · 滚轮缩放·拖动·双击';
  [...$('lbStrip').children].forEach((t,i)=>t.classList.toggle('sel',i===lbIdx));
  lbReset();
}
// 缩放：滚轮缩放(1~6x)、放大后拖动平移、双击放大/还原
$('lbBig').addEventListener('wheel',e=>{ e.preventDefault();
  lbScale=Math.max(1,Math.min(6, lbScale*(e.deltaY<0?1.18:0.85)));
  if(lbScale===1){lbX=0;lbY=0;} lbApply(); },{passive:false});
$('lbBig').addEventListener('mousedown',e=>{ if(lbScale<=1)return; lbDrag=true; lbSx=e.clientX-lbX; lbSy=e.clientY-lbY; lbApply(); e.preventDefault(); });
window.addEventListener('mousemove',e=>{ if(!lbDrag)return; lbX=e.clientX-lbSx; lbY=e.clientY-lbSy; lbApply(); });
window.addEventListener('mouseup',()=>{ if(lbDrag){lbDrag=false; lbApply();} });
$('lbBig').addEventListener('dblclick',()=>{ lbScale=lbScale>1?1:2.5; lbX=0; lbY=0; lbApply(); });
$('lbx').onclick=()=>$('lb').classList.remove('on');
$('lb').onclick=e=>{ if(e.target===$('lb'))$('lb').classList.remove('on'); };
document.addEventListener('keydown',e=>{ if(!$('lb').classList.contains('on'))return;
  if(e.key==='Escape')$('lb').classList.remove('on');
  if(e.key==='ArrowRight'){lbIdx=(lbIdx+1)%lbShots.length;renderLB();}
  if(e.key==='ArrowLeft'){lbIdx=(lbIdx-1+lbShots.length)%lbShots.length;renderLB();} });

// 识别目标切换：POST 后端换 endpoint/model，免重启、下一轮识别生效；按钮状态由 tick 对账
$('seg').addEventListener('click', async e=>{
  const b=e.target.closest('button');
  if(!b||b.disabled||b.classList.contains('sel'))return;
  try{
    const r=await fetch('/api/recog/target',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify({target:b.dataset.t})});
    if(r.ok) tick(); else console.log('识别目标切换失败：', await r.text());
  }catch(e){/* 网络失败忽略，按钮状态由下个 tick 恢复 */}
});

// 清空：后端清卡 + 前端清 feed/state + 关 lightbox
$('clr').onclick=async()=>{
  try{ await fetch('/api/recog/clear',{method:'POST'}); }catch(e){}
  feed.querySelectorAll('.rcard').forEach(n=>n.remove()); state.clear();
  $('lb').classList.remove('on'); $('empty').style.display='block';
};

// —— 智能跳顶：接近顶部(<100px)时新内容自动回顶；正在下翻则浮出「↑ 有新卡」不打断 ——
function bumpScroll(){
  if(feed.scrollTop<100){ feed.scrollTop=0; $('newpill').classList.remove('on'); }
  else $('newpill').classList.add('on');
}
$('newpill').onclick=()=>{ feed.scrollTo({top:0,behavior:'smooth'}); $('newpill').classList.remove('on'); };
feed.addEventListener('scroll',()=>{ if(feed.scrollTop<10) $('newpill').classList.remove('on'); });

let curDev=null;   // 卡片流跟随服务端选中设备：切换时列表自动换成新设备的桶（旧卡不丢，切回来还在）
async function tick(){
 try{
  const d=await(await fetch('/api/recog/list',{cache:'no-store'})).json();
  const live=$('live');
  if(!d.enabled){ live.className='live off'; live.innerHTML='<i></i>未接入';
    $('empty').textContent='识别服务未接入（RECOG_ENDPOINT 未配置）'; }
  else { live.className='live'; live.innerHTML='<i></i>识别中'; }
  // 识别目标：模型名 + 分段按钮状态（选中高亮；未配置 endpoint 的目标置灰）
  if(d.model) $('model').textContent=d.model;
  const tg=d.targets||{};
  [...$('seg').children].forEach(b=>{ const t=b.dataset.t;
    b.classList.toggle('sel', t===d.target);
    b.disabled=!tg[t];
    b.title=tg[t]?'':'未配置该目标的 endpoint（服务器 .env）'; });
  // 设备切换（在 /panel 下拉触发）：关掉图集弹层，卡片由下方 seen 对账逻辑自动整体换桶
  const dev=d.device||null;
  if(curDev!==null && dev!==curDev) $('lb').classList.remove('on');
  curDev=dev; $('devlab').textContent=dev||'–';
  const cards=d.cards||[];
  const seen=new Set(cards.map(c=>c.id));
  for(const [id,st] of state){ if(!seen.has(id)){ st.el.remove(); state.delete(id); } }
  $('empty').style.display = cards.length ? 'none' : 'block';
  let changed=false;
  // 新卡：逆序 prepend 使最新在最上
  cards.filter(c=>!state.has(c.id)).reverse().forEach(c=>{
    const el=cardEl(c); feed.insertBefore(el, feed.firstChild);
    const shots=c.shots||[]; state.set(c.id,{el, rev:c.rev||0, shots});
    renderStack(el, c.name||'', shots); changed=true;
  });
  // 已存在卡：rev 变化=去重合并了新缩略图 → 置顶 + 更新叠卡/帧信息 + 闪 + “更新”提示（内容不改）
  cards.forEach(c=>{ const st=state.get(c.id); if(!st)return;
    if((c.rev||0)!==st.rev){ st.rev=c.rev||0; st.shots=c.shots||[];
      if(feed.firstChild!==st.el) feed.insertBefore(st.el, feed.firstChild);
      const ms=st.el.querySelectorAll('.meta span');
      if(ms[0]) ms[0].textContent='帧 '+(c.frame||''); if(ms[1]) ms[1].textContent=c.t||'';
      latFill(st.el, c);
      renderStack(st.el, c.name||'', st.shots); flashUpdate(st.el); changed=true; } });
  if(changed) bumpScroll();
 }catch(e){}
}
setInterval(tick,700); tick();
</script>
</body></html>"""


@app.get("/recog", response_class=HTMLResponse)
def recog_page():
    """实验识别卡片流页（首页右栏 iframe）。"""
    return EXP_RECOG_PAGE
