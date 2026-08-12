"""
感知链路实验台（8061）——与产线 8060 完全隔离的检测/分割实验编排服务。

定位（为什么独立成进程）：
- 8060 的感知编排（LA 检测 → 点云叠框 → 触发 VLM 识别）是产线，/experience 演示页直接依赖；
- 实验（换检测组合、调 SAM3 阈值、试 embedding match 等）需要频繁改代码/重启，
  独立进程随便折腾，8060 与 /experience 一秒不中断；
- 不加载任何大模型：帧从 8060 的 HTTP 接口拉，SAM3 / LocateAnything 都是 5090 上
  已常驻的 HTTP 模型服务，直接调（SAM3 用无状态的 /v1/segment，不碰产线的流式 session）。

GPU 负载账：增量 = 本服务采样率 × 实验栈调用（默认 1 帧/秒，可调/可总开关关停），
DA3 不重复跑，显存零新增（embedding 小模型接入后另算）。

页面（/）仿 8060 风格：左=实验链路标注帧实况，右=链路配置 + 逐轮结果流水。
"""

import base64
import json
import os
import re
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from fastapi import Body, FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, Response

app = FastAPI(title="感知链路实验台")

# ── 上游服务地址（与 8060 同一套 .env 约定，run-exp.sh 会加载仓根 .env） ──
FRAME_SOURCE = os.environ.get("EXP_FRAME_SOURCE", "http://127.0.0.1:8060").rstrip("/")
SAM3_ENDPOINT = os.environ.get("SAM3_ENDPOINT", "http://127.0.0.1:8013").rstrip("/")
LOCATE_ENDPOINT = os.environ.get("EXP_LOCATE_ENDPOINT",
                                 "http://127.0.0.1:8000/v1/chat/completions")
LOCATE_MODEL = "nvidia/LocateAnything-3B"
SAM3_TIMEOUT = float(os.environ.get("EXP_SAM3_TIMEOUT", "30"))
LOCATE_TIMEOUT = float(os.environ.get("EXP_LOCATE_TIMEOUT", "20"))

# ── 实验链路配置（页面可改，全部热生效；本服务自己的状态，与 8060 无任何共享） ──
_cfg = {
    "enabled": False,          # 总开关：关=不拉帧不推理，零 GPU 增量（默认关，开着白烧）
    "interval": 1.0,           # 采样间隔（秒）：两轮实验之间的最小间隔
    "sam3_on": True,           # SAM3 /v1/segment 检测
    "sam3_queries": "food",    # SAM3 文本查询，分号/逗号分隔多查询（每查询一次调用）
    "sam3_thresh": 0.5,        # SAM3 score 阈值（客户端过滤；低于阈值的画灰框便于调参）
    "sam3_mask": True,         # 是否叠加 mask 半透明高亮（关了只画框，绘制更快）
    "la_on": False,            # LocateAnything 检测（对照用，默认关——实验主题就是不走 LA）
    "la_queries": "food; bottle; glass",
    "embed_on": False,         # embedding 匹配插槽（见 _embed_match，未接入模型前无输出）
}
_cfg_lock = threading.Lock()

# ── 实验产物（页面轮询）：最新标注帧 + 最近若干轮结果记录 ──
_out = {"jpg": None, "seq": 0, "src_seq": -1}   # seq=标注帧自增号；src_seq=消费到的 8060 帧号
_rounds = []                                    # 逐轮结果（最新在后），只留最近 60 轮
_out_lock = threading.Lock()
ROUNDS_KEEP = 60

# SAM3 各查询的框/掩码配色（RGB），循环取用；LA 统一红色系对照
_SAM3_COLORS = [(52, 199, 89), (255, 159, 10), (10, 132, 255), (191, 90, 242), (255, 214, 10)]
_LA_COLOR = (255, 69, 58)
_CUT_COLOR = (142, 142, 147)   # 低于阈值被裁掉的实例：灰


def _split_queries(s):
    """查询串 → 查询列表（分号/逗号分隔，去空白去空项）。"""
    return [q.strip() for q in str(s).replace("，", ",").replace("；", ";")
            .replace(",", ";").split(";") if q.strip()]


def _http_json(url, payload=None, timeout=10.0):
    """极简 HTTP JSON 调用（与 8060 同款 urllib 风格，不引新依赖）。"""
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


# ══════════════════════════════════════════════════════════════════════
# 帧来源：拉 8060 的选中设备最新帧（只读接口，对产线零影响）
# ══════════════════════════════════════════════════════════════════════
def _fetch_frame():
    """向 8060 拉最新帧：返回 (rgb ndarray, 帧号) 或 (None, -1)。"""
    try:
        st = _http_json(FRAME_SOURCE + "/api/frame/status", timeout=5.0)
        if not st.get("has_frame"):
            return None, -1
        seq = int(st.get("seq", 0))
        req = urllib.request.Request(FRAME_SOURCE + "/api/frame/latest?t=%d" % seq)
        with urllib.request.urlopen(req, timeout=5.0) as r:
            raw = r.read()
        arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            return None, -1
        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB), seq
    except Exception as e:
        print(f"[exp] 拉帧失败：{type(e).__name__}: {e}", flush=True)
        return None, -1


def _b64_jpg(rgb):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf.tobytes()).decode() if ok else None


# ══════════════════════════════════════════════════════════════════════
# 检测级：SAM3（无状态 /v1/segment，不碰产线流式 session）与 LocateAnything
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


def _sam3_segment(rgb, query):
    """单查询 SAM3 分割：返回实例列表（含归一化 box、score、mask RLE），失败返回 []。
    SAM3 服务端有全局 GPU 锁，多查询串行发送、不并发排队。"""
    b64 = _b64_jpg(rgb)
    if not b64:
        return []
    try:
        r = _http_json(SAM3_ENDPOINT + "/v1/segment",
                       {"image_b64": b64, "text": query}, timeout=SAM3_TIMEOUT)
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
# embedding 匹配插槽：接小模型（SigLIP 之类）做特征匹配时只需实现这一个函数。
# 输入=检测框裁剪图（RGB ndarray），输出={"name":..,"score":..} 或 None（无匹配/未接入）。
# 模型加载请放模块级懒加载（首次调用才加载），避免服务启动就占显存。
# ══════════════════════════════════════════════════════════════════════
def _embed_match(crop_rgb):
    return None   # 未接入：占位返回 None，页面上该列显示「插槽未接入」


# ══════════════════════════════════════════════════════════════════════
# 实验主循环：采样帧 → 按配置跑各级 → 画标注帧 + 记一轮结果
# ══════════════════════════════════════════════════════════════════════
def _draw_round(rgb, sam3_res, la_res, thresh, with_mask):
    """标注帧绘制：SAM3 实例（过阈值=彩色框+可选 mask 高亮、低于阈值=灰框），LA=红框。"""
    out = rgb.copy()
    H, W = out.shape[:2]
    lw = max(2, W // 400)
    for qi, (query, insts) in enumerate(sam3_res):
        color = _SAM3_COLORS[qi % len(_SAM3_COLORS)]
        for ins in insts:
            bx = ins.get("box_xywh_norm") or []
            if len(bx) != 4:
                continue
            x, y, w, h = bx
            p1, p2 = (int(x * W), int(y * H)), (int((x + w) * W), int((y + h) * H))
            kept = ins.get("score", 0) >= thresh
            c = color if kept else _CUT_COLOR
            cv2.rectangle(out, p1, p2, c, lw if kept else max(1, lw // 2))
            label = "%s %.2f" % (query, ins.get("score", 0))
            cv2.putText(out, label, (p1[0], max(14, p1[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2, cv2.LINE_AA)
            if kept and with_mask:
                rle = ins.get("mask_rle") or {}
                if rle.get("counts") and rle.get("size"):
                    try:
                        m = _rle_decode(rle["size"], rle["counts"])
                        if m.shape != (H, W):
                            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                        sel = m.astype(bool)
                        out[sel] = (out[sel] * 0.55 + np.array(color) * 0.45).astype(np.uint8)
                    except Exception:
                        pass   # 单实例 mask 解码失败只丢高亮，不丢框
    for query, boxes in la_res:
        for (x1, y1, x2, y2) in boxes:
            p1, p2 = (int(x1 * W), int(y1 * H)), (int(x2 * W), int(y2 * H))
            cv2.rectangle(out, p1, p2, _LA_COLOR, lw)
            cv2.putText(out, "LA:" + query, (p1[0], min(H - 6, p2[1] + 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, _LA_COLOR, 2, cv2.LINE_AA)
    return out


def _run_round(rgb, src_seq, cfg):
    """跑一轮实验链路，返回该轮结果记录（同时更新最新标注帧）。"""
    rec = {"t": time.strftime("%H:%M:%S"), "src_seq": src_seq,
           "sam3": [], "la": [], "embed": [], "ms": {}}
    sam3_res, la_res = [], []
    if cfg["sam3_on"]:
        t0 = time.time()
        for q in _split_queries(cfg["sam3_queries"]):   # 服务端有 GPU 锁，串行不并发排队
            insts = _sam3_segment(rgb, q)
            kept = [i for i in insts if i.get("score", 0) >= cfg["sam3_thresh"]]
            sam3_res.append((q, insts))
            rec["sam3"].append({"query": q, "total": len(insts), "kept": len(kept),
                                "scores": [round(i.get("score", 0), 3) for i in insts]})
        rec["ms"]["sam3"] = round((time.time() - t0) * 1000)
    if cfg["la_on"]:
        t0 = time.time()
        qs = _split_queries(cfg["la_queries"])
        with ThreadPoolExecutor(max_workers=max(1, len(qs))) as ex:   # LA 有 LB，可并发
            results = list(ex.map(lambda q: (q, _locate_one(rgb, q)), qs))
        la_res = results
        rec["la"] = [{"query": q, "boxes": len(bs)} for q, bs in results]
        rec["ms"]["la"] = round((time.time() - t0) * 1000)
    if cfg["embed_on"]:
        t0 = time.time()
        H, W = rgb.shape[:2]
        for q, insts in sam3_res:
            for ins in insts:
                bx = ins.get("box_xywh_norm") or []
                if len(bx) != 4 or ins.get("score", 0) < cfg["sam3_thresh"]:
                    continue
                x, y, w, h = bx
                crop = rgb[max(0, int(y * H)):int((y + h) * H), max(0, int(x * W)):int((x + w) * W)]
                m = _embed_match(crop) if crop.size else None
                rec["embed"].append({"from": q, "match": m or "插槽未接入"})
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
    n_sam3 = sum(r["kept"] for r in rec["sam3"])
    n_la = sum(r["boxes"] for r in rec["la"])
    print("[exp] 实验一轮：帧%d · sam3 %d 命中(%dms) · la %d 框(%dms)" % (
        src_seq, n_sam3, rec["ms"].get("sam3", 0), n_la, rec["ms"].get("la", 0)), flush=True)


def _worker():
    """后台主循环：按采样间隔消费 8060 新帧；总开关关 = 完全静默。"""
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
        rgb, seq = _fetch_frame()
        if rgb is None or seq == last_seq:   # 无帧/没有新帧：小睡等下一轮
            time.sleep(0.3)
            continue
        last_seq, last_run = seq, time.time()
        try:
            _run_round(rgb, seq, cfg)
        except Exception as e:
            print(f"[exp] 实验轮异常：{type(e).__name__}: {e}", flush=True)


threading.Thread(target=_worker, daemon=True).start()


# ══════════════════════════════════════════════════════════════════════
# API
# ══════════════════════════════════════════════════════════════════════
@app.get("/api/exp/status")
def exp_status():
    """页面轮询：配置 + 最新标注帧号 + 最近轮次记录 + 上游健康。"""
    with _cfg_lock:
        cfg = dict(_cfg)
    with _out_lock:
        seq, src_seq = _out["seq"], _out["src_seq"]
        rounds = list(_rounds[-20:])
    return JSONResponse({"config": cfg, "seq": seq, "src_seq": src_seq,
                         "rounds": rounds})


@app.get("/api/exp/health")
def exp_health():
    """上游依赖健康：8060 帧源 / SAM3 / LA（页面顶栏指示灯）。"""
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
    """最新标注帧 JPEG。"""
    with _out_lock:
        jpg = _out["jpg"]
    if not jpg:
        return JSONResponse({"error": "暂无标注帧（总开关未开或还没消费到帧）"}, status_code=404)
    return Response(jpg, media_type="image/jpeg")


# ══════════════════════════════════════════════════════════════════════
# 页面（仿 8060 深色风格）：左=实验标注帧实况，右=链路配置 + 逐轮结果流水
# ══════════════════════════════════════════════════════════════════════
EXP_PAGE = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>感知链路实验台 · 8061</title>
<style>
 *{box-sizing:border-box}
 html,body{margin:0;height:100%;font-family:system-ui,-apple-system,'Segoe UI',sans-serif;background:#0d0d0f;color:#e5e5ea}
 .top{height:48px;display:flex;align-items:center;gap:14px;padding:0 18px;background:#1c1c1e;font-size:14px}
 .top b{font-size:15px}
 .top .tag{font-size:12px;color:#9a9aa0}
 .dot{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:5px;background:#8e8e93}
 .dot.ok{background:#34c759}.dot.bad{background:#ff453a}
 .hlth{margin-left:auto;font-size:12px;color:#9a9aa0;display:flex;gap:14px;align-items:center}
 .wrap{display:flex;height:calc(100% - 48px)}
 .pane{min-width:0;display:flex;flex-direction:column}
 .pane.l{flex:1 1 62%;border-right:1px solid #2c2c2e}
 .pane.r{flex:1 1 38%;overflow-y:auto;padding:14px 16px}
 .bar{height:34px;flex:none;display:flex;align-items:center;padding:0 14px;background:#141416;font-size:13px;font-weight:600;border-bottom:1px solid #2c2c2e}
 #live{flex:1;display:flex;align-items:center;justify-content:center;background:#050505;overflow:hidden}
 #live img{max-width:100%;max-height:100%;display:none}
 #wait{font-size:13px;color:#6e6e73}
 h3{font-size:13px;margin:16px 0 8px;color:#e5e5ea;border-top:1px solid #2c2c2e;padding-top:14px}
 h3:first-child{border-top:0;padding-top:0;margin-top:2px}
 .fld{margin:8px 0;font-size:13px;color:#c7c7cc;display:flex;align-items:center;gap:8px;flex-wrap:wrap}
 .fld label{min-width:98px;color:#9a9aa0}
 .fld input[type=text]{flex:1;min-width:160px;background:#1c1c1e;border:1px solid #3a3a3c;color:#e5e5ea;border-radius:6px;padding:5px 8px;font-size:13px}
 .fld input[type=number]{width:70px;background:#1c1c1e;border:1px solid #3a3a3c;color:#e5e5ea;border-radius:6px;padding:5px 8px;font-size:13px}
 .fld input[type=range]{flex:1;min-width:120px}
 .fld b{color:#6ab7ff;font-variant-numeric:tabular-nums}
 .sw{position:relative;width:40px;height:22px;flex:none}
 .sw input{opacity:0;width:0;height:0}
 .sw i{position:absolute;inset:0;border-radius:99px;background:#3a3a3c;transition:.2s;cursor:pointer}
 .sw i:before{content:'';position:absolute;left:2px;top:2px;width:18px;height:18px;border-radius:50%;background:#fff;transition:.2s}
 .sw input:checked+i{background:#34c759}
 .sw input:checked+i:before{transform:translateX(18px)}
 .hint{font-size:11.5px;color:#6e6e73;line-height:1.5;margin:4px 0 0}
 .rounds{margin-top:6px}
 .rd{font-size:12px;font-family:ui-monospace,Menlo,monospace;color:#c7c7cc;padding:6px 8px;border-bottom:1px solid #232325;line-height:1.55}
 .rd .t{color:#6e6e73}
 .rd .q{color:#34c759}.rd .cut{color:#8e8e93}.rd .la{color:#ff6961}.rd .em{color:#bf5af2}
 .badge{display:inline-block;font-size:11px;padding:1px 8px;border-radius:99px;background:#2c2c2e;color:#9a9aa0;margin-left:8px}
</style></head><body>
<div class="top"><b>感知链路实验台</b><span class="tag">8061 · 与产线 8060 / experience 完全隔离 · 帧取自 8060 只读接口</span>
 <span class="hlth"><span><span class="dot" id="h_frame"></span>帧源8060</span>
  <span><span class="dot" id="h_sam3"></span>SAM3</span>
  <span><span class="dot" id="h_la"></span>LA</span></span></div>
<div class="wrap">
 <div class="pane l">
  <div class="bar">实验标注帧 <span class="badge" id="seqinfo">–</span></div>
  <div id="live"><img id="anno" alt=""><span id="wait">总开关未开，或还没消费到帧</span></div>
 </div>
 <div class="pane r">
  <h3>总控</h3>
  <div class="fld"><label>总开关</label><span class="sw"><input type="checkbox" id="c_enabled"><i></i></span>
   <span class="hint">关 = 不拉帧不推理，GPU 零增量</span></div>
  <div class="fld"><label>采样间隔</label><input type="number" id="c_interval" min="0.2" step="0.1"> 秒</div>
  <h3>SAM3（/v1/segment 无状态，不碰产线流式）</h3>
  <div class="fld"><label>启用</label><span class="sw"><input type="checkbox" id="c_sam3_on"><i></i></span></div>
  <div class="fld"><label>查询词</label><input type="text" id="c_sam3_queries" placeholder="food; drink"></div>
  <div class="fld"><label>score 阈值</label><input type="range" id="c_sam3_thresh" min="0" max="1" step="0.05"><b id="v_thresh">0.50</b></div>
  <div class="fld"><label>mask 高亮</label><span class="sw"><input type="checkbox" id="c_sam3_mask"><i></i></span>
   <span class="hint">低于阈值的实例画灰框，便于观察阈值卡掉了什么</span></div>
  <h3>LocateAnything（对照，默认关）</h3>
  <div class="fld"><label>启用</label><span class="sw"><input type="checkbox" id="c_la_on"><i></i></span></div>
  <div class="fld"><label>查询词</label><input type="text" id="c_la_queries"></div>
  <h3>embedding 匹配（插槽）</h3>
  <div class="fld"><label>启用</label><span class="sw"><input type="checkbox" id="c_embed_on"><i></i></span>
   <span class="hint">接入小模型前无输出；实现 exp_app.py 的 _embed_match() 即接通</span></div>
  <h3>逐轮结果（最新在上）</h3>
  <div class="rounds" id="rounds"></div>
 </div>
</div>
<script>
const $=id=>document.getElementById(id);
const BOOLS=['enabled','sam3_on','sam3_mask','la_on','embed_on'];
const TEXTS=['sam3_queries','la_queries'];
let inited=false,lastSeq=-1,pushTimer=null;

function push(){
  clearTimeout(pushTimer);
  pushTimer=setTimeout(()=>{
    const body={};
    BOOLS.forEach(k=>body[k]=$('c_'+k).checked);
    TEXTS.forEach(k=>body[k]=$('c_'+k).value);
    body.interval=+$('c_interval').value||1.0;
    body.sam3_thresh=+$('c_sam3_thresh').value;
    fetch('/api/exp/config',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify(body)}).catch(()=>{});
  },200);
}
BOOLS.concat(TEXTS).forEach(k=>$('c_'+k).addEventListener('change',push));
$('c_interval').addEventListener('change',push);
$('c_sam3_thresh').addEventListener('input',()=>{$('v_thresh').textContent=(+$('c_sam3_thresh').value).toFixed(2);push();});

function fillCfg(c){   // 只在首次回填控件，之后以页面为准（避免打字被轮询覆盖）
  if(inited)return;inited=true;
  BOOLS.forEach(k=>$('c_'+k).checked=!!c[k]);
  TEXTS.forEach(k=>$('c_'+k).value=c[k]||'');
  $('c_interval').value=c.interval;
  $('c_sam3_thresh').value=c.sam3_thresh;$('v_thresh').textContent=(+c.sam3_thresh).toFixed(2);
}
function rdLine(r){
  let s='<span class="t">'+r.t+' · 帧'+r.src_seq+'</span> ';
  (r.sam3||[]).forEach(x=>{s+=' <span class="q">'+x.query+' '+x.kept+'/'+x.total+'</span>'
    +(x.scores.length?' <span class="cut">['+x.scores.join(', ')+']</span>':'');});
  (r.la||[]).forEach(x=>{s+=' <span class="la">LA:'+x.query+'×'+x.boxes+'</span>';});
  if((r.embed||[]).length)s+=' <span class="em">embed×'+r.embed.length+'</span>';
  const ms=r.ms||{};s+=' <span class="t">('+Object.entries(ms).map(([k,v])=>k+' '+v+'ms').join(' · ')+')</span>';
  return '<div class="rd">'+s+'</div>';
}
async function tick(){
  try{
    const s=await(await fetch('/api/exp/status',{cache:'no-store'})).json();
    fillCfg(s.config);
    $('seqinfo').textContent=s.seq?('第 '+s.seq+' 轮 · 消费到 8060 帧 '+s.src_seq):'–';
    if(s.seq&&s.seq!==lastSeq){lastSeq=s.seq;
      $('anno').src='/api/exp/latest?t='+s.seq;
      $('anno').style.display='block';$('wait').style.display='none';}
    $('rounds').innerHTML=(s.rounds||[]).slice().reverse().map(rdLine).join('');
  }catch(e){}
}
async function hlth(){
  try{
    const h=await(await fetch('/api/exp/health',{cache:'no-store'})).json();
    [['h_frame',h.frame],['h_sam3',h.sam3],['h_la',h.la]].forEach(([id,ok])=>{
      $(id).className='dot '+(ok?'ok':'bad');});
  }catch(e){}
}
setInterval(tick,800);tick();
setInterval(hlth,5000);hlth();
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def home():
    return EXP_PAGE
