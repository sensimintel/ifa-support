# -*- coding: utf-8 -*-
"""VLM 识别观测日志的环形缓冲与响应投影（纯逻辑，无 cv2/torch 依赖，可单测）。

每一轮识别请求都留一条痕：请求图（缩略 dataURI）、prompt 原文、模型原始返回、
解析出的每一项、以及五道去重闸门的逐项判定。浅体验区控制面（superadmin
`/ifa-support/experience`）据此回答三个排障问题：
  · 这一轮到底把什么图发出去了（食物在不在画面里、糊没糊）；
  · 模型原样答了什么（漏识别是模型没认出来，还是解析/闸门把它吃了）；
  · 每一项最后落到哪张卡（新建还是并入，被哪道闸门拦下）。

图片编码留在调用方（app.py 用 _thumb_uri），本模块只搬运字符串——这样测试
不需要 OpenCV，也方便以后换编码口径。
"""
import threading
import time

# 列表态与详情态的字段划分：列表要能秒级轮询，大字段（prompt 全文 / 原始返回 /
# 候选参考图 dataURI）一律留给详情接口
_HEAD_KEYS = ("id", "ts", "device", "trigger", "img_orig", "img_boxed",
              "n_food", "n_drink", "outcome", "timings")
_REQ_KEYS = ("label", "model", "endpoint", "direct", "n_images",
             "max_tokens", "temperature", "stream", "img_full_px")
# 详情才给的大字段（原尺寸图本身不进 JSON，走 /api/recoglog/{id}/image/{kind} 端点：
# 详情响应从几百 KB 回到几 KB，点开大图时才拉那一张）
_REQ_DETAIL_KEYS = ("prompt",)
_RESP_KEYS = ("ok", "error", "llm_ms", "wait_ms", "n_items")


class RecogLog:
    """线程安全的定长日志缓冲（最新在前）。

    典型用法：worker 开一条 `begin()` → 识别调用填 `entry["req"]/["resp"]`
    → 逐项判定 append 进 `entry["outcome"]` → `commit()`。失败轮同样 commit，
    「这一轮压根没调通」正是最该被看见的日志。"""

    def __init__(self, max_items=30, raw_max=20000, full_keep=8):
        self.max_items = int(max_items)
        self.raw_max = int(raw_max)
        # 只有最近 full_keep 条留「送 VLM 的原尺寸图」（每条两张、每张几百 KB）：
        # 全留会把缓冲撑到十几 MB，而翻到很旧的日志时通常只关心判定不关心原图
        self.full_keep = int(full_keep)
        self._items = []
        self._lock = threading.Lock()
        # 与锁共用一把 Condition：控制面挂长轮询在这上面等，有新日志立刻被唤醒，
        # 不必靠固定间隔去猜（1.5s 轮询在展台上肉眼可见地"慢半拍"）
        self._cv = threading.Condition(self._lock)
        self._seq = 0
        self._gen = 0

    def begin(self, device, trigger, candidates, n_food=0, n_drink=0,
              img_orig=None, img_boxed=None) -> dict:
        """开一条日志（请求侧快照）。候选参考图直接引用卡片上那份 dataURI 字符串
        （同一对象，不重复编码、不多占内存）。返回的 dict 由调用方继续填。"""
        with self._lock:
            self._seq += 1
            entry_id = self._seq
        return {
            "id": entry_id, "ts": None, "device": device, "trigger": trigger,
            "img_orig": img_orig, "img_boxed": img_boxed,
            "n_food": n_food, "n_drink": n_drink,
            "candidates": [{"id": c.get("id"), "name": c.get("name", ""),
                            "type": c.get("type", ""), "ref_img": c.get("ref_img")}
                           for c in (candidates or [])],
            "req": None, "resp": None, "outcome": [], "timings": {},
        }

    def set_response(self, entry, ok, error="", raw="", items=None, timings=None) -> None:
        """写响应侧。raw 超长截断——单条几十 KB 的思考链会把缓冲吃光。
        timings 是识别调用内部的分段（encode/http/parse），并进 entry["timings"]。"""
        if entry is None:
            return
        raw = raw or ""
        entry["resp"] = {"ok": bool(ok), "error": error or "",
                         "raw": raw[:self.raw_max], "truncated": len(raw) > self.raw_max,
                         "items": list(items or []), "n_items": len(items or [])}
        if timings:
            entry.setdefault("timings", {}).update(timings)

    def commit(self, entry) -> None:
        with self._cv:
            self._gen += 1
            self._cv.notify_all()      # 唤醒所有挂着的长轮询
            self._items.insert(0, entry)
            del self._items[self.max_items:]
            for old in self._items[self.full_keep:]:      # 滑出窗口的条目丢掉原尺寸图
                req = old.get("req")
                if req:
                    req.pop("img_full", None)
                    req.pop("img_boxed_full", None)

    def list(self, device=None, limit=12):
        """返回 (投影后的列表, 过滤后的总条数)。limit 钳制在 [1, max_items]。"""
        with self._lock:
            items = list(self._items)
        if device:
            items = [it for it in items if it.get("device") == device]
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 12
        limit = min(max(limit, 1), self.max_items)
        return [project(it) for it in items[:limit]], len(items)

    def get(self, entry_id):
        """按 id 取详情投影；已滚出缓冲返回 None。"""
        with self._lock:
            hit = next((it for it in self._items if it.get("id") == entry_id), None)
        return project(hit, detail=True) if hit is not None else None

    def full_image(self, entry_id, kind="orig"):
        """取送 VLM 的原尺寸图 dataURI（kind=orig|boxed）；已滚出留存窗口返回 None。"""
        key = "img_boxed_full" if kind == "boxed" else "img_full"
        with self._lock:
            hit = next((it for it in self._items if it.get("id") == entry_id), None)
        return (hit.get("req") or {}).get(key) if hit else None

    def gen(self) -> int:
        """变更版本号：每 commit 一条 +1。长轮询用它判断"我看到的还是不是最新的"。"""
        with self._lock:
            return self._gen

    def wait_for(self, since_gen, timeout_s) -> int:
        """挂起直到版本号变化或超时，返回当前版本号。

        since_gen 与当前版本不同（含调用方给 -1 的首帧）时立即返回——保证第一拍
        永远拿得到数据，只有"我已是最新"的连接才真正挂起。
        """
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        with self._cv:
            while self._gen == since_gen:
                remain = deadline - time.monotonic()
                if remain <= 0:
                    break
                self._cv.wait(remain)
            return self._gen

    def clear(self) -> None:
        with self._cv:
            self._gen += 1
            self._cv.notify_all()
            self._items.clear()

    def __len__(self):
        with self._lock:
            return len(self._items)


def project(entry, detail=False) -> dict:
    """条目 → 响应体。detail=False 剥掉大字段，只留长度与摘要。"""
    req = entry.get("req") or {}
    resp = entry.get("resp") or {}
    out = {k: entry.get(k) for k in _HEAD_KEYS}
    out["req"] = {k: req.get(k) for k in _REQ_KEYS}
    out["req"]["prompt_len"] = len(req.get("prompt") or "")
    out["resp"] = {k: resp.get(k) for k in _RESP_KEYS}
    out["resp"]["truncated"] = bool(resp.get("truncated"))
    out["n_candidates"] = len(entry.get("candidates") or [])
    out["has_full"] = bool(req.get("img_full"))          # 还能不能点开看原尺寸图
    out["has_boxed_full"] = bool(req.get("img_boxed_full"))
    if detail:
        for k in _REQ_DETAIL_KEYS:
            out["req"][k] = req.get(k) or ""
        out["resp"]["raw"] = resp.get("raw") or ""
        out["resp"]["items"] = resp.get("items") or []
        out["candidates"] = entry.get("candidates") or []
    else:
        # 列表态候选只给名字（参考图留给详情），够看出"这轮带了谁去对照"
        out["candidates"] = [{"id": c.get("id"), "name": c.get("name"),
                              "type": c.get("type")}
                             for c in (entry.get("candidates") or [])]
    return out
