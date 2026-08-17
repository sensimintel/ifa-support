# -*- coding: utf-8 -*-
"""主链路「直传 VLM 识别」的配置态（纯逻辑，无 DA3/torch 依赖，可单测）。

背景：识别触发历来挂在 SAM3 上——单目链取 SAM3 流式液体框、双目链取左目 IR 的
SAM3 命中，SAM3 没定位到东西就完全不识别。直传把这层前置依赖去掉：设备深度图
主链路按固定间隔取该设备最新 RGB 帧直接送 VLM，问「画面里可能有什么食物」。

本模块只负责配置的规范化 / 钳制 / 落盘，触发线程与 worker 池在 app.py。
"""
import json
import threading
from pathlib import Path

# 直传默认口径：开启、0.5s 一发、并发 1（同一时刻只 1 个 VLM 请求在飞，
# 间隔到了但上一轮没回就丢旧帧用最新帧——即「串行·最新优先」）
DEFAULTS = {"on": True, "interval_s": 0.5, "concurrency": 1}

# 数值键取值范围：interval 下限 0.2s（再密也快不过 RGB 推帧），
# concurrency 上限 6（每路都是一次完整 VLM 多图请求，再高就是纯烧 GPU）
LIMITS = {"interval_s": (0.2, 10.0), "concurrency": (1, 6)}


def normalize(patch, base=None) -> dict:
    """merge-patch 语义：base 上叠 patch，非法值忽略、越界钳制，返回新配置字典。

    patch 非字典、键不认识、值转不成数字一律跳过（保持 base 的值），
    不抛异常——控制面来的是不可信输入，坏值不该打断链路。"""
    out = dict(DEFAULTS if base is None else base)
    for key in DEFAULTS:
        out.setdefault(key, DEFAULTS[key])
    if not isinstance(patch, dict):
        return out
    if "on" in patch:
        val = patch["on"]
        if isinstance(val, str):
            out["on"] = val.strip().lower() in ("1", "true", "on", "yes")
        else:
            out["on"] = bool(val)
    for key, (lo, hi) in LIMITS.items():
        if key not in patch:
            continue
        try:
            val = float(patch[key])
        except (TypeError, ValueError):
            continue
        if val != val:            # NaN
            continue
        val = min(hi, max(lo, val))
        out[key] = int(round(val)) if isinstance(DEFAULTS[key], int) else round(val, 2)
    return out


def should_trigger(direct: bool, has_detections: bool, direct_on: bool) -> bool:
    """识别触发闸门（单点真相，app.py 的 _maybe_recognize 只认这一处规则）。

    direct=True（直传口径）：恒放行——节奏由直传触发线程按 interval_s 控制，
      不再叠 RECOG_MIN_INTERVAL 那套 SAM3 时代的节流。
    direct=False（SAM3 口径）：要有检测证据，且直传处于关闭状态——直传开着时
      SAM3 触发整体让位，识别卡片只留直传一个来源，两边节奏不打架。"""
    if direct:
        return True
    return bool(has_detections) and not direct_on


class DirectConfig:
    """带锁的直传配置 + 落盘持久化（服务重启不回落默认，学 sam3_score_cfg 的做法）。

    读侧（触发线程每 tick 读一次）走 snapshot()，写侧走 update()；
    落盘失败只打日志不抛——配置在内存里已经生效，落盘只是重启后的记忆。"""

    def __init__(self, path):
        self._path = Path(path)
        self._lock = threading.Lock()
        self._cfg = dict(DEFAULTS)
        self.load()

    def load(self) -> dict:
        """回读落盘配置（文件不存在/损坏一律用默认，不抛）。"""
        try:
            saved = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return self.snapshot()
        with self._lock:
            self._cfg = normalize(saved, DEFAULTS)
            return dict(self._cfg)

    def save(self) -> None:
        """原子落盘（先写 .tmp 再 replace，防半截文件）。"""
        with self._lock:
            body = json.dumps(self._cfg, ensure_ascii=False, indent=2) + "\n"
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(body, encoding="utf-8")
        tmp.replace(self._path)

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._cfg)

    def update(self, patch) -> dict:
        """merge-patch 更新并落盘，返回新配置。"""
        with self._lock:
            self._cfg = normalize(patch, self._cfg)
            cfg = dict(self._cfg)
        try:
            self.save()
        except OSError as e:
            print(f"[da3-web] 直传识别配置落盘失败（内存已生效）：{type(e).__name__}: {e}",
                  flush=True)
        return cfg

    def enabled(self) -> bool:
        with self._lock:
            return bool(self._cfg["on"])

    def interval_s(self) -> float:
        with self._lock:
            return float(self._cfg["interval_s"])

    def concurrency(self) -> int:
        with self._lock:
            return int(self._cfg["concurrency"])
