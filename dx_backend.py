"""深体验区（superadmin ifa-support 页面）专属后端，独立于 8060 DA3 服务，跑在 8070。

职责：
1. 四通道食物秤（一台 SJ101T2_CH4_ETH 模块，通道 1..4 → 寄存器 addr 0/2/4/6）：
   后台线程轮询缓存实时读数；「清空」= 软件去皮（记当前 raw 为皮重）。
2. 桌边分组绑定配置：每条桌边一组「手机 / 项链 / 秤通道」，可查可改、落盘持久化。
   这份绑定信息是深体验区的编排事实源——services 之后可按设备号反查所在组，
   进而知道该设备的推理要读哪个秤通道（见 GET /api/groups/resolve）。

持久化：单文件 dx_data.json（分组配置 + 各通道皮重），原子写（tmp+rename），
重启不丢；文件 gitignore，不进代码仓。

零重依赖：fastapi + uvicorn（复用 da3 conda 环境），Modbus TCP 为手写 socket。
启动：./run-dx.sh 或 systemd dx-backend.service（见 deploy.sh）。
"""

import json
import socket
import struct
import threading
import time
from pathlib import Path

from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(title="深体验区后端")
# 局域网演示服务：放开跨域，供 superadmin(18091) 等同网页面直调
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ══════════════════════════════════════════════════════════════════════
# 持久化状态：分组配置 + 皮重
# ══════════════════════════════════════════════════════════════════════
DATA_FILE = Path(__file__).resolve().parent / "dx_data.json"
EDGES = (1, 2, 3, 4)

# 分组允许编辑的字段（除 edge 外全部可改）
GROUP_EDITABLE_FIELDS = ("label", "phone_device_id", "necklace_device_id", "scale_channel")

_state_lock = threading.Lock()


def _default_state():
    """默认状态：桌边 N 绑秤通道 N，设备号留空待配置。"""
    return {
        "groups": [
            {"edge": e, "label": f"桌边 {e}",
             "phone_device_id": "", "necklace_device_id": "", "scale_channel": e}
            for e in EDGES
        ],
        # 皮重按通道存 raw 值（与读数同分度），服务重启不丢
        "tare_raw": {str(ch): 0 for ch in EDGES},
    }


def _load_state():
    """启动加载持久化文件；文件缺失/损坏时回落默认并立即落盘。"""
    if DATA_FILE.exists():
        try:
            state = json.loads(DATA_FILE.read_text(encoding="utf-8"))
            # 结构兜底：缺 key 用默认补齐，防旧版本文件升级后崩
            default = _default_state()
            for key, value in default.items():
                state.setdefault(key, value)
            return state
        except Exception:
            pass
    state = _default_state()
    _save_state(state)
    return state


def _save_state(state):
    """原子落盘：先写临时文件再 rename，避免写一半掉电产生坏文件。"""
    tmp = DATA_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(DATA_FILE)


_state = _load_state()


def _group_of(edge):
    return next((g for g in _state["groups"] if g["edge"] == edge), None)


# ══════════════════════════════════════════════════════════════════════
# 四通道食物秤：Modbus TCP 轮询（契约与硬件铭牌一致，实测确认于 2026-08-04）
#   192.168.0.80:502  unit=1  FC3  通道 N → addr (N-1)*2
#   32 位有符号 · 字序 HH-LL · 分度 0.1（raw/10 = 克）
# ══════════════════════════════════════════════════════════════════════
SCALE_HOST = "192.168.0.80"
SCALE_PORT = 502
SCALE_UNIT = 1
SCALE_DIVISION = 0.1
SCALE_CHANNELS = (1, 2, 3, 4)
SCALE_POLL_INTERVAL = 0.4   # 后台轮询间隔（秒）
SCALE_TIMEOUT = 1.2         # 单次 Modbus 读超时（秒）

_scale_latest = {ch: {"ok": False, "raw": None} for ch in SCALE_CHANNELS}
_scale_lock = threading.Lock()


def _recv_exact(sock, n):
    """从 socket 精确读取 n 个字节。"""
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("连接被对端关闭")
        buf += chunk
    return buf


def _read_scale_raws():
    """一次 FC3 读 addr 0 起 8 个保持寄存器，解码四个通道的有符号 32 位 raw（HH-LL）。"""
    req = struct.pack(">HHHBBHH", 1, 0, 6, SCALE_UNIT, 0x03, 0, 8)
    with socket.create_connection((SCALE_HOST, SCALE_PORT), timeout=SCALE_TIMEOUT) as sock:
        sock.settimeout(SCALE_TIMEOUT)
        sock.sendall(req)
        head = _recv_exact(sock, 9)          # MBAP(7)+功能码(1)+字节数(1)
        if head[7] & 0x80:
            raise IOError(f"Modbus 异常响应，功能码 0x{head[7]:02X}")
        data = _recv_exact(sock, head[8])
    regs = struct.unpack(">" + "H" * (head[8] // 2), data)
    raws = {}
    for i, ch in enumerate(SCALE_CHANNELS):
        high, low = regs[2 * i], regs[2 * i + 1]
        raw_u = ((high << 16) | low) & 0xFFFFFFFF
        raws[ch] = struct.unpack(">i", struct.pack(">I", raw_u))[0]
    return raws


def _scale_poller():
    """后台线程：周期性读四通道并缓存；失败整组标离线、保留上次 raw。"""
    while True:
        try:
            raws = _read_scale_raws()
            with _scale_lock:
                for ch, raw in raws.items():
                    _scale_latest[ch] = {"ok": True, "raw": raw}
        except Exception:
            with _scale_lock:
                for ch in SCALE_CHANNELS:
                    _scale_latest[ch] = {"ok": False, "raw": _scale_latest[ch].get("raw")}
        time.sleep(SCALE_POLL_INTERVAL)


threading.Thread(target=_scale_poller, daemon=True).start()


def _channel_reading(ch):
    """组装单通道读数（net=去皮净重，gross=毛重，单位克）。调用方需自持锁语义：此处各自加锁。"""
    with _scale_lock:
        st = dict(_scale_latest[ch])
    with _state_lock:
        tare = _state["tare_raw"].get(str(ch), 0)
    raw = st["raw"]
    return {
        "channel": ch,
        "ok": st["ok"],
        "net": round((raw - tare) * SCALE_DIVISION, 1) if raw is not None else None,
        "gross": round(raw * SCALE_DIVISION, 1) if raw is not None else None,
        "tare_g": round(tare * SCALE_DIVISION, 1),
    }


# ══════════════════════════════════════════════════════════════════════
# API
# ══════════════════════════════════════════════════════════════════════
@app.get("/api/health")
def api_health():
    with _scale_lock:
        online = sum(1 for ch in SCALE_CHANNELS if _scale_latest[ch]["ok"])
    return {"ok": True, "scales_online": online, "ts": time.time()}


@app.get("/api/groups")
def api_groups():
    """四条桌边的分组绑定配置（手机 / 项链 / 秤通道）。"""
    with _state_lock:
        return JSONResponse({"groups": _state["groups"]})


@app.put("/api/groups/{edge}")
def api_group_update(edge: int, patch: dict = Body(...)):
    """更新一条桌边的绑定（支持部分字段），立即落盘。"""
    if edge not in EDGES:
        return JSONResponse({"ok": False, "error": f"桌边 {edge} 不存在（合法 1~4）"},
                            status_code=404)
    unknown = [k for k in patch if k not in GROUP_EDITABLE_FIELDS]
    if unknown:
        return JSONResponse({"ok": False, "error": f"不支持的字段：{unknown}"}, status_code=400)
    if "scale_channel" in patch:
        if not isinstance(patch["scale_channel"], int) or patch["scale_channel"] not in SCALE_CHANNELS:
            return JSONResponse({"ok": False, "error": "scale_channel 必须是 1~4 的整数"},
                                status_code=400)
    for key in ("label", "phone_device_id", "necklace_device_id"):
        if key in patch and not isinstance(patch[key], str):
            return JSONResponse({"ok": False, "error": f"{key} 必须是字符串"}, status_code=400)
    with _state_lock:
        group = _group_of(edge)
        group.update({k: v for k, v in patch.items() if k in GROUP_EDITABLE_FIELDS})
        _save_state(_state)
        return JSONResponse({"ok": True, "group": dict(group)})


@app.get("/api/groups/resolve")
def api_group_resolve(device_id: str):
    """按设备号（手机或项链）反查所在分组——services 由此得知该设备对应读哪个秤通道。"""
    with _state_lock:
        for group in _state["groups"]:
            if device_id and device_id in (group["phone_device_id"], group["necklace_device_id"]):
                return JSONResponse({"ok": True, "group": dict(group)})
    return JSONResponse({"ok": False, "error": f"没有分组绑定设备 {device_id}"}, status_code=404)


@app.get("/api/food-scales")
def api_food_scales():
    """四通道实时读数。"""
    return JSONResponse({"scales": [_channel_reading(ch) for ch in SCALE_CHANNELS],
                         "ts": time.time()})


@app.post("/api/food-scales/{channel}/tare")
def api_food_scale_tare(channel: int):
    """清空（软件去皮）：把该通道当前 raw 记为皮重并落盘。"""
    if channel not in SCALE_CHANNELS:
        return JSONResponse({"ok": False, "error": f"通道 {channel} 不存在（合法 1~4）"},
                            status_code=404)
    with _scale_lock:
        st = dict(_scale_latest[channel])
    if not st["ok"] or st["raw"] is None:
        return JSONResponse({"ok": False, "error": "秤当前离线，无法去皮"}, status_code=409)
    with _state_lock:
        _state["tare_raw"][str(channel)] = st["raw"]
        _save_state(_state)
    return JSONResponse({"ok": True, "channel": channel,
                         "tare_g": round(st["raw"] * SCALE_DIVISION, 1)})
