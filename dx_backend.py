"""深体验区（superadmin ifa-support 页面）专属后端，独立于 8060 DA3 服务，跑在 8070。

职责：
1. 四通道食物秤（一台 SJ101T2_CH4_ETH 模块，通道 1..4 → 寄存器 addr 0/2/4/6）：
   后台线程轮询缓存实时读数；「清空」= 软件去皮（记当前 raw 为皮重）。
   注意 ok 只表示模块可达（四通道整组同生共死）；通道是否真插了称重传感器
   硬件报不出来（空通道浮空输入照样出稳定读数），靠人工标注 connected 区分。
2. 桌边分组绑定配置：每条桌边一组「手机 / 项链 / 秤通道」，可查可改、落盘持久化。
   这份绑定信息是深体验区的编排事实源——services 按项链 device_id 反查所在组，
   即可得知要读哪个秤通道、以及把通知推给哪台手机（见 GET /api/groups/resolve）。
   三层身份（详见 NETWORK.md）：
     · 秤   = 通道号 1..4，固定不变
     · 项链 = 蓝牙名（odyss-XXXX），随帧上报在 camera_info.device_id，跨手机稳定
     · 手机 = client_id（App 每个请求的 X-Client-Id），推送定向用它；
              不能用 FCM token 或 target_id——token 会轮换、target_id 由它派生
3. 在线项链列表（代理 8060 帧中继）与绑定变更流水，供控制面下拉选择与追踪配对历史。

持久化：dx_data.json（分组配置 + 各通道皮重）原子写（tmp+rename）；
dx_pairing_log.jsonl 追加写绑定变更流水。两者均 gitignore，不进代码仓。

零重依赖：fastapi + uvicorn（复用 da3 conda 环境），Modbus TCP 为手写 socket。
启动：./run-dx.sh 或 systemd dx-backend.service（见 deploy.sh）。
"""

import json
import os
import select
import socket
import struct
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

app = FastAPI(title="深体验区后端")
# 局域网演示服务：放开跨域，供 superadmin(18091) 等同网页面直调
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ══════════════════════════════════════════════════════════════════════
# 持久化状态：分组配置 + 皮重
# ══════════════════════════════════════════════════════════════════════
DATA_FILE = Path(__file__).resolve().parent / "dx_data.json"
# 绑定变更流水（append-only）：追踪项链/手机什么时候换到了哪条桌边
PAIRING_LOG_FILE = Path(__file__).resolve().parent / "dx_pairing_log.jsonl"
EDGES = (1, 2, 3, 4)

# 分组允许编辑的字段（除 edge 外全部可改）
GROUP_EDITABLE_FIELDS = (
    "label", "phone_no", "phone_identity", "phone_client_id", "phone_user_id",
    "phone_udid", "phone_serial", "phone_build",
    "necklace_device_id", "scale_channel",
)
# 会写入配对流水的字段（只关心「谁换到了哪条桌边」这类物理配对）
PAIRING_TRACKED_FIELDS = ("necklace_device_id", "phone_client_id")

# 在线项链来源：8060 帧中继按 camera_info.device_id 分桶维护的设备表（60s 无新帧即下线）。
# 走宿主 localhost——控制面只经 nginx /dx-api/ 访问 8070，8060 那条 ufw 放行早已移除，
# 让 8070 代理比再开一条对外通路省事。
NECKLACE_SOURCE_URL = os.environ.get(
    "NECKLACE_SOURCE_URL", "http://127.0.0.1:8060/api/frame/status")
# 取某个项链最新一帧原图（image/jpeg）。同样经本服务转发，理由见 api_necklace_frame。
NECKLACE_FRAME_URL = os.environ.get(
    "NECKLACE_FRAME_URL", "http://127.0.0.1:8060/api/frame/latest")
NECKLACE_SOURCE_TIMEOUT = 2.0
# 判定「项链在线」的最大数据龄（秒）：超过这么久没有新帧就不算在线。
#
# 故意不改帧中继自己的 DEVICE_TTL(60s)——那是它的桶清理策略，8060 的 /experience
# 多设备下拉、/panel 的选中设备回落都依赖它，调小会让那些页面在短暂停传时就丢设备。
# 这里只在本接口按更严的阈值过滤，影响面收在深体验区内。
NECKLACE_ONLINE_MAX_AGE = float(os.environ.get("NECKLACE_ONLINE_MAX_AGE", "15"))
# ifa 演示状态机（项链 ready / meal_in_progress / meal_complete）：状态机本体在
# local-stack 的 odyss-services（宿主 18090），这里只做控制面代理——控制面依旧
# 免认证走 /dx-api/，由本服务补上 services 的 service token（X-Odyss-Service-Token）。
# token 必须与 local-stack runtime-config 的 infra.http.service_tokens.ifa_demo_control 一致。
IFA_SERVICES_BASE_URL = os.environ.get(
    "IFA_SERVICES_BASE_URL", "http://127.0.0.1:18090")
IFA_DEMO_CONTROL_TOKEN = os.environ.get(
    "IFA_DEMO_CONTROL_TOKEN", "local-ifa-demo-control-token")
IFA_SERVICES_TIMEOUT = 5.0

# 帧中继对「camera_info 缺失或其中没有 device_id」的兜底桶名。
# 它不是真实身份：多个未识别设备的帧会混进同一个桶，绑定它毫无意义，故不进候选。
# 但它出现本身是个运维信号——说明有项链的蓝牙名没被 App 侧的 BleDeviceIdentityCache
# 缓存到（回落顺序：缓存 → 已连接设备 → 配置值 → "unknown"），该项链需要重连一次。
UNKNOWN_NECKLACE = "unknown"

_state_lock = threading.Lock()


def _default_group(edge):
    """一条桌边的默认绑定。三层身份的取值来源见 NETWORK.md：
    秤=通道号（固定）；项链=蓝牙名（如 odyss-0F0B）；手机=client_id（X-Client-Id）。
    """
    return {
        "edge": edge,
        "label": f"桌边 {edge}",
        "scale_channel": edge,
        # 手机编号：纯人类标签，方便现场喊「1 号机」；定向一律用 phone_client_id
        "phone_no": edge,
        # 账号（手机号/邮箱）：控制面据此查该手机的推送目标；4 台演示手机各登一个账号
        "phone_identity": "",
        # 手机唯一身份：App 每个请求的 X-Client-Id。推送定向用它，不用 FCM token
        # （token 会轮换，target_id 由 token_hash 派生也会跟着变）
        "phone_client_id": "",
        "phone_user_id": "",
        # 手机硬件 UDID（如 00008150-001D15E43478401C）：Mac 上 xcodebuild / run-ios
        # 装机的 build 目标身份，跟机身走、重装 App 不变——与 client_id 互补
        "phone_udid": "",
        # 手机序列号：机身「设置→通用→关于本机」可查，现场人肉对照哪台是几号机用
        "phone_serial": "",
        # 当前装机 build 描述（自由文本，如 "fe 75f0eb62 + hub f92b405 @ 2026-08-10"）
        "phone_build": "",
        # 项链蓝牙名，随帧上报在 camera_info.device_id
        "necklace_device_id": "",
    }


def _default_state():
    """默认状态：桌边 N 绑秤通道 N，设备身份留空待配置。"""
    return {
        "groups": [_default_group(e) for e in EDGES],
        # 皮重按通道存 raw 值（与读数同分度），服务重启不丢
        "tare_raw": {str(ch): 0 for ch in EDGES},
        # 每通道「已接秤」人工标注：模块上没插称重传感器的通道，浮空输入照样出
        # 像模像样的读数，硬件层报不出断线（实测扫遍寄存器区无每通道状态标志），
        # 只能靠现场接线的人标注。默认 True 保持旧行为（全部当已接）。
        "scale_connected": {str(ch): True for ch in EDGES},
    }


def _migrate_groups(state):
    """把旧结构升级到新结构，返回是否发生改动。

    旧版把手机记成 phone_device_id 自由文本（实际存的是 "iPhone-5 (192.168.0.243)"
    这类带 IP 的描述，IP 会变、也无法用于推送定向），这里直接丢弃并打日志留痕，
    改由 phone_client_id 承担手机身份。
    """
    changed = False
    for group in state.get("groups", []):
        legacy = group.pop("phone_device_id", None)
        if legacy:
            print(f"[迁移] 桌边 {group.get('edge')} 的旧 phone_device_id={legacy!r} 已丢弃，"
                  f"请在控制面重新绑定手机（改用 client_id 定向）", flush=True)
            changed = True
        for key, value in _default_group(group.get("edge") or 0).items():
            if key not in group:
                group[key] = value
                changed = True
    return changed


def _load_state():
    """启动加载持久化文件；文件缺失/损坏时回落默认并立即落盘。"""
    if DATA_FILE.exists():
        try:
            state = json.loads(DATA_FILE.read_text(encoding="utf-8"))
            # 结构兜底：缺 key 用默认补齐，防旧版本文件升级后崩
            default = _default_state()
            for key, value in default.items():
                state.setdefault(key, value)
            if _migrate_groups(state):
                _save_state(state)
            return state
        except Exception:
            pass
    state = _default_state()
    _save_state(state)
    return state


def _append_pairing_log(edge, field, old, new):
    """把一次绑定变更追加到配对流水。写失败只打日志，绝不影响绑定本身。"""
    try:
        record = {"ts": time.time(), "edge": edge, "field": field,
                  "old": old, "new": new}
        with PAIRING_LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"[配对流水] 写入失败（绑定已生效，仅流水缺失）：{exc}", flush=True)


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
#   <SCALE_HOST>:502  unit=1  FC3  通道 N → addr (N-1)*2
#   32 位有符号 · 字序 HH-LL · 分度 0.1（raw/10 = 克）
#
# 连接必须复用：实测该模块在新建 TCP 连接后的**第一个**请求要 6 秒左右才应答，
# 之后同一条连接上稳定在 20ms。旧实现每次读都新建连接且只给 1.2s 超时，
# 必然卡在 6 秒门槛前放弃，表现为「秤一直离线」（2026-08-05 抓包定位）。
#
# 地址随现场网络而变，用环境变量覆盖、不改代码：
#   SCALE_HOST=192.168.0.80 ./run-dx.sh
# 网络形态与各地址的由来见 NETWORK.md。
# ══════════════════════════════════════════════════════════════════════
SCALE_HOST = os.environ.get("SCALE_HOST", "192.168.100.80")
SCALE_PORT = int(os.environ.get("SCALE_PORT", "502"))
SCALE_UNIT = 1
SCALE_DIVISION = 0.1
SCALE_CHANNELS = (1, 2, 3, 4)
# 后台轮询间隔（秒）。Modbus 一次读只要 ~20ms，而模块说明书要求的最小请求间隔是 50ms，
# 取 0.2s 仍有 10 倍余量；端到端延迟里这一层贡献平均 100ms。
SCALE_POLL_INTERVAL = 0.2
SCALE_CONNECT_TIMEOUT = 5.0   # 建立 TCP 连接的超时（秒）
SCALE_FIRST_TIMEOUT = 15.0    # 新连接上首个请求的超时——设备实测约需 6s
SCALE_READ_TIMEOUT = 3.0      # 连接热起来之后的常规读超时
SCALE_MAX_BACKOFF = 5.0       # 连不上时的最大重试间隔（秒）

# read_at = 这批 raw 实际被读到的时刻，用于让调用方判断数据新鲜度
# （接口的 ts 是响应生成时刻，两者不是一回事）
_scale_latest = {ch: {"ok": False, "raw": None, "read_at": None} for ch in SCALE_CHANNELS}
_scale_lock = threading.Lock()


def _drain(sock):
    """清掉 socket 里残留的响应，再发新请求。

    该模块固定回 transaction_id=1，客户端无从识别响应错位；若上一轮的响应还躺在
    接收缓冲里，本轮就会读到那一帧陈旧数据、白白多担一轮延迟（实测连续读时出现过
    0.1ms 的「秒回」，正是读到了积压帧）。
    """
    try:
        while True:
            ready, _, _ = select.select([sock], [], [], 0)
            if not ready:
                break
            if not sock.recv(4096):
                break
    except OSError:
        pass


def _recv_exact(sock, n):
    """从 socket 精确读取 n 个字节。"""
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("连接被对端关闭")
        buf += chunk
    return buf


class _ScaleLink:
    """保持一条到秤模块的常连 Modbus TCP 连接。

    只在后台轮询线程内使用，故不额外加锁。任一次读失败都会关闭连接，
    下一轮自动重连；重连后的首个请求会再次享受 SCALE_FIRST_TIMEOUT。
    """

    def __init__(self):
        self._sock = None
        self._warm = False        # 这条连接是否已成功收到过响应

    def close(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        self._sock = None
        self._warm = False

    def read_holding(self, addr, count):
        """FC3 读保持寄存器，返回寄存器元组；失败时关闭连接并向上抛。"""
        if self._sock is None:
            self._sock = socket.create_connection(
                (SCALE_HOST, SCALE_PORT), timeout=SCALE_CONNECT_TIMEOUT)
            self._warm = False
        sock = self._sock
        sock.settimeout(SCALE_READ_TIMEOUT if self._warm else SCALE_FIRST_TIMEOUT)
        try:
            _drain(sock)   # 丢掉上一轮可能积压的响应，确保这次读到的是新鲜帧
            # 该模块固定回 transaction_id=1，这里也固定发 1，避免事务号比对错位
            sock.sendall(struct.pack(">HHHBBHH", 1, 0, 6, SCALE_UNIT, 0x03, addr, count))
            head = _recv_exact(sock, 9)      # MBAP(7)+功能码(1)+字节数(1)
            if head[7] & 0x80:
                raise IOError(f"Modbus 异常响应，异常码 0x{head[8]:02X}")
            data = _recv_exact(sock, head[8])
        except Exception:
            self.close()
            raise
        self._warm = True
        return struct.unpack(">" + "H" * (head[8] // 2), data)


_scale_link = _ScaleLink()


def _read_scale_raws():
    """一次 FC3 读 addr 0 起 8 个保持寄存器，解码四个通道的有符号 32 位 raw（HH-LL）。"""
    regs = _scale_link.read_holding(0, 8)
    raws = {}
    for i, ch in enumerate(SCALE_CHANNELS):
        high, low = regs[2 * i], regs[2 * i + 1]
        raw_u = ((high << 16) | low) & 0xFFFFFFFF
        raws[ch] = struct.unpack(">i", struct.pack(">I", raw_u))[0]
    return raws


def _scale_poller():
    """后台线程：周期性读四通道并缓存；失败整组标离线、保留上次 raw。"""
    backoff = SCALE_POLL_INTERVAL
    while True:
        try:
            raws = _read_scale_raws()
            read_at = time.time()
            with _scale_lock:
                for ch, raw in raws.items():
                    _scale_latest[ch] = {"ok": True, "raw": raw, "read_at": read_at}
            backoff = SCALE_POLL_INTERVAL
        except Exception:
            with _scale_lock:
                for ch in SCALE_CHANNELS:
                    # 掉线时保留上次 raw 与其 read_at——读数是旧的，新鲜度必须能看出来
                    _scale_latest[ch] = {"ok": False, "raw": _scale_latest[ch].get("raw"),
                                         "read_at": _scale_latest[ch].get("read_at")}
            # 秤不可达时逐步退避，避免每轮都卡在建连/首包的长超时上空转
            backoff = min(backoff * 2, SCALE_MAX_BACKOFF)
        time.sleep(backoff)


threading.Thread(target=_scale_poller, daemon=True).start()


def _channel_reading(ch):
    """组装单通道读数（net=去皮净重，gross=毛重，单位克）。调用方需自持锁语义：此处各自加锁。"""
    with _scale_lock:
        st = dict(_scale_latest[ch])
    with _state_lock:
        tare = _state["tare_raw"].get(str(ch), 0)
        connected = bool(_state["scale_connected"].get(str(ch), True))
    raw = st["raw"]
    read_at = st.get("read_at")
    return {
        "channel": ch,
        # ok 只表示「秤模块可达」——一台模块四个通道整组同生共死；
        # 该通道是否真插了秤看 connected（人工标注，见 _default_state 注释）
        "ok": st["ok"],
        "connected": connected,
        "net": round((raw - tare) * SCALE_DIVISION, 1) if raw is not None else None,
        "gross": round(raw * SCALE_DIVISION, 1) if raw is not None else None,
        "tare_g": round(tare * SCALE_DIVISION, 1),
        # 这批读数被采到的时刻，以及到现在过了多久。调用方据此判断新鲜度——
        # 响应里的 ts 只是响应生成时刻，掉线时读数会停在旧值上而 ts 照常前进。
        "read_at": read_at,
        "age_s": round(time.time() - read_at, 3) if read_at else None,
    }


# ══════════════════════════════════════════════════════════════════════
# API
# ══════════════════════════════════════════════════════════════════════
@app.get("/api/health")
def api_health():
    with _state_lock:
        connected = {ch: bool(_state["scale_connected"].get(str(ch), True))
                     for ch in SCALE_CHANNELS}
    with _scale_lock:
        # 只数「已接秤且模块可达」的通道——未接秤的空通道不算在线
        online = sum(1 for ch in SCALE_CHANNELS
                     if connected[ch] and _scale_latest[ch]["ok"])
    return {"ok": True, "scales_online": online, "ts": time.time()}


@app.get("/api/groups")
def api_groups():
    """四条桌边的分组绑定配置（手机 / 项链 / 秤通道）。"""
    with _state_lock:
        return JSONResponse({"groups": _state["groups"]})


@app.put("/api/groups/{edge}")
def api_group_update(edge: int, patch: dict = Body(...)):
    """更新一条桌边的绑定（支持部分字段），立即落盘并记配对流水。"""
    if edge not in EDGES:
        return JSONResponse({"ok": False, "error": f"桌边 {edge} 不存在（合法 1~4）"},
                            status_code=404)
    unknown = [k for k in patch if k not in GROUP_EDITABLE_FIELDS]
    if unknown:
        return JSONResponse({"ok": False, "error": f"不支持的字段：{unknown}"}, status_code=400)
    for key, allowed in (("scale_channel", SCALE_CHANNELS), ("phone_no", EDGES)):
        if key in patch and (not isinstance(patch[key], int) or patch[key] not in allowed):
            return JSONResponse({"ok": False, "error": f"{key} 必须是 1~4 的整数"},
                                status_code=400)
    for key in ("label", "phone_identity", "phone_client_id", "phone_user_id",
                "phone_udid", "phone_serial", "phone_build", "necklace_device_id"):
        if key in patch and not isinstance(patch[key], str):
            return JSONResponse({"ok": False, "error": f"{key} 必须是字符串"}, status_code=400)

    with _state_lock:
        # 同一项链 / 同一手机不能同时绑在两条桌边，否则 resolve 会出现歧义
        for key, name_cn in (("necklace_device_id", "项链"), ("phone_client_id", "手机")):
            value = str(patch.get(key, "") or "").strip()
            if not value:
                continue
            conflict = next((g for g in _state["groups"]
                             if g["edge"] != edge and g.get(key) == value), None)
            if conflict:
                return JSONResponse(
                    {"ok": False,
                     "error": f"{name_cn} {value} 已绑在桌边 {conflict['edge']}，请先在那边解绑"},
                    status_code=409)
        group = _group_of(edge)
        # 先算出物理配对的变化，落盘后再写流水（不在持锁期间做文件 IO）
        changes = [(k, group.get(k, ""), patch[k]) for k in PAIRING_TRACKED_FIELDS
                   if k in patch and patch[k] != group.get(k, "")]
        group.update({k: v for k, v in patch.items() if k in GROUP_EDITABLE_FIELDS})
        _save_state(_state)
        updated = dict(group)

    for field, old, new in changes:
        _append_pairing_log(edge, field, old, new)
    return JSONResponse({"ok": True, "group": updated})


@app.get("/api/groups/resolve")
def api_group_resolve(device_id: str = "", client_id: str = ""):
    """按项链设备号或手机 client_id 反查所在分组。

    services 的典型用法：拿到项链 device_id（camera_info.device_id）→ 得到该桌边的
    秤通道，以及要把通知推给哪台手机（phone_client_id + phone_user_id）。
    """
    device_id = (device_id or "").strip()
    client_id = (client_id or "").strip()
    if not device_id and not client_id:
        return JSONResponse({"ok": False, "error": "需要 device_id（项链）或 client_id（手机）"},
                            status_code=400)
    with _state_lock:
        for group in _state["groups"]:
            if device_id and device_id == group.get("necklace_device_id"):
                return JSONResponse({"ok": True, "group": dict(group)})
            if client_id and client_id == group.get("phone_client_id"):
                return JSONResponse({"ok": True, "group": dict(group)})
    missing = device_id or client_id
    return JSONResponse({"ok": False, "error": f"没有分组绑定 {missing}"}, status_code=404)


@app.get("/api/necklaces/online")
def api_necklaces_online():
    """在线项链列表：代理 8060 帧中继的设备表，并标注每个项链当前绑在哪条桌边。

    「在线」的唯一依据是 **NECKLACE_ONLINE_MAX_AGE 秒内有新帧到达**（默认 15s）——
    它不代表设备通电、BLE 已连接或 App 里显示绑定成功，只代表这个 device_id 最近
    确实有图片传进来。帧中继自己的桶保留 60s，这里按更严的阈值过滤。

    项链身份取 camera_info.device_id（蓝牙名，如 odyss-0F0B），跨手机稳定。
    帧中继不可达时返回空列表 + error 说明，让控制面回落到手工输入而不是报错。
    """
    devices, error = [], ""
    try:
        with urllib.request.urlopen(NECKLACE_SOURCE_URL, timeout=NECKLACE_SOURCE_TIMEOUT) as resp:
            devices = (json.loads(resp.read().decode("utf-8")) or {}).get("devices") or []
    except (urllib.error.URLError, OSError, ValueError) as exc:
        error = f"帧中继（{NECKLACE_SOURCE_URL}）不可达：{exc}"
    with _state_lock:
        bound = {g["necklace_device_id"]: g["edge"] for g in _state["groups"]
                 if g.get("necklace_device_id")}
    items, unknown_active = [], False
    for dev in devices:
        device_id = str((dev or {}).get("device_id") or "").strip()
        if not device_id:
            continue
        age = (dev or {}).get("age")
        # age 未知一律当不在线：宁可少列一个，也不要把停传的项链说成在线
        if age is None or age > NECKLACE_ONLINE_MAX_AGE:
            continue
        if device_id == UNKNOWN_NECKLACE:
            unknown_active = True   # 不进候选，但要报出去让人去修那个项链
            continue
        items.append({"device_id": device_id, "age": age, "seq": dev.get("seq"),
                      "fps": dev.get("fps"), "bound_edge": bound.get(device_id)})
    return JSONResponse({"necklaces": items, "error": error,
                         "max_age_s": NECKLACE_ONLINE_MAX_AGE,
                         "unknown_active": unknown_active, "ts": time.time()})


@app.get("/api/necklaces/frame")
def api_necklace_frame(device: str = ""):
    """代理某个项链的最新帧原图（image/jpeg），供控制面的 <img> 直接展示。

    为什么经本服务转发、而不让控制面直连 8060：superadmin 的 nginx 只反代了
    /dx-api/ → 宿主 8070，8060 对容器网络的 ufw 放行早已移除，加回去要动部署机
    防火墙与 nginx 配置。而这里的帧很小（几十 KB、1~2 fps），转发开销可忽略。

    404 原样透出（该项链暂无帧），让前端显示占位而不是当成错误。
    """
    device = (device or "").strip()
    if not device:
        return JSONResponse({"ok": False, "error": "需要 device 参数"}, status_code=400)
    url = "%s?device=%s" % (NECKLACE_FRAME_URL, urllib.parse.quote(device))
    try:
        with urllib.request.urlopen(url, timeout=NECKLACE_SOURCE_TIMEOUT) as resp:
            data = resp.read()
            content_type = resp.headers.get("Content-Type") or "image/jpeg"
            seq = resp.headers.get("X-Frame-Seq") or "0"
    except urllib.error.HTTPError as exc:
        return JSONResponse({"ok": False, "error": "该项链暂无帧"}, status_code=exc.code)
    except (urllib.error.URLError, OSError) as exc:
        return JSONResponse({"ok": False, "error": "帧中继不可达：%s" % exc}, status_code=502)
    # no-store：帧是一直在变的，缓存住就成了静态图
    return Response(content=data, media_type=content_type,
                    headers={"Cache-Control": "no-store", "X-Frame-Seq": seq})


def _ifa_services_request(method, path, body=None):
    """调 services 的 ifa 端点（演示状态机/餐段），统一补 service token 与错误封装。

    返回 (payload, status)：payload 是 services 响应 data 段（或错误说明），
    status 是要透传给控制面的 HTTP 状态码。services 不可达按 502 报出。
    """
    url = IFA_SERVICES_BASE_URL + path
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "X-Odyss-Service-Token": IFA_DEMO_CONTROL_TOKEN,
        "Content-Type": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=IFA_SERVICES_TIMEOUT) as resp:
            payload = json.loads(resp.read().decode("utf-8")) or {}
            return payload.get("data") or {}, resp.status
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode("utf-8")).get("msg") or str(exc)
        except (ValueError, OSError):
            detail = str(exc)
        return {"error": detail}, exc.code
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return {"error": "services（%s）不可达：%s" % (IFA_SERVICES_BASE_URL, exc)}, 502


@app.get("/api/necklaces/{device_id}/meal-state")
def api_necklace_meal_state_get(device_id: str):
    """查询某条项链的演示状态机状态（tracked=false 表示未参与演示）。"""
    device_id = (device_id or "").strip()
    if not device_id:
        return JSONResponse({"ok": False, "error": "需要 device_id"}, status_code=400)
    data, status = _ifa_services_request(
        "GET", "/api/v1/ifa/devices/%s/meal-state" % urllib.parse.quote(device_id))
    ok = status < 400
    return JSONResponse({"ok": ok, **data}, status_code=status if not ok else 200)


@app.put("/api/necklaces/{device_id}/meal-state")
def api_necklace_meal_state_set(device_id: str, body: dict = Body(...)):
    """设置某条项链的演示状态（控制面用：设 ready 与 reset 都是 state=ready）。"""
    device_id = (device_id or "").strip()
    if not device_id:
        return JSONResponse({"ok": False, "error": "需要 device_id"}, status_code=400)
    state = str((body or {}).get("state") or "").strip()
    if not state:
        return JSONResponse({"ok": False, "error": "需要 state"}, status_code=400)
    data, status = _ifa_services_request(
        "PUT", "/api/v1/ifa/devices/%s/meal-state" % urllib.parse.quote(device_id), {"state": state})
    ok = status < 400
    return JSONResponse({"ok": ok, **data}, status_code=status if not ok else 200)


@app.get("/api/necklaces/{device_id}/meal-segments")
def api_necklace_meal_segments(device_id: str, limit: int = 10):
    """列出某条项链最近的演示餐段（meal 分析链的取帧时间窗）。"""
    device_id = (device_id or "").strip()
    if not device_id:
        return JSONResponse({"ok": False, "error": "需要 device_id"}, status_code=400)
    data, status = _ifa_services_request(
        "GET", "/api/v1/ifa/devices/%s/meal-segments?limit=%d" % (urllib.parse.quote(device_id), limit))
    ok = status < 400
    return JSONResponse({"ok": ok, **data}, status_code=status if not ok else 200)


@app.post("/api/necklaces/{device_id}/meal-segments/{segment_id}/analyze")
def api_necklace_meal_segment_analyze(device_id: str, segment_id: str):
    """手动触发某段餐段的整餐分析（segment_id 可为 latest）。受理即返回，
    分析在 services 后台执行，结果落 meal fact。"""
    device_id = (device_id or "").strip()
    segment_id = (segment_id or "latest").strip()
    if not device_id:
        return JSONResponse({"ok": False, "error": "需要 device_id"}, status_code=400)
    # segment_id 形如 ifa-seg:<device>:<millis>，冒号必须原样透传：默认 quote 会把它
    # 编成 %3A，services 侧路由参数拿到的就不是原始 id，查不到段。
    data, status = _ifa_services_request(
        "POST", "/api/v1/ifa/devices/%s/meal-segments/%s/analyze" % (
            urllib.parse.quote(device_id), urllib.parse.quote(segment_id, safe=":")))
    ok = status < 400
    return JSONResponse({"ok": ok, **data}, status_code=status if not ok else 202)


@app.get("/api/pairing-log")
def api_pairing_log(limit: int = 50):
    """最近的绑定变更流水（倒序）：追踪一天里项链/手机什么时候换到了哪条桌边。"""
    limit = max(1, min(limit, 500))
    if not PAIRING_LOG_FILE.exists():
        return JSONResponse({"records": []})
    try:
        lines = PAIRING_LOG_FILE.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return JSONResponse({"records": [], "error": f"流水读取失败：{exc}"})
    records = []
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except ValueError:
            continue
        if len(records) >= limit:
            break
    return JSONResponse({"records": records})


@app.get("/api/food-scales")
def api_food_scales():
    """四通道实时读数。"""
    return JSONResponse({"scales": [_channel_reading(ch) for ch in SCALE_CHANNELS],
                         "ts": time.time()})


@app.put("/api/food-scales/{channel}/connected")
def api_food_scale_connected(channel: int, patch: dict = Body(...)):
    """标注某通道是否真插了称重传感器（现场接线事实，硬件报不出来只能人标）。"""
    if channel not in SCALE_CHANNELS:
        return JSONResponse({"ok": False, "error": f"通道 {channel} 不存在（合法 1~4）"},
                            status_code=404)
    if not isinstance(patch.get("connected"), bool):
        return JSONResponse({"ok": False, "error": "connected 必须是布尔值"}, status_code=400)
    with _state_lock:
        _state["scale_connected"][str(channel)] = patch["connected"]
        _save_state(_state)
    return JSONResponse({"ok": True, "channel": channel, "connected": patch["connected"]})


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
