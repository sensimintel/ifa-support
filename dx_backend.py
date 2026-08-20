"""深体验区（superadmin ifa-support 页面）专属后端，独立于 8060 DA3 服务，跑在 8070。

职责：
1. 四通道食物秤（一台 SJ101T2_CH4_ETH 模块，通道 1..4 → 寄存器 addr 0/2/4/6）：
   后台线程轮询缓存实时读数；「清空」= 软件去皮（记当前 raw 为皮重）。
   注意 ok 只表示模块可达（四通道整组同生共死）；通道是否真插了称重传感器
   硬件报不出来（空通道浮空输入照样出稳定读数），靠人工标注 connected 区分。
2. 设备编排，分两层存：
   · **手机台账（phones）**：一台手机一行、机号做主键的全局资产表，与桌边无关，
     可以多于四行（备用机不占桌边）。手填的只有机号 / 序列号 / 账号三样。
   · **桌边分组（groups）**：每条桌边一组「机号 / 项链 / 秤通道」，手机只写一个机号
     引用台账，身份读时展开。
   拆两层是因为「这台手机是谁」与「它今天摆在哪条边」是两件事：旧结构把身份抄在每条
   桌边里，六个字段各自能改、彼此无约束，换一次手机要同时改六处，改漏一处就分叉。
   这份编排是深体验区的事实源——秤事件按「通道→桌边→项链」反查后才带得上归属，
   绑错就是把克数记到别人的餐上（见 _necklace_of_channel 与 GET /api/groups/resolve）。
   三层身份（详见 NETWORK.md）：
     · 秤   = 通道号 1..4，固定不变
     · 项链 = 蓝牙名（odyss-XXXX），随帧上报在 camera_info.device_id，跨手机稳定
     · 手机 = 机号（现场喊的「3 号机」）指向台账；台账里序列号认机身、
              client_id（X-Client-Id）认 App 身份。推送定向用 client_id，
              不能用 FCM token 或 target_id——token 会轮换、target_id 由它派生
3. 在线项链列表（代理 8060 帧中继）与绑定变更流水，供控制面下拉选择与追踪配对历史。

持久化：dx_data.json（分组配置 + 各通道皮重）原子写（tmp+rename）；
dx_pairing_log.jsonl 追加写绑定变更流水。两者均 gitignore，不进代码仓。

零重依赖：fastapi + uvicorn（复用 da3 conda 环境），Modbus TCP 为手写 socket。
启动：./run-dx.sh 或 systemd dx-backend.service（见 deploy.sh）。
"""

import collections
import datetime
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

import dx_scale_events as sev
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

# 分组允许编辑的字段。手机身份不在其中——桌边只写一个机号引用台账（见 PHONES 段注释），
# 「这台手机是谁」只有台账那一处能改，避免同一个事实在四条桌边里各存一份、各改各的。
GROUP_EDITABLE_FIELDS = ("label", "phone_no", "necklace_device_id", "scale_channel")
# 桌边读出去时从台账展开的手机字段。**只读**：写入口一律走 /api/phones，
# 这里保留旧字段名是为了不惊动下游（ios-build 拿 phone_udid/phone_serial 定位设备、
# 深区页面显示 phone_client_id），换成新名字得同时改好几处，收益却是零。
PHONE_VIEW_FIELDS = ("phone_identity", "phone_client_id", "phone_user_id",
                     "phone_udid", "phone_serial", "phone_build", "phone_device_name")
# 会写入配对流水的字段（只关心「谁换到了哪条桌边」这类物理配对）。
# scale_channel 也在其中：秤事件按「通道→桌边→项链」实时反查、没有快照，
# 演示中途改通道会把前后事件分给两条不同项链，事后只能靠流水看出这里动过手。
PAIRING_TRACKED_FIELDS = ("necklace_device_id", "phone_no", "scale_channel")

# ══════════════════════════════════════════════════════════════════════
# 手机台账（phones）：一台手机一行，全局资产，不属于任何一条桌边
# ══════════════════════════════════════════════════════════════════════
# **主键是机号**（现场喊的「3 号机」），不是序列号。用机号做主键有个直接好处：
# 「机号唯一」由主键天然保证，撞号的数据根本存不进来——旧结构里两条桌边都写着
# 「3 号机」，光看编号分不出哪台是哪台，白跑过几轮装机。
#
# 序列号是这一行的属性而非主键：它确实是唯一确定一台机身的东西（机身「设置→通用→
# 关于本机」可查），但现场没人按序列号喊人。两者的分工是「机号用来指认，序列号用来核对」。
#
# 手填的只有三样：机号 / 序列号 / 账号。udid 与 device_name 装机时由 Mac 回填
# （xcrun devicectl 一条命令同时给出两者），resolved 那组由控制面按账号解析。
PHONE_EDITABLE_FIELDS = ("no", "serial", "identity", "udid", "device_name",
                         "build", "resolved")
# 台账里必须互不重复的属性（空值不参与判重：还没填的行不算撞车）
PHONE_UNIQUE_FIELDS = (("serial", "序列号"), ("identity", "账号"))
# 解析结果（控制面按账号查 services 的推送目标后回写）。这一组刻意不做校验：
# 它们是 services 的事实的副本，本服务无从判断对错，存下来只为让页面能显示、能比对。
PHONE_RESOLVED_FIELDS = ("user_id", "client_id", "platform", "last_seen_at", "resolved_at")

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
# ifa 演示状态机（项链 standby / ready / meal_in_progress / analyzing /
# report_published / failed，一次「开轮→开餐→分析→出报告」为一轮 cycle；
# standby 是控制面「归零」的落点，那时还没有人认领这一轮，进食信号不算数）：
# 状态机本体在 local-stack 的 odyss-services
# （宿主 18090），这里只做控制面代理——控制面依旧
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
        # 手机：只存机号，指向台账里那一行。0 = 这条桌边还没摆手机。
        # 身份（序列号 / 账号 / client_id / UDID / build）一律从台账读时展开，
        # 不在这里存副本——存了就会有两份各自能改的事实，迟早对不上。
        "phone_no": 0,
        # 项链蓝牙名，随帧上报在 camera_info.device_id
        "necklace_device_id": "",
    }


def _default_phone(no):
    """台账里一台手机的默认行。手填的只有 no / serial / identity 三样。"""
    return {
        # 机号：主键，也是桌边引用它的键。现场喊的「3 号机」就是它
        "no": no,
        # 机身序列号：唯一确定一台机身，「设置→通用→关于本机」可查
        "serial": "",
        # 这台登的账号（手机号 / 邮箱）：控制面据此解析出推送目标
        "identity": "",
        # 硬件 UDID：Mac 上 run-ios 的装机目标。与 device_name 一起由装机流程回填
        # （xcrun devicectl device info details 一条命令同时给出两者）
        "udid": "",
        # 设备名（如 "Xiao 17 iPhone"）：现场认机用，同样装机时回填
        "device_name": "",
        # 当前装机 build 描述（自由文本）
        "build": "",
        # 按账号解析出来的 services 侧事实，控制面查完回写。空 dict = 还没解析过
        "resolved": {},
    }


# 一次性机号纠正表：旧结构的 phone_no 没有唯一性校验，桌边 2 与桌边 3 都写着「3 号机」，
# 迁移到「机号做主键」时撞号的数据进不来，必须先裁决。
# 裁决依据是序列号而不是猜——这份对应关系由用户 2026-08-19 当面确认，并写在 ios-build 台账里。
# 迁移跑过一次之后这张表就没用了，下次动这块代码时可以直接删掉。
_LEGACY_PHONE_NO_BY_SERIAL = {
    "LXWVK71CP9": 1,
    "MVM4N0XTYQ": 2,
    "HK3H3FK6KW": 3,
    "FMKMQ62JK0": 4,
    "DCJWF5W0M4": 5,   # 备用机，不占桌边
}


def _default_state():
    """默认状态：桌边 N 绑秤通道 N，设备身份留空待配置。"""
    return {
        "groups": [_default_group(e) for e in EDGES],
        # 手机台账：全局资产表，一台一行，可以多于四行（备用机不占桌边）
        "phones": [],
        # 皮重按通道存 raw 值（与读数同分度），服务重启不丢
        "tare_raw": {str(ch): 0 for ch in EDGES},
        # 每通道「已接秤」人工标注：模块上没插称重传感器的通道，浮空输入照样出
        # 像模像样的读数，硬件层报不出断线（实测扫遍寄存器区无每通道状态标志），
        # 只能靠现场接线的人标注。默认 True 保持旧行为（全部当已接）。
        "scale_connected": {str(ch): True for ch in EDGES},
    }


def _migrate_groups(state):
    """把旧结构升级到新结构，返回是否发生改动。

    两轮历史包袱：
      · 更早的版本把手机记成 phone_device_id 自由文本（存的是 "iPhone-5 (192.168.0.243)"
        这类带 IP 的描述，IP 会变、也无法用于推送定向），直接丢弃并打日志留痕。
      · 上一版把手机身份的六个字段抄在每条桌边里（phone_serial / phone_udid /
        phone_identity / phone_client_id / phone_user_id / phone_build）。它们现在搬进台账，
        由 _migrate_phones 先搬走，这里只负责把桌边上的残留字段清干净。
    """
    changed = False
    for group in state.get("groups", []):
        legacy = group.pop("phone_device_id", None)
        if legacy:
            print(f"[迁移] 桌边 {group.get('edge')} 的旧 phone_device_id={legacy!r} 已丢弃，"
                  f"请在控制面重新绑定手机（改用机号引用台账）", flush=True)
            changed = True
        # 台账搬完后，桌边上抄的那份手机身份就是死数据，留着只会让人以为还能在这改
        for key in PHONE_VIEW_FIELDS:
            if group.pop(key, None) is not None:
                changed = True
        for key, value in _default_group(group.get("edge") or 0).items():
            if key not in group:
                group[key] = value
                changed = True
    return changed


def _migrate_phones(state):
    """把桌边里抄的手机身份搬进台账，返回是否发生改动。

    只在台账还不存在时搬一次（已有台账说明搬过了，再搬会拿旧数据盖掉后来的编辑）。

    机号撞号必须在这里裁决：旧的 phone_no 没有唯一性校验，桌边 2 与桌边 3 都写着 3，
    而机号是台账主键，撞号的两行合不成两台机。裁决依据是序列号（见
    _LEGACY_PHONE_NO_BY_SERIAL），认不出序列号的才退回原编号，再撞就顺延到下一个空号——
    顺延是最后兜底，会打日志要求人去核对，而不是让它悄悄顶着一个错编号跑下去。
    """
    if "phones" in state:
        return False
    phones, used_nos, changed = [], set(), False
    for group in state.get("groups", []):
        serial = str(group.get("phone_serial") or "").strip()
        legacy = {key: group.get(key) for key in PHONE_VIEW_FIELDS}
        # 一条身份都没填过的桌边不生成台账行，免得凭空多出几台不存在的空手机。
        # 它旧的 phone_no 也必须清掉——那是个没有台账行撑着的编号，留着就是一条悬空引用。
        if not any(str(value or "").strip() for value in legacy.values()):
            if group.get("phone_no"):
                group["phone_no"] = 0
                changed = True
            continue
        wanted = _LEGACY_PHONE_NO_BY_SERIAL.get(serial) or int(group.get("phone_no") or 0)
        no = wanted
        if no <= 0 or no in used_nos:
            no = next(n for n in range(1, 100) if n not in used_nos)
            print(f"[迁移] 桌边 {group.get('edge')} 的机号 {wanted} 无法使用"
                  f"（重复或缺失），暂定为 {no} 号机，序列号 {serial or '未填'}；"
                  f"请在控制面核对这一行", flush=True)
        elif no != int(group.get("phone_no") or 0):
            print(f"[迁移] 桌边 {group.get('edge')} 的机号按序列号 {serial} 纠正："
                  f"{group.get('phone_no')} → {no}", flush=True)
        used_nos.add(no)
        phone = _default_phone(no)
        phone.update({
            "serial": serial,
            "identity": str(group.get("phone_identity") or "").strip(),
            "udid": str(group.get("phone_udid") or "").strip(),
            "build": str(group.get("phone_build") or ""),
        })
        # 旧结构里 client_id 与 user_id 是查询出来的结果，原样搬进 resolved；
        # 解析时刻旧数据里没有，留空即可——控制面会显示「未解析过」提示人刷一次
        for key, value in (("client_id", group.get("phone_client_id")),
                           ("user_id", group.get("phone_user_id"))):
            if str(value or "").strip():
                phone["resolved"][key] = str(value).strip()
        phones.append(phone)
        group["phone_no"] = no
        changed = True
    state["phones"] = sorted(phones, key=lambda item: item["no"])
    return changed or bool(phones)


def _load_state():
    """启动加载持久化文件；文件缺失/损坏时回落默认并立即落盘。"""
    if DATA_FILE.exists():
        try:
            state = json.loads(DATA_FILE.read_text(encoding="utf-8"))
            # 台账迁移必须跑在结构兜底之前：兜底会把 phones 补成空数组，
            # 而「有没有 phones 这个键」正是判断搬没搬过的唯一依据，补完就再也搬不动了
            changed = _migrate_phones(state)
            # 结构兜底：缺 key 用默认补齐，防旧版本文件升级后崩
            default = _default_state()
            for key, value in default.items():
                state.setdefault(key, value)
            if _migrate_groups(state) or changed:
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


def _phone_of(no):
    """按机号取台账行；没有返回 None。调用方需自持 _state_lock。"""
    if not no:
        return None
    return next((p for p in _state.get("phones", []) if p.get("no") == no), None)


def _group_view(group):
    """桌边对外的样子：自身字段 + 从台账展开的手机身份。

    展开而不是让调用方自己去查台账，是为了让 GET /api/groups 的响应形状与旧版一致——
    ios-build 靠 phone_udid/phone_serial 定位设备、深区页面显示 phone_client_id，
    它们都不该因为存储结构变了而跟着改。**这些字段只读**：写入口只有 /api/phones。
    """
    view = dict(group)
    phone = _phone_of(group.get("phone_no"))
    resolved = (phone or {}).get("resolved") or {}
    view.update({
        "phone_identity": (phone or {}).get("identity", ""),
        "phone_client_id": resolved.get("client_id", ""),
        "phone_user_id": resolved.get("user_id", ""),
        "phone_udid": (phone or {}).get("udid", ""),
        "phone_serial": (phone or {}).get("serial", ""),
        "phone_build": (phone or {}).get("build", ""),
        "phone_device_name": (phone or {}).get("device_name", ""),
    })
    return view


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
# 两条后台线程（秤轮询、事件上报）在模块导入时就起跑，这对测试是灾难：轮询会去连真的
# 秤模块并在连不上时给检测器打 gap，上报线程会异步把事件队列搬空再退回——断言看到的是
# 一个自己会动的状态，跑得慢一点结果就变了。DX_BACKGROUND_THREADS=0 把它们按住，
# 只留纯函数与接口，测试自己喂数据。生产不设这个变量，行为不变。
BACKGROUND_THREADS = os.environ.get("DX_BACKGROUND_THREADS", "1") not in ("0", "false", "False")

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


# ══════════════════════════════════════════════════════════════════════
# 秤事件：平台检测 → 本地留痕 → 上报 services
# ══════════════════════════════════════════════════════════════════════
# 秤读数本身只回答「此刻多重」，而餐段要的是「每次变化了多少、变化了多大范围」。
# 切变化的逻辑在 dx_scale_events.py（纯逻辑、有单测），这里只负责喂样本、留痕、上报。
#
# 三条刻意的设计：
#   · **全程录，不看餐段边界**。餐段起止在 services，dx 这边压根不知道；而且全程录
#     之后，改阈值、重跑分析、甚至改餐段边界都不用重新采一次数据。
#   · **只上报原始变化，不上报分类**。是蓝莓还是爆米花由 services 按可调阈值读时算；
#     在这里判死了，现场调完阈值就再也算不回历史。
#   · **检测跑毛重不跑净重**。软件去皮会让净重瞬间跳几百克，跑净重会凭空造出一次
#     几百克的假取食；而去皮量在相邻两个平台的差里本来就抵消掉了。
SCALE_STABLE_WINDOW_S = float(os.environ.get("SCALE_STABLE_WINDOW_S", "1.0"))
SCALE_STABLE_EPSILON_G = float(os.environ.get("SCALE_STABLE_EPSILON_G", "0.2"))
SCALE_LIFT_THRESHOLD_G = float(os.environ.get("SCALE_LIFT_THRESHOLD_G", "20.0"))
SCALE_ABSENT_TIMEOUT_S = float(os.environ.get("SCALE_ABSENT_TIMEOUT_S", "300"))
# 原始采样落盘，供离线回放调参（第一版阈值一定是错的，没有回放就只能靠现场重现）。
# 只写已接秤的通道，5 Hz 一天约 20 MB/通道，按日切文件，旧文件由运维清理。
SCALE_RAW_LOG = os.environ.get("SCALE_RAW_LOG", "1") not in ("0", "false", "False")
SCALE_RAW_LOG_DIR = Path(os.environ.get("SCALE_RAW_LOG_DIR",
                                        str(Path(__file__).resolve().parent)))
# 上报积压上限：services 长时间不可达时最多缓这么多条，超了丢最旧的。
# 保新不保全是有意的——现场关心「刚才那几口记上没有」，一小时前的积压补上去也没人看。
SCALE_UPLOAD_QUEUE_MAX = int(os.environ.get("SCALE_UPLOAD_QUEUE_MAX", "2000"))
SCALE_UPLOAD_BATCH = 50
SCALE_EVENT_RING = 500          # 控制面查「最近事件」用的内存环
SCALE_EVENTS_PATH = "/api/v1/ifa/scale-events"

_scale_detectors = {
    ch: sev.ScaleEventDetector(
        ch,
        sample_interval_s=SCALE_POLL_INTERVAL,
        stable_window_s=SCALE_STABLE_WINDOW_S,
        stable_epsilon_g=SCALE_STABLE_EPSILON_G,
        lift_threshold_g=SCALE_LIFT_THRESHOLD_G,
        absent_timeout_s=SCALE_ABSENT_TIMEOUT_S,
    )
    for ch in SCALE_CHANNELS
}
_scale_events_recent = collections.deque(maxlen=SCALE_EVENT_RING)
_scale_upload_queue = collections.deque()
_scale_events_lock = threading.Lock()
_scale_upload_wake = threading.Event()
_scale_online = True
_scale_upload_dropped = 0
_raw_log_warned = False


def _necklace_of_channel(ch):
    """按秤通道反查绑在同一条桌边上的项链蓝牙名；没绑返回空串。"""
    with _state_lock:
        for group in _state["groups"]:
            if group.get("scale_channel") == ch:
                return (group.get("necklace_device_id") or "").strip()
    return ""


def _iso_utc(ts):
    """epoch 秒 → services 认的 RFC3339 UTC 串。"""
    return datetime.datetime.fromtimestamp(
        ts, datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def _raw_log_path(ts):
    return SCALE_RAW_LOG_DIR / ("dx_scale_raw_%s.jsonl" % datetime.datetime.fromtimestamp(
        ts, datetime.timezone.utc).strftime("%Y%m%d"))


def _append_raw_samples(read_at, samples):
    """把已接秤通道的原始采样追加到当日文件。失败只提醒一次，绝不影响检测。"""
    global _raw_log_warned
    if not SCALE_RAW_LOG or not samples:
        return
    try:
        line = json.dumps({"ts": round(read_at, 3), "g": samples}, ensure_ascii=False)
        with _raw_log_path(read_at).open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        _raw_log_warned = False
    except OSError as exc:
        if not _raw_log_warned:
            print(f"[秤原始采样] 落盘失败（检测与上报不受影响）：{exc}", flush=True)
            _raw_log_warned = True


def _emit_scale_events(events):
    """给检测器吐出来的事件补上归属，留进内存环并排进上报队列。"""
    global _scale_upload_dropped
    if not events:
        return
    enriched = []
    for event in events:
        record = dict(event)
        record["device_id"] = _necklace_of_channel(event["channel"])
        record["started_at_iso"] = _iso_utc(event["started_at"])
        record["occurred_at_iso"] = _iso_utc(event["occurred_at"])
        enriched.append(record)
    with _scale_events_lock:
        for record in enriched:
            _scale_events_recent.append(record)
            # 没绑项链的通道照样留痕，但不上报：services 按项链归属，
            # 硬塞一条空 device_id 只会往事实表里灌无主数据
            if not record["device_id"]:
                continue
            if len(_scale_upload_queue) >= SCALE_UPLOAD_QUEUE_MAX:
                _scale_upload_queue.popleft()
                _scale_upload_dropped += 1
            _scale_upload_queue.append(record)
    _scale_upload_wake.set()
    for record in enriched:
        print("[秤事件] 通道%s %s %s→%s Δ=%s g 项链=%s" % (
            record["channel"], record["kind"], record["before_g"], record["after_g"],
            record["delta_g"], record["device_id"] or "未绑定"), flush=True)


def _feed_detectors(read_at, raws):
    """把这一轮读数喂给各通道检测器（跑毛重），并落原始采样。"""
    global _scale_online
    if not _scale_online:
        # 掉线期间发生的事情无从归属：可能有人吃了半碗，也可能什么都没动。
        # 与其把重连后的读数差当成一次取食，不如显式记一条 resync 说明这里有个洞。
        for ch in SCALE_CHANNELS:
            _scale_detectors[ch].mark_gap(read_at)
        _scale_online = True
    with _state_lock:
        connected = {ch: bool(_state["scale_connected"].get(str(ch), True))
                     for ch in SCALE_CHANNELS}
    events, samples = [], {}
    for ch, raw in sorted(raws.items()):
        # 没插传感器的通道浮空输入照样出稳定读数，喂进去只会造一堆无主事件
        if not connected.get(ch, True):
            continue
        gross = round(raw * SCALE_DIVISION, 1)
        samples[str(ch)] = gross
        events.extend(_scale_detectors[ch].feed(read_at, gross))
    _append_raw_samples(read_at, samples)
    _emit_scale_events(events)


def _mark_detector_gap():
    """整组掉线：打断档标记，重连后由 _feed_detectors 落实成一条 resync。"""
    global _scale_online
    _scale_online = False


def _scale_upload_payload(batch):
    """组装上报请求体。services 只认这 8 个字段——本地留痕多带的 duration_s、
    两个 epoch 时刻等都是排障用的，不往事实表里塞。"""
    return {"events": [{
        "device_id": r["device_id"],
        "scale_channel": r["channel"],
        "kind": r["kind"],
        "started_at": r["started_at_iso"],
        "occurred_at": r["occurred_at_iso"],
        "before_g": r["before_g"],
        "after_g": r["after_g"],
        "delta_g": r["delta_g"],
    } for r in batch]}


def _scale_uploader():
    """把积压事件批量 POST 给 services；可重试的失败原样退回队首并退避。"""
    backoff = 1.0
    while True:
        _scale_upload_wake.wait(timeout=5.0)
        _scale_upload_wake.clear()
        while True:
            with _scale_events_lock:
                batch = [_scale_upload_queue.popleft()
                         for _ in range(min(SCALE_UPLOAD_BATCH, len(_scale_upload_queue)))]
            if not batch:
                backoff = 1.0
                break
            _, status = _ifa_services_request(
                "POST", SCALE_EVENTS_PATH, _scale_upload_payload(batch))
            if status < 400:
                backoff = 1.0
                continue
            if 400 <= status < 500:
                # 4xx 是这批数据本身的问题，重试多少次都一样；留日志然后丢掉，
                # 否则一条坏记录会把后面所有事件永远堵在队列里
                print("[秤事件] services 拒收 %d 条（status=%s），丢弃" % (len(batch), status),
                      flush=True)
                continue
            with _scale_events_lock:
                _scale_upload_queue.extendleft(reversed(batch))
            print("[秤事件] 上报失败 status=%s，%d 条退回队列，%.0fs 后重试" % (
                status, len(batch), backoff), flush=True)
            time.sleep(backoff)
            backoff = min(backoff * 2, 30.0)
            break


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
            # 检测放在缓存之后：读数先可见，事件慢一步不影响控制面实时读数
            _feed_detectors(read_at, raws)
            backoff = SCALE_POLL_INTERVAL
        except Exception:
            with _scale_lock:
                for ch in SCALE_CHANNELS:
                    # 掉线时保留上次 raw 与其 read_at——读数是旧的，新鲜度必须能看出来
                    _scale_latest[ch] = {"ok": False, "raw": _scale_latest[ch].get("raw"),
                                         "read_at": _scale_latest[ch].get("read_at")}
            _mark_detector_gap()
            # 秤不可达时逐步退避，避免每轮都卡在建连/首包的长超时上空转
            backoff = min(backoff * 2, SCALE_MAX_BACKOFF)
        time.sleep(backoff)


if BACKGROUND_THREADS:
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
    """四条桌边的分组绑定配置（手机 / 项链 / 秤通道）。

    手机身份从台账展开后一并返回，形状与旧版一致——下游（ios-build 取 UDID/序列号、
    深区页面显示 client_id）不用感知存储结构变了。要改这些值请走 /api/phones。
    """
    with _state_lock:
        return JSONResponse({"groups": [_group_view(g) for g in _state["groups"]]})


@app.put("/api/groups/{edge}")
def api_group_update(edge: int, patch: dict = Body(...)):
    """更新一条桌边的绑定（支持部分字段），立即落盘并记配对流水。

    只接受四个字段：label / phone_no / necklace_device_id / scale_channel。
    手机身份字段一律拒绝并指路台账——「这台手机是谁」只有台账那一处能改。
    """
    if edge not in EDGES:
        return JSONResponse({"ok": False, "error": f"桌边 {edge} 不存在（合法 1~4）"},
                            status_code=404)
    stale = [k for k in patch if k in PHONE_VIEW_FIELDS]
    if stale:
        return JSONResponse(
            {"ok": False,
             "error": f"手机身份不在桌边上改：{stale} 属于手机台账，请改 /api/phones，"
                      f"桌边这边只需选机号（phone_no）"},
            status_code=400)
    unknown = [k for k in patch if k not in GROUP_EDITABLE_FIELDS]
    if unknown:
        return JSONResponse({"ok": False, "error": f"不支持的字段：{unknown}"}, status_code=400)
    # 0 = 这条桌边不接秤。它不只是「可以没有」——四条桌边占满四路通道时，
    # 想把两条边的通道对调就必须先有一个地方能把通道放下，否则中间态必然撞唯一性。
    if "scale_channel" in patch and (not isinstance(patch["scale_channel"], int)
                                     or isinstance(patch["scale_channel"], bool)
                                     or patch["scale_channel"] not in (0,) + SCALE_CHANNELS):
        return JSONResponse({"ok": False, "error": "scale_channel 必须是 0~4 的整数（0 = 不接秤）"},
                            status_code=400)
    # phone_no 不再限制 1~4：它是台账主键，而台账放得下不占桌边的备用机。
    # 0 是有意义的取值——这条桌边还没摆手机。
    if "phone_no" in patch and (not isinstance(patch["phone_no"], int)
                                or isinstance(patch["phone_no"], bool)
                                or patch["phone_no"] < 0):
        return JSONResponse({"ok": False, "error": "phone_no 必须是非负整数（0 = 未绑手机）"},
                            status_code=400)
    for key in ("label", "necklace_device_id"):
        if key in patch and not isinstance(patch[key], str):
            return JSONResponse({"ok": False, "error": f"{key} 必须是字符串"}, status_code=400)

    with _state_lock:
        if patch.get("phone_no") and not _phone_of(patch["phone_no"]):
            return JSONResponse(
                {"ok": False,
                 "error": f"台账里没有 {patch['phone_no']} 号机，请先在手机台账里登记这台机器"},
                status_code=400)
        # 同一项链 / 同一手机 / 同一路秤都不能同时挂在两条桌边上。
        # 秤通道这条尤其要紧：秤事件按「通道→桌边→项链」反查、命中第一条就返回，
        # 两条桌边共用一路时，那一路的克数会静默记到编号靠前那条边的项链上。
        for key, name_cn in (("necklace_device_id", "项链"), ("phone_no", "手机"),
                             ("scale_channel", "秤通道")):
            value = patch.get(key)
            if isinstance(value, str):
                value = value.strip()
            if not value:      # 空串 / 0 / None 都表示解绑，不参与判重
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
        updated = _group_view(group)

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
            view = _group_view(group)
            if device_id and device_id == view.get("necklace_device_id"):
                return JSONResponse({"ok": True, "group": view})
            # client_id 现在存在台账的 resolved 里，靠展开后的视图比对
            if client_id and client_id == view.get("phone_client_id"):
                return JSONResponse({"ok": True, "group": view})
    missing = device_id or client_id
    return JSONResponse({"ok": False, "error": f"没有分组绑定 {missing}"}, status_code=404)


# ══════════════════════════════════════════════════════════════════════
# 手机台账：一台手机一行，机号做主键
# ══════════════════════════════════════════════════════════════════════
def _phone_view(phone):
    """台账行对外的样子：自身字段 + 它当前绑在哪条桌边（没绑为 None）。

    列表与单条更新必须走同一个函数——少了 bound_edge 的那一份会让控制面把「已上桌」
    显示成「未上桌」，还会多出一个删除按钮（2026-08-20 现场撞到：点一次「查询」，
    那一行就从桌边 3 变成未上桌）。调用方需自持 _state_lock。
    """
    bound = next((g["edge"] for g in _state["groups"] if g.get("phone_no") == phone.get("no")), None)
    return dict(phone, bound_edge=bound)


@app.get("/api/phones")
def api_phones():
    """手机台账全表，按机号升序；每行附带它当前绑在哪条桌边（没绑为 null）。"""
    with _state_lock:
        phones = [_phone_view(p) for p in _state.get("phones", [])]
    return JSONResponse({"phones": sorted(phones, key=lambda item: item.get("no") or 0)})


@app.put("/api/phones/{no}")
def api_phone_upsert(no: int, patch: dict = Body(...)):
    """新增或更新一台手机（upsert，支持部分字段）。

    机号是主键，所以「改机号」是一次搬家而不是改一个属性：body 里带 no 即表示改号，
    本服务会把引用它的桌边一并更新——否则那条桌边会指向一个不存在的机号。
    """
    if no <= 0:
        return JSONResponse({"ok": False, "error": "机号必须是正整数"}, status_code=400)
    unknown = [k for k in patch if k not in PHONE_EDITABLE_FIELDS]
    if unknown:
        return JSONResponse({"ok": False, "error": f"不支持的字段：{unknown}"}, status_code=400)
    for key in ("serial", "identity", "udid", "device_name", "build"):
        if key in patch and not isinstance(patch[key], str):
            return JSONResponse({"ok": False, "error": f"{key} 必须是字符串"}, status_code=400)
    if "resolved" in patch and not isinstance(patch["resolved"], dict):
        return JSONResponse({"ok": False, "error": "resolved 必须是对象"}, status_code=400)
    new_no = no
    if "no" in patch:
        if not isinstance(patch["no"], int) or isinstance(patch["no"], bool) or patch["no"] <= 0:
            return JSONResponse({"ok": False, "error": "机号必须是正整数"}, status_code=400)
        new_no = patch["no"]

    with _state_lock:
        phones = _state.setdefault("phones", [])
        phone = _phone_of(no)
        if new_no != no and _phone_of(new_no):
            return JSONResponse(
                {"ok": False, "error": f"{new_no} 号机已存在，机号不能重复"}, status_code=409)
        # 序列号与账号在台账内必须唯一：它们是「唯一确定一台手机」这件事的两个抓手，
        # 重复了就等于说不清面前这台到底是哪一行。空值不参与判重（还没填的行不算撞车）。
        for key, name_cn in PHONE_UNIQUE_FIELDS:
            value = str(patch.get(key, "") or "").strip()
            if not value:
                continue
            conflict = next((p for p in phones
                             if p.get("no") != no and str(p.get(key) or "").strip() == value), None)
            if conflict:
                return JSONResponse(
                    {"ok": False,
                     "error": f"{name_cn} {value} 已登记在 {conflict['no']} 号机名下"},
                    status_code=409)
        created = phone is None
        if created:
            phone = _default_phone(no)
            phones.append(phone)
        for key in ("serial", "identity", "udid", "device_name"):
            if key in patch:
                phone[key] = patch[key].strip()
        if "build" in patch:
            phone["build"] = patch["build"]
        if "resolved" in patch:
            # 整组替换而不是合并：解析是一次性的快照，半新半旧的 resolved
            # 会让人以为 client_id 与 last_seen 是同一次查出来的
            phone["resolved"] = {k: v for k, v in patch["resolved"].items()
                                 if k in PHONE_RESOLVED_FIELDS}
        if new_no != no:
            phone["no"] = new_no
            for group in _state["groups"]:
                if group.get("phone_no") == no:
                    group["phone_no"] = new_no
        phones.sort(key=lambda item: item.get("no") or 0)
        _save_state(_state)
        updated = _phone_view(phone)
    return JSONResponse({"ok": True, "phone": updated, "created": created})


@app.delete("/api/phones/{no}")
def api_phone_delete(no: int):
    """从台账里删掉一台手机。还绑在桌边上时拒绝——先在那条桌边解绑再删。"""
    with _state_lock:
        phone = _phone_of(no)
        if phone is None:
            return JSONResponse({"ok": False, "error": f"台账里没有 {no} 号机"}, status_code=404)
        bound = next((g for g in _state["groups"] if g.get("phone_no") == no), None)
        if bound:
            return JSONResponse(
                {"ok": False, "error": f"{no} 号机还绑在桌边 {bound['edge']}，请先在那边解绑"},
                status_code=409)
        _state["phones"] = [p for p in _state["phones"] if p.get("no") != no]
        _save_state(_state)
    return JSONResponse({"ok": True, "no": no})


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
    """查询某条项链的演示状态机状态（tracked=false 表示未参与演示）。

    services 的 data 段整体透传，不逐字段挑选——状态取值（ready / meal_in_progress /
    analyzing / report_published / failed）和一轮演示的 cycle_id、cycle_started_at
    都由 services 定义，本服务只加一个 ok 标志，services 加字段这里不用跟着改。
    """
    device_id = (device_id or "").strip()
    if not device_id:
        return JSONResponse({"ok": False, "error": "需要 device_id"}, status_code=400)
    data, status = _ifa_services_request(
        "GET", "/api/v1/ifa/devices/%s/meal-state" % urllib.parse.quote(device_id))
    ok = status < 400
    return JSONResponse({"ok": ok, **data}, status_code=status if not ok else 200)


@app.put("/api/necklaces/{device_id}/meal-state")
def api_necklace_meal_state_set(device_id: str, body: dict = Body(...)):
    """设置某条项链的演示状态（控制面主用两个取值）。

    · state=standby 是「归零」：换周期、清角色、把 App 弹回宣言页，但**不开窗**。
      这是控制面的主按钮，桌边最常按的就是它。
    · state=ready 是「手动开轮」兜底：定下取帧窗口的左端。正常场次由访客在 App 上
      点 Start your trip 完成，只有访客不碰手机、工作人员替他跑全程时才用它。

    persona_id 可选，且只在 state=ready（手动开轮）时有意义：一轮演示绑定一个深区
    角色与至多一份报告，App 侧靠这条绑定在用户离开深区再回来时落回「那个人的报告」。
    归零刻意忽略它（后端也会忽略）：那个动作就是「清空场子等下一位访客」。
    """
    device_id = (device_id or "").strip()
    if not device_id:
        return JSONResponse({"ok": False, "error": "需要 device_id"}, status_code=400)
    state = str((body or {}).get("state") or "").strip()
    if not state:
        return JSONResponse({"ok": False, "error": "需要 state"}, status_code=400)
    payload = {"state": state}
    persona_id = str((body or {}).get("persona_id") or "").strip()
    if persona_id:
        payload["persona_id"] = persona_id
    data, status = _ifa_services_request(
        "PUT", "/api/v1/ifa/devices/%s/meal-state" % urllib.parse.quote(device_id), payload)
    ok = status < 400
    return JSONResponse({"ok": ok, **data}, status_code=status if not ok else 200)


@app.delete("/api/necklaces/{device_id}/meal-state")
def api_necklace_meal_state_delete(device_id: str):
    """让某条项链退出演示状态机，回到原有逻辑。

    「退出演示」与「本轮失败」在数据上是两回事：前者是这台设备根本不参与，
    后者是参与了但这一轮没跑出结果。控制面必须能表达前者，否则现场只能把
    不参与的设备也标成 failed，看板上分不清哪台是真出问题。
    """
    device_id = (device_id or "").strip()
    if not device_id:
        return JSONResponse({"ok": False, "error": "需要 device_id"}, status_code=400)
    data, status = _ifa_services_request(
        "DELETE", "/api/v1/ifa/devices/%s/meal-state" % urllib.parse.quote(device_id))
    ok = status < 400
    return JSONResponse({"ok": ok, **data}, status_code=status if not ok else 200)


@app.get("/api/necklaces/{device_id}/meal-segments")
def api_necklace_meal_segments(device_id: str, limit: int = 10):
    """列出某条项链最近的演示餐段（meal 分析链的取帧时间窗）。

    同样整体透传 services 的 data 段：每个餐段除时间窗外还带所属轮次 cycle_id、
    是否因超过窗口上限被截断 window_truncated、报告状态 report{...} 与本轮实际生效的
    深区分析参数 params{...}，这些都由 services 决定，本服务不解释、不裁剪。
    """
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


@app.post("/api/necklaces/{device_id}/close-meal")
def api_necklace_close_meal(device_id: str):
    """强制关餐：把该项链正在进行的这一餐立刻收尾，交由 services 走后续分析。

    现场兜底用——App 或 Live Activity 上的「结束用餐」点不动时，控制面从这里补一刀。
    无请求体；响应形如 {"device_id": "...", "closed": true}，整体透传 services 的 data 段。
    关不关得掉、当前状态允不允许关，全由 services 判断（本服务不做状态判断），
    不允许时它回 4xx，这里原样透传状态码与 error 说明。
    """
    device_id = (device_id or "").strip()
    if not device_id:
        return JSONResponse({"ok": False, "error": "需要 device_id"}, status_code=400)
    data, status = _ifa_services_request(
        "POST", "/api/v1/ifa/devices/%s/close-meal" % urllib.parse.quote(device_id))
    ok = status < 400
    return JSONResponse({"ok": ok, **data}, status_code=status if not ok else 200)


@app.get("/api/fallback-report/foods")
def api_fallback_report_foods():
    """兜底食物表：控制面据此渲染「推送报告」的克数输入框。

    食物清单与每 100 g 营养常量都由 services 定义（那边才是算这份报告的人），
    本服务只透传，避免两处各存一份口径、现场对不上账。
    """
    data, status = _ifa_services_request("GET", "/api/v1/ifa/fallback-report/foods")
    ok = status < 400
    return JSONResponse({"ok": ok, **data}, status_code=status if not ok else 200)


@app.post("/api/necklaces/{device_id}/fallback-report")
def api_necklace_fallback_report(device_id: str, body: dict = Body(...)):
    """推送兜底报告：按填入的克数直接给这一轮出一份报告（不走任何大模型）。

    请求体 {"items": [{"food_key": "blueberry", "grams": 120}, ...]}，原样透传。
    什么状态能推、克数合不合法全由 services 判断——它才知道这一轮停在哪、
    食物表里有什么，在代理这层再抄一份规则必然与它漂移。
    """
    device_id = (device_id or "").strip()
    if not device_id:
        return JSONResponse({"ok": False, "error": "需要 device_id"}, status_code=400)
    body = body or {}
    # 只做一处结构性把关：items 必须是数组，否则误传的裸参数会被当成空清单发出去。
    if not isinstance(body.get("items"), list):
        return JSONResponse({"ok": False, "error": "需要 items 数组"}, status_code=400)
    data, status = _ifa_services_request(
        "POST", "/api/v1/ifa/devices/%s/fallback-report" % urllib.parse.quote(device_id), body)
    ok = status < 400
    return JSONResponse({"ok": ok, **data}, status_code=status if not ok else 200)


@app.get("/api/necklaces/{device_id}/scale-summary")
def api_necklace_scale_summary(device_id: str, segment_id: str = ""):
    """某台项链某一段的秤汇总：两种食物各吃了多少克、逐条事件的判定、闭合自检。

    segment_id 留空取最新一段。分类是 services 按当前阈值**读时算**的，
    所以控制面改完阈值直接重拉这个接口就能看到新的账，不用重新采数据。
    """
    device_id = (device_id or "").strip()
    if not device_id:
        return JSONResponse({"ok": False, "error": "需要 device_id"}, status_code=400)
    path = "/api/v1/ifa/devices/%s/scale-summary" % urllib.parse.quote(device_id)
    if segment_id.strip():
        path += "?segment_id=" + urllib.parse.quote(segment_id.strip())
    data, status = _ifa_services_request("GET", path)
    ok = status < 400
    return JSONResponse({"ok": ok, **data}, status_code=status if not ok else 200)


@app.get("/api/necklaces/{device_id}/scale-events")
def api_necklace_scale_events(device_id: str, limit: int = 100):
    """某台项链最近的秤事件（与餐段无关），排障用。

    与本服务的 /api/scale-events 分工：那个查的是 dx 本地内存环（检测出来了没有），
    这个查的是 services 落库的（上报上去了没有）。两边对不上就说明上报这一段有问题。
    """
    device_id = (device_id or "").strip()
    if not device_id:
        return JSONResponse({"ok": False, "error": "需要 device_id"}, status_code=400)
    data, status = _ifa_services_request(
        "GET", "/api/v1/ifa/devices/%s/scale-events?limit=%d" % (
            urllib.parse.quote(device_id), max(1, min(limit, 1000))))
    ok = status < 400
    return JSONResponse({"ok": ok, **data}, status_code=status if not ok else 200)


@app.get("/api/necklaces/{device_id}/params")
def api_necklace_params_get(device_id: str):
    """查询该项链当前生效的深区分析参数（取帧窗口 / 帧数 / 图片规格 / 模型 / prompt）。

    响应带 scope：global 表示这台项链没有自己的覆盖值、吃的是全局默认，device 表示
    已按设备覆盖过。整体透传 services 的 data 段——参数有哪些项由 services 定义，
    这里不维护字段白名单，免得 services 加一项就要跟着改一次代理。
    """
    device_id = (device_id or "").strip()
    if not device_id:
        return JSONResponse({"ok": False, "error": "需要 device_id"}, status_code=400)
    data, status = _ifa_services_request(
        "GET", "/api/v1/ifa/devices/%s/params" % urllib.parse.quote(device_id))
    ok = status < 400
    return JSONResponse({"ok": ok, **data}, status_code=status if not ok else 200)


@app.put("/api/necklaces/{device_id}/params")
def api_necklace_params_set(device_id: str, body: dict = Body(...)):
    """覆盖该项链的深区分析参数，支持只传要改的那几项（部分更新）。

    请求体 {"params": {...}}，原样透传给 services，本服务不校验参数名与取值范围——
    合法值域（窗口上限、帧数上限、模型与 prompt 名等）只有 services 知道，
    在代理这层再抄一份校验规则必然与它漂移，不如让 services 回 4xx 再透传出去。
    """
    device_id = (device_id or "").strip()
    if not device_id:
        return JSONResponse({"ok": False, "error": "需要 device_id"}, status_code=400)
    body = body or {}
    # 只做一处结构性把关：params 必须是对象。少了它，误传的裸参数会被静默当成空更新。
    if not isinstance(body.get("params"), dict):
        return JSONResponse({"ok": False, "error": "需要 params 对象"}, status_code=400)
    data, status = _ifa_services_request(
        "PUT", "/api/v1/ifa/devices/%s/params" % urllib.parse.quote(device_id), body)
    ok = status < 400
    return JSONResponse({"ok": ok, **data}, status_code=status if not ok else 200)


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


@app.get("/api/scale-events")
def api_scale_events(limit: int = 100, channel: int = 0, countable_only: bool = False):
    """最近的秤事件（倒序）+ 各通道检测器现状 + 上报积压。

    这是排障的第一现场：现场说「我明明拿了一颗，报告里没有」，先看这里有没有出事件，
    再决定是检测的问题（没切出来）还是归属的问题（切出来了但没绑项链、没上报）。

    countable_only 只留可计入摄入的两类（step / lift_return）；其余（lift、
    absent_step、lift_expired、resync）是诊断线索，不参与克数统计。
    """
    limit = max(1, min(limit, SCALE_EVENT_RING))
    with _scale_events_lock:
        records = list(_scale_events_recent)
        pending = len(_scale_upload_queue)
        dropped = _scale_upload_dropped
    if channel:
        records = [r for r in records if r["channel"] == channel]
    if countable_only:
        records = [r for r in records if r["kind"] in sev.COUNTABLE_KINDS]
    records = list(reversed(records))[:limit]
    return JSONResponse({
        "events": records,
        "detectors": [_scale_detectors[ch].snapshot() for ch in SCALE_CHANNELS],
        "upload": {"pending": pending, "dropped": dropped,
                   "queue_max": SCALE_UPLOAD_QUEUE_MAX},
        "params": {
            "poll_interval_s": SCALE_POLL_INTERVAL,
            "stable_window_s": SCALE_STABLE_WINDOW_S,
            "stable_epsilon_g": SCALE_STABLE_EPSILON_G,
            "lift_threshold_g": SCALE_LIFT_THRESHOLD_G,
            "absent_timeout_s": SCALE_ABSENT_TIMEOUT_S,
        },
        "raw_log": str(_raw_log_path(time.time())) if SCALE_RAW_LOG else "",
        "ts": time.time(),
    })


# 上报线程放在模块末尾再起：它要用 _ifa_services_request，而那个函数定义在本文件
# 靠后的位置——在定义之前就把线程拉起来，第一条事件会撞上 NameError。
if BACKGROUND_THREADS:
    threading.Thread(target=_scale_uploader, daemon=True).start()
