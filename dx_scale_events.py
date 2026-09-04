# -*- coding: utf-8 -*-
"""秤事件检测：把 5 Hz 的连续读数切成「一次变化」。

**为什么不能逐样本做差分**：手碰一下桌子、手指在碗沿搭一下，读数就会抖出几十次
跳变，逐样本差分会把它们全当成取食。这里只认**稳定平台**——滑窗内峰峰值不超过 ε
才算平台成立，一次事件就是相邻两个平台之间的净差，中间那段过程整段丢弃，
不管它抖得多难看。

三个状态：

  STABLE  滑窗内峰峰值 ≤ ε，平台成立
  MOVING  正在变化，不产出任何值
  ABSENT  跌幅超过离台阈，容器被端走了，这期间的读数与桌上还剩多少食物无关

**ABSENT 是「端起碗吃再放回」能被识别出来的全部机制**：跌幅越过离台阈就把离台前的
平台压成锚点，等碗回来时用「新平台 − 锚点」一次性结算这期间吃掉的总量——碗在手上
那段秤读数没有意义，状态机干脆不看。

**离台阈的余量随展台食物变化，换食物时必须重估**。旧展台（爆米花 + 蓝莓）一次取食
0.2–3 g、端碗几十到几百克，两者差两个数量级，阈值取 20 g 怎么定都对。换成草莓 +
蓝莓之后一颗草莓就有 10–25 g、抓一把三四十克，20 g 的阈值会把「拿一颗大草莓」判成
离台：那一克不但不计入，状态机还会就此进 ABSENT，后面每一次取食都变成
absent_step（不计入），整段账掉到接近零。阈值因此提到 60 g。

60 这个数依赖**容器本身足够重**：碗连内容物几百克时，「抓一把」（≤40 g）与「端起碗」
（≥200 g）之间还有五倍余量。换成轻质纸杯（几十克）这层区分就不成立了——那种情况下
端碗与抓一把在秤上长得一样，只能靠加一个秤通道或换重容器解决，调阈值救不了。

**本模块只吐原始变化，不做分类**（是草莓还是蓝莓、要不要抵消手扶碗那一对），
分类阈值在 services 侧现场可调；一旦在这里把结论写死，改阈值就再也算不回历史了。

**检测跑在毛重上而不是净重**：软件去皮会让净重瞬间跳变，跑在净重上会凭空造出一次
几百克的假事件；而去皮量在相邻两个平台的差里本来就会抵消掉。
"""
import collections
import statistics

# 事件类型。只有 step 与 lift_return 携带可计入摄入的量，其余都只是诊断线索。
KIND_STEP = "step"                                  # 普通跃变：取走（负）或放回（正）
KIND_LIFT = "lift"                                  # 容器离台，压栈锚点，本身不含量
KIND_LIFT_RETURN = "lift_return"                    # 容器回台，delta = 新平台 − 锚点
KIND_ABSENT_STEP = "absent_step"                    # 离台期间的跃变，多半是动了别的东西
KIND_LIFT_EXPIRED = "lift_expired"                  # 离台太久，锚点作废（碗被端走了）
KIND_LIFT_RETURN_UNANCHORED = "lift_return_unanchored"  # 锚点已作废后才回台，算不出净差
KIND_RESYNC = "resync"                              # 秤掉线重连后的重新基线，期间的变化无从归属

# 计入摄入的事件类型：services 侧也有同一份判断，这里导出供上报方过滤与自检。
COUNTABLE_KINDS = (KIND_STEP, KIND_LIFT_RETURN)

DEFAULT_STABLE_WINDOW_S = 1.0
DEFAULT_STABLE_EPSILON_G = 0.2
# 离台阈：跌幅越过它就判「容器被端起」。取值依据与换食物时的重估口径见模块头注释。
# 20 → 60 是随展台从「爆米花 + 蓝莓」换成「草莓 + 蓝莓」改的：一颗草莓 10–25 g、
# 抓一把三四十克，留在 20 会把取食判成离台，整段账掉到接近零。
DEFAULT_LIFT_THRESHOLD_G = 60.0
DEFAULT_ABSENT_TIMEOUT_S = 300.0
# 上报下限：小于半个分度（0.1 g）的平台差是量化噪声，不值得占一条记录。
# 这不是方案里那个「死区」——死区是 services 侧可调的语义阈值（默认 0.2 g），
# 留在那边才能在改小之后还算得回历史；这里只挡掉物理上无意义的那一档。
DEFAULT_REPORT_FLOOR_G = 0.05


class ScaleEventDetector:
    """单通道平台法事件检测器。纯内存、零 I/O，喂样本、收事件。"""

    def __init__(self, channel, sample_interval_s=0.2,
                 stable_window_s=DEFAULT_STABLE_WINDOW_S,
                 stable_epsilon_g=DEFAULT_STABLE_EPSILON_G,
                 lift_threshold_g=DEFAULT_LIFT_THRESHOLD_G,
                 absent_timeout_s=DEFAULT_ABSENT_TIMEOUT_S,
                 report_floor_g=DEFAULT_REPORT_FLOOR_G):
        self.channel = channel
        # 平台至少要持续一整个滑窗才算成立，所以窗口长度就是「最短平台时长」
        self._window_len = max(2, int(round(stable_window_s / max(sample_interval_s, 1e-6))))
        self._eps = stable_epsilon_g
        self._lift = lift_threshold_g
        self._absent_timeout = absent_timeout_s
        self._floor = report_floor_g
        self._samples = collections.deque(maxlen=self._window_len)
        self._state = "init"
        self._plateau = None            # 当前平台值（毛重克）
        self._plateau_at = None
        self._prev_plateau = None       # 离开的那个平台值，等新平台成立时用来结算
        self._left_at = None            # 离开平台的时刻
        self._from_absent = False       # 这次 MOVING 是从 ABSENT 出发的吗
        self._anchor = None             # 离台锚点：碗离台前桌上有多重
        self._anchor_at = None
        self._resync_pending = False

    # ── 对外 ────────────────────────────────────────────────────────────
    def feed(self, ts, gross_g):
        """喂一个采样，返回本次产生的事件列表（通常为空）。"""
        events = []
        self._samples.append(gross_g)
        if len(self._samples) < self._window_len:
            return events
        values = list(self._samples)
        stable = (max(values) - min(values)) <= self._eps
        if stable:
            value = round(statistics.median(values), 3)
            if self._state == "init":
                events.extend(self._baseline(value, ts))
            elif self._state == "moving":
                events.extend(self._settle(value, ts))
            # STABLE / ABSENT：平台维持。**刻意不刷新平台值**——跟着滑窗中位数
            # 慢慢走，几分钟的零点漂移就会被悄悄吸收进平台，最后谁也说不清那几克
            # 去哪了；让它攒到越过 ε、走一次 MOVING，漂移才会显式变成一条记录。
        elif self._state in ("stable", "absent"):
            self._prev_plateau = self._plateau
            self._from_absent = (self._state == "absent")
            self._left_at = ts
            self._state = "moving"
        events.extend(self._expire_anchor(ts))
        return events

    def mark_gap(self, ts):
        """秤掉线：丢掉窗口与平台，重连后的第一个平台只当重新基线。

        掉线期间发生的事情无从归属——可能有人吃了半碗，也可能什么都没动。
        与其把重连后的读数差当成一次取食，不如显式记一条 resync 说明这里有个洞。
        """
        self._samples.clear()
        if self._state != "init":
            self._prev_plateau = self._plateau
            self._resync_pending = True
        self._state = "init"
        self._plateau = None
        self._anchor = None
        self._anchor_at = None
        self._from_absent = False
        return []

    def snapshot(self):
        """当前内部状态，供控制面与排障查看。"""
        return {
            "channel": self.channel,
            "state": self._state,
            "plateau_g": self._plateau,
            "anchor_g": self._anchor,
            "anchor_at": self._anchor_at,
            "window_len": self._window_len,
        }

    # ── 内部 ────────────────────────────────────────────────────────────
    def _baseline(self, value, ts):
        """首个平台（或掉线重连后的第一个平台）成立。"""
        events = []
        # 重连后读数没变就没有洞要记：一条 Δ=0 的 resync 只会把真正有断档的那几条
        # 淹掉。四通道整组同生共死，一次掉线会同时惊动所有通道，噪声是成倍的。
        if (self._resync_pending and self._prev_plateau is not None
                and abs(value - self._prev_plateau) >= self._floor):
            events.append(self._event(KIND_RESYNC, self._left_at or ts, ts,
                                      self._prev_plateau, value))
        self._resync_pending = False
        self._enter_stable(value, ts)
        return events

    def _settle(self, value, ts):
        """MOVING 结束、新平台成立：把这次跃变判成哪一类。"""
        prev = self._prev_plateau
        started_at = self._left_at if self._left_at is not None else ts
        delta = round(value - prev, 3)

        if self._from_absent:
            if delta >= self._lift:
                if self._anchor is None:
                    # 锚点已经超时作废，只能承认这次回台算不出净差
                    self._enter_stable(value, ts)
                    return [self._event(KIND_LIFT_RETURN_UNANCHORED, started_at, ts, prev, value)]
                net_before = self._anchor
                self._anchor = None
                self._anchor_at = None
                self._enter_stable(value, ts)
                # 净差用锚点而不是离台平台：中间那段桌上只剩别的东西，跟吃了多少无关
                return [self._event(KIND_LIFT_RETURN, started_at, ts, net_before, value)]
            # 还在离台状态里动了点别的（比如挪了另一个盘子），照记但不计入
            self._enter_absent(value, ts)
            return [self._event(KIND_ABSENT_STEP, started_at, ts, prev, value)]

        if delta <= -self._lift:
            self._anchor = prev
            self._anchor_at = started_at
            self._enter_absent(value, ts)
            return [self._event(KIND_LIFT, started_at, ts, prev, value)]

        self._enter_stable(value, ts)
        if abs(delta) < self._floor:
            return []
        return [self._event(KIND_STEP, started_at, ts, prev, value)]

    def _expire_anchor(self, ts):
        """离台太久：碗多半被端走了，锚点作废，免得几小时后一次无关的放置被结算成一餐。"""
        if self._anchor is None or self._anchor_at is None:
            return []
        if ts - self._anchor_at <= self._absent_timeout:
            return []
        expired = self._anchor
        self._anchor = None
        self._anchor_at = None
        return [self._event(KIND_LIFT_EXPIRED, ts, ts, expired, self._plateau)]

    def _enter_stable(self, value, ts):
        self._state = "stable"
        self._plateau = value
        self._plateau_at = ts

    def _enter_absent(self, value, ts):
        self._state = "absent"
        self._plateau = value
        self._plateau_at = ts

    def _event(self, kind, started_at, occurred_at, before, after):
        delta = None
        if before is not None and after is not None:
            delta = round(after - before, 3)
        return {
            "channel": self.channel,
            "kind": kind,
            "started_at": started_at,
            "occurred_at": occurred_at,
            "before_g": before,
            "after_g": after,
            "delta_g": delta,
            "duration_s": round(occurred_at - started_at, 3),
        }
