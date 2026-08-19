# -*- coding: utf-8 -*-
"""秤事件检测器单测：平台法切变化、端碗识别、掉线重连。

核心用例是 `FigureOneReplayTest`——把设计稿里那个 250 秒餐段的 9 次跃变原样回放，
断言事件序列与闭合不变式。设计稿改了、这条用例就该跟着改，反之亦然。
"""
import unittest

import dx_scale_events as sev

DT = 0.2


def feed_plateaus(det, plateaus, t0=0.0, dt=DT):
    """按 [(平台值 g, 保持秒数)] 喂样本，返回 (事件列表, 结束时刻)。

    平台之间不插斜坡：一次瞬时跳变会让滑窗里新旧值混在一起、峰峰值超过 ε，
    检测器自然进 MOVING，等窗口被新值填满再成立新平台——这正是真实信号的形状。
    """
    events, t = [], t0
    for value, hold in plateaus:
        for _ in range(max(1, int(round(hold / dt)))):
            events.extend(det.feed(round(t, 3), value))
            t = round(t + dt, 3)
    return events, t


def kinds_and_deltas(events):
    return [(e["kind"], e["delta_g"]) for e in events]


class PlateauStepTest(unittest.TestCase):
    def test_取食产生一条_step_事件(self):
        det = sev.ScaleEventDetector(1)
        events, _ = feed_plateaus(det, [(280.3, 5), (278.5, 5)])
        self.assertEqual(kinds_and_deltas(events), [(sev.KIND_STEP, -1.8)])
        self.assertEqual(events[0]["before_g"], 280.3)
        self.assertEqual(events[0]["after_g"], 278.5)

    def test_爆米花级别的亚克变化也切得出来(self):
        det = sev.ScaleEventDetector(1)
        events, _ = feed_plateaus(det, [(280.3, 5), (279.9, 5)])
        self.assertEqual(kinds_and_deltas(events), [(sev.KIND_STEP, -0.4)])

    def test_峰峰值不超过_ε_的抖动不算变化(self):
        # 0.15 g < ε(0.2)：窗口从头到尾都稳定，压根不会进 MOVING
        det = sev.ScaleEventDetector(1)
        events, _ = feed_plateaus(det, [(280.3, 5), (280.15, 5)])
        self.assertEqual(events, [])

    def test_平台维持期间不刷新平台值(self):
        # 长时间静置不应产生任何事件，也不该把平台悄悄挪走
        det = sev.ScaleEventDetector(1)
        events, _ = feed_plateaus(det, [(280.3, 60)])
        self.assertEqual(events, [])
        self.assertEqual(det.snapshot()["plateau_g"], 280.3)


class LiftReturnTest(unittest.TestCase):
    def test_端起碗再放回结算出净差(self):
        det = sev.ScaleEventDetector(1)
        events, _ = feed_plateaus(det, [(278.1, 5), (11.4, 20), (274.9, 5)])
        self.assertEqual(kinds_and_deltas(events),
                         [(sev.KIND_LIFT, -266.7), (sev.KIND_LIFT_RETURN, -3.2)])
        # 净差锚在离台前的平台上，而不是碗在手上那段的读数
        self.assertEqual(events[1]["before_g"], 278.1)
        self.assertEqual(events[1]["after_g"], 274.9)

    def test_离台期间动了别的东西不计入(self):
        det = sev.ScaleEventDetector(1)
        events, _ = feed_plateaus(det, [(278.1, 5), (11.4, 5), (13.0, 5), (274.9, 5)])
        kinds = [e["kind"] for e in events]
        self.assertEqual(kinds, [sev.KIND_LIFT, sev.KIND_ABSENT_STEP, sev.KIND_LIFT_RETURN])
        # 锚点不受离台期间那次跃变影响，净差仍然对着 278.1 算
        self.assertEqual(events[2]["delta_g"], -3.2)

    def test_离台超时后锚点作废(self):
        det = sev.ScaleEventDetector(1, absent_timeout_s=3.0)
        events, _ = feed_plateaus(det, [(278.1, 5), (11.4, 10)])
        self.assertEqual([e["kind"] for e in events],
                         [sev.KIND_LIFT, sev.KIND_LIFT_EXPIRED])

    def test_锚点作废之后回台算不出净差(self):
        det = sev.ScaleEventDetector(1, absent_timeout_s=3.0)
        events, _ = feed_plateaus(det, [(278.1, 5), (11.4, 10), (274.9, 5)])
        self.assertEqual([e["kind"] for e in events],
                         [sev.KIND_LIFT, sev.KIND_LIFT_EXPIRED,
                          sev.KIND_LIFT_RETURN_UNANCHORED])
        # 这条不可计入：真实净差已经无从得知，不能拿它冒充一次取食
        self.assertNotIn(sev.KIND_LIFT_RETURN_UNANCHORED, sev.COUNTABLE_KINDS)


class GapTest(unittest.TestCase):
    def test_掉线重连记一条_resync_而不是一次取食(self):
        det = sev.ScaleEventDetector(1)
        events, t = feed_plateaus(det, [(280.3, 5)])
        self.assertEqual(events, [])
        det.mark_gap(t)
        events, _ = feed_plateaus(det, [(240.0, 5)], t0=t)
        self.assertEqual([e["kind"] for e in events], [sev.KIND_RESYNC])
        self.assertNotIn(sev.KIND_RESYNC, sev.COUNTABLE_KINDS)

    def test_掉线期间读数没变就不记_resync(self):
        # 四通道整组同生共死，一次掉线会惊动所有通道；没变化的通道不该留痕
        det = sev.ScaleEventDetector(1)
        _, t = feed_plateaus(det, [(280.3, 5)])
        det.mark_gap(t)
        events, _ = feed_plateaus(det, [(280.3, 5)], t0=t)
        self.assertEqual(events, [])


class FigureOneReplayTest(unittest.TestCase):
    """设计稿图 1 的 250 秒餐段：9 次跃变、含一次端碗与一次抓一把。"""

    PLATEAUS = [
        (280.3, 42),   # 开轮静置
        (278.5, 19),   # ① 取一颗蓝莓 −1.8
        (278.1, 34),   # ② 取一颗爆米花 −0.4
        (279.0, 2),    # ③ 手压在碗沿 +0.9
        (278.1, 33),   # ④ 手松开 −0.9
        (11.4, 58),    # ⑤ 端起碗离台
        (274.9, 17),   # ⑥ 碗回台，净差 −3.2
        (273.1, 13),   # ⑦ 取一颗蓝莓 −1.8
        (272.5, 14),   # ⑧ 取一颗爆米花 −0.6
        (269.1, 18),   # ⑨ 抓一把爆米花 −3.4
    ]

    def setUp(self):
        self.det = sev.ScaleEventDetector(1)
        self.events, self.end_ts = feed_plateaus(self.det, self.PLATEAUS)

    def test_事件序列与设计稿一致(self):
        self.assertEqual(kinds_and_deltas(self.events), [
            (sev.KIND_STEP, -1.8),
            (sev.KIND_STEP, -0.4),
            (sev.KIND_STEP, 0.9),
            (sev.KIND_STEP, -0.9),
            (sev.KIND_LIFT, -266.7),
            (sev.KIND_LIFT_RETURN, -3.2),
            (sev.KIND_STEP, -1.8),
            (sev.KIND_STEP, -0.6),
            (sev.KIND_STEP, -3.4),
        ])

    def test_闭合不变式_可计入事件之和等于首末平台之差(self):
        # 这条不变式是整条链的自检：对不上就说明事件流有洞（漏跃变 / 未闭合的离台 /
        # 皮重被人动过）。lift 本身不参与——它那 266.7 g 已经被 lift_return 的净差覆盖。
        countable = sum(e["delta_g"] for e in self.events
                        if e["kind"] in sev.COUNTABLE_KINDS)
        self.assertAlmostEqual(countable, 269.1 - 280.3, places=3)

    def test_整段跑完落在稳定平台上且没有悬空锚点(self):
        snap = self.det.snapshot()
        self.assertEqual(snap["state"], "stable")
        self.assertEqual(snap["plateau_g"], 269.1)
        # 锚点必须已经结算掉：留着它说明有一次离台没闭合，这一段的克数就是不完整的
        self.assertIsNone(snap["anchor_g"])


if __name__ == "__main__":
    unittest.main()
