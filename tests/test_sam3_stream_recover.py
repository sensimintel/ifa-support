# -*- coding: utf-8 -*-
"""SAM3 流式 session 降级/恢复状态机 + triton autotune 并发锁的纯逻辑单测。

不依赖 GPU：导入 sam3_server 前逐个探测重依赖（torch/sam3/pycocotools/PIL/numpy/
prometheus/uvicorn），本机缺失的用 MagicMock 桩顶替（与 test_recog_direct 的
"纯逻辑直接测"同思路，只是 sam3_server 与重依赖同文件，需 sys.modules 桩）；
fastapi/pydantic 用真实包（模块里定义了 BaseModel 子类与路由，桩替会炸）。
推理路径（_step_incremental / _step_replay / _close_live）全部 mock 掉，
只测「进入 replay → 计步 → 尝试恢复 → 成功/失败」的状态转移与日志轨迹。
"""
import importlib
import importlib.util
import sys
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]

# 本机可能缺失的重依赖：能真实导入就用真的，导不动就桩替
_MAYBE_MISSING = ["torch", "numpy", "PIL", "PIL.Image", "pycocotools", "pycocotools.mask",
                  "prometheus_fastapi_instrumentator", "uvicorn"]
# sam3 上游包本机必无（只在 5090 的 sam3-env 里），一律桩替；桩存在 ⇒ _INCR_IMPORTS_OK=True
_ALWAYS_STUB = ["sam3", "sam3.model", "sam3.model.data_misc",
                "sam3.model.utils", "sam3.model.utils.misc"]


def _load_server_module():
    """按路径加载 model/sam3/sam3_server.py（重依赖先桩后载）。"""
    for name in _MAYBE_MISSING:
        try:
            importlib.import_module(name)
        except Exception:
            sys.modules[name] = mock.MagicMock(name=name)
    for name in _ALWAYS_STUB:
        sys.modules[name] = mock.MagicMock(name=name)
    spec = importlib.util.spec_from_file_location(
        "sam3_server_under_test", ROOT / "model" / "sam3" / "sam3_server.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


srv = _load_server_module()


def _make_session(impl="incremental"):
    """复刻 stream_start 建出的 session 状态字典（与生产字段一一对应）。"""
    return {
        "text": "food", "window": 5, "forget_frames": 30,
        "ring": [], "next_global": 0, "registry": {}, "next_pub": 1,
        "live_sid": None, "gen_frames": 0, "gen_base_global": 0, "id_map": {},
        "impl": impl, "degrade_count": 0, "replay_steps": 0,
        "last_ts": 0.0, "step_lock": threading.Lock(),
    }


RECOVER_EVERY = 5   # 测试里统一用小周期，跑得快


class DegradeStateMachineTest(unittest.TestCase):
    """降级状态机：进入 replay → 计步 → 尝试恢复 → 成功/失败。"""

    def setUp(self):
        self.s = _make_session()
        self.inc = mock.patch.object(srv, "_step_incremental").start()
        self.rep = mock.patch.object(srv, "_step_replay").start()
        self.close = mock.patch.object(srv, "_close_live").start()
        mock.patch.object(srv, "STREAM_INCREMENTAL", True).start()
        mock.patch.object(srv, "_INCR_IMPORTS_OK", True).start()
        mock.patch.object(srv, "STREAM_RECOVER_EVERY", RECOVER_EVERY).start()
        self.addCleanup(mock.patch.stopall)
        self.inc.return_value = ["增量结果"]
        self.rep.return_value = ["重放结果"]

    def _step(self):
        # 模拟 stream_frame 的一次步进（帧号推进；图像用哨兵对象即可）
        g = self.s["next_global"]
        out = srv._step_with_fallback(self.s, mock.sentinel.img, g, "sid-测试")
        self.s["next_global"] = g + 1
        return out

    def test_健康增量路径不降级(self):
        for _ in range(3):
            self.assertEqual(self._step(), ["增量结果"])
        self.assertEqual(self.s["impl"], "incremental")
        self.assertEqual(self.s["degrade_count"], 0)
        self.rep.assert_not_called()

    def test_增量抛错当步降级并用replay补齐(self):
        self.inc.side_effect = KeyError("labels_ptr")
        with self.assertLogs("sam3-server", level="ERROR") as logs:
            out = self._step()
        self.assertEqual(out, ["重放结果"])            # 本步结果由 replay 补齐，请求不失败
        self.assertEqual(self.s["impl"], "replay")
        self.assertEqual(self.s["degrade_count"], 1)
        self.assertEqual(self.s["replay_steps"], 0)    # 恢复计步从 0 重新起算
        self.assertIsNone(self.s["live_sid"])
        self.close.assert_called_once_with(self.s)     # 降级时活会话必须关掉
        self.assertTrue(any("降级 replay" in m for m in logs.output))

    def _degrade_once(self):
        """先制造一次降级，把 session 打进 replay 态。"""
        self.inc.side_effect = KeyError("labels_ptr")
        self._step()
        self.assertEqual(self.s["impl"], "replay")
        self.inc.reset_mock()
        self.close.reset_mock()

    def test_replay满周期尝试恢复成功(self):
        self._degrade_once()
        self.inc.side_effect = None
        self.inc.return_value = ["恢复后的增量结果"]
        # 前 N-1 步维持 replay，不碰增量
        for _ in range(RECOVER_EVERY - 1):
            self.assertEqual(self._step(), ["重放结果"])
        self.inc.assert_not_called()
        self.assertEqual(self.s["replay_steps"], RECOVER_EVERY - 1)
        # 第 N 步触发恢复尝试并成功切回增量
        with self.assertLogs("sam3-server", level="INFO") as logs:
            out = self._step()
        self.assertEqual(out, ["恢复后的增量结果"])
        self.assertEqual(self.s["impl"], "incremental")
        self.assertEqual(self.s["degrade_count"], 1)   # 恢复成功不清降级史，只切实现
        self.assertEqual(self.s["replay_steps"], 0)
        self.inc.assert_called_once()
        self.assertTrue(any("尝试重建增量 session 恢复" in m for m in logs.output))
        self.assertTrue(any("恢复成功" in m for m in logs.output))
        # 恢复后继续正常增量
        self.assertEqual(self._step(), ["恢复后的增量结果"])

    def test_恢复失败继续replay并重新计步(self):
        self._degrade_once()
        self.inc.side_effect = KeyError("labels_ptr")  # 恢复尝试仍会炸
        with self.assertLogs("sam3-server", level="ERROR") as logs:
            for _ in range(RECOVER_EVERY):
                out = self._step()
        self.assertEqual(out, ["重放结果"])            # 失败当步也由 replay 补齐
        self.assertEqual(self.s["impl"], "replay")
        self.assertEqual(self.s["degrade_count"], 2)   # 失败计入降级史
        self.assertEqual(self.s["replay_steps"], 0)    # 计步清零，重新等下个周期
        self.assertEqual(self.inc.call_count, 1)       # 周期内只试了一次
        self.assertTrue(any("增量恢复尝试失败" in m for m in logs.output))
        # 再满一个周期会再试（不会因为失败过就放弃）
        for _ in range(RECOVER_EVERY):
            self._step()
        self.assertEqual(self.inc.call_count, 2)

    def test_天生replay的session不做恢复尝试(self):
        # 一开始就是 replay（增量关闭/上游 import 失败）的 session：degrade_count=0，恒走 replay
        self.s = _make_session(impl="replay")
        for _ in range(RECOVER_EVERY * 3):
            self.assertEqual(self._step(), ["重放结果"])
        self.inc.assert_not_called()
        self.assertEqual(self.s["replay_steps"], 0)

    def test_增量总开关关闭时降级后不恢复(self):
        self._degrade_once()
        with mock.patch.object(srv, "STREAM_INCREMENTAL", False):
            for _ in range(RECOVER_EVERY * 2):
                self.assertEqual(self._step(), ["重放结果"])
        self.inc.assert_not_called()

    def test_恢复周期可配置(self):
        self._degrade_once()
        self.inc.side_effect = None
        with mock.patch.object(srv, "STREAM_RECOVER_EVERY", 3):
            self._step(); self._step()
            self.inc.assert_not_called()
            self._step()                               # 第 3 步即触发恢复
        self.inc.assert_called_once()
        self.assertEqual(self.s["impl"], "incremental")


class AutotuneLockTest(unittest.TestCase):
    """triton Autotuner.run 并发锁：装上后调度必互斥（根因侧的修复）。"""

    def _fake_triton(self):
        """构造假的 triton.runtime.autotuner 模块链 + 带并发探测的 Autotuner。"""
        autotuner_mod = types.ModuleType("triton.runtime.autotuner")

        class Autotuner:
            # 类级并发探测：run 里记录同时在跑的线程峰值
            _active = 0
            _max_active = 0
            _guard = threading.Lock()

            def run(self, *args, **kwargs):
                cls = type(self)
                with cls._guard:
                    cls._active += 1
                    cls._max_active = max(cls._max_active, cls._active)
                time.sleep(0.005)                      # 拉开窗口让竞态有机会暴露
                with cls._guard:
                    cls._active -= 1
                return "内核结果"

        autotuner_mod.Autotuner = Autotuner
        triton_mod = types.ModuleType("triton")
        runtime_mod = types.ModuleType("triton.runtime")
        triton_mod.runtime = runtime_mod
        runtime_mod.autotuner = autotuner_mod
        return ({"triton": triton_mod, "triton.runtime": runtime_mod,
                 "triton.runtime.autotuner": autotuner_mod}, Autotuner)

    def test_安装后并发调用互斥(self):
        modules, Autotuner = self._fake_triton()
        with mock.patch.dict(sys.modules, modules):
            self.assertTrue(srv._install_triton_autotune_lock())
            results = []
            threads = [threading.Thread(target=lambda: results.append(Autotuner().run()))
                       for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        self.assertEqual(results, ["内核结果"] * 8)     # 包锁后结果不变
        self.assertEqual(Autotuner._max_active, 1)      # 全程无并发互踩

    def test_重复安装幂等不叠包(self):
        modules, Autotuner = self._fake_triton()
        with mock.patch.dict(sys.modules, modules):
            self.assertTrue(srv._install_triton_autotune_lock())
            wrapped = Autotuner.run
            self.assertTrue(srv._install_triton_autotune_lock())
            self.assertIs(Autotuner.run, wrapped)       # 第二次安装不再包一层

    def test_无triton时优雅跳过(self):
        # sys.modules 置 None 会让 import 直接抛 ImportError，模拟 triton 不存在
        with mock.patch.dict(sys.modules, {"triton": None, "triton.runtime": None,
                                           "triton.runtime.autotuner": None}):
            self.assertFalse(srv._install_triton_autotune_lock())


if __name__ == "__main__":
    unittest.main()
