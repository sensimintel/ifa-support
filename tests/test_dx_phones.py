# -*- coding: utf-8 -*-
"""手机台账与桌边引用的测试：机号做主键、身份只有一处能改、迁移不猜机号。

覆盖四件容易悄悄坏掉的事：
  1. 台账 upsert 的唯一性（机号 / 序列号 / 账号）与改号时的级联
  2. 桌边只认机号，手机身份字段一律拒绝，读出来仍展开成旧形状
  3. 秤通道、项链、手机各自的独占——尤其是秤通道，共用会让克数静默记错人
  4. 旧结构迁移：身份搬进台账、撞号按序列号裁决、桌边只剩机号

导入前把 SCALE_HOST 指到 127.0.0.1：dx_backend 一导入就拉起 Modbus 轮询线程，
不改的话在部署机上跑测试会去连真的秤模块，跟正在服务的那条常连接抢连接。
"""
import os
import unittest

os.environ.setdefault("SCALE_HOST", "127.0.0.1")
os.environ.setdefault("DX_BACKGROUND_THREADS", "0")

from fastapi.testclient import TestClient  # noqa: E402

import dx_backend  # noqa: E402


class DxStateTestBase(unittest.TestCase):
    """给每个用例一份干净的内存状态，并掐掉落盘——测试不该写仓目录里的 dx_data.json。"""

    def setUp(self):
        self._origin = dx_backend._state
        dx_backend._state = {
            "groups": [dx_backend._default_group(e) for e in dx_backend.EDGES],
            "phones": [],
            "tare_raw": {str(ch): 0 for ch in dx_backend.EDGES},
            "scale_connected": {str(ch): True for ch in dx_backend.EDGES},
        }
        self.addCleanup(setattr, dx_backend, "_state", self._origin)
        self._save = dx_backend._save_state
        dx_backend._save_state = lambda state: None
        self.addCleanup(setattr, dx_backend, "_save_state", self._save)
        self._log = dx_backend._append_pairing_log
        self.pairing = []
        dx_backend._append_pairing_log = lambda *args: self.pairing.append(args)
        self.addCleanup(setattr, dx_backend, "_append_pairing_log", self._log)
        self.client = TestClient(dx_backend.app)

    def add_phone(self, no, **fields):
        res = self.client.put(f"/api/phones/{no}", json=fields)
        self.assertEqual(res.status_code, 200, res.text)
        return res.json()["phone"]


class PhoneLedgerTest(DxStateTestBase):
    def test_upsert_creates_then_updates_same_row(self):
        created = self.add_phone(2, serial="MVM4N0XTYQ", identity="test3@odyss.dev")
        self.assertEqual(created["no"], 2)
        self.assertEqual(created["serial"], "MVM4N0XTYQ")

        res = self.client.put("/api/phones/2", json={"build": "hub f200890 @ 08-12"})
        self.assertEqual(res.status_code, 200)
        self.assertFalse(res.json()["created"])
        # 部分更新不该把没提到的字段清掉
        self.assertEqual(res.json()["phone"]["serial"], "MVM4N0XTYQ")
        self.assertEqual(len(self.client.get("/api/phones").json()["phones"]), 1)

    def test_serial_and_identity_must_be_unique(self):
        self.add_phone(1, serial="LXWVK71CP9", identity="test2@odyss.dev")
        dup_serial = self.client.put("/api/phones/2", json={"serial": "LXWVK71CP9"})
        self.assertEqual(dup_serial.status_code, 409)
        self.assertIn("1 号机", dup_serial.json()["error"])
        dup_identity = self.client.put("/api/phones/2", json={"identity": "test2@odyss.dev"})
        self.assertEqual(dup_identity.status_code, 409)

    def test_phone_no_is_free_of_the_four_edge_limit(self):
        """台账要放得下不占桌边的备用机，机号不跟着桌边数封顶。"""
        self.assertEqual(self.add_phone(5, serial="DCJWF5W0M4")["no"], 5)

    def test_renumbering_carries_the_edge_reference_along(self):
        """改机号是搬家：引用它的桌边必须跟着走，否则那条边指向一个不存在的机号。"""
        self.add_phone(3, serial="HK3H3FK6KW")
        self.client.put("/api/groups/1", json={"phone_no": 3})
        res = self.client.put("/api/phones/3", json={"no": 7})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["phone"]["no"], 7)
        self.assertEqual(self.client.get("/api/groups").json()["groups"][0]["phone_no"], 7)
        self.assertIsNone(dx_backend._phone_of(3))

    def test_renumbering_onto_an_occupied_number_is_refused(self):
        self.add_phone(1, serial="LXWVK71CP9")
        self.add_phone(2, serial="MVM4N0XTYQ")
        self.assertEqual(self.client.put("/api/phones/2", json={"no": 1}).status_code, 409)

    def test_delete_refused_while_bound_to_an_edge(self):
        self.add_phone(1, serial="LXWVK71CP9")
        self.client.put("/api/groups/2", json={"phone_no": 1})
        bound = self.client.delete("/api/phones/1")
        self.assertEqual(bound.status_code, 409)
        self.assertIn("桌边 2", bound.json()["error"])

        self.client.put("/api/groups/2", json={"phone_no": 0})
        self.assertEqual(self.client.delete("/api/phones/1").status_code, 200)
        self.assertEqual(self.client.get("/api/phones").json()["phones"], [])

    def test_resolved_is_replaced_wholesale(self):
        """解析结果是一次查询的快照，半新半旧会让人以为几个值是同一次查出来的。"""
        self.add_phone(1, serial="LXWVK71CP9")
        self.client.put("/api/phones/1", json={
            "resolved": {"client_id": "OLD", "user_id": "user-1", "last_seen_at": "07-31"}})
        res = self.client.put("/api/phones/1", json={"resolved": {"client_id": "NEW"}})
        self.assertEqual(res.json()["phone"]["resolved"], {"client_id": "NEW"})

    def test_upsert_response_carries_bound_edge_too(self):
        """列表与单条更新必须同形状：少了 bound_edge 那一行就会显示成「未上桌」。"""
        self.add_phone(1, serial="LXWVK71CP9")
        self.client.put("/api/groups/3", json={"phone_no": 1})
        res = self.client.put("/api/phones/1", json={"identity": "test2@odyss.dev"})
        self.assertEqual(res.json()["phone"]["bound_edge"], 3)
        # 换成解析结果回写（控制面点「查询」走的就是这条）同样要带上
        res = self.client.put("/api/phones/1", json={"resolved": {"client_id": "X"}})
        self.assertEqual(res.json()["phone"]["bound_edge"], 3)

    def test_bound_edge_is_reported_per_phone(self):
        self.add_phone(1, serial="LXWVK71CP9")
        self.add_phone(5, serial="DCJWF5W0M4")
        self.client.put("/api/groups/3", json={"phone_no": 1})
        by_no = {p["no"]: p for p in self.client.get("/api/phones").json()["phones"]}
        self.assertEqual(by_no[1]["bound_edge"], 3)
        self.assertIsNone(by_no[5]["bound_edge"])


class GroupBindingTest(DxStateTestBase):
    def test_phone_identity_fields_are_refused_on_the_edge(self):
        """身份只有台账一处能改——桌边这边收到就报错并指路，而不是默默存第二份。"""
        res = self.client.put("/api/groups/1", json={"phone_serial": "LXWVK71CP9"})
        self.assertEqual(res.status_code, 400)
        self.assertIn("/api/phones", res.json()["error"])

    def test_group_view_expands_identity_from_the_ledger(self):
        """读出来的形状要和旧版一致：ios-build 靠 phone_udid/phone_serial 定位设备。"""
        self.add_phone(1, serial="LXWVK71CP9", identity="test2@odyss.dev",
                       udid="00008150-001D15E43478401C", device_name="Test·iPhone",
                       build="原生 9242286f @ 08-13")
        self.client.put("/api/phones/1", json={
            "resolved": {"client_id": "CA2ACFAD", "user_id": "user-2a92"}})
        self.client.put("/api/groups/1", json={"phone_no": 1})

        group = self.client.get("/api/groups").json()["groups"][0]
        self.assertEqual(group["phone_serial"], "LXWVK71CP9")
        self.assertEqual(group["phone_udid"], "00008150-001D15E43478401C")
        self.assertEqual(group["phone_identity"], "test2@odyss.dev")
        self.assertEqual(group["phone_client_id"], "CA2ACFAD")
        self.assertEqual(group["phone_user_id"], "user-2a92")
        self.assertEqual(group["phone_device_name"], "Test·iPhone")

    def test_unknown_phone_no_is_refused(self):
        res = self.client.put("/api/groups/1", json={"phone_no": 9})
        self.assertEqual(res.status_code, 400)
        self.assertIn("9 号机", res.json()["error"])

    def test_zero_phone_no_means_unbound(self):
        self.add_phone(1, serial="LXWVK71CP9")
        self.client.put("/api/groups/1", json={"phone_no": 1})
        self.assertEqual(self.client.put("/api/groups/1", json={"phone_no": 0}).status_code, 200)
        self.assertEqual(self.client.get("/api/groups").json()["groups"][0]["phone_serial"], "")

    def test_one_phone_cannot_sit_on_two_edges(self):
        self.add_phone(1, serial="LXWVK71CP9")
        self.client.put("/api/groups/1", json={"phone_no": 1})
        clash = self.client.put("/api/groups/2", json={"phone_no": 1})
        self.assertEqual(clash.status_code, 409)

    def test_scale_channel_is_exclusive(self):
        """两条桌边共用一路秤时，克数会静默记到编号靠前那条边的项链上。"""
        clash = self.client.put("/api/groups/2", json={"scale_channel": 1})
        self.assertEqual(clash.status_code, 409)
        self.assertIn("秤通道", clash.json()["error"])

    def test_channels_can_be_swapped_via_the_unassigned_state(self):
        """四路占满时也要能对调：先把一边设成 0（不接秤），腾出通道再占过去。"""
        self.assertEqual(self.client.put("/api/groups/1",
                                         json={"scale_channel": 0}).status_code, 200)
        self.assertEqual(self.client.put("/api/groups/2",
                                         json={"scale_channel": 1}).status_code, 200)
        self.assertEqual(self.client.put("/api/groups/1",
                                         json={"scale_channel": 2}).status_code, 200)
        channels = [g["scale_channel"] for g in self.client.get("/api/groups").json()["groups"]]
        self.assertEqual(channels[:2], [2, 1])

    def test_pairing_log_records_channel_and_phone_moves(self):
        """演示中途改通道会把前后事件分给两条项链，事后只能靠流水看出这里动过手。"""
        self.add_phone(1, serial="LXWVK71CP9")
        self.client.put("/api/groups/1", json={"scale_channel": 0})
        self.client.put("/api/groups/1", json={"phone_no": 1})
        self.client.put("/api/groups/1", json={"necklace_device_id": "odyss-0F0B"})
        logged = [(edge, field, old, new) for edge, field, old, new in self.pairing]
        self.assertEqual(logged, [
            (1, "scale_channel", 1, 0),
            (1, "phone_no", 0, 1),
            (1, "necklace_device_id", "", "odyss-0F0B"),
        ])

    def test_resolve_still_finds_by_client_id(self):
        """services 侧的反查契约不变，即使 client_id 已经搬到台账里。"""
        self.add_phone(2, serial="MVM4N0XTYQ")
        self.client.put("/api/phones/2", json={"resolved": {"client_id": "D8F8327D"}})
        self.client.put("/api/groups/2", json={"phone_no": 2,
                                               "necklace_device_id": "odyss-0F28"})
        by_device = self.client.get("/api/groups/resolve?device_id=odyss-0F28")
        self.assertEqual(by_device.status_code, 200)
        self.assertEqual(by_device.json()["group"]["scale_channel"], 2)
        by_client = self.client.get("/api/groups/resolve?client_id=D8F8327D")
        self.assertEqual(by_client.status_code, 200)
        self.assertEqual(by_client.json()["group"]["edge"], 2)


class MigrationTest(unittest.TestCase):
    """旧结构（身份抄在桌边上）升级到台账的一次性搬运。"""

    def legacy_state(self):
        return {"groups": [
            {"edge": 1, "label": "桌边 1", "scale_channel": 1, "phone_no": 1,
             "phone_identity": "test2@odyss.dev", "phone_client_id": "CA2ACFAD",
             "phone_user_id": "user-2a92", "phone_udid": "00008150-001D15E43478401C",
             "phone_serial": "LXWVK71CP9", "phone_build": "原生 9242286f @ 08-13",
             "necklace_device_id": "odyss-0F0B"},
            # 现场实际状态：这一条的 phone_no 写错成 3，与桌边 3 撞号
            {"edge": 2, "label": "桌边 2", "scale_channel": 2, "phone_no": 3,
             "phone_identity": "test3@odyss.dev", "phone_client_id": "5AFD5F1D",
             "phone_user_id": "user-975e", "phone_udid": "00008150-000225C13493401C",
             "phone_serial": "MVM4N0XTYQ", "phone_build": "原生 19bddd3b @ 08-12",
             "necklace_device_id": "odyss-0F28"},
            {"edge": 3, "label": "桌边 3", "scale_channel": 3, "phone_no": 3,
             "phone_identity": "", "phone_client_id": "", "phone_user_id": "",
             "phone_udid": "00008150-0006504E0C7B401C", "phone_serial": "HK3H3FK6KW",
             "phone_build": "fe045ac2 @ 08-19", "necklace_device_id": ""},
            {"edge": 4, "label": "桌边 4", "scale_channel": 4, "phone_no": 4,
             "phone_identity": "", "phone_client_id": "", "phone_user_id": "",
             "phone_udid": "", "phone_serial": "", "phone_build": "",
             "necklace_device_id": ""},
        ]}

    def migrated(self):
        state = self.legacy_state()
        dx_backend._migrate_phones(state)
        dx_backend._migrate_groups(state)
        return state

    def test_identity_moves_into_the_ledger(self):
        phones = {p["no"]: p for p in self.migrated()["phones"]}
        self.assertEqual(phones[1]["serial"], "LXWVK71CP9")
        self.assertEqual(phones[1]["identity"], "test2@odyss.dev")
        self.assertEqual(phones[1]["resolved"]["client_id"], "CA2ACFAD")
        self.assertEqual(phones[1]["resolved"]["user_id"], "user-2a92")

    def test_clashing_numbers_are_settled_by_serial(self):
        """桌边 2 与桌边 3 都写着 3 号机，按序列号裁决：MVM4N0XTYQ 是 2 号机。"""
        state = self.migrated()
        by_serial = {p["serial"]: p["no"] for p in state["phones"]}
        self.assertEqual(by_serial["MVM4N0XTYQ"], 2)
        self.assertEqual(by_serial["HK3H3FK6KW"], 3)
        self.assertEqual([g["phone_no"] for g in state["groups"]], [1, 2, 3, 0])

    def test_edges_keep_only_the_reference(self):
        for group in self.migrated()["groups"]:
            for key in dx_backend.PHONE_VIEW_FIELDS:
                self.assertNotIn(key, group)
            self.assertIn("phone_no", group)

    def test_edge_without_any_identity_gets_no_ledger_row(self):
        """桌边 4 一条身份都没填过，不该凭空多出一台空手机。"""
        state = self.migrated()
        self.assertEqual(len(state["phones"]), 3)
        self.assertEqual(state["groups"][3]["phone_no"], 0)

    def test_migration_runs_once(self):
        state = self.migrated()
        state["phones"][0]["identity"] = "changed@odyss.dev"
        self.assertFalse(dx_backend._migrate_phones(state))
        self.assertEqual(state["phones"][0]["identity"], "changed@odyss.dev")


if __name__ == "__main__":
    unittest.main()
