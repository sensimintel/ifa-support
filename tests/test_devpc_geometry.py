# -*- coding: utf-8 -*-
"""设备点云几何模块（devpc.py）单测：meta 校验、反投影坐标系、量程/边缘剔除、
降采样与取色。numpy 不在测试环境时整档跳过（跑全量用 uv --with numpy）。"""
import unittest

try:
    import numpy as np
except ImportError:  # pragma: no cover - 无 numpy 的精简测试环境
    np = None

if np is not None:
    import devpc


def _meta(**kw):
    m = {"fx": 500.0, "fy": 500.0, "cx": 2.5, "cy": 2.5,
         "width": 5, "height": 5, "stride": 1, "depth_scale": 1.0}
    m.update(kw)
    return m


@unittest.skipIf(np is None, "numpy 不可用（精简测试环境），跳过几何单测")
class DevpcMetaTest(unittest.TestCase):
    def test_parse_meta_roundtrip_and_reject(self):
        m = devpc.parse_meta(_meta())
        self.assertEqual(m["stride"], 1)
        self.assertEqual(m["width"], 5)
        self.assertAlmostEqual(m["fx"], 500.0)
        # 缺字段 / 非法值一律 ValueError
        for bad in ({}, _meta(fx=0), _meta(fy=-1), _meta(stride=0),
                    _meta(depth_scale=0), _meta(width="x")):
            with self.assertRaises(ValueError):
                devpc.parse_meta(bad)

    def test_fov_from_intrinsics(self):
        # fy = height/2 → 垂直 FOV = 2·atan(1) = 90°
        self.assertAlmostEqual(
            devpc.fov_y_deg({"height": 480, "fy": 240.0}), 90.0, places=2)


@unittest.skipIf(np is None, "numpy 不可用（精简测试环境），跳过几何单测")
class DevpcBuildPointsTest(unittest.TestCase):
    def test_unproject_center_pixel_to_gltf_frame(self):
        # 光轴上的像素（u=cx, v=cy）深度 1m → 相机系 (0,0,1) → glTF (0,0,-1)
        d = np.zeros((5, 5), np.uint16)
        d[2, 2] = 1000
        rgb = np.zeros((5, 5, 3), np.uint8)
        rgb[2, 2] = (10, 20, 30)
        pts, cols = devpc.build_points(d, rgb, devpc.parse_meta(_meta()))
        self.assertEqual(pts.shape, (1, 3))
        np.testing.assert_allclose(pts[0], [0.0, 0.0, -1.0], atol=1e-6)
        self.assertEqual(list(cols[0]), [10, 20, 30, 255])

    def test_range_and_holes_dropped(self):
        # 0 值空洞与超量程（>8m）都不出点
        d = np.zeros((5, 5), np.uint16)
        d[1, 1] = 9000    # 9m 超上限
        d[3, 3] = 10      # 1cm 低于下限
        pts, _cols = devpc.build_points(
            d, np.zeros((5, 5, 3), np.uint8), devpc.parse_meta(_meta()))
        self.assertEqual(pts.shape[0], 0)

    def test_edge_jump_culled(self):
        # 相邻像素 1m↔2m 的跳变（前后景交界）：两侧都剔除；远离交界的平滑区保留
        d = np.zeros((5, 5), np.uint16)
        d[0, 0], d[0, 1] = 1000, 2000
        d[4, 3], d[4, 4] = 1500, 1500
        pts, _cols = devpc.build_points(
            d, np.zeros((5, 5, 3), np.uint8), devpc.parse_meta(_meta()))
        self.assertEqual(pts.shape[0], 2)
        np.testing.assert_allclose(pts[:, 2], [-1.5, -1.5], atol=1e-6)

    def test_stride_maps_color_from_full_res(self):
        # stride=2：降采样格 (1,1) 对应全分辨率像素 (2,2)，取色也从那取
        d = np.zeros((2, 2), np.uint16)
        d[1, 1] = 1000
        rgb = np.zeros((4, 4, 3), np.uint8)
        rgb[2, 2] = (7, 8, 9)
        meta = devpc.parse_meta(_meta(width=4, height=4, stride=2,
                                      cx=2.0, cy=2.0))
        pts, cols = devpc.build_points(d, rgb, meta)
        self.assertEqual(pts.shape[0], 1)
        self.assertEqual(list(cols[0][:3]), [7, 8, 9])
        # u = 1·2+0.5 = 2.5 → x = (2.5-2.0)·1/500
        np.testing.assert_allclose(pts[0], [0.001, -0.001, -1.0], atol=1e-6)

    def test_max_points_downsample(self):
        d = np.full((10, 10), 1000, np.uint16)
        pts, cols = devpc.build_points(
            d, np.zeros((10, 10, 3), np.uint8),
            devpc.parse_meta(_meta(width=10, height=10, cx=5.0, cy=5.0)),
            max_points=17)
        self.assertEqual(pts.shape[0], 17)
        self.assertEqual(cols.shape, (17, 4))


if __name__ == "__main__":
    unittest.main()
