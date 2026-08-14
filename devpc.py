# -*- coding: utf-8 -*-
"""设备硬件深度点云（/experience「设备点云」来源）的几何构建模块。

链路：mini 端 cam-pusher 把「对齐到彩色相机的原始 uint16 深度（按 stride 降采样）
+ 同帧 RGB JPEG + 彩色相机内参」按需（页面选中该来源时）POST 到 8060 的
/api/devpc/frame；本模块负责把深度图反投影成带顶点色的 3D 点，供 app.py 用
_write_pointcloud_glb 导出 GLB、前端 model-viewer 作为背景渲染。

与 DA3 点云的关系：几何公式完全同源（深度 + 内参反投影），只是深度来源从模型
推理换成相机硬件测距——真实米制尺度、帧间稳定、不吃 GPU；代价是硬件深度存在
空洞（黑色/反光/遮挡阴影区无值）与边缘飞点，后者由本模块的边缘剔除压掉。

刻意只依赖 numpy + 标准库（不 import torch / cv2 / DA3）：JPEG/PNG 编解码留在
app.py（5090 端有 cv2），本模块因此可以在本地无 GPU 环境下被单测直接引用。
"""
import math

import numpy as np

# 深度有效量程（米）：下限剔除 0 值/贴脸噪声，上限剔除硬件测距远端的大噪声点。
# 硬件深度是真实米制，固定量程即可（不需要 DA3 那套 conf 分位裁剪）
Z_MIN_M = 0.05
Z_MAX_M = 8.0
# 边缘剔除：相邻像素深度跳变超过 max(比例×深度, 绝对值) 即视为前后景交界，
# 交界两侧的点都剔掉——压掉硬件深度在物体轮廓处的"飞点拉丝"
EDGE_RATIO = 0.04
EDGE_ABS_M = 0.03
# 单帧点数预算：超出按均匀随机降采样（G335 对齐深度 stride 3 后 ~10 万点，通常不触发）
MAX_POINTS = 160_000


def parse_meta(meta: dict) -> dict:
    """校验并规整 /api/devpc/frame 随帧上报的 meta 字典，返回构建所需字段。

    必填：fx/fy/cx/cy（对齐后全分辨率彩色相机内参，像素）、width/height（对齐后
    全分辨率）、stride（深度降采样步长）、depth_scale（深度值 → 毫米的换算系数）。
    字段缺失/非法抛 ValueError（调用方转 400）。"""
    out = {}
    for k in ("fx", "fy", "cx", "cy", "depth_scale"):
        try:
            v = float(meta[k])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"meta 缺少或非法：{k}")
        if not math.isfinite(v) or (k in ("fx", "fy", "depth_scale") and v <= 0):
            raise ValueError(f"meta 字段非法：{k}={meta.get(k)!r}")
        out[k] = v
    for k in ("width", "height", "stride"):
        try:
            v = int(meta[k])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"meta 缺少或非法：{k}")
        if not 1 <= v <= 16384:
            raise ValueError(f"meta 字段非法：{k}={meta.get(k)!r}")
        out[k] = v
    return out


def fov_y_deg(meta: dict) -> float:
    """由内参求真实相机垂直 FOV（度）：前端据此把点云相机摆回拍摄视角。"""
    return round(float(np.degrees(2 * np.arctan(meta["height"] / (2 * meta["fy"])))), 2)


def _edge_mask(z: np.ndarray) -> np.ndarray:
    """标记前后景交界两侧的像素（True=剔除）。z 为米制深度，0/NaN 视为无效。"""
    valid = z > 0
    edge = np.zeros(z.shape, dtype=bool)
    # 水平/垂直相邻像素对：两侧都有效且跳变超阈值 → 两侧同标
    for axis in (0, 1):
        a = z.take(range(z.shape[axis] - 1), axis=axis)
        b = z.take(range(1, z.shape[axis]), axis=axis)
        both = (a > 0) & (b > 0)
        thr = np.maximum(EDGE_RATIO * np.minimum(a, b), EDGE_ABS_M)
        jump = both & (np.abs(a - b) > thr)
        if axis == 0:
            edge[:-1, :] |= jump
            edge[1:, :] |= jump
        else:
            edge[:, :-1] |= jump
            edge[:, 1:] |= jump
    return edge & valid


def build_points(depth_u16: np.ndarray, rgb: np.ndarray, meta: dict,
                 max_points: int = MAX_POINTS):
    """深度图 + RGB → (点坐标, 顶点色) 供 GLB 导出。

    depth_u16  (H',W') uint16，已按 meta.stride 降采样、已对齐到彩色相机坐标系
    rgb        (H,W,3) uint8 RGB，对齐后全分辨率彩色帧（取色用）
    meta       parse_meta 的产物

    返回 (pts(N,3) float32 glTF 坐标系, cols(N,4) uint8 RGBA)；有效点为空时 N=0。
    坐标系与 DA3 点云 GLB 对齐：相机在原点、朝 -Z 看（即相机系 (X,Y,Z) → glTF
    (X,-Y,-Z)），前端 applyExpView 的取景数学直接复用。"""
    stride = meta["stride"]
    z = depth_u16.astype(np.float32) * (meta["depth_scale"] / 1000.0)  # 米
    valid = (z > Z_MIN_M) & (z < Z_MAX_M)
    valid &= ~_edge_mask(np.where(valid, z, np.float32(0.0)))
    idx = np.flatnonzero(valid.reshape(-1))
    if idx.size == 0:
        return (np.zeros((0, 3), np.float32), np.zeros((0, 4), np.uint8))
    if idx.size > max_points:
        # 均匀随机降采样：固定种子保证同一帧重算结果稳定（帧间本就整帧重建）
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(idx, size=max_points, replace=False))

    h, w = depth_u16.shape
    ii, jj = idx // w, idx % w
    # 降采样格 (i,j) 对应全分辨率像素 (i·stride, j·stride)，取该像素中心反投影
    u = jj.astype(np.float32) * stride + 0.5
    v = ii.astype(np.float32) * stride + 0.5
    zz = z.reshape(-1)[idx]
    x = (u - meta["cx"]) * zz / meta["fx"]
    y = (v - meta["cy"]) * zz / meta["fy"]
    pts = np.stack([x, -y, -zz], axis=1).astype(np.float32)   # 相机系 → glTF 系

    # 顶点色：同位置全分辨率 RGB 像素直取（深度已对齐彩色相机，零错位）
    rv = np.clip(ii * stride, 0, rgb.shape[0] - 1)
    ru = np.clip(jj * stride, 0, rgb.shape[1] - 1)
    cols = np.empty((idx.size, 4), np.uint8)
    cols[:, :3] = rgb[rv, ru]
    cols[:, 3] = 255
    return pts, cols
