# mac-cam · Mac 相机推流到 8060

把接在 Mac 上的 Orbbec 相机接入 5090 的 da3-web(8060)：

- `push_rgb.py`：抓任一相机的 RGB 帧，按 `/panel` 控制面的「推帧 fps」推到 `/api/frame`，
  作为一台虚拟设备出现在设备下拉里（与手机 App 同款链路，随时切换互不干扰）。
- `push_astra.py`：Astra Pro Plus 专用——RGB 推帧之外，同时用 Orbbec SDK v1 抓**真深度**，
  反投影成点云 GLB 直传 `/api/frame/product`。选中该设备时，面板「DA3 产物」区被
  真深度点云接管（跳过 DA3 单目估计；停止推流约 10s 后自动恢复 DA3）。

## 安装

```bash
cd tools/mac-cam && ./setup.sh    # 需要 uv；建 .venv(Python 3.11) 并装依赖
```

## 运行

```bash
# Gemini 335 的 RGB 推流（设备名按 AVFoundation 名称模糊匹配）
.venv/bin/python push_rgb.py --camera "Gemini 335" --device-id mac-g335

# Astra：RGB 推帧 + 真深度点云直传（Astra 的 RGB 在系统里叫「USB相机」）
.venv/bin/python push_astra.py
```

默认服务端 `http://192.168.100.50:8060`，可用 `--server` 覆盖。

## 帧率调节（按设备，两台摄像机各调各的）

推帧频率与点云直传间隔都是 **per-device 配置**，脚本每 2s 轮询
`/api/frame/status` 的 `devices[].config` 生效（服务端不可达时自动重试）。调节入口：

- 8060 `/panel` 设备栏的 **推帧 fps** 滑条（作用于当前选中设备）；
- 8060 `/experience` 右下「调节」抽屉的 **数据源帧率** 区（RGB 推帧 fps + 点云直传间隔）；
- 直接调接口（18091 控制面经 superadmin `/da3-api` 反代走的就是这条）：
  `POST /api/frame/device-config`，body 形如
  `{"device_id":"macmini-astra","config":{"push_fps":5,"product_interval":1.5}}`
  （merge-patch：只改传入的键，键值传 null 恢复兜底）。

没有下发配置时兜底：全局 `config.push_fps`（旧口径）→ 脚本 `--fps` / `--product-interval`。
点云直传的接管窗 hold 随间隔联动（3 倍且不小于 10s），间隔调大不会让 DA3 中途抢回槽位。

## 注意

- macOS 上 Astra 的深度是私有协议，无需 root；Gemini 335 的深度是 UVC，要 root 才能
  抢占系统驱动，故 `push_rgb.py` 只推它的 RGB。
- 点云 GLB 坐标与 8060 DA3 产物同约定（相机在原点、点云在 -Z 前方、Y 向上），
  面板的固定视角/相机跟随逻辑无需区分来源。
