# mac-mini 摄像头推帧器（cam-pusher）

展会拓扑里 mac mini 的唯一职责：把机身上插着的两台 Orbbec 相机的彩色流，按 8060
现有入帧契约推给 5090 的 da3-web（`POST /api/frame`，multipart：`image` +
`camera_info` JSON 内 `device_id`）。每台相机独立成一个设备桶，`/panel` 下拉可在
相机与手机 App 之间切换，**8060 侧零改动**，手机 App 通路继续并存。

## 硬件与设备号

| 相机 | PID | device_id | 彩色流 |
|---|---|---|---|
| Orbbec Gemini 335 | `0x0800` | `macmini-g335` | 1280×720 MJPG（JPEG 直传不转码） |
| Astra Pro Plus | `0x060F` | `macmini-astra` | 默认 profile，非 MJPG 时 cv2 转码 |

彩色流之外同时开**硬件深度流**：深度帧在 mini 端做固定量程伪彩（近亮暖/远暗冷/
无效点黑，默认 0.2~2m，`DEPTH_MIN_M`/`DEPTH_MAX_M` 可调）后随彩色帧一并 POST
（multipart 可选字段 `depth`），供 `/panel` 左上角「相机硬件深度图」格展示；深度不
参与 DA3 处理（DA3 点云仍由 RGB 算）。深度与彩色 FOV 不同、纯展示不做 D2C 对齐。
默认 3fps/台（面板滑杆可调），LAN 带宽占用约几百 KB/s。

## 运行形态

- **LaunchDaemon**（`/Library/LaunchDaemons/life.odyss.cam-pusher.plist`）：root 运行
  （libusb 直读 USB 必须 root，这也是不用用户级 LaunchAgent 的原因）、开机自启、
  崩溃自拉起。日志在 mini 的 `~/Library/Logs/cam-pusher.log`。
- venv 在 mini 的 `~/cam-pusher-venv`（Python 3.11——`pyorbbecsdk==1.3.2` 的
  macOS arm64 wheel 只配 3.11）。**不需要 brew / Xcode CLT / ffmpeg**。
- 配置走 mini 部署目录 `~/cam-pusher/.env`（gitignore，不进仓）：
  `RELAY_URL`（默认 `http://192.168.0.50:8060`）、`PUSH_FPS`（**兜底值**，默认 3）、
  `JPEG_QUALITY`（默认 80）。
- **推帧频率的权威来源是 8060 的 per-device 配置，两台相机各调各的**：`/panel`
  设备栏滑杆与 `/experience`「调节」抽屉的「数据源帧率」区都 POST
  `/api/frame/device-config`（按 `device_id` merge-patch）；推帧器每 2s 轮询
  `/api/frame/status` 的 `devices[].config.push_fps` 热生效（钳制 0.2~15fps），
  取值优先级 per-device ＞ 全局 `config.push_fps`（旧口径）＞ `PUSH_FPS`；
  8060 不可达时沿用最后值。

## 部署（推送式：开发机 → mini）

mini 不装 git（无 Xcode CLT，macOS 26.1 的 softwareupdate 目录里无 CLT 条目、
无法无头安装），代码由开发机 rsync 推送：

```bash
# 开发机上，本目录执行；MINI_SUDO_PASS 提供 mini 的 sudo 密码则全程非交互
MINI_SUDO_PASS=... ./deploy.sh
```

流程：rsync 代码 → `setup.sh` 装 venv 依赖（幂等）→ 装/更新 LaunchDaemon 并
`kickstart` 重启 → 轮询 8060 `/api/frame/status` 直到出现 `macmini-*` 设备帧。

## 展会搬迁「一步拉起」

网络按仓根 [`NETWORK.md`](../NETWORK.md) 搬（整段复刻 `192.168.0.0/24`）。mini 已配
`pmset sleep 0 + autorestart 1`（防睡眠、断电来电自动开机），搬过去**插电即活**：
开机 → LaunchDaemon 自动起推帧器 → 相机帧出现在 8060。无需任何人工步骤。

验证一条就够：

```bash
curl -s http://192.168.0.50:8060/api/frame/status | grep -o 'macmini-[a-z0-9]*'
```

## 排障

- 8060 无 `macmini-*` 帧：ssh mac-mini 看 `~/Library/Logs/cam-pusher.log`；
  常见原因依次是 相机被拔 / RELAY_URL 不可达（查网线与网段，见 NETWORK.md §6）/
  daemon 未跑（`sudo launchctl print system/life.odyss.cam-pusher`）。
- 单相机没帧、另一台正常：几乎必是 USB 线/口问题，线程会每 5s 自动重连，插稳即恢复。
- **勿把 `_open_color_pipeline` 改成 `enable_stream(OBSensorType...)` 简写**：该
  重载在 pyorbbecsdk 1.3.2 上会段错误（2026-08-13 实测），必须显式取 stream profile。
- mini 连不上：先 `ping OdyssdeMac-mini.local`（DHCP 地址可能变），机器睡眠断电见
  `ssh-mac-mini` skill 的排障节。
