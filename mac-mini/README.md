# mac-mini 摄像头推帧器（cam-pusher）

展会拓扑里 mac mini 的唯一职责：把机身上插着的两台 Orbbec 相机的帧推给 5090 的
da3-web。每个推帧节拍推两路（同一节拍 → 同设备两路 fps 天然一致）：

1. **RGB 彩色帧（+硬件深度）** → `POST /api/frame`（multipart：`image` +
   `camera_info` JSON 内 `device_id`，与手机 App 契约完全一致，8060 单目链路零改动）。
   带深度的相机同请求附可选字段 `depth`：深度帧在 mini 端做**伪彩渲染**（默认
   固定量程 TURBO：近亮暖/远暗冷/无效点黑，默认 0.2~2m，`DEPTH_MIN_M`/`DEPTH_MAX_M`
   可调；固定量程防手/物进出画面时整图颜色跳变），供 `/panel`「原设备深度图」格与
   `/experience`「设备深度图」来源展示，不参与 DA3 处理，纯展示不做 D2C 对齐。
   渲染参数（`depth_*` 键：色彩映射/方向/量程或分位自适应/gamma/直方图均衡/孔洞
   填充/时空域滤波/边缘描边/等值线/无效点色/JPEG 质量/深度独立帧率）由 8060 的
   per-device `device-config` 下发，随现有 2s 配置轮询热生效（约 2~4s），未下发
   即全默认=历史行为；
2. **双目辅助帧** → `POST /api/frame/aux`（仅 G335：multipart `left`/`right` 灰度
   JPEG，`camera_info` 带 `stereo_supported`/`baseline_mm`/`laser_mode`），供 8060
   的双目 DA3 点云链路（DA3 双视角推理 + SAM3 左目染色）。

每台相机独立成一个设备桶，`/panel` 下拉可在相机与手机 App 之间切换，手机 App 通路继续并存。

## 硬件与设备号

| 相机 | PID | device_id | 推送流 |
|---|---|---|---|
| Orbbec Gemini 335 | `0x0800` | `macmini-g335` | 彩色 1280×720 MJPG 直传 + 左/右 IR + 硬件深度 |
| Astra Pro Plus | `0x060F` | `macmini-astra` | 彩色（默认 profile，非 MJPG 转码）+ 硬件深度；无双目（`stereo_supported=false`） |

### G335 激光策略（`.env` 的 `LASER_MODE`）

喂 DA3 的双目 IR 要**无散斑**，硬件深度要**有散斑**，两者冲突。默认
`interleave`：用 `OB_PROP_LASER_ON_OFF_PATTERN_INT` 让投射器逐帧交替开关，按帧
元数据 `LASER_STATUS` 分拣——无光帧取 IR、有光帧取深度；属性设置失败自动退回
`on`（散斑 IR 直喂 DA3）。可显式设 `on` / `off` 做对比实验。

> 2026-08-13 实测：pyorbbecsdk 1.3.2 + G335 固件不支持该属性（`Property is not
> supported! propertyId: 3`），实际运行自动退回 `on`——硬件深度质量优先，双目 IR
> 带散斑喂 DA3。SDK 标定基线实测读出 50.49mm（与规格 50mm 吻合）。

默认 3fps/台（兜底值；实际按设备跟随 `device-config`，见下）。双目辅助帧新增约
0.3~0.5MB/帧，LAN 带宽合计约 1-2 MB/s。任一辅助流开不出来只降级该流（日志告警），
RGB 主链路不受影响。

## 运行形态

- **LaunchDaemon**（`/Library/LaunchDaemons/life.odyss.cam-pusher.plist`）：root 运行
  （libusb 直读 USB 必须 root，这也是不用用户级 LaunchAgent 的原因）、开机自启、
  崩溃自拉起。日志在 mini 的 `~/Library/Logs/cam-pusher.log`。
- venv 在 mini 的 `~/cam-pusher-venv`（Python 3.11——`pyorbbecsdk==1.3.2` 的
  macOS arm64 wheel 只配 3.11）。不需要 brew / ffmpeg；Xcode CLT 只为 git（已装，
  2026-08-13）。
- 配置走 mini 部署目录 `~/ifa-support/mac-mini/.env`（gitignore，不进仓）：
  `RELAY_URL`（默认 `http://192.168.0.50:8060`）、`PUSH_FPS`（**兜底值**，默认 3）、
  `JPEG_QUALITY`（默认 80）。
- **推帧频率的权威来源是 8060 的 per-device 配置，两台相机各调各的**：`/panel`
  设备栏滑杆与 `/experience`「调节」抽屉的「数据源帧率」区都 POST
  `/api/frame/device-config`（按 `device_id` merge-patch）；推帧器每 2s 轮询
  `/api/frame/status` 的 `devices[].config.push_fps` 热生效（钳制 0.2~30fps），
  取值优先级 per-device ＞ 全局 `config.push_fps`（旧口径）＞ `PUSH_FPS`；
  8060 不可达时沿用最后值。

## 部署（git 部署源模式，与 5090 同款）

mini 的 `~/ifa-support` 是本仓 checkout，**只 pull、不 commit/push**（GitHub 侧
只读 deploy key `mac-mini-deploy-readonly`）。开发流程：

```
本地改代码 → push 到 GitHub → mini 上（或开发机远程触发）执行 deploy.sh
```

```bash
# mini 上：
cd ~/ifa-support/mac-mini && MINI_SUDO_PASS=... ./deploy.sh
# 或开发机远程触发：
ssh mac-mini 'MINI_SUDO_PASS=... ~/ifa-support/mac-mini/deploy.sh'
```

流程：`git pull --ff-only` → `setup.sh` 装 venv 依赖（幂等）→ 装/更新 LaunchDaemon
并 `kickstart` 重启 → 轮询 8060 `/api/frame/status` 直到出现 `macmini-*` 设备帧。

> 历史：2026-08-13 前 mini 无 Xcode CLT（macOS 26.1 的 softwareupdate 目录无 CLT
> 条目、无法无头安装），部署走开发机 rsync 推送；CLT 经 GUI 装好后已切回 git 模式。

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
