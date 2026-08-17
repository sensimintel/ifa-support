# ifa-support

> 本仓同时是 Odyss 本地化组件的**编排管理仓**：全套管理规范见 [`MANAGEMENT.md`](MANAGEMENT.md)，全栈本地化 SOP（一套拉起）见 [`local-stack/`](local-stack/README.md)，演示现场网络方案（展会拓扑与公司内模拟）见 [`NETWORK.md`](NETWORK.md)。以下为 DA3 演示服务说明。

5090 服务器上跑在 `0.0.0.0:8060` 的运维/演示 Web 服务（原目录 `~/da3-web`），单文件 FastAPI 应用，纯服务端渲染、零前端构建。

- **`/` 根路径**：直接跳 `/experience`（主链路展示页）；调试页从 `/panel` 顶部导航进。
- **`/experience` 浅体验区展示页**：IFA 展台品牌化全屏 UI（Figma「IFA 专项 · 浅体验区」实现）。全屏背景为硬件深度的两种来源（默认「设备深度图」=mini 端伪彩深度帧，可切「设备点云」=硬件真深度反投影彩色点云 GLB；任何情况不展示设备 RGB 原图，产物未就绪时保持黑场。DA3/SAM3 派生的四种点云来源已于 2026-08-17 去冗余下线）。「设备深度图」来源（仅 G335 等带硬件深度的相机）直接展示 mini 端伪彩深度帧，抽屉配全套渲染调节（色彩映射/方向/量程或自动分位/gamma/均衡/填洞/时空域滤波/描边/等值线/无效点色/JPEG 质量/深度帧率——经 `device-config` 下发到推流端渲染，约 2~4s 生效、/panel「原设备深度图」格同步变化）与本页即时显示调节（CSS 亮度/对比度/饱和/色相/反色/模糊/透明度/缩放/镜像/旋转/像素化）；另有「点云化样式」区（对图片类背景通用）：canvas 像素格化 + mask 圆点镂空，保留原色彩把画面离散成等距圆点（开关/点距/圆径占比/背景色，浏览器端即时生效）。右下「调节」抽屉顶部是**识别触发（主链路直传 VLM）**——按可调间隔（默认 0.5s）取当前设备的最新 RGB 帧做识别，两种口径共用同一条触发线程：**开**=整帧直送 Qwen VLM 问画面里有什么食物；**关**=同一帧先过 SAM3（生产词表 + food 词，直接跑彩色帧），认出食物/饮品才带框送 VLM（SAM3 只当前置门，命中后照样出识别卡，省画面空时的 VLM 调用）。另可调并发上限（1=串行·最新优先，调大才真按间隔多路齐发），配置走 `/api/recog/direct/config` 落盘全局共享，详见 `docs/sam3-usage-map.md`；其下是**数据源帧率**（per-device：RGB 推帧 fps + Astra 类设备的点云直传间隔，写 `/api/frame/device-config`，推流端轮询生效，两台摄像机各调各的），其下点渲染样式区仅 GLB 类来源（设备点云）展示；右侧状态区在「Place your food here」待机态与识别成功态（浅色玻璃卡片：名称 / 英文短描述 / 卡路里 / 蛋白·碳水·脂肪 / Good·Neutral·Bad 健康分级，营养数字按画面可见份量估算）间切换，识别成功时画面上食物位置处另有定位小标（名称 + 卡路里，锚在 VLM box、随去重合并刷新位置）；另有流水视图（临时按钮进入）展示当日识别记录。品牌字体（ABC Arizona Serif / Seabirds，trial 版）在 `static/fonts/`，经 `/static` 提供。`?demo=1` 为无后端目检模式。
- **`/panel` 扩展面板（深度 / 点云 / 网格）**：浏览器上传一张图 + 选产物类型 + 调参，用 Depth Anything 3（DA3NESTED-GIANT-LARGE-1.1）出三种产物：
  - **深度图**：彩色深度图（越亮 = 越近）；
  - **点云 + 相机（GLB）**：DA3 官方 `scene.glb` 导出（点云 + 相机线框），网页内 `<model-viewer>` 可鼠标 3D 转视角；
  - **网格 mesh（GLB）**：由深度反投影自建带顶点色的三角网格，同样可 3D 转视角。
  - 可调参数：`process_res`、`conf_thresh_percentile`、`num_max_points`、`show_cameras`。`/api/infer` 出 JSON，GLB 经 `/glb/{token}/scene.glb` 提供（只保留最近若干次、自动清理）。

> **电子秤不在本服务**：四通道食物秤（净重 / 毛重、软件去皮）与桌边分组由独立的 dx-backend 提供（`0.0.0.0:8070`，见 `dx_backend.py`），8060 已无 `/weight` 与 `/api/weights`。秤的地址与网络前提见 [`NETWORK.md`](NETWORK.md)。

> **模型单例 + GPU 共存**：进程内只加载一份 DA3 模型权重，并用一把 GPU 锁串行化推理——5090 显存与产线服务共享。模型懒加载：启动不占显存，首次推理才加载一份（约 6.5GB，推理峰值约 8.6GB@process_res=504）。process_res 调太高或产线显存吃紧时可能 OOM，此时调低 process_res 重试。

## 文件

| 文件 | 说明 |
|---|---|
| `app.py` | 8060 的全部服务端逻辑（FastAPI 应用 `app:app`，深度推理 + 内嵌 HTML 页面） |
| `static/` | 静态资源（`/experience` 用的品牌字体等），经 `/static` 路径提供 |
| `recog_direct.py` | 主链路「直传 VLM 识别」的配置态（开关/间隔/并发 + 触发闸门，纯逻辑可单测），落盘 `recog_direct_cfg.json` |
| `recog_log.py` | VLM 识别观测日志的环形缓冲与响应投影（纯逻辑、可单测）；接口 `/api/recoglog/*`，消费方是 superadmin 浅体验区控制面的「VLM 识别日志」页签 |
| `docs/sam3-usage-map.md` | **SAM3 使用路径地图**：SAM3 被哪些链路消费、各自产出什么、识别触发的两种口径与开关速查 |
| `run.sh` | 用 `da3` conda 环境在 `0.0.0.0:8060` 起服务的启动脚本 |
| `dx_backend.py` | **深体验区后端（8070）**：四通道食物秤读数与软件去皮、桌边分组绑定，独立于 8060 |
| `run-dx.sh` | 起 8070 的启动脚本（复用 `da3` conda 环境） |
| `dx-backend.service` | 8070 的 systemd 单元 |
| `deploy.sh` | 5090 上一键部署：`git pull` + 重启服务（8060 与 8070 一并重启，systemd 优先） |
| `da3-web.service` | 可选 systemd 单元（正规化开机自启/重启） |
| `requirements.txt` | pip 依赖（不含 `depth_anything_3`，见下） |
| `model/` | **两个模型服务（DA3 / SAM3）的一键拉起脚本与部署信息**，含 SAM3 推理服务源码（流式长记忆版）与 systemd 单元，见 `model/README.md` |
| `mac-mini/` | **mac mini 摄像头推帧器（cam-pusher）**：把 mini 上两台 Orbbec 相机（Gemini 335 / Astra Pro Plus）的彩色流按 `/api/frame` 契约推给 8060，LaunchDaemon 常驻、开发机 rsync 推送部署，见 `mac-mini/README.md` |

## 运行

```bash
./run.sh
# 等价于：
# export HF_HOME=/home/odyss/Depth-Anything-3/models
# python -m uvicorn app:app --host 0.0.0.0 --port 8060
```

局域网内访问 `http://<5090局域网IP>:8060`。

## 部署（git 部署源模式）

5090 上的运行目录 `~/da3-web` 是本仓的 checkout，**只 pull、不 commit/push**（用只读 deploy key）。开发流程：

```
本地改代码 → push 到 GitHub → 登录 5090 → cd ~/da3-web && ./deploy.sh
```

`deploy.sh` 会 `git pull --ff-only` 后重启 8060 服务并做健康检查。

首次把 5090 目录接成 checkout / 配 deploy key 的步骤，见部署纪律：deploy key 为**只读**，5090 不承担任何提交。

## 外部依赖（不随本仓分发）

本服务只包含应用代码，运行还需要 5090 上的以下外部资源：

1. **DA3 源码**：`app.py` 通过 `sys.path` 引用 `/home/odyss/Depth-Anything-3/src` 的 `depth_anything_3` 包（不在 PyPI），只用其推理 API（官方 Gradio 应用已于 2026-08-17 去冗余下线）。
2. **模型权重**：`/home/odyss/Depth-Anything-3/models/DA3NESTED-GIANT-LARGE-1.1`。
3. **conda 环境**：`da3`（含 torch/CUDA 等）。
4. **电子秤硬件**（dx-backend / 8070 用，非 8060）：一台四通道称重变送模块 `SJ101T2_CH4_ETH`，需与 5090 在同一局域网可达（静态 `192.168.0.80`，Modbus TCP 502，通道 1..4 → 寄存器 addr 0/2/4/6）。网络前提见 [`NETWORK.md`](NETWORK.md)。

如需迁移到其他机器，上述路径（`app.py` 中的 `DA3_ROOT` / `MODEL_DIR`、`run.sh` 中的 `HF_HOME` 与 conda python 路径）需相应调整。
