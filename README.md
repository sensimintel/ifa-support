# ifa-support

> 本仓同时是 Odyss 本地化组件的**编排管理仓**：全套管理规范见 [`MANAGEMENT.md`](MANAGEMENT.md)，全栈本地化 SOP（一套拉起）见 [`local-stack/`](local-stack/README.md)，演示现场网络方案（展会拓扑与公司内模拟）见 [`NETWORK.md`](NETWORK.md)。以下为 DA3 演示服务说明。

5090 服务器上跑在 `0.0.0.0:8060` 的运维/演示 Web 服务（原目录 `~/da3-web`），单文件 FastAPI 应用，纯服务端渲染、零前端构建。一个端口同时挂官方 Gradio 与自研面板：

- **`/` 分栏首页**：左右两栏 iframe 对比 —— 左栏嵌官方 Gradio UI（`/gradio`），右栏嵌自研扩展面板（`/panel`）。顶栏有「浅体验区展示页」入口。
- **`/experience` 浅体验区展示页**：IFA 展台品牌化全屏 UI（Figma「IFA 专项 · 浅体验区」实现）。全屏背景为实时点云（默认 SAM3 高亮点云，右下临时按钮可在 高亮点云/LA点云/SAM3点云 三种来源间切换；任何情况不展示设备原图，产物未就绪时保持黑场）；右侧状态区在「Place your food here」待机态与识别成功态（浅色玻璃卡片：名称 / 英文短描述 / 卡路里 / 蛋白·碳水·脂肪 / Good·Neutral·Bad 健康分级，营养数字按画面可见份量估算）间切换，识别成功时画面上食物位置处另有定位小标（名称 + 卡路里，锚在 VLM box、随去重合并刷新位置）；另有流水视图（临时按钮进入）展示当日识别记录。品牌字体（ABC Arizona Serif / Seabirds，trial 版）在 `static/fonts/`，经 `/static` 提供。`?demo=1` 为无后端目检模式。
- **`/gradio` 官方 Gradio UI**：通过 `gr.mount_gradio_app` 挂在同一 FastAPI 上（点云 / 网格 / 3D 量距等）。app.py 内含 gradio 6 兼容 shim，静默丢弃已废弃 kwargs，避免改动上游 DA3 源码。
- **`/panel` 扩展面板（深度 / 点云 / 网格）**：浏览器上传一张图 + 选产物类型 + 调参，用 Depth Anything 3（DA3NESTED-GIANT-LARGE-1.1）出三种产物：
  - **深度图**：彩色深度图（越亮 = 越近）；
  - **点云 + 相机（GLB）**：DA3 官方 `scene.glb` 导出（点云 + 相机线框），网页内 `<model-viewer>` 可鼠标 3D 转视角；
  - **网格 mesh（GLB）**：由深度反投影自建带顶点色的三角网格，同样可 3D 转视角。
  - 可调参数：`process_res`、`conf_thresh_percentile`、`num_max_points`、`show_cameras`。`/api/infer` 出 JSON，GLB 经 `/glb/{token}/scene.glb` 提供（只保留最近若干次、自动清理）。

> **电子秤不在本服务**：四通道食物秤（净重 / 毛重、软件去皮）与桌边分组由独立的 dx-backend 提供（`0.0.0.0:8070`，见 `dx_backend.py`），8060 已无 `/weight` 与 `/api/weights`。秤的地址与网络前提见 [`NETWORK.md`](NETWORK.md)。

> **模型单例 + GPU 共存**：`/gradio` 与 `/panel` **共用同一份 DA3 模型权重**（官方 UI 的 `ModelInference.initialize_model` 被改为复用本服务的共享单例），并用一把 GPU 锁串行化推理——因为 5090 显存与产线服务共享，进程内加载两份权重（约 2×6.5GB）会撑爆显存。模型懒加载：启动不占显存，首次推理才加载一份（约 6.5GB，推理峰值约 8.6GB@process_res=504）。process_res 调太高或产线显存吃紧时可能 OOM，此时调低 process_res 重试。

## 文件

| 文件 | 说明 |
|---|---|
| `app.py` | 8060 的全部服务端逻辑（FastAPI 应用 `app:app`，深度推理 + 内嵌 HTML 页面） |
| `static/` | 静态资源（`/experience` 用的品牌字体等），经 `/static` 路径提供 |
| `run.sh` | 用 `da3` conda 环境在 `0.0.0.0:8060` 起服务的启动脚本 |
| `exp_app.py` | **感知链路实验台（8061）**：与产线 8060 完全隔离的「完整识别链路」实验环境，UI 复刻 8060 主页——左栏 设备帧/实验检测标注帧/点云产物(共享 8060 产线 DA3 产物)，右栏 完整 VLM 识别卡片流（与 /recog 同款 UI，prompt/解析/五道合并闸门移植自产线、可独立魔改）。链路=SAM3+VLM（不走 LA）：「调节」抽屉热改 采样率/SAM3 食物·液体查询词/score 阈值/presence α·检测阈值口径/识别节流，留 embedding 匹配插槽（`_embed_match()`）。帧取自 8060 只读接口，SAM3 走无状态 `/v1/segment` 不碰产线流式，不加载新模型。手动起停：`nohup ./run-exp.sh > exp.log 2>&1 &` |
| `run-exp.sh` | 起 8061 实验台的启动脚本（复用 `da3` conda 环境与仓根 `.env`） |
| `dx_backend.py` | **深体验区后端（8070）**：四通道食物秤读数与软件去皮、桌边分组绑定，独立于 8060 |
| `run-dx.sh` | 起 8070 的启动脚本（复用 `da3` conda 环境） |
| `dx-backend.service` | 8070 的 systemd 单元 |
| `deploy.sh` | 5090 上一键部署：`git pull` + 重启服务（8060 与 8070 一并重启，systemd 优先） |
| `da3-web.service` | 可选 systemd 单元（正规化开机自启/重启） |
| `requirements.txt` | pip 依赖（不含 `depth_anything_3`，见下） |
| `model/` | **三个模型服务（DA3 / LocateAnything / SAM3）的一键拉起脚本与部署信息**，含 SAM3 推理服务源码（流式长记忆版）与 systemd 单元，见 `model/README.md` |

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

1. **DA3 源码**：`app.py` 通过 `sys.path` 引用 `/home/odyss/Depth-Anything-3/src` 的 `depth_anything_3` 包（不在 PyPI），既用其推理 API，也用其自带的官方 Gradio 应用 `depth_anything_3.app.gradio_app`。
2. **模型权重**：`/home/odyss/Depth-Anything-3/models/DA3NESTED-GIANT-LARGE-1.1`。
3. **conda 环境**：`da3`（含 torch/CUDA 等）。
4. **电子秤硬件**（dx-backend / 8070 用，非 8060）：一台四通道称重变送模块 `SJ101T2_CH4_ETH`，需与 5090 在同一局域网可达（静态 `192.168.0.80`，Modbus TCP 502，通道 1..4 → 寄存器 addr 0/2/4/6）。网络前提见 [`NETWORK.md`](NETWORK.md)。

如需迁移到其他机器，上述路径（`app.py` 中的 `DA3_ROOT` / `MODEL_DIR`、`run.sh` 中的 `HF_HOME` 与 conda python 路径）需相应调整。
