# ifa-support

5090 服务器上跑在 `0.0.0.0:8060` 的运维/演示 Web 服务（原目录 `~/da3-web`），单文件 FastAPI 应用，纯服务端渲染、零前端构建。一个端口同时挂官方 Gradio 与自研面板：

- **`/` 分栏首页**：左右两栏 iframe 对比 —— 左栏嵌官方 Gradio UI（`/gradio`），右栏嵌自研扩展面板（`/panel`）。
- **`/gradio` 官方 Gradio UI**：通过 `gr.mount_gradio_app` 挂在同一 FastAPI 上（点云 / 网格 / 3D 量距等）。app.py 内含 gradio 6 兼容 shim，静默丢弃已废弃 kwargs，避免改动上游 DA3 源码。
- **`/panel` 扩展面板（深度 / 点云 / 网格）**：浏览器上传一张图 + 选产物类型 + 调参，用 Depth Anything 3（DA3NESTED-GIANT-LARGE-1.1）出三种产物：
  - **深度图**：彩色深度图（越亮 = 越近）；
  - **点云 + 相机（GLB）**：DA3 官方 `scene.glb` 导出（点云 + 相机线框），网页内 `<model-viewer>` 可鼠标 3D 转视角；
  - **网格 mesh（GLB）**：由深度反投影自建带顶点色的三角网格，同样可 3D 转视角。
  - 可调参数：`process_res`、`conf_thresh_percentile`、`num_max_points`、`show_cameras`。`/api/infer` 出 JSON，GLB 经 `/glb/{token}/scene.glb` 提供（只保留最近若干次、自动清理）。
- **`/weight` 电子秤实时重量**：后台线程手写 Modbus TCP 轮询两台电子秤（秤A SJ101CX @ 192.168.0.80、秤B Y31X04 @ 192.168.0.90），`/api/weights` 出 JSON，看板页每 0.5s 刷新并画迷你趋势线。

> **模型单例 + GPU 共存**：`/gradio` 与 `/panel` **共用同一份 DA3 模型权重**（官方 UI 的 `ModelInference.initialize_model` 被改为复用本服务的共享单例），并用一把 GPU 锁串行化推理——因为 5090 显存与产线服务共享，进程内加载两份权重（约 2×6.5GB）会撑爆显存。模型懒加载：启动不占显存，首次推理才加载一份（约 6.5GB，推理峰值约 8.6GB@process_res=504）。process_res 调太高或产线显存吃紧时可能 OOM，此时调低 process_res 重试。

## 文件

| 文件 | 说明 |
|---|---|
| `app.py` | 全部服务端逻辑（FastAPI 应用 `app:app`，含深度推理 + 电子秤模块 + 内嵌 HTML 页面） |
| `run.sh` | 用 `da3` conda 环境在 `0.0.0.0:8060` 起服务的启动脚本 |
| `deploy.sh` | 5090 上一键部署：`git pull` + 重启服务（systemd 优先，否则 kill+nohup） |
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
4. **电子秤硬件**：需与两台秤在同一局域网可达（192.168.0.80 / 192.168.0.90，Modbus TCP 502）。

如需迁移到其他机器，上述路径（`app.py` 中的 `DA3_ROOT` / `MODEL_DIR`、`run.sh` 中的 `HF_HOME` 与 conda python 路径）需相应调整。
