# ifa-support

5090 服务器上跑在 `0.0.0.0:8060` 的运维/演示 Web 服务（原目录 `~/da3-web`），单文件 FastAPI 应用，纯服务端渲染、零前端构建。一个端口同时挂官方 Gradio 与自研面板：

- **`/` 分栏首页**：左右两栏 iframe 对比 —— 左栏嵌官方 Gradio UI（`/gradio`），右栏嵌自研扩展面板（`/panel`）。
- **`/gradio` 官方 Gradio UI**：通过 `gr.mount_gradio_app` 挂在同一 FastAPI 上，复用本地模型档（点云 / 网格 / 3D 量距等）。app.py 内含 gradio 6 兼容 shim，静默丢弃已废弃 kwargs，避免改动上游 DA3 源码。
- **`/panel` 深度图**：浏览器上传一张图 → 用 Depth Anything 3（DA3NESTED-GIANT-LARGE-1.1）返回彩色深度图（越亮 = 越近）。模型懒加载到 GPU。
- **`/weight` 电子秤实时重量**：后台线程手写 Modbus TCP 轮询两台电子秤（秤A SJ101CX @ 192.168.0.80、秤B Y31X04 @ 192.168.0.90），`/api/weights` 出 JSON，看板页每 0.5s 刷新并画迷你趋势线。

> 模型均为**懒加载**：启动时不占显存，`/gradio` 与 `/panel` 各自在首次推理时加载一份权重。

## 文件

| 文件 | 说明 |
|---|---|
| `app.py` | 全部服务端逻辑（FastAPI 应用 `app:app`，含深度推理 + 电子秤模块 + 内嵌 HTML 页面） |
| `run.sh` | 用 `da3` conda 环境在 `0.0.0.0:8060` 起服务的启动脚本 |
| `requirements.txt` | pip 依赖（不含 `depth_anything_3`，见下） |

## 运行

```bash
./run.sh
# 等价于：
# export HF_HOME=/home/odyss/Depth-Anything-3/models
# python -m uvicorn app:app --host 0.0.0.0 --port 8060
```

局域网内访问 `http://<5090局域网IP>:8060`。

## 外部依赖（不随本仓分发）

本服务只包含应用代码，运行还需要 5090 上的以下外部资源：

1. **DA3 源码**：`app.py` 通过 `sys.path` 引用 `/home/odyss/Depth-Anything-3/src` 的 `depth_anything_3` 包（不在 PyPI），既用其推理 API，也用其自带的官方 Gradio 应用 `depth_anything_3.app.gradio_app`。
2. **模型权重**：`/home/odyss/Depth-Anything-3/models/DA3NESTED-GIANT-LARGE-1.1`。
3. **conda 环境**：`da3`（含 torch/CUDA 等）。
4. **电子秤硬件**：需与两台秤在同一局域网可达（192.168.0.80 / 192.168.0.90，Modbus TCP 502）。

如需迁移到其他机器，上述路径（`app.py` 中的 `DA3_ROOT` / `MODEL_DIR`、`run.sh` 中的 `HF_HOME` 与 conda python 路径）需相应调整。
