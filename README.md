# ifa-support

5090 服务器上跑在 `0.0.0.0:8060` 的极简运维/演示 Web 服务（原目录 `~/da3-web`），单文件 FastAPI 应用，纯服务端渲染、零前端构建。包含两个页面：

- **`/` 深度图**：浏览器上传一张图 → 用 Depth Anything 3（DA3NESTED-GIANT-LARGE-1.1）返回彩色深度图（越亮 = 越近）。模型启动时常驻 GPU，之后每次请求只做推理。
- **`/weight` 电子秤实时重量**：后台线程手写 Modbus TCP 轮询两台电子秤（秤A SJ101CX @ 192.168.0.80、秤B Y31X04 @ 192.168.0.90），`/api/weights` 出 JSON，看板页每 0.5s 刷新并画迷你趋势线。

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

1. **DA3 源码**：`app.py` 通过 `sys.path` 引用 `/home/odyss/Depth-Anything-3/src` 的 `depth_anything_3` 包（不在 PyPI）。
2. **模型权重**：`/home/odyss/Depth-Anything-3/models/DA3NESTED-GIANT-LARGE-1.1`。
3. **conda 环境**：`da3`（含 torch/CUDA 等）。
4. **电子秤硬件**：需与两台秤在同一局域网可达（192.168.0.80 / 192.168.0.90，Modbus TCP 502）。

如需迁移到其他机器，上述路径（`app.py` 中的 `DA3_ROOT` / `MODEL_DIR`、`run.sh` 中的 `HF_HOME` 与 conda python 路径）需相应调整。
