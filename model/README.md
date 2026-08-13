# model/ · 模型与服务一键拉起手册

本目录沉淀 ifa-support 演示栈依赖的**两个模型服务**（DA3 / SAM3）的全部部署脚本与信息。

**第一性目标**：在一台「稳定、有外网、显存足够」的新服务器上，凭本目录 + 本仓根目录，
一次性拉起所有模型、对应服务 gateway 与全部依赖，复刻 5090 现网的完整演示栈。

## 架构与端口

```
设备帧 → app.py(:8060, 本仓根目录, DA3 进程内推理 + 各页面 gateway)
              ├─ SAM3            http://127.0.0.1:8013   (sam3_server.py, systemd)
              └─ 识别 Qwen3-VL    RECOG_ENDPOINT          (外部服务, 不在本仓, 经 .env 配置)
```

| 端口 | 服务 | 来源 |
|---|---|---|
| 8060 | ifa web（本仓 `app.py`，含 DA3 推理） | 本仓根目录 `run.sh` / `da3-web.service` |
| 8013 | SAM3 | `model/sam3/` |

## 硬件 / 系统前置

- NVIDIA 驱动 ≥ 580（SAM3 环境用 torch cu130 需 CUDA 13；DA3 环境用 cu128。5090 现网驱动 580.95）
- 外网：github.com、huggingface.co（国内可 `export HF_ENDPOINT=https://hf-mirror.com`；
  **例外：`facebook/sam3` 是 gated 仓，必须官方源 + `HF_TOKEN`**，镜像上没有）
- 磁盘：权重合计约 18GB（DA3 nested giant ~13GB、sam3.pt 3.45GB）+ 各 venv 若干 GB

## 显存预算（5090 32GB 现网实测）

| 服务 | 显存 | 控制手段 |
|---|---|---|
| DA3（app.py 进程内） | ~9.2G | `process_res` 越高越吃 |
| SAM3 | 实占 ~4.2G，上限 9G | `SAM3_MEM_FRACTION=0.28`；流式窗口 `window` 控瞬时占用 |

## 拉起顺序（新机器从零）

```bash
# 1. DA3：上游源码 + 权重 + venv（app.py 进程内 import，无独立服务）
model/da3/setup.sh

# 2. SAM3：venv + 权重(需 HF_TOKEN) + systemd 服务
model/sam3/setup.sh

# 3. web 本体：本仓根目录（先按新机器路径改 run.sh 里的 python 路径与 .env）
cp da3-web.service /etc/systemd/system/ && systemctl enable --now da3-web
```

`.env`（仓根，gitignore）契约：`RECOG_ENDPOINT` / `RECOG_API_KEY` / `RECOG_MODEL`（识别 Qwen，
外部服务）；`SAM3_ENDPOINT`（默认 `http://127.0.0.1:8013`）。

## 已知缺口（如实记录）

- **识别 Qwen3-VL 不在本仓**：GCP g4-01 的 Qwen3.6-35B 属外部依赖，只经 `.env` 接入。

## 5090 现网对照（排障用）

| 内容 | 5090 路径 |
|---|---|
| 本仓 checkout | `~/da3-web`（部署机只 pull，禁 commit/push） |
| DA3 源码+权重 | `~/Depth-Anything-3`（models/ 下权重），conda env `da3`(py3.10) |
| SAM3 venv | `~/sam3-env`(py3.12)，权重 `~/models/sam3/sam3.pt` |
| SAM3 服务 | systemd `sam3.service`（应指向 `~/da3-web/model/sam3/sam3_server.py`） |
