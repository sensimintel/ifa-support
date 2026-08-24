# model/ · 模型与服务一键拉起手册

本目录沉淀 ifa-support 演示栈依赖的**两个模型服务**（DA3 / SAM3）的全部部署脚本与信息。

**第一性目标**：在一台「稳定、有外网、显存足够」的新服务器上，凭本目录 + 本仓根目录，
一次性拉起所有模型、对应服务 gateway 与全部依赖，复刻 5090 现网的完整演示栈。
（「5090」沿用主机称谓；2026-08-24 起双卡：**RTX Pro 6000 Blackwell 96GB（独占 vLLM）+ RTX 5090 32GB（SAM3/DA3）**，不做全文改名。）

## 架构与端口

```
设备帧 → app.py(:8060, 本仓根目录, DA3 进程内推理 + 各页面 gateway)
              ├─ SAM3            http://127.0.0.1:8013   (sam3_server.py, systemd)
              └─ 识别 Qwen3-VL    RECOG_ENDPOINT          (本机 vLLM 裸进程 :8000, 不在本仓, 经 .env 配置)
```

| 端口 | 服务 | 来源 |
|---|---|---|
| 8060 | ifa web（本仓 `app.py`，含 DA3 推理） | 本仓根目录 `run.sh` / `da3-web.service` |
| 8013 | SAM3 | `model/sam3/` |

## 硬件 / 系统前置

- NVIDIA 驱动 ≥ 580（SAM3 环境用 torch cu130 需 CUDA 13；DA3 环境用 cu128。现网驱动版本以 `nvidia-smi` 实测为准——2026-08 换卡 RTX Pro 6000 后驱动已随卡更新，不要按旧记录写死）
- 外网：github.com、huggingface.co（国内可 `export HF_ENDPOINT=https://hf-mirror.com`；
  **例外：`facebook/sam3` 是 gated 仓，必须官方源 + `HF_TOKEN`**，镜像上没有）
- 磁盘：权重合计约 18GB（DA3 nested giant ~13GB、sam3.pt 3.45GB）+ 各 venv 若干 GB

## 显存预算（2026-08-24 起双卡现网口径）

双卡布局：GPU0 = RTX Pro 6000 Blackwell 96GB **独占给 vLLM**；GPU1 = RTX 5090 32GB 归 SAM3 + DA3。三个 unit 都在 systemd 里钉卡，任何重启都各归各卡：sam3/da3-web 用 `CUDA_VISIBLE_DEVICES=<GPU UUID>`（torch 认 UUID；CUDA 默认枚举「最快卡优先」，裸 index 不稳定）；vllm 用 `CUDA_DEVICE_ORDER=PCI_BUS_ID` + `CUDA_VISIBLE_DEVICES=0`（**vLLM 按整数解析该变量、不认 UUID**——UUID 会 ValidationError 崩溃循环，2026-08-24 实测；PCI 序稳定，0=02:00.0=Pro 6000）。

| 服务 | 落卡 | 显存 | 控制手段 |
|---|---|---|---|
| 本机 vLLM（`:8000`，Qwen3.6-35B-A3B-FP8，不在本目录管理） | GPU0 Pro 6000（`GPU-5938e8ce…`） | ~86G | `gpu-memory-utilization 0.90`（独占卡口径） |
| SAM3 | GPU1 RTX 5090（`GPU-565f6e8d…`） | 实占 ~7G，上限 0.9×31.8G≈28.6G | `SAM3_MEM_FRACTION=0.9`（按可见卡基数）；流式窗口 `window` 控瞬时占用 |
| DA3（app.py 进程内，懒加载） | GPU1 RTX 5090（同上） | 峰值 ~8.6G@process_res=504 | `process_res` 越高越吃；不为其预留（显存不足任其失败，OOM 时调低重试） |

GPU1 上 SAM3 常态 ~7G + DA3 峰值 ~8.6G，32G 内互不影响；SAM3 上限 0.9 是隔离兜底而非预留。旧「96G 单卡共享」与更早的「32G 紧张 / OOM 风险」口径均作废。

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

- **识别 Qwen3-VL 不在本仓**：现为 5090 宿主裸进程 `vllm serve :8000`（Qwen3.6-35B-A3B-FP8，别名 `gemini-3.1-pro-preview`），本机 vLLM 启动配置即模型名正源，本服务只经 `.env` 接入；旧的 GCP g4-01 链路（frp / 反向 SSH 隧道）已废弃。

## 5090 现网对照（排障用）

| 内容 | 5090 路径 |
|---|---|
| 本仓 checkout | `~/da3-web`（部署机只 pull，禁 commit/push） |
| DA3 源码+权重 | `~/Depth-Anything-3`（models/ 下权重），conda env `da3`(py3.10) |
| SAM3 venv | `~/sam3-env`(py3.12)，权重 `~/models/sam3/sam3.pt` |
| SAM3 服务 | systemd `sam3.service`（应指向 `~/da3-web/model/sam3/sam3_server.py`） |
