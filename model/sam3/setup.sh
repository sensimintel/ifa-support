#!/usr/bin/env bash
# SAM3 一键准备：venv + 依赖 + 权重下载 + systemd 服务安装。
#
# ⚠ facebook/sam3 是 HF gated 仓：权重下载必须官方源 + 有权限的 HF_TOKEN（镜像上没有）。
#   先 export HF_TOKEN=hf_xxx 再跑本脚本。
#
# 可覆盖的环境变量：
#   SAM3_ENV     venv 目录     默认 ~/sam3-env（需系统有 python3.12）
#   SAM3_CKPT_DIR 权重目录     默认 ~/models/sam3
set -euo pipefail

SAM3_ENV="${SAM3_ENV:-$HOME/sam3-env}"
SAM3_CKPT_DIR="${SAM3_CKPT_DIR:-$HOME/models/sam3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "!! 未设置 HF_TOKEN（facebook/sam3 是 gated 仓，必须带 token 走官方源下载）" >&2
  exit 1
fi

echo "==> 1/4 创建 venv（python3.12）并安装依赖（torch cu130，需驱动支持 CUDA 13）"
if [ ! -x "${SAM3_ENV}/bin/python" ]; then
  python3.12 -m venv "${SAM3_ENV}"
fi
"${SAM3_ENV}/bin/pip" install --upgrade pip >/dev/null
"${SAM3_ENV}/bin/pip" install -r "${SCRIPT_DIR}/requirements.txt"

echo "==> 2/4 下载权重 sam3.pt（3.45GB，gated，走官方源）"
mkdir -p "${SAM3_CKPT_DIR}"
HF_ENDPOINT=https://huggingface.co "${SAM3_ENV}/bin/hf" download facebook/sam3 sam3.pt \
  --local-dir "${SAM3_CKPT_DIR}" --token "${HF_TOKEN}" 2>/dev/null \
  || { "${SAM3_ENV}/bin/pip" install "huggingface-hub[cli]>=0.36,<1" >/dev/null; \
       HF_ENDPOINT=https://huggingface.co "${SAM3_ENV}/bin/hf" download facebook/sam3 sam3.pt \
         --local-dir "${SAM3_CKPT_DIR}" --token "${HF_TOKEN}"; }

echo "==> 3/4 安装 systemd 服务（代码指向本仓 checkout 的 model/sam3/sam3_server.py）"
sudo cp "${SCRIPT_DIR}/sam3.service" /etc/systemd/system/sam3.service
sudo systemctl daemon-reload
sudo systemctl enable --now sam3.service

echo "==> 4/4 健康检查（模型加载约 1 分钟）"
for i in $(seq 1 30); do
  if curl -sf http://127.0.0.1:8013/health | grep -q '"ok"'; then
    echo "SAM3 OK：http://127.0.0.1:8013（/v1/segment /v1/track /v1/stream/*）"
    exit 0
  fi
  sleep 4
done
echo "!! SAM3 健康检查超时，journalctl -u sam3 -f 排障" >&2
exit 1
