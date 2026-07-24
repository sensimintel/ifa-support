#!/usr/bin/env bash
# LocateAnything 一键引导（薄封装）。
#
# LA 的部署产物（vLLM + ViT gateway ×2 + nginx LB 的 compose、模型下载、健康检查）
# **版本化在 odyss-models 仓的 deploy/gpu5090/**——那是生产 IaC，本脚本刻意不复制一份
# 进本仓（避免两处分叉腐烂），只做：克隆 → 下载模型 → 起 LA 三件套。
#
# 前置：docker + nvidia-container-toolkit；github 私仓 sensimintel/odyss-models 的访问权。
#
# ⚠ 已知缺口（来自 odyss-models scripts/download_models.sh 头注）：
#   vLLM 侧需要「LocateAnything-3B 拆出的纯 Qwen2 目录」+「独立 embed 权重」两个拆分产物，
#   干净机拆分脚本在 odyss-models 标注待补。新机器需先从 5090 拷贝：
#     scp -r odyss@192.168.0.50:~/locateanything-vllm/locate_qwen2_model  <本机路径>
#     scp    odyss@192.168.0.50:~/locateanything-vllm/qwen2_embed_tokens.safetensors <本机路径>
#   再在 deploy/gpu5090/.env 里把 LOCATEANYTHING_QWEN2_DIR / LOCATEANYTHING_EMBED_PATH 指过去。
#
# 可覆盖的环境变量：
#   MODELS_REPO_DIR  odyss-models 克隆位置   默认 ~/odyss-models
set -euo pipefail

MODELS_REPO_DIR="${MODELS_REPO_DIR:-$HOME/odyss-models}"

echo "==> 1/3 克隆/更新 odyss-models（LA 部署 IaC 所在仓）"
if [ ! -d "${MODELS_REPO_DIR}/.git" ]; then
  git clone git@github.com:sensimintel/odyss-models.git "${MODELS_REPO_DIR}"
else
  git -C "${MODELS_REPO_DIR}" pull --ff-only
fi
cd "${MODELS_REPO_DIR}/deploy/gpu5090"

echo "==> 2/3 下载 LA 全模型 + SigLIP2（拆分产物见脚本头注的缺口说明）"
./scripts/download_models.sh

echo "==> 3/3 起 LA 三件套（vLLM + gateway-1 + nginx LB；gateway-2 视显存自行加）"
docker compose -f compose.gpu.yml up -d locateanything-vllm locateanything-gateway-1 locateanything-lb

echo "==> 健康检查（vLLM 首次加载模型需数分钟）"
for i in $(seq 1 60); do
  if curl -sf http://127.0.0.1:8010/health >/dev/null 2>&1; then
    echo "LA gateway OK（入口 http://127.0.0.1:8000，OpenAI 兼容 /v1/chat/completions，model=nvidia/LocateAnything-3B）"
    exit 0
  fi
  sleep 10
done
echo "!! LA 健康检查超时，用 docker logs locateanything-vllm / locateanything-server-1 排障" >&2
exit 1
