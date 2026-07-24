#!/usr/bin/env bash
# 【有公网的机器】制作离线部署包：拉齐全部 base 镜像并 docker save，
# 连同 local-stack 目录与 artifacts 打成一个 tar，之后目标机完全不需要公网。
# 前置：已执行 build-artifacts.sh（artifacts/ 就绪），本机 docker 可用且能访问镜像源。
set -euo pipefail

STACK_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUNDLE_NAME="odyss-local-stack-bundle-$(date +%Y%m%d).tar"

[ -f "$STACK_DIR/artifacts/bin/odyss-services" ] || { echo "artifacts 未就绪，先执行 build-artifacts.sh"; exit 1; }

# 与 docker-compose.yml 保持一致的 base 镜像清单（版本 pin 死，离线环境不漂移）
IMAGES=(
  "alpine:3.20"
  "postgres:18.4"
  "valkey/valkey:9.1.0"
  "minio/minio:latest"
  "minio/mc:latest"
  "nginx:alpine"
)

echo "== 拉取 base 镜像"
for img in "${IMAGES[@]}"; do
  docker pull "$img"
done

echo "== 导出镜像为 tar"
mkdir -p "$STACK_DIR/images"
docker save -o "$STACK_DIR/images/base-images.tar" "${IMAGES[@]}"
ls -lh "$STACK_DIR/images/base-images.tar"

echo "== 打离线包（local-stack 全目录：compose/Dockerfile/配置/脚本/artifacts/images）"
pushd "$STACK_DIR/.." > /dev/null
tar cf "$BUNDLE_NAME" \
  local-stack/docker-compose.yml \
  local-stack/services.Dockerfile \
  local-stack/config/runtime-config.template.yaml \
  local-stack/superadmin \
  local-stack/scripts \
  local-stack/artifacts \
  local-stack/images \
  local-stack/README.md
popd > /dev/null

echo "== 完成：$(cd "$STACK_DIR/.." && pwd)/$BUNDLE_NAME"
echo "拷到目标机后：tar xf $BUNDLE_NAME && cd local-stack && ./scripts/bootstrap.sh"
