#!/usr/bin/env bash
# 【有公网的机器】制作离线部署包（唯一入口：./stack bundle）：按 manifest.env 拉齐
# 全部 base 镜像并 docker save，连同 local-stack 目录与 artifacts 打成一个 tar，
# 之后目标机完全不需要公网。frpc 镜像一并打入（real 模式即插即用）。
# 前置：已执行 ./stack build（artifacts/ 就绪），本机 docker 可用且能访问镜像源。
set -euo pipefail

STACK_DIR="$(cd "$(dirname "$0")/.." && pwd)"
set -a
. "$STACK_DIR/manifest.env"
set +a
BUNDLE_NAME="odyss-local-stack-bundle-$(date +%Y%m%d).tar"

[ -f "$STACK_DIR/artifacts/bin/odyss-services" ] || { echo "artifacts 未就绪，先执行 ./stack build"; exit 1; }

# base 镜像清单唯一来源：manifest.env（compose 经 ${IMG_*} 引用同一批值，不漂移）
IMAGES=(
  "$IMG_ALPINE"
  "$IMG_POSTGRES"
  "$IMG_REDIS"
  "$IMG_MINIO"
  "$IMG_MINIO_MC"
  "$IMG_NGINX"
  "$IMG_FRPC"
)

echo "== 拉取 base 镜像（manifest.env 清单）"
for img in "${IMAGES[@]}"; do
  docker pull "$img"
done

echo "== 导出镜像为 tar"
mkdir -p "$STACK_DIR/images"
docker save -o "$STACK_DIR/images/base-images.tar" "${IMAGES[@]}"
ls -lh "$STACK_DIR/images/base-images.tar"

echo "== 打离线包（stack 入口 + manifest + 模板 + 脚本 + artifacts + images）"
pushd "$STACK_DIR/.." > /dev/null
tar cf "$BUNDLE_NAME" \
  local-stack/stack \
  local-stack/manifest.env \
  local-stack/stack.env.example \
  local-stack/docker-compose.yml \
  local-stack/services.Dockerfile \
  local-stack/config/runtime-config.template.yaml \
  local-stack/config/frpc-visitor.template.toml \
  local-stack/superadmin/superadmin.conf \
  local-stack/superadmin/superadmin-runtime-config.template.json \
  local-stack/scripts \
  local-stack/artifacts \
  local-stack/images \
  local-stack/README.md
popd > /dev/null

echo "== 完成：$(cd "$STACK_DIR/.." && pwd)/$BUNDLE_NAME"
echo "拷到目标机后：tar xf $BUNDLE_NAME && cd local-stack && ./stack bootstrap"
