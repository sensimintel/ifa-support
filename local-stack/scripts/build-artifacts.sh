#!/usr/bin/env bash
# 【开发机 · 需要公网与代码仓】构建全部业务产物到 artifacts/：
#   - odyss-services / odyss-migrate / mockllm 三个 linux/amd64 静态二进制
#   - superadmin 前端 dist
# 前置：本机有 Go 工具链与 Node，且 clone 了 odyss-services（ifa 分支）与 odyss-superadmin（main 分支）。
set -euo pipefail

STACK_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SERVICES_REPO="${SERVICES_REPO:-$STACK_DIR/../../odyss-services}"
SUPERADMIN_REPO="${SUPERADMIN_REPO:-$STACK_DIR/../../odyss-superadmin}"

echo "== 校验代码仓路径"
[ -d "$SERVICES_REPO/cmd/odyss-services" ] || { echo "找不到 odyss-services 仓：$SERVICES_REPO（可用环境变量 SERVICES_REPO 覆盖）"; exit 1; }
[ -f "$SUPERADMIN_REPO/package.json" ] || { echo "找不到 odyss-superadmin 仓：$SUPERADMIN_REPO（可用环境变量 SUPERADMIN_REPO 覆盖）"; exit 1; }

mkdir -p "$STACK_DIR/artifacts/bin"

echo "== 构建 Go 二进制（linux/amd64，静态链接）"
pushd "$SERVICES_REPO" > /dev/null
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o "$STACK_DIR/artifacts/bin/odyss-services" ./cmd/odyss-services
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o "$STACK_DIR/artifacts/bin/odyss-migrate" ./cmd/odyss-migrate
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o "$STACK_DIR/artifacts/bin/mockllm" ./tests/e2e/mockllm
popd > /dev/null

echo "== 构建 superadmin 前端"
pushd "$SUPERADMIN_REPO" > /dev/null
[ -d node_modules ] || npm install
npm run build
popd > /dev/null
rm -rf "$STACK_DIR/artifacts/superadmin-dist"
cp -R "$SUPERADMIN_REPO/dist" "$STACK_DIR/artifacts/superadmin-dist"
# dist 自带的 Cloudflare 版 runtime-config 去掉，运行时由 compose 挂载本目录的版本
rm -f "$STACK_DIR/artifacts/superadmin-dist"/superadmin-runtime-config*.json

echo "== 完成，产物列表："
ls -lh "$STACK_DIR/artifacts/bin"
du -sh "$STACK_DIR/artifacts/superadmin-dist"
