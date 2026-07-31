#!/usr/bin/env bash
# 【开发机 · 需要公网与代码仓】构建 lumen overlay 产物到 lumen/artifacts/：
#   - odyss-trace-collector / odyss-lumen-observation 两个 linux/amd64 静态二进制
#   - db/migrations 迁移 SQL（随产物同步到目标机由 lumen-migrate 应用）
# 版本纪律同主栈：目标分支以 manifest.env 的 LUMEN_REF 为准；仓路径可用
# stack.env / 环境变量 LUMEN_REPO 覆盖；分支不一致时告警不阻断。
set -euo pipefail

LUMEN_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STACK_DIR="$(cd "$LUMEN_DIR/.." && pwd)"
set -a
. "$STACK_DIR/manifest.env"
[ -f "$STACK_DIR/stack.env" ] && . "$STACK_DIR/stack.env"
set +a
LUMEN_REPO="${LUMEN_REPO:-$STACK_DIR/../../odyss-lumen}"

echo "== 校验 lumen 仓路径与分支（manifest：lumen=${LUMEN_REF}）"
[ -d "$LUMEN_REPO/tracing/collector" ] || { echo "找不到 odyss-lumen 仓：$LUMEN_REPO（stack.env 里 LUMEN_REPO 覆盖）"; exit 1; }
cur=$(git -C "$LUMEN_REPO" branch --show-current 2>/dev/null || echo "?")
[ "$cur" = "$LUMEN_REF" ] || echo "  ⚠️ odyss-lumen 当前分支 $cur ≠ manifest 要求 ${LUMEN_REF}（开发迭代可继续；发布前请对齐）"

mkdir -p "$LUMEN_DIR/artifacts/bin"

echo "== 构建 lumen 二进制（linux/amd64，静态链接）"
pushd "$LUMEN_REPO" > /dev/null
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o "$LUMEN_DIR/artifacts/bin/odyss-trace-collector" ./tracing/collector/cmd/odyss-trace-collector
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o "$LUMEN_DIR/artifacts/bin/odyss-lumen-observation" ./platform/cmd/server
popd > /dev/null

echo "== 同步迁移 SQL"
rm -rf "$LUMEN_DIR/artifacts/migrations"
mkdir -p "$LUMEN_DIR/artifacts/migrations"
cp "$LUMEN_REPO"/db/migrations/*.sql "$LUMEN_DIR/artifacts/migrations/"

echo "== 完成，产物列表："
ls -lh "$LUMEN_DIR/artifacts/bin" "$LUMEN_DIR/artifacts/migrations"
