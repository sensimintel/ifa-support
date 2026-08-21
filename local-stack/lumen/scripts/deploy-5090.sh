#!/usr/bin/env bash
# 【开发机】把 lumen overlay 部署 / 更新到 5090 孤本栈（odyss-ifa-*），并同步 superadmin
# 的无状态文件（nginx conf + runtime-config，MANAGEMENT.md §5 允许的对齐范围）：
#   1. rsync overlay（compose / Dockerfile / 配置 / 二进制 / 迁移）到目标机 ~/odyss-ifa-lumen
#   2. 目标机 docker build lumen 镜像 → docker compose up -d（幂等）
#   3. 同步 superadmin.conf 与渲染后的 superadmin-runtime-config.json → 重启 superadmin 容器
# 前置：已跑 ./scripts/build-lumen-artifacts.sh；目标机可 SSH（默认 odyss@192.168.100.50）。
set -euo pipefail

LUMEN_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STACK_DIR="$(cd "$LUMEN_DIR/.." && pwd)"
REMOTE="${LUMEN_DEPLOY_TARGET:-odyss@192.168.100.50}"
REMOTE_DIR="${LUMEN_DEPLOY_DIR:-/home/odyss/odyss-ifa-lumen}"
SUPERADMIN_DIR="${LUMEN_DEPLOY_SUPERADMIN_DIR:-/home/odyss/odyss-services-ifa/superadmin}"
STACK_ID="${STACK_ID:-ifa-5090}"
STACK_LABEL="${STACK_LABEL:-5090 本地栈}"

[ -f "$LUMEN_DIR/artifacts/bin/odyss-trace-collector" ] || { echo "缺 lumen 产物，先跑 scripts/build-lumen-artifacts.sh"; exit 1; }

echo "== 1/4 同步 overlay 到 $REMOTE:$REMOTE_DIR"
ssh "$REMOTE" "mkdir -p '$REMOTE_DIR'"
rsync -az --delete \
  "$LUMEN_DIR/docker-compose.yml" \
  "$LUMEN_DIR/lumen.Dockerfile" \
  "$LUMEN_DIR/config" \
  "$LUMEN_DIR/scripts" \
  "$LUMEN_DIR/artifacts" \
  "$REMOTE:$REMOTE_DIR/"

echo "== 2/4 目标机构建镜像并拉起 overlay"
ssh "$REMOTE" "cd '$REMOTE_DIR' && docker build -t odyss-lumen:ifa-stack -f lumen.Dockerfile . && docker compose up -d"

echo "== 3/4 同步 superadmin 无状态文件（nginx conf + runtime-config）"
sed -e "s|__STACK_ID__|$STACK_ID|" -e "s|__STACK_LABEL__|$STACK_LABEL|" \
  "$STACK_DIR/superadmin/superadmin-runtime-config.template.json" > /tmp/superadmin-runtime-config.rendered.json
rsync -az "$STACK_DIR/superadmin/superadmin.conf" "$REMOTE:$SUPERADMIN_DIR/superadmin.conf"
# 渲染后的 runtime-config 必须进 webroot（dist/）才能被前端 fetch 到；
# superadmin/ 层再留一份作为「当前生效配置」的可读孤本。注意 dist 同步用了 --delete，
# 每次重发 dist 后都要重跑本脚本把它补回去。
rsync -az /tmp/superadmin-runtime-config.rendered.json "$REMOTE:$SUPERADMIN_DIR/superadmin-runtime-config.json"
rsync -az /tmp/superadmin-runtime-config.rendered.json "$REMOTE:$SUPERADMIN_DIR/dist/superadmin-runtime-config.json"
rm -f /tmp/superadmin-runtime-config.rendered.json
ssh "$REMOTE" "docker restart odyss-ifa-superadmin > /dev/null"

echo "== 4/4 探活"
ssh "$REMOTE" "docker ps --format '{{.Names}}\t{{.Status}}' | grep lumen; curl -sf http://127.0.0.1:18091/lumen/api/health && echo ' <- observation 经 superadmin 反代 OK'"
echo "完成。services 的 span 会经网络别名 lumen-collector.odyss.internal 自动开始上报（无需重启 services）。"
