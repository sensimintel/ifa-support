#!/usr/bin/env bash
# 【目标机】从 backup.sh 产出的备份目录恢复数据（postgres + minio + 运行配置）。
# 用法：./scripts/restore.sh backups/<时间戳>
# 破坏性操作：会清空并覆盖当前数据库与 minio 数据，执行前确认备份目录无误。
set -euo pipefail

STACK_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${1:?用法: restore.sh <备份目录，如 backups/20260724-030000>}"
if [ -f "$SRC/odyss_services.dump" ]; then
  SRC_ABS="$(cd "$SRC" && pwd)"
else
  SRC_ABS="$STACK_DIR/$SRC"
fi
[ -f "$SRC_ABS/odyss_services.dump" ] || { echo "备份目录里没有 odyss_services.dump：$SRC_ABS"; exit 1; }

read -r -p "将覆盖当前全部数据，确认恢复自 $SRC_ABS ？(yes/no) " ANSWER
[ "$ANSWER" = "yes" ] || { echo "已取消"; exit 1; }

echo "== 停业务容器（存储容器保持运行）"
docker compose -f "$STACK_DIR/docker-compose.yml" stop odyss-services superadmin llm-mock

echo "== 恢复 postgres"
docker exec odyss-local-postgres psql -U odyss -d postgres \
  -c "DROP DATABASE IF EXISTS odyss_services WITH (FORCE); CREATE DATABASE odyss_services OWNER odyss;"
docker exec -i odyss-local-postgres pg_restore -U odyss -d odyss_services --no-owner < "$SRC_ABS/odyss_services.dump"

echo "== 恢复 minio 数据卷"
docker compose -f "$STACK_DIR/docker-compose.yml" stop minio
docker run --rm -v odyss-local-minio-data:/data -v "$SRC_ABS":/backup:ro alpine:3.20 \
  sh -c "rm -rf /data/* && tar xzf /backup/minio-data.tgz -C /data"

echo "== 恢复运行配置（如备份中有）"
[ -f "$SRC_ABS/runtime-config.yaml" ] && cp "$SRC_ABS/runtime-config.yaml" "$STACK_DIR/config/runtime-config.yaml"

echo "== 重启全栈"
docker compose -f "$STACK_DIR/docker-compose.yml" up -d
echo "== 恢复完成"
