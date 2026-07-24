#!/usr/bin/env bash
# 【目标机】数据备份：postgres 逻辑备份（pg_dump 自定义格式）+ minio 数据卷 tar。
# 在线执行，无需停机。备份落 backups/<时间戳>/，默认滚动保留最近 7 份。
# 建议配 cron 每日执行，例如：
#   0 3 * * * /path/to/local-stack/scripts/backup.sh >> /path/to/local-stack/backups/backup.log 2>&1
set -euo pipefail

STACK_DIR="$(cd "$(dirname "$0")/.." && pwd)"
KEEP="${KEEP:-7}"
TS=$(date +%Y%m%d-%H%M%S)
DEST="$STACK_DIR/backups/$TS"
mkdir -p "$DEST"

echo "== 备份 postgres（pg_dump -Fc）"
docker exec odyss-local-postgres pg_dump -U odyss -d odyss_services -Fc > "$DEST/odyss_services.dump"

echo "== 备份 minio 数据卷"
docker run --rm -v odyss-local-minio-data:/data:ro -v "$DEST":/backup alpine:3.20 \
  tar czf /backup/minio-data.tgz -C /data .

echo "== 备份运行配置（含密钥，注意保管）"
cp "$STACK_DIR/config/runtime-config.yaml" "$DEST/runtime-config.yaml"

echo "== 滚动清理（保留最近 $KEEP 份）"
ls -1d "$STACK_DIR"/backups/*/ 2>/dev/null | sort | head -n -"$KEEP" | xargs -r rm -rf

echo "== 完成：$DEST"
ls -lh "$DEST"
