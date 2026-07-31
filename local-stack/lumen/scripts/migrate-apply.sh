#!/bin/sh
# lumen 迁移应用（容器内执行）：按文件名顺序应用 /migrate/migrations/*.sql，
# 以 lumen_local_migrations 记账表保证幂等——已应用的版本跳过，重复 up 安全。
# 每个迁移单事务应用（对应 manifest transaction_mode: automatic）。
set -eu

psql -v ON_ERROR_STOP=1 -q -c \
  "CREATE TABLE IF NOT EXISTS lumen_local_migrations (version integer PRIMARY KEY, file text NOT NULL, applied_at timestamptz NOT NULL DEFAULT now())"

for f in $(ls /migrate/migrations/*.sql | sort); do
  base=$(basename "$f")
  version=$(printf '%s' "$base" | sed -n 's/^0*\([0-9][0-9]*\)_.*/\1/p')
  if [ -z "$version" ]; then
    echo "跳过无版本前缀的文件：$base"
    continue
  fi
  applied=$(psql -tA -c "SELECT 1 FROM lumen_local_migrations WHERE version = $version")
  if [ "$applied" = "1" ]; then
    echo "已应用，跳过：$base"
    continue
  fi
  echo "应用迁移：$base"
  psql -v ON_ERROR_STOP=1 -q --single-transaction -f "$f"
  psql -v ON_ERROR_STOP=1 -q -c \
    "INSERT INTO lumen_local_migrations (version, file) VALUES ($version, '$base')"
done

echo "lumen 迁移全部应用完成"
