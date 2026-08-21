#!/usr/bin/env bash
# 【开发机】把 ifa 分支最新的业务栈更新到 5090 演示机（唯一入口：./stack deploy-5090）：
#   1. 校验本地两仓停在 ifa tip 且工作树干净（部署产物必须可追溯到某个 commit）
#   2. 本机构建 Go 二进制与 superadmin dist（复用 ./stack build）
#   3. 产物 rsync 到目标机临时上下文 → docker build 出 odyss-services:ifa
#   4. 跑 migrate → 只重建 services / llm-mock 两个无状态容器
#   5. 发 superadmin dist（bind mount，换文件即生效）
#   6. 探活
#
# ⚠️ 铁律：5090 的 ~/odyss-services-ifa 是**手工演化的孤本**，postgres 与 minio
# 都没有命名卷（见 MANAGEMENT.md §5），数据全在容器可写层里。所以本脚本对 compose
# 的每一次调用都必须带 --no-deps 并显式点名服务，**绝不允许出现裸 `docker compose up -d`**
# ——那会按 depends_on 连带重建 postgres/minio，一次误操作就把整个演示库清空。
# 改这个脚本时请守住这条：任何新增的 compose 调用都要点名服务 + --no-deps。
#
# 前置：本机有 Go 工具链与 Node；能免密 SSH 到目标机。
set -euo pipefail

STACK_DIR="$(cd "$(dirname "$0")/.." && pwd)"
set -a
. "$STACK_DIR/manifest.env"
[ -f "$STACK_DIR/stack.env" ] && . "$STACK_DIR/stack.env"
set +a

SERVICES_REPO="${SERVICES_REPO:-$STACK_DIR/../../odyss-services}"
SUPERADMIN_REPO="${SUPERADMIN_REPO:-$STACK_DIR/../../odyss-superadmin}"
# 默认走局域网直连；不在局域网时用 DEPLOY_TARGET=odyss-server-frpc ./stack deploy-5090 走 frp
REMOTE="${DEPLOY_TARGET:-odyss@192.168.100.50}"
REMOTE_DIR="${DEPLOY_REMOTE_DIR:-/home/odyss/odyss-services-ifa}"
# 构建上下文放 /tmp：目标机上不留配置孤本，用完即删（MANAGEMENT.md §6）
BUILD_CTX="/tmp/odyss-services-build-ctx"
STAMP="$(date +%Y%m%d-%H%M)"
IMAGE="odyss-services:ifa"

# 本脚本只重建这两个无状态容器；postgres/redis/minio/llm-switch/llm-tunnel 一律不碰
RECREATE_SERVICES="odyss-services llm-mock"

echo "== 1/6 校验本地仓停在 ${SERVICES_REF} tip"
check_repo_ref() {
  local repo="$1" want="$2" name
  name="$(basename "$repo")"
  [ -d "$repo/.git" ] || { echo "  ✗ 找不到仓：$repo"; exit 1; }
  local cur
  cur="$(git -C "$repo" branch --show-current 2>/dev/null || echo '?')"
  [ "$cur" = "$want" ] || { echo "  ✗ $name 当前分支 $cur ≠ $want（部署只发 $want 上的代码）"; exit 1; }
  if [ -n "$(git -C "$repo" status --porcelain)" ]; then
    # 未提交改动发上去就追溯不到 commit，现场出问题无从对版本
    [ "${ALLOW_DIRTY:-0}" = "1" ] || { echo "  ✗ $name 工作树不干净（确认要发未提交改动时用 ALLOW_DIRTY=1）"; exit 1; }
    echo "  ⚠️ $name 工作树不干净，按 ALLOW_DIRTY=1 继续"
  fi
  git -C "$repo" fetch -q origin "$want"
  local head remote
  head="$(git -C "$repo" rev-parse HEAD)"
  remote="$(git -C "$repo" rev-parse "origin/$want")"
  if [ "$head" != "$remote" ]; then
    [ "${ALLOW_BEHIND:-0}" = "1" ] || { echo "  ✗ $name 与 origin/$want 不一致（先 pull；确有意发本地版本时用 ALLOW_BEHIND=1）"; exit 1; }
    echo "  ⚠️ $name 与 origin/$want 不一致，按 ALLOW_BEHIND=1 继续"
  fi
  echo "  ✓ $name @ $(git -C "$repo" log --oneline -1)"
}
check_repo_ref "$SERVICES_REPO" "$SERVICES_REF"
check_repo_ref "$SUPERADMIN_REPO" "$SUPERADMIN_REF"

echo "== 2/6 本机构建产物"
"$STACK_DIR/scripts/build-artifacts.sh"

echo "== 3/6 同步产物到 $REMOTE 并构建镜像"
ssh "$REMOTE" "rm -rf '$BUILD_CTX' && mkdir -p '$BUILD_CTX/artifacts/bin' '$BUILD_CTX/config'"
rsync -az "$STACK_DIR/artifacts/bin/" "$REMOTE:$BUILD_CTX/artifacts/bin/"
rsync -az "$STACK_DIR/services.Dockerfile" "$REMOTE:$BUILD_CTX/"
# 镜像里烘焙的 runtime-config 是 migrate 实际读的那份（migrate 没有 bind mount），
# 必须用目标机现场生效的配置，不能从开发机推——那份含密钥且只在目标机上有。
ssh "$REMOTE" "cp '$REMOTE_DIR/config/runtime-config.yaml' '$BUILD_CTX/config/runtime-config.yaml'"
ssh "$REMOTE" "cd '$BUILD_CTX' && docker build -q -t '$IMAGE' -f services.Dockerfile . && rm -rf '$BUILD_CTX'"

echo "== 4/6 部署前备份数据库（孤本无命名卷，重建前必留退路）"
# 用 -Fc（custom 压缩格式）与目录里既有的 pg-backup-*.dump 保持同一格式：
# 明文 pg_dumpall 会大 20 倍（280M vs 6.2G），几次部署就能把盘吃满。
# 恢复：docker exec -i odyss-ifa-postgres pg_restore -U odyss -d odyss_services --clean < <dump>
ssh "$REMOTE" "docker exec odyss-ifa-postgres pg_dump -U odyss -Fc odyss_services > '$REMOTE_DIR/pg-backup-predeploy-$STAMP.dump' && ls -lh '$REMOTE_DIR/pg-backup-predeploy-$STAMP.dump'"
# 滚动保留最近 3 份部署前备份，避免无人清理把盘撑爆
ssh "$REMOTE" "cd '$REMOTE_DIR' && ls -1t pg-backup-predeploy-*.dump 2>/dev/null | tail -n +4 | xargs -r rm -f"

echo "== 5/6 跑 migration 并重建 $RECREATE_SERVICES"
# --no-deps：不碰 postgres/minio（见文件头铁律）
ssh "$REMOTE" "cd '$REMOTE_DIR' && docker compose up -d --no-deps --force-recreate migrate"
migrate_code="$(ssh "$REMOTE" "docker wait odyss-ifa-migrate")"
if [ "$migrate_code" != "0" ]; then
  echo "  ✗ migrate 退出码 $migrate_code，部署中止（services 未动，仍是旧版本）"
  ssh "$REMOTE" "docker logs --tail 40 odyss-ifa-migrate"
  exit 1
fi
echo "  ✓ migration 完成"
ssh "$REMOTE" "cd '$REMOTE_DIR' && docker compose up -d --no-deps --force-recreate $RECREATE_SERVICES"

echo "== 6/6 发 superadmin dist 并探活"
# dist 是 nginx 的只读 bind mount，换文件即生效，不需重启容器。
# 备份一份便于回滚；runtime-config 是目标机侧数据，本地 build 不产出，必须排除后补回。
ssh "$REMOTE" "cd '$REMOTE_DIR/superadmin' && cp -r dist 'dist.bak-$STAMP' && ls -1dt dist.bak-* 2>/dev/null | tail -n +4 | xargs -r rm -rf"
rsync -az --delete --exclude 'superadmin-runtime-config*.json' \
  "$STACK_DIR/artifacts/superadmin-dist/" "$REMOTE:$REMOTE_DIR/superadmin/dist/"
ssh "$REMOTE" "cp '$REMOTE_DIR/superadmin/superadmin-runtime-config.json' '$REMOTE_DIR/superadmin/dist/superadmin-runtime-config.json'"

sleep 3
ssh "$REMOTE" "
  echo '--- 容器状态'
  docker ps --filter name=odyss-ifa- --format '{{.Names}}\t{{.Status}}' | grep -E 'services|superadmin|postgres|minio'
  echo '--- services 探活'
  curl -sf -o /dev/null -w 'services health: %{http_code}\n' http://127.0.0.1:18090/api/v1/base/health || echo 'services 探活失败'
  curl -sf -o /dev/null -w 'superadmin: %{http_code}\n' http://127.0.0.1:18091/ || echo 'superadmin 探活失败'
"
echo "完成。浏览器打开 http://192.168.100.50:18091 需**硬刷新**（index.html 会被缓存，旧 JS 会让人以为没部署上）。"
