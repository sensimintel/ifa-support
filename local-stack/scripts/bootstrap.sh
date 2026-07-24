#!/usr/bin/env bash
# 【目标机 · 无需公网】一键拉起整套本地闭环栈：
#   导入镜像 → 生成密钥与配置 → 构建 services 镜像 → 起栈（自动 migrate）→ 创建 admin 账号。
# 可重复执行（幂等）：已有配置/账号会跳过，起栈动作只做增量。
# 前置：目标机已装 docker 与 docker compose 插件（离线装机自带即可），openssl 可用。
set -euo pipefail

STACK_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$STACK_DIR"

ADMIN_EMAIL="${ADMIN_EMAIL:-admin@odyss.local}"
ADMIN_PASSWORD="${ADMIN_PASSWORD:-Odyss@Local1}"
API="http://127.0.0.1:18090/api/v1"

echo "== 1/6 校验前置"
command -v docker >/dev/null || { echo "docker 不可用"; exit 1; }
docker compose version >/dev/null || { echo "docker compose 插件不可用"; exit 1; }
[ -f artifacts/bin/odyss-services ] || { echo "artifacts 缺失（应从离线包解出）"; exit 1; }
[ -d artifacts/superadmin-dist ] || { echo "superadmin dist 缺失（应从离线包解出）"; exit 1; }

echo "== 2/6 导入 base 镜像（本机已有的自动跳过）"
if [ -f images/base-images.tar ]; then
  docker load -i images/base-images.tar
else
  echo "  images/base-images.tar 不存在，假定镜像已在本机（docker images 自查）"
fi

echo "== 3/6 生成运行配置（已存在则跳过）"
if [ ! -f config/runtime-config.yaml ]; then
  KEY_B64=$(openssl genrsa 2048 2>/dev/null | base64 | tr -d '\n')
  TOKEN_SECRET=$(openssl rand -hex 32)
  KEY_ID="local-$(hostname)-$(date +%Y%m%d)"
  sed -e "s|__PASSWORD_KEY_B64__|$KEY_B64|" \
      -e "s|__TOKEN_SECRET__|$TOKEN_SECRET|" \
      -e "s|__PASSWORD_KEY_ID__|$KEY_ID|" \
      config/runtime-config.template.yaml > config/runtime-config.yaml
  echo "  已生成 config/runtime-config.yaml（key_id=$KEY_ID）"
else
  echo "  config/runtime-config.yaml 已存在，跳过"
fi

echo "== 4/6 构建 services 镜像"
docker build -t odyss-services:local -f services.Dockerfile .

echo "== 5/6 起栈（migrate 自动执行后 services 才启动）"
docker compose up -d
echo "  等待 services 健康……"
for i in $(seq 1 60); do
  if curl -sf "$API/base/health" > /dev/null 2>&1; then break; fi
  sleep 2
done
curl -sf "$API/base/health" > /dev/null || { echo "services 未就绪，查日志：docker logs odyss-local-services"; exit 1; }
echo "  services 健康检查通过"

echo "== 6/6 创建 admin 账号（已存在则跳过）"
LOGIN_CODE=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$API/auth/login" \
  -H 'Content-Type: application/json' \
  -d "{\"email\":\"$ADMIN_EMAIL\",\"password\":\"$ADMIN_PASSWORD\"}")
if [ "$LOGIN_CODE" != "200" ]; then
  CODE=$(curl -s -X POST "$API/auth/send-code" -H 'Content-Type: application/json' \
    -d "{\"email\":\"$ADMIN_EMAIL\",\"type\":\"login\"}" | sed -n 's/.*"code":"\([0-9]*\)".*/\1/p')
  [ -n "$CODE" ] || { echo "取验证码失败（确认 dev_expose_code=true）"; exit 1; }
  curl -sf -X POST "$API/auth/register" -H 'Content-Type: application/json' \
    -d "{\"email\":\"$ADMIN_EMAIL\",\"code\":\"$CODE\",\"password\":\"$ADMIN_PASSWORD\"}" > /dev/null \
    || { echo "注册失败"; exit 1; }
  echo "  已注册 $ADMIN_EMAIL"
else
  echo "  账号已存在"
fi
docker exec odyss-local-postgres psql -U odyss -d odyss_services \
  -c "UPDATE users SET is_superuser=true WHERE email='$ADMIN_EMAIL'" > /dev/null

IP=$(hostname -I 2>/dev/null | awk '{print $1}')
echo ""
echo "======== 拉起完成 ========"
echo "App 后端地址（App 内自定义 base url）：http://${IP:-<本机IP>}:18090"
echo "Superadmin 控制台：http://${IP:-<本机IP>}:18091"
echo "Admin 账号：$ADMIN_EMAIL / $ADMIN_PASSWORD"
echo "普通测试账号可随时自助注册（验证码从 send-code 响应返回，见 README）"
