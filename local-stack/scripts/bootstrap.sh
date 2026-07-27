#!/usr/bin/env bash
# 【目标机 · 无需公网】一键拉起/更新整套本地闭环栈（唯一入口：./stack bootstrap）：
#   导入镜像 → 密钥就位 → 按 LLM_MODE 渲染全部运行配置 → 构建 services 镜像
#   → 起栈（自动 migrate）→ 创建 admin 账号。
# 可重复执行（幂等）：密钥只生成一次并持久化在 config/secrets.env；配置每次从模板
# 重渲染（换 LLM_MODE / 改 stack.env 后重跑本脚本即完成切换）。
# 前置：目标机已装 docker 与 docker compose 插件，openssl 可用。
set -euo pipefail

STACK_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$STACK_DIR"

# ---- 装载 manifest 与本机覆盖（经 ./stack 进来时已注入，独立执行也成立）----
set -a
. "$STACK_DIR/manifest.env"
[ -f "$STACK_DIR/stack.env" ] && . "$STACK_DIR/stack.env"
set +a
LLM_MODE="${LLM_MODE:-mock}"
STACK_ID="${STACK_ID:-local-stack}"
STACK_LABEL="${STACK_LABEL:-本地栈}"
ADMIN_EMAIL="${ADMIN_EMAIL:-admin@odyss.local}"
ADMIN_PASSWORD="${ADMIN_PASSWORD:-Odyss@Local1}"
API="http://127.0.0.1:18090/api/v1"
COMPOSE_PROFILES=""
[ "$LLM_MODE" = "real" ] && COMPOSE_PROFILES="real-vlm"
export COMPOSE_PROFILES

echo "== 1/7 校验前置（LLM_MODE=$LLM_MODE）"
command -v docker >/dev/null || { echo "docker 不可用"; exit 1; }
docker compose version >/dev/null || { echo "docker compose 插件不可用"; exit 1; }
[ -f artifacts/bin/odyss-services ] || { echo "artifacts 缺失（应从离线包解出或先 ./stack build）"; exit 1; }
[ -d artifacts/superadmin-dist ] || { echo "superadmin dist 缺失（应从离线包解出或先 ./stack build）"; exit 1; }
if [ "$LLM_MODE" = "real" ]; then
  for v in VLM_API_KEY FRP_SERVER_ADDR FRP_SERVER_PORT FRP_TOKEN FRP_STCP_SERVER_NAME FRP_STCP_SECRET; do
    [ -n "${!v:-}" ] || { echo "real 模式缺少 stack.env 配置项：$v（见 stack.env.example）"; exit 1; }
  done
fi

echo "== 2/7 导入 base 镜像（本机已有的自动跳过）"
if [ -f images/base-images.tar ]; then
  docker load -i images/base-images.tar
else
  echo "  images/base-images.tar 不存在，假定镜像已在本机（docker images 自查）"
fi

echo "== 3/7 身份密钥就位（只生成一次，换模式重渲染不换密钥）"
if [ ! -f config/secrets.env ]; then
  if [ -f config/runtime-config.yaml ]; then
    # 从存量配置迁移密钥（老部署升级路径），保证已注册账号与 App 登录不失效
    PASSWORD_KEY_ID=$(sed -n 's/^ *key_id: *//p' config/runtime-config.yaml | head -1)
    PASSWORD_KEY_B64=$(sed -n 's/^ *private_key_b64: *"\(.*\)"/\1/p' config/runtime-config.yaml | head -1)
    TOKEN_SECRET=$(sed -n 's/^ *token_secret: *//p' config/runtime-config.yaml | head -1)
    [ -n "$PASSWORD_KEY_B64" ] && [ -n "$TOKEN_SECRET" ] \
      || { echo "存量 runtime-config.yaml 里提取密钥失败，请人工核对后重试"; exit 1; }
    echo "  已从存量 runtime-config.yaml 迁移密钥（key_id=$PASSWORD_KEY_ID）"
  else
    PASSWORD_KEY_ID="local-$(hostname)-$(date +%Y%m%d)"
    PASSWORD_KEY_B64=$(openssl genrsa 2048 2>/dev/null | base64 | tr -d '\n')
    TOKEN_SECRET=$(openssl rand -hex 32)
    echo "  已生成新密钥（key_id=$PASSWORD_KEY_ID）"
  fi
  {
    echo "PASSWORD_KEY_ID=$PASSWORD_KEY_ID"
    echo "PASSWORD_KEY_B64=$PASSWORD_KEY_B64"
    echo "TOKEN_SECRET=$TOKEN_SECRET"
  } > config/secrets.env
  chmod 600 config/secrets.env
else
  echo "  config/secrets.env 已存在，沿用"
fi
. config/secrets.env

echo "== 4/7 按 LLM_MODE 渲染运行配置"
if [ "$LLM_MODE" = "real" ]; then
  LLM_BASE_URL="http://llm-tunnel:28000/v1"
  LLM_API_KEY="$VLM_API_KEY"
  REALTIME_MODEL="$VLM_MODEL"
  MEAL_MODEL="$VLM_MODEL"
  CHUNK_MODEL="$VLM_MODEL"
  CHUNK_INHOUSE_ENABLED=true
  sed -e "s|__FRP_SERVER_ADDR__|$FRP_SERVER_ADDR|" \
      -e "s|__FRP_SERVER_PORT__|$FRP_SERVER_PORT|" \
      -e "s|__FRP_TOKEN__|$FRP_TOKEN|" \
      -e "s|__FRP_STCP_SERVER_NAME__|$FRP_STCP_SERVER_NAME|" \
      -e "s|__FRP_STCP_SECRET__|$FRP_STCP_SECRET|" \
      config/frpc-visitor.template.toml > config/frpc-visitor.toml
  chmod 600 config/frpc-visitor.toml
  echo "  real 模式：LLM 指向 llm-tunnel（$VLM_MODEL）；已渲染 frpc-visitor.toml"
  echo "  ⚠️ 远端 vLLM 必须以别名同时应答 $VLM_REQUIRED_ALIAS（见 manifest.env 注释），否则整餐分析 404"
else
  LLM_BASE_URL="http://llm-mock:8081/v1"
  LLM_API_KEY="local-mock-key"
  REALTIME_MODEL="local-mock-realtime-v1"
  MEAL_MODEL="local-mock-meal-v1"
  CHUNK_MODEL="local-mock-chunk-v1"
  CHUNK_INHOUSE_ENABLED=false
  # 残留的 real 模式渲染物清掉，防误用
  rm -f config/frpc-visitor.toml
fi
sed -e "s|__PASSWORD_KEY_B64__|$PASSWORD_KEY_B64|" \
    -e "s|__TOKEN_SECRET__|$TOKEN_SECRET|" \
    -e "s|__PASSWORD_KEY_ID__|$PASSWORD_KEY_ID|" \
    -e "s|__LLM_BASE_URL__|$LLM_BASE_URL|" \
    -e "s|__LLM_API_KEY__|$LLM_API_KEY|" \
    -e "s|__REALTIME_MODEL__|$REALTIME_MODEL|" \
    -e "s|__MEAL_MODEL__|$MEAL_MODEL|" \
    -e "s|__CHUNK_MODEL__|$CHUNK_MODEL|" \
    -e "s|__CHUNK_INHOUSE_ENABLED__|$CHUNK_INHOUSE_ENABLED|" \
    config/runtime-config.template.yaml > config/runtime-config.yaml
chmod 600 config/runtime-config.yaml
sed -e "s|__STACK_ID__|$STACK_ID|" \
    -e "s|__STACK_LABEL__|$STACK_LABEL|" \
    superadmin/superadmin-runtime-config.template.json > config/superadmin-runtime-config.json

echo "== 5/7 构建 services 镜像（配置烧入镜像，渲染后必须重建）"
docker build -t odyss-services:local -f services.Dockerfile .

echo "== 6/7 起栈（migrate 自动执行后 services 才启动）"
docker compose up -d
# nginx 配置/前端产物是文件挂载，内容变化不触发容器重建，restart 保证生效
docker compose restart superadmin > /dev/null
echo "  等待 services 健康……"
for i in $(seq 1 60); do
  if curl -sf "$API/base/health" > /dev/null 2>&1; then break; fi
  sleep 2
done
curl -sf "$API/base/health" > /dev/null || { echo "services 未就绪，查日志：docker logs odyss-local-services"; exit 1; }
echo "  services 健康检查通过"

echo "== 7/7 创建 admin 账号（已存在则跳过）"
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
echo "======== 拉起完成（LLM_MODE=$LLM_MODE）========"
echo "App 后端地址（App 内自定义 base url）：http://${IP:-<本机IP>}:18090"
echo "Superadmin 控制台：http://${IP:-<本机IP>}:18091"
echo "Admin 账号：$ADMIN_EMAIL / $ADMIN_PASSWORD"
echo "普通测试账号可随时自助注册（验证码从 send-code 响应返回，见 README）"
