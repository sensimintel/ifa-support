#!/usr/bin/env bash
# 一步拉起 GCP 监控看板：持久 IAP 隧道（nohup 常驻，幂等）+ 本机 Grafana（docker）。
# 拉起后用 http://<本机内网IP>:${GRAFANA_PORT} 访问（默认 3001）。
#
# 前置：① 本机装了 gcloud 并已登录 GCP Workforce 身份（见 README「认证与持久化」）；
#       ② 本机装了 docker + docker compose。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# 首次无 .env 则从样例拷一份
if [ ! -f .env ]; then
  echo "==> 未见 .env，从 .env.example 拷贝一份（按需改端口/密码后可重跑）"
  cp .env.example .env
fi
set -a; . ./.env; set +a
GRAFANA_PORT="${GRAFANA_PORT:-3001}"
LOCAL_PROM_PORT="${LOCAL_PROM_PORT:-19090}"

# ---- 1. 隧道（幂等）：已在跑则复用，否则 nohup 拉起 ----
if pgrep -f "grafana-gcp/tunnel/gcp-tunnel.sh" >/dev/null 2>&1; then
  echo "==> IAP 隧道已在运行，复用"
else
  echo "==> nohup 拉起 IAP 隧道（日志 tunnel/tunnel.log）"
  nohup bash "$HERE/tunnel/gcp-tunnel.sh" > "$HERE/tunnel/tunnel.log" 2>&1 &
fi

# ---- 2. 等隧道落地口就绪（Prometheus /-/ready 经隧道可达）----
echo "==> 等待隧道就绪：curl 127.0.0.1:${LOCAL_PROM_PORT}/-/ready（至多 60s）"
ok=0
for i in $(seq 1 30); do
  if curl -fsS -o /dev/null -m 3 "http://127.0.0.1:${LOCAL_PROM_PORT}/-/ready" 2>/dev/null; then
    echo "    隧道就绪（约 $((i * 2))s）"; ok=1; break
  fi
  sleep 2
done
if [ "$ok" != 1 ]; then
  echo "!! 隧道 60s 内未就绪。多为 gcloud 凭证过期或未登录——请看 tunnel/tunnel.log，" >&2
  echo "   并按 README「认证与持久化」重跑浏览器 SSO 登录后再 ./up.sh。" >&2
  exit 1
fi

# ---- 3. 本机 Grafana ----
DC="docker compose"; docker compose version >/dev/null 2>&1 || DC="docker-compose"
echo "==> 拉起本机 Grafana（$DC up -d）"
$DC up -d

# ---- 4. Grafana 健康检查 ----
echo "==> 健康检查 http://127.0.0.1:${GRAFANA_PORT}/api/health（至多 60s）"
for i in $(seq 1 30); do
  if curl -fsS -o /dev/null -m 3 "http://127.0.0.1:${GRAFANA_PORT}/api/health" 2>/dev/null; then
    IP="$(hostname -I 2>/dev/null | awk '{print $1}')"; IP="${IP:-<本机内网IP>}"
    echo "    Grafana 就绪（约 $((i * 2))s）"
    echo "==> 完成。访问：http://${IP}:${GRAFANA_PORT}  （看板：gpu-g4-01 vLLM 观测）"
    echo "    admin 账号见 .env 的 GRAFANA_ADMIN_USER/PASSWORD"
    exit 0
  fi
  sleep 2
done
echo "!! Grafana 60s 内未就绪，请看 $DC logs grafana" >&2
exit 1
