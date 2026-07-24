#!/usr/bin/env bash
# 一步拉起：frp STCP 隧道(把 g4-01 Prometheus 拉到本机) + 本机统一 Grafana。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
if [ ! -f .env ]; then
  echo "==> 未见 .env，从 .env.example 拷贝一份。请填 FRP_TOKEN / STCP_SECRET / GRAFANA_ADMIN_PASSWORD 后重跑。"
  cp .env.example .env
  exit 1
fi
set -a; . ./.env; set +a
G4_PROM_LOCAL_PORT="${G4_PROM_LOCAL_PORT:-29090}"
LOCAL_PROM_PORT="${LOCAL_PROM_PORT:-9091}"
GRAFANA_PORT="${GRAFANA_PORT:-3001}"

# 1) frp 隧道（后台常驻）
if pgrep -f "grafana-gcp/tunnel/frp-tunnel.sh" >/dev/null 2>&1 || pgrep -f "grafana-gcp/tunnel/frpc " >/dev/null 2>&1; then
  echo "==> frp 隧道已在运行，复用"
else
  echo "==> nohup 拉起 frp 隧道（日志 tunnel/tunnel.log）"
  nohup bash "$HERE/tunnel/frp-tunnel.sh" > "$HERE/tunnel/tunnel.log" 2>&1 &
fi

# 2) 等隧道就绪（g4-01 Prometheus 经隧道可达）
echo "==> 等隧道就绪：curl 127.0.0.1:${G4_PROM_LOCAL_PORT}/-/ready（至多 60s）"
ok=0
for i in $(seq 1 30); do
  if curl -fsS -o /dev/null -m 3 "http://127.0.0.1:${G4_PROM_LOCAL_PORT}/-/ready" 2>/dev/null; then
    echo "    隧道就绪（约 $((i*2))s）"; ok=1; break
  fi
  sleep 2
done
[ "$ok" = 1 ] || { echo "!! 隧道未就绪，检查 tunnel/tunnel.log 与 .env 的 FRP_TOKEN/STCP_SECRET，以及 g4-01 端 frpc"; exit 1; }

# 3) 检查本机 Prometheus（数据源①）
if curl -fsS -o /dev/null -m 3 "http://127.0.0.1:${LOCAL_PROM_PORT}/-/ready" 2>/dev/null; then
  echo "==> 本机 Prometheus(${LOCAL_PROM_PORT}) OK"
else
  echo "!! 警告：本机 Prometheus(${LOCAL_PROM_PORT}) 不可达，数据源① 会空（不影响 g4-01 看板）"
fi

# 4) 起统一 Grafana
echo "==> docker compose up -d grafana"
docker compose up -d
echo ""
echo "==> 完成。访问 http://<本机内网IP>:${GRAFANA_PORT}（在 5090 即 http://192.168.0.50:${GRAFANA_PORT}）"
echo "    看板：/d/g4-01（g4-01 vLLM&GPU）、/d/locateanything（5090 LocateAnything）"
