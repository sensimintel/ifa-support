#!/usr/bin/env bash
# 一步拉起：本机统一 Grafana（唯一数据源 = 本机 Prometheus 127.0.0.1:9091）。
# g4 联邦观测（frp STCP 隧道 + ifa-grafana-tunnel systemd 服务）已废弃，本脚本不再涉及。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
if [ ! -f .env ]; then
  echo "==> 未见 .env，从 .env.example 拷贝一份。请填 GRAFANA_ADMIN_PASSWORD 后重跑。"
  cp .env.example .env; exit 1
fi
set -a; . ./.env; set +a
LOCAL_PROM_PORT="${LOCAL_PROM_PORT:-9091}"; GRAFANA_PORT="${GRAFANA_PORT:-3001}"

# 1) 检查本机 Prometheus（唯一数据源）
curl -fsS -o /dev/null -m3 "http://127.0.0.1:${LOCAL_PROM_PORT}/-/ready" 2>/dev/null \
  && echo "==> 本机 Prometheus(${LOCAL_PROM_PORT}) OK" \
  || echo "!! 警告：本机 Prometheus(${LOCAL_PROM_PORT}) 不可达（看板会空）"

# 2) 起统一 Grafana
echo "==> docker compose up -d"; docker compose up -d
echo ""; echo "==> 完成。访问 http://<本机内网IP>:${GRAFANA_PORT}（在 5090 即 http://192.168.100.50:${GRAFANA_PORT}）"
echo "    看板：/d/g4-01（本机 Qwen vLLM）、/d/gpu5090-server、/d/sam3"
