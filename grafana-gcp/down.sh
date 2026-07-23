#!/usr/bin/env bash
# 停掉 GCP 监控看板：停本机 Grafana + 停 nohup 拉起的 IAP 隧道。
# （若隧道是用 systemd 单元跑的，请改用 sudo systemctl stop gcp-grafana-tunnel）
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

DC="docker compose"; docker compose version >/dev/null 2>&1 || DC="docker-compose"
echo "==> 停 Grafana（$DC down）"
$DC down || true

echo "==> 停 IAP 隧道（nohup 进程）"
pkill -f "grafana-gcp/tunnel/gcp-tunnel.sh" 2>/dev/null && echo "    已停" || echo "    未见运行中的隧道进程"

echo "==> 完成"
