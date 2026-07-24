#!/usr/bin/env bash
# 停：Grafana + frp 隧道。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
echo "==> 停 Grafana"; docker compose down 2>/dev/null || true
echo "==> 停 frp 隧道"; pkill -f "grafana-gcp/tunnel/frpc " 2>/dev/null || true; pkill -f "grafana-gcp/tunnel/frp-tunnel.sh" 2>/dev/null || true
echo "==> 已停"
