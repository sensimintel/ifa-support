#!/usr/bin/env bash
# 停：Grafana + frp 隧道(systemd 服务)。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
echo "==> 停 Grafana"; docker compose down 2>/dev/null || true
echo "==> 停并禁用 frp 隧道服务"; sudo systemctl disable --now ifa-grafana-tunnel 2>/dev/null || true
echo "==> 已停"
