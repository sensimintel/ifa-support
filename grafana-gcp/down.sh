#!/usr/bin/env bash
# 停：统一 Grafana。（g4 隧道 systemd 服务 ifa-grafana-tunnel 已废弃，若历史机器上
# 仍残留，可手动 sudo systemctl disable --now ifa-grafana-tunnel 清理。）
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
echo "==> 停 Grafana"; docker compose down 2>/dev/null || true
echo "==> 已停"
