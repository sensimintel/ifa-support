#!/bin/bash
# LaunchDaemon 入口：source 本目录 .env（可选）后以 venv python 前台运行推帧器。
# 由 /Library/LaunchDaemons/life.odyss.cam-pusher.plist 承载（root，KeepAlive）。
set -euo pipefail
cd "$(dirname "$0")"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

PUSHER_PYTHON="${PUSHER_PYTHON:-/Users/odyss/cam-pusher-venv/bin/python}"
exec "$PUSHER_PYTHON" -u cam_pusher.py
