#!/usr/bin/env bash
# 用 da3 conda 环境在 0.0.0.0:8070 起深体验区后端（秤 + 分组绑定配置）
set -e
PY=/home/odyss/miniconda3/envs/da3/bin/python
cd /home/odyss/da3-web
exec "$PY" -m uvicorn dx_backend:app --host 0.0.0.0 --port 8070
