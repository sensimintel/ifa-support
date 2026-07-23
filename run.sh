#!/usr/bin/env bash
# 用 da3 conda 环境在 0.0.0.0:8060 起 DA3 深度图 web 服务
set -e
export HF_HOME=/home/odyss/Depth-Anything-3/models
PY=/home/odyss/miniconda3/envs/da3/bin/python
cd /home/odyss/da3-web
# 加载本地 .env（gitignore 的运维配置，如识别服务 RECOG_ENDPOINT/RECOG_API_KEY/RECOG_MODEL），存在才读
set -a; [ -f .env ] && . ./.env; set +a
exec "$PY" -m uvicorn app:app --host 0.0.0.0 --port 8060
