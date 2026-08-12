#!/usr/bin/env bash
# 用 da3 conda 环境在 0.0.0.0:8061 起「感知链路实验台」（exp_app.py）。
# 实验用途、手动起停，不进 deploy.sh，不设 systemd：
#   起：nohup ./run-exp.sh > exp.log 2>&1 &
#   停：pkill -f 'uvicorn exp_app:app'
set -e
PY=/home/odyss/miniconda3/envs/da3/bin/python
cd /home/odyss/da3-web
# 与 8060 共用仓根 .env（SAM3_ENDPOINT 等），存在才读
set -a; [ -f .env ] && . ./.env; set +a
exec "$PY" -m uvicorn exp_app:app --host 0.0.0.0 --port 8061
