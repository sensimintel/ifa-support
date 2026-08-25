#!/usr/bin/env bash
# 在 0.0.0.0:8060 起 da3-web 演示服务（DA3 模型已于 2026-08-25 退役；
# 服务名/conda 环境名沿用 da3 历史称谓，环境本身是本服务的 python 运行时，不要删）
set -e
PY=/home/odyss/miniconda3/envs/da3/bin/python
cd /home/odyss/da3-web
# 加载本地 .env（gitignore 的运维配置，如识别服务 RECOG_ENDPOINT/RECOG_API_KEY/RECOG_MODEL），存在才读
set -a; [ -f .env ] && . ./.env; set +a
exec "$PY" -m uvicorn app:app --host 0.0.0.0 --port 8060
