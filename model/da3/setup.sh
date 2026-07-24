#!/usr/bin/env bash
# DA3（Depth Anything 3）一键准备：上游源码 + venv 依赖 + 权重下载。
# DA3 没有独立服务进程——本仓 app.py 通过 sys.path 引 DA3/src 在进程内推理，
# 本脚本只负责把「源码 + 环境 + 权重」备齐。
#
# 可覆盖的环境变量：
#   DA3_ROOT   源码目录            默认 ~/Depth-Anything-3
#   DA3_ENV    venv 目录           默认 ~/da3-env（5090 现网用 conda env `da3`，等价）
#   DA3_COMMIT 上游 commit 钉版     默认 eeb8a87（5090 现网实测版本）
#   HF_ENDPOINT 国内可设 https://hf-mirror.com 加速权重下载
set -euo pipefail

DA3_ROOT="${DA3_ROOT:-$HOME/Depth-Anything-3}"
DA3_ENV="${DA3_ENV:-$HOME/da3-env}"
DA3_COMMIT="${DA3_COMMIT:-eeb8a87}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> 1/4 拉取 DA3 上游源码（钉版 ${DA3_COMMIT}）"
if [ ! -d "${DA3_ROOT}/.git" ]; then
  git clone https://github.com/ByteDance-Seed/Depth-Anything-3 "${DA3_ROOT}"
fi
git -C "${DA3_ROOT}" fetch --all --quiet
git -C "${DA3_ROOT}" checkout --quiet "${DA3_COMMIT}"

echo "==> 2/4 创建 venv 并安装依赖（python ≥3.10）"
if [ ! -x "${DA3_ENV}/bin/python" ]; then
  python3 -m venv "${DA3_ENV}"
fi
"${DA3_ENV}/bin/pip" install --upgrade pip >/dev/null
"${DA3_ENV}/bin/pip" install -r "${SCRIPT_DIR}/requirements.txt"
# DA3 包本体装成 editable（--no-deps 保住上面的 pin 不被上游 setup.py 拉偏）
"${DA3_ENV}/bin/pip" install --no-deps -e "${DA3_ROOT}"

echo "==> 3/4 下载权重 DA3NESTED-GIANT-LARGE-1.1（约 13GB，落 ${DA3_ROOT}/models/）"
export HF_HOME="${DA3_ROOT}/models"
"${DA3_ENV}/bin/pip" install "huggingface-hub[cli]>=0.36,<1" >/dev/null
"${DA3_ENV}/bin/hf" download depth-anything/DA3NESTED-GIANT-LARGE-1.1 \
  --local-dir "${DA3_ROOT}/models/DA3NESTED-GIANT-LARGE-1.1"

echo "==> 4/4 验证：venv 里 import + 权重目录存在"
"${DA3_ENV}/bin/python" -c "
import sys; sys.path.append('${DA3_ROOT}/src')
from depth_anything_3.api import DepthAnything3
print('DA3 import OK')"
ls "${DA3_ROOT}/models/DA3NESTED-GIANT-LARGE-1.1" >/dev/null && echo "权重目录 OK"

echo "完成。app.py 期望的路径：DA3_ROOT=${DA3_ROOT}（app.py 顶部常量），运行解释器用 ${DA3_ENV}/bin/python（改仓根 run.sh 的 PY 变量）。"
