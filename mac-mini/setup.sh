#!/bin/bash
# mac mini 上一步装好推帧器运行环境（幂等，可反复执行）。
# 前置：mini 已装 uv（~/.local/bin，机器初始化时已就位）。不需要 brew / Xcode CLT。
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

VENV="$HOME/cam-pusher-venv"

if [ ! -x "$VENV/bin/python" ]; then
  echo "==> 创建 venv（Python 3.11：pyorbbecsdk 1.3.2 的 macOS arm64 wheel 只配 3.11）"
  uv venv --python 3.11 "$VENV"
fi

echo "==> 安装依赖（版本组合已在本机验证：两台相机均正常出帧）"
uv pip install --python "$VENV/bin/python" \
  'pyorbbecsdk==1.3.2' numpy opencv-python requests

"$VENV/bin/python" -c "import pyorbbecsdk, cv2, numpy, requests; print('==> 依赖 OK')"
