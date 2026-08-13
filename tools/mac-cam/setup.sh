#!/bin/bash
# 建本目录下的 .venv（Python 3.11：pyorbbecsdk v1 只有到 cp311 的 mac wheel）并装依赖
set -euo pipefail
cd "$(dirname "$0")"
uv venv -p 3.11 .venv
uv pip install -p .venv/bin/python \
  pyorbbecsdk==1.3.2 opencv-python numpy requests trimesh pyobjc-framework-AVFoundation
echo "OK：用 .venv/bin/python 运行 push_rgb.py / push_astra.py"
