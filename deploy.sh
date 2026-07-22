#!/usr/bin/env bash
# ifa-support 部署脚本：在 5090 上拉取最新代码并重启 8060 服务。
# 用法：在 checkout 目录（~/da3-web）执行 ./deploy.sh
# 纪律：部署机只 pull、不 commit/push（deploy key 只读）。
set -euo pipefail

cd "$(dirname "$0")"

echo "==> 拉取最新代码 (git pull --ff-only)"
git pull --ff-only

# 优先用 systemd（若已安装 da3-web.service），否则回退到 kill + nohup 手动重启
if systemctl list-unit-files 2>/dev/null | grep -q '^da3-web\.service'; then
  echo "==> 通过 systemd 重启 da3-web.service"
  sudo systemctl restart da3-web.service
  sleep 2
  systemctl --no-pager --lines=5 status da3-web.service || true
else
  echo "==> 未装 systemd 单元，回退到 kill + nohup 重启"
  # 精确匹配本目录跑的 uvicorn 8060 进程
  pids=$(pgrep -f 'uvicorn app:app .*--port 8060' || true)
  if [ -n "$pids" ]; then
    echo "    结束旧进程: $pids"
    kill $pids 2>/dev/null || true
    sleep 2
  fi
  echo "    以 nohup 重新拉起（日志写入 serve.log）"
  nohup ./run.sh > serve.log 2>&1 &
  sleep 2
fi

echo "==> 健康检查 http://127.0.0.1:8060/（导入 torch/gradio + 构建 Gradio app 需若干秒，轮询至多 60s）"
for i in $(seq 1 20); do
  code=$(curl -fsS -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8060/ 2>/dev/null || true)
  if [ "$code" = "200" ]; then
    echo "    HTTP 200（约 $((i * 3))s 就绪）"
    echo "==> 部署完成"
    exit 0
  fi
  sleep 3
done
echo "!! 60s 内健康检查未通过，请查看 serve.log 或 systemd 日志" >&2
exit 1
