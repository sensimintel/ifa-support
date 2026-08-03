#!/usr/bin/env bash
# ifa-support 部署脚本：在 5090 上拉取最新代码并重启 8060 服务。
# 用法：在 checkout 目录（~/da3-web）执行 ./deploy.sh
# 纪律：部署机只 pull、不 commit/push（deploy key 只读）。
set -euo pipefail

cd "$(dirname "$0")"

echo "==> 拉取最新代码 (git pull --ff-only)"
git pull --ff-only

# 唯一重启路径：systemd。严禁 kill + nohup 手工拉起——2026-08-03 曾因此产生脱离
# systemd 的游离进程（丢失崩溃自愈与开机自启，体检报 systemd inactive 而页面仍 200）。
if ! systemctl list-unit-files da3-web.service --no-legend 2>/dev/null | grep -q .; then
  echo "!! 未安装 da3-web.service，拒绝以 nohup 方式拉起（不再提供该兜底）。" >&2
  echo "   先安装单元：sudo cp da3-web.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable da3-web" >&2
  echo "   注意：daemon-reload 会吊销容器内 NVML，之后需 docker restart locateanything-gpu-exporter（见 tools/health/README.md）" >&2
  exit 1
fi

# 归位：若有脱离 systemd 的游离 uvicorn 占着 8060，先精确清掉，否则 restart 会因端口被占失败。
# 匹配模式必须带端口——不带端口会误伤 LA gateway 等同为「uvicorn app:app」的进程。
main_pid=$(systemctl show da3-web -p MainPID --value 2>/dev/null || echo 0)
for pid in $(pgrep -f 'uvicorn app:app --host 0\.0\.0\.0 --port 8060' || true); do
  if [ "$pid" != "$main_pid" ]; then
    echo "==> 清理游离进程 $pid（非 systemd MainPID，管理态漂移归位）"
    kill "$pid" 2>/dev/null || true
  fi
done
sleep 1

echo "==> 通过 systemd 重启 da3-web.service"
sudo systemctl restart da3-web.service
sleep 2
systemctl --no-pager --lines=5 status da3-web.service || true

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
echo "!! 60s 内健康检查未通过，请查看 journalctl -u da3-web" >&2
exit 1
