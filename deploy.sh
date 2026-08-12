#!/usr/bin/env bash
# ifa-support 部署脚本：在 5090 上拉取最新代码并重启 8060 服务。
# 用法：在 checkout 目录（~/da3-web）执行 ./deploy.sh
# 纪律：部署机只 pull、不 commit/push（deploy key 只读）。
set -euo pipefail

cd "$(dirname "$0")"

# systemd 单元是否已安装。不能写 `systemctl list-unit-files | grep -q`：grep -q 命中即关闭
# 管道，systemctl 吃 SIGPIPE 非零退出，set -o pipefail 下整条判定**偶发**为假 → 误走 nohup
# 回退，孤儿进程占住端口挡住 systemd 实例（2026-08-12 两次实锤）。命令替换全量读输出无此坑。
has_unit() { [ -n "$(systemctl list-unit-files "$1.service" --no-legend 2>/dev/null)" ]; }

echo "==> 拉取最新代码 (git pull --ff-only)"
git pull --ff-only

# 优先用 systemd（若已安装 da3-web.service），否则回退到 kill + nohup 手动重启
if has_unit da3-web; then
  echo "==> 通过 systemd 重启 da3-web.service"
  # 先清历史误走回退分支留下的 nohup 孤儿：占住 8060 会让 systemd 实例绑定失败进崩溃循环，
  # 而健康检查打在孤儿进程上"假绿"（部署显示成功、页面却是旧代码）
  main_pid=$(systemctl show -p MainPID --value da3-web.service 2>/dev/null || echo 0)
  for pid in $(pgrep -f 'uvicorn app:app .*--port 8060' || true); do
    if [ "$pid" != "$main_pid" ]; then
      echo "    清理占用 8060 的孤儿进程: $pid"
      kill "$pid" 2>/dev/null || true
    fi
  done
  sleep 1
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

# 深体验区后端 8070（dx-backend.service）：已安装 systemd 单元则一并重启
if has_unit dx-backend; then
  echo "==> 通过 systemd 重启 dx-backend.service (8070)"
  sudo systemctl restart dx-backend.service
  sleep 1
else
  echo "==> 未装 dx-backend.service，跳过 8070（首次安装见 dx-backend.service 头部注释）"
fi

echo "==> 健康检查 http://127.0.0.1:8060/（导入 torch/gradio + 构建 Gradio app 需若干秒，轮询至多 60s）"
ok8060=0
for i in $(seq 1 20); do
  code=$(curl -fsS -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8060/ 2>/dev/null || true)
  if [ "$code" = "200" ]; then
    echo "    8060 HTTP 200（约 $((i * 3))s 就绪）"
    ok8060=1
    break
  fi
  sleep 3
done
if [ "$ok8060" != "1" ]; then
  echo "!! 8060 在 60s 内健康检查未通过，请查看 serve.log 或 systemd 日志" >&2
  exit 1
fi

if has_unit dx-backend; then
  echo "==> 健康检查 http://127.0.0.1:8070/api/health"
  for i in $(seq 1 5); do
    if curl -fsS --max-time 3 http://127.0.0.1:8070/api/health > /dev/null 2>&1; then
      echo "    8070 就绪"
      echo "==> 部署完成"
      exit 0
    fi
    sleep 2
  done
  echo "!! 8070 健康检查未通过：journalctl -u dx-backend -n 50" >&2
  exit 1
fi
echo "==> 部署完成"
