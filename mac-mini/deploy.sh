#!/bin/bash
# mac mini 上一键部署：git pull + 环境准备 + LaunchDaemon 安装/重启 + 健康检查。
# 与 5090 的仓根 deploy.sh 同款「git 部署 checkout」模式：mini 的 ~/ifa-support
# 只 pull、不 commit/push（GitHub 侧只读 deploy key：mac-mini-deploy-readonly）。
# 用法：
#   mini 上执行：cd ~/ifa-support/mac-mini && ./deploy.sh
#   开发机远程触发：ssh mac-mini 'MINI_SUDO_PASS=… ~/ifa-support/mac-mini/deploy.sh'
# sudo：优先读环境变量 MINI_SUDO_PASS（非交互），否则交互输入。
set -euo pipefail
cd "$(dirname "$0")"

PLIST=life.odyss.cam-pusher.plist
LABEL=life.odyss.cam-pusher

run_sudo() {
  if [ -n "${MINI_SUDO_PASS:-}" ]; then
    echo "$MINI_SUDO_PASS" | sudo -S -p '' "$@"
  else
    sudo "$@"
  fi
}

echo "==> 拉取最新代码 (git pull --ff-only)"
git -C .. pull --ff-only

echo "==> 准备运行环境（venv + 依赖，幂等）"
bash ./setup.sh

echo "==> 安装/更新 LaunchDaemon 并重启"
run_sudo cp "$PLIST" "/Library/LaunchDaemons/$PLIST"
run_sudo chown root:wheel "/Library/LaunchDaemons/$PLIST"
# 必须 bootout 再 bootstrap：kickstart -k 只重启进程、不重读 plist，改过
# ProgramArguments 等定义不 bootout 就不生效（2026-08-13 实锤：进程仍跑在旧路径）
run_sudo launchctl bootout "system/$LABEL" 2>/dev/null || true
run_sudo launchctl bootstrap system "/Library/LaunchDaemons/$PLIST"

echo "==> 健康检查：等 8060 出现 macmini-* 设备帧（至多 30s）"
RELAY_URL="${RELAY_URL:-http://192.168.0.50:8060}"
for i in $(seq 1 10); do
  if curl -fsS --max-time 5 "$RELAY_URL/api/frame/status" 2>/dev/null | grep -q 'macmini-'; then
    echo "    8060 已收到 macmini 设备帧"
    echo "==> 部署完成"
    exit 0
  fi
  sleep 3
done
echo "!! 30s 内 8060 未见 macmini-* 帧：看 ~/Library/Logs/cam-pusher.log" >&2
exit 1
