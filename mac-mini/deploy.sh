#!/bin/bash
# 从开发机把推帧器部署/更新到 mac mini（推送式部署）。
# 用法（在开发机、本目录执行）：
#   ./deploy.sh              # 目标默认 ssh 别名 mac-mini
#   ./deploy.sh <ssh目标>    # 覆盖目标
# 说明：
#   · mini 不装 git/Xcode CLT，代码经 rsync 推送（macOS 原生自带 rsync/ssh）；
#   · 首次安装 LaunchDaemon 需要 mini 的 sudo：优先读环境变量 MINI_SUDO_PASS，
#     否则走交互式 sudo（ssh -t）；
#   · 日常运行不依赖本脚本——mini 开机由 LaunchDaemon 自动拉起。
set -euo pipefail
cd "$(dirname "$0")"

TARGET="${1:-mac-mini}"
DEST_DIR="cam-pusher"
PLIST=life.odyss.cam-pusher.plist
LABEL=life.odyss.cam-pusher

echo "==> 同步代码到 $TARGET:~/$DEST_DIR/（保留对端 .env）"
rsync -a --delete --exclude '.env' \
  cam_pusher.py run-pusher.sh setup.sh "$PLIST" README.md \
  "$TARGET:$DEST_DIR/"

echo "==> 准备运行环境（venv + 依赖，幂等）"
ssh "$TARGET" "bash ~/$DEST_DIR/setup.sh"

echo "==> 安装/更新 LaunchDaemon 并重启"
if [ -n "${MINI_SUDO_PASS:-}" ]; then
  # 非交互：密码经 stdin 交给 sudo -S（-p '' 抑制提示串混进输出）
  ssh "$TARGET" "echo '$MINI_SUDO_PASS' | sudo -S -p '' bash -c '
    set -e
    cp ~odyss/$DEST_DIR/$PLIST /Library/LaunchDaemons/$PLIST
    chown root:wheel /Library/LaunchDaemons/$PLIST
    launchctl bootstrap system /Library/LaunchDaemons/$PLIST 2>/dev/null || true
    launchctl kickstart -k system/$LABEL
  '"
else
  ssh -t "$TARGET" "sudo bash -c '
    set -e
    cp ~odyss/$DEST_DIR/$PLIST /Library/LaunchDaemons/$PLIST
    chown root:wheel /Library/LaunchDaemons/$PLIST
    launchctl bootstrap system /Library/LaunchDaemons/$PLIST 2>/dev/null || true
    launchctl kickstart -k system/$LABEL
  '"
fi

echo "==> 健康检查：等 8060 出现 macmini-* 设备帧（至多 30s）"
RELAY_URL="${RELAY_URL:-http://192.168.0.50:8060}"
for i in $(seq 1 10); do
  status=$(ssh "$TARGET" "curl -fsS --max-time 5 '$RELAY_URL/api/frame/status'" 2>/dev/null || true)
  if echo "$status" | grep -q 'macmini-'; then
    echo "    8060 已收到 macmini 设备帧"
    echo "==> 部署完成"
    exit 0
  fi
  sleep 3
done
echo "!! 30s 内 8060 未见 macmini-* 帧：ssh $TARGET 后看 ~/Library/Logs/cam-pusher.log" >&2
exit 1
