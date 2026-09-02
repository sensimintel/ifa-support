#!/usr/bin/env bash
# =============================================================================
# 5090 重启后一键恢复浅体验区（8060）——在 5090 桌面终端直接执行，无参数。
#   ~/da3-web/tools/health/after-reboot.sh
#
# 做三件事：
#   1. 按依赖顺序等 8060 链路就绪（da3-web → SAM3 → 本机 vLLM → mac-mini 帧），每个单元
#      超时未就绪就按 README 的「唯一拉起入口」重启一次再等（不重复重启、不裸 kill）。
#   2. 跑同目录 check.sh 做全栈体检（只读），原样打印。
#   3. 有桌面时用 firefox 打开 http://localhost:8060/experience；无 DISPLAY 只打印地址。
#
# 明确不做：不 docker compose up -d（孤本 pg/minio 无命名卷，连带重建等于清库，且 8060
# 不依赖业务栈）；不重启 frpc；不裸探秤；不 pkill。体检中业务栈 FAIL 仅报告。
# 参数（环境变量）：KIOSK=1 全屏展台模式；NO_OPEN=1 不开浏览器。
# =============================================================================
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
URL=http://localhost:8060/experience
FAILED=()

log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
code() { curl -sS -o /dev/null -m "${2:-5}" -w '%{http_code}' "$1" 2>/dev/null || echo 000; }

# wait_for <标签> <超时秒> <探测函数>：轮询直到探测函数返回 0
wait_for() {
  local label=$1 timeout=$2 probe=$3 t=0
  while [ "$t" -lt "$timeout" ]; do
    if "$probe"; then log "$label 就绪（${t}s）"; return 0; fi
    sleep 3; t=$((t + 3))
  done
  return 1
}

# ensure <标签> <unit> <超时秒> <探测函数>：unit 不 active 先拉起；超时未就绪拉起一次再等同样时长
ensure() {
  local label=$1 unit=$2 timeout=$3 probe=$4
  if ! systemctl is-active --quiet "$unit"; then
    log "$label: systemd $(systemctl is-active "$unit" 2>/dev/null)，拉起 $unit"
    sudo systemctl restart "$unit"
  fi
  wait_for "$label" "$timeout" "$probe" && return 0
  log "$label: ${timeout}s 未就绪，重启 $unit 后再等一轮"
  sudo systemctl restart "$unit"
  wait_for "$label" "$timeout" "$probe" && return 0
  log "$label: 仍未就绪——journalctl -u $unit -n 50"
  FAILED+=("$label")
  return 1
}

p_web()  { [ "$(code $URL)" = 200 ]; }
p_sam3() { [ "$(code http://127.0.0.1:8013/health 8)" = 200 ]; }   # 直探 SAM3 本体（.env SAM3_ENDPOINT）
p_vllm() { [[ "$(code http://127.0.0.1:8000/v1/models)" =~ ^(200|401)$ ]]; }
p_frame() {
  curl -sS -m 8 http://127.0.0.1:8060/api/frame/status 2>/dev/null | python3 -c '
import json,sys
d=json.load(sys.stdin)
a=[float(x.get("age",1e9)) for x in d.get("devices",[]) if str(x.get("device_id","")).startswith("macmini")]
raise SystemExit(0 if a and min(a)<=30 else 1)' 2>/dev/null
}

log "开机 $(uptime -s)，开始恢复 8060 链路"
ensure "da3-web 8060"      da3-web 60  p_web
ensure "SAM3 8013"         sam3    120 p_sam3
ensure "本机 vLLM 8000"    vllm    240 p_vllm
# mac-mini 帧源在 192.168.100.3（cam-pusher LaunchDaemon 自愈），5090 侧无拉起入口，只等不修
wait_for "mac-mini 帧链路" 60 p_frame || { log "mac-mini 无帧/帧龄>30s：查 mini 上电/网线，仍断走 mini deploy.sh"; FAILED+=("mac-mini 帧链路"); }

echo; log "===== 全栈体检 check.sh ====="
bash "$HERE/check.sh"; CHECK_RC=$?

echo
if [ "${#FAILED[@]}" -gt 0 ]; then log "8060 链路未就绪项：${FAILED[*]}"; fi
[ "$CHECK_RC" = 0 ] && log "体检：全部健康" || log "体检：存在 FAIL（见上，按 README 单元入口处理后重跑）"
log "浅体验区：$URL   （局域网：http://192.168.100.50:8060/experience）"

if [ "${NO_OPEN:-0}" != 1 ] && [ -n "${DISPLAY:-}" ] && command -v firefox >/dev/null; then
  if [ "${KIOSK:-0}" = 1 ]; then setsid firefox --kiosk "$URL" >/dev/null 2>&1 &
  else setsid firefox "$URL" >/dev/null 2>&1 & fi
  log "已在桌面打开浏览器"
fi

[ "${#FAILED[@]}" = 0 ] && [ "$CHECK_RC" = 0 ]
