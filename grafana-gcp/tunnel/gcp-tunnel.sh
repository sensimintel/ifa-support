#!/usr/bin/env bash
# 持久 IAP SSH 隧道：把 GCP gpu-g4-01 上 loopback 的 Prometheus(${REMOTE_PROM_PORT})
# 转发到本机 127.0.0.1:${LOCAL_PROM_PORT}，供本机 Grafana 作数据源消费。
#
# 为什么要隧道：gpu-g4-01 无公网 SSH 入口，观测栈全绑 127.0.0.1，唯一入口是 gcloud IAP 隧道
# （见 odyss-models/deploy/gcp-g4）。这里用 gcloud compute ssh --tunnel-through-iap -- -L 做端口转发。
#
# 断线自愈：ssh 每次退出即 5s 后重连（前台常驻，供 up.sh 的 nohup 或 systemd 调用）。
#
# 认证前提：本机已 gcloud 登录 Workforce 身份（yu.ji@odyss.life）。
#   ⚠️ Workforce 联邦凭证会过期，过期后隧道会持续重连失败——需本人重跑浏览器 SSO：
#      gcloud auth login --login-config=<odyss-gcp-login.json>
#   （见 README「认证与持久化」。凭证在时隧道全自动，无需人工。）
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 读上级 .env（若有）
if [ -f "$HERE/../.env" ]; then
  set -a; . "$HERE/../.env"; set +a
fi

GCP_PROJECT="${GCP_PROJECT:-pelagic-pod-489307-g3}"
GCP_ZONE="${GCP_ZONE:-asia-southeast1-b}"
GCP_VM="${GCP_VM:-gpu-g4-01}"
REMOTE_PROM_PORT="${REMOTE_PROM_PORT:-9090}"
LOCAL_PROM_PORT="${LOCAL_PROM_PORT:-19090}"

command -v gcloud >/dev/null 2>&1 \
  || { echo "[隧道] 未找到 gcloud，请先安装 Google Cloud SDK 并登录 Workforce 身份（见 README）。" >&2; exit 1; }

echo "[隧道] 目标 ${GCP_VM}(${GCP_ZONE}) 127.0.0.1:${REMOTE_PROM_PORT} → 本机 127.0.0.1:${LOCAL_PROM_PORT}"

while true; do
  # 认证探测：无 active 凭证时不疯狂重试刷屏，等 60s 再看（等待用户重新 SSO 登录）
  if ! gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null | grep -q .; then
    echo "[隧道] gcloud 无 active 凭证——请重跑浏览器 SSO 登录（见 README「认证与持久化」）。60s 后重试。" >&2
    sleep 60
    continue
  fi

  echo "[隧道] 建立中 $(date '+%F %T') ..."
  # -N: 不执行远端命令，仅转发；-T: 不分配伪终端
  # ExitOnForwardFailure: 端口占用/转发失败即退出（触发重连），不留半死连接
  # ServerAlive*: 30s 无响应连发 3 次探测，判死则断开重连
  gcloud compute ssh "$GCP_VM" \
    --project="$GCP_PROJECT" --zone="$GCP_ZONE" \
    --tunnel-through-iap --quiet \
    -- -N -T \
       -o ExitOnForwardFailure=yes \
       -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
       -L "127.0.0.1:${LOCAL_PROM_PORT}:127.0.0.1:${REMOTE_PROM_PORT}" \
    2>&1 | sed 's/^/[隧道] /'

  echo "[隧道] 断开（$(date '+%F %T')），5s 后重连 ..." >&2
  sleep 5
done
