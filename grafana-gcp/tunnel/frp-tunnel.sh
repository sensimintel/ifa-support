#!/usr/bin/env bash
# 常驻 frp STCP visitor：把 g4-01 的 Prometheus 落地到本机 127.0.0.1:${G4_PROM_LOCAL_PORT}。
# 只用 frp 服务器控制口(7000)，不依赖公网高位端口，也不需要 gcloud/IAP。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
set -a; . ../.env; set +a
FRP_VERSION="${FRP_VERSION:-0.65.0}"
export FRP_SERVER_ADDR FRP_SERVER_PORT FRP_TOKEN STCP_SERVER_NAME STCP_SECRET G4_PROM_LOCAL_PORT

# 下载 frpc（若缺）
if [ ! -x "$HERE/frpc" ]; then
  echo "==> 下载 frpc ${FRP_VERSION}"
  curl -sL "https://github.com/fatedier/frp/releases/download/v${FRP_VERSION}/frp_${FRP_VERSION}_linux_amd64.tar.gz" -o /tmp/frp-gcp.tgz
  tar xzf /tmp/frp-gcp.tgz -C /tmp
  cp "/tmp/frp_${FRP_VERSION}_linux_amd64/frpc" "$HERE/frpc"
  rm -rf /tmp/frp-gcp.tgz "/tmp/frp_${FRP_VERSION}_linux_amd64"
fi

# 渲染 frpc.toml
envsubst < "$HERE/frpc.toml.tmpl" > "$HERE/frpc.toml"
echo "==> 启动 frpc（visitor g4-prometheus → 127.0.0.1:${G4_PROM_LOCAL_PORT}）"
exec "$HERE/frpc" -c "$HERE/frpc.toml"
