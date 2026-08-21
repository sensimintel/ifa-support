#!/usr/bin/env bash
# 一步拉起：frp STCP 隧道(systemd 服务，把 g4-01 Prometheus 拉到本机) + 本机统一 Grafana。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
if [ ! -f .env ]; then
  echo "==> 未见 .env，从 .env.example 拷贝一份。请填 FRP_TOKEN / STCP_SECRET / GRAFANA_ADMIN_PASSWORD 后重跑。"
  cp .env.example .env; exit 1
fi
set -a; . ./.env; set +a
G4_PROM_LOCAL_PORT="${G4_PROM_LOCAL_PORT:-29090}"; LOCAL_PROM_PORT="${LOCAL_PROM_PORT:-9091}"; GRAFANA_PORT="${GRAFANA_PORT:-3001}"
FRP_VERSION="${FRP_VERSION:-0.65.0}"
export FRP_SERVER_ADDR FRP_SERVER_PORT FRP_TOKEN STCP_SERVER_NAME STCP_SECRET G4_PROM_LOCAL_PORT

# 1) 准备 frpc：下载二进制 + 渲染配置
if [ ! -x "$HERE/tunnel/frpc" ]; then
  echo "==> 下载 frpc ${FRP_VERSION}"
  curl -sL "https://github.com/fatedier/frp/releases/download/v${FRP_VERSION}/frp_${FRP_VERSION}_linux_amd64.tar.gz" -o /tmp/frp-gcp.tgz
  tar xzf /tmp/frp-gcp.tgz -C /tmp && cp "/tmp/frp_${FRP_VERSION}_linux_amd64/frpc" "$HERE/tunnel/frpc"
  rm -rf /tmp/frp-gcp.tgz "/tmp/frp_${FRP_VERSION}_linux_amd64"
fi
envsubst < "$HERE/tunnel/frpc.toml.tmpl" > "$HERE/tunnel/frpc.toml"

# 2) 隧道做成 systemd 服务(clean env 不继承 shell 的 mihomo http_proxy；开机自启+崩溃自愈)
echo "==> 安装并启动 systemd 隧道服务 ifa-grafana-tunnel"
sudo tee /etc/systemd/system/ifa-grafana-tunnel.service >/dev/null <<UNIT
[Unit]
Description=ifa grafana-gcp frp STCP tunnel (g4-01 Prometheus -> 127.0.0.1:${G4_PROM_LOCAL_PORT})
After=network.target
[Service]
Type=simple
WorkingDirectory=$HERE/tunnel
ExecStart=$HERE/tunnel/frpc -c $HERE/tunnel/frpc.toml
Restart=on-failure
RestartSec=5
[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable --now ifa-grafana-tunnel
sudo systemctl restart ifa-grafana-tunnel

# 3) 等隧道就绪
echo "==> 等隧道就绪：curl 127.0.0.1:${G4_PROM_LOCAL_PORT}/-/ready（至多 60s）"
ok=0; for i in $(seq 1 30); do curl -fsS -o /dev/null -m3 "http://127.0.0.1:${G4_PROM_LOCAL_PORT}/-/ready" 2>/dev/null && { echo "    就绪(约$((i*2))s)"; ok=1; break; }; sleep 2; done
[ "$ok" = 1 ] || { echo "!! 隧道未就绪，看 journalctl -u ifa-grafana-tunnel（注意 7890 代理报错）与 .env"; exit 1; }

# 4) 检查本机 Prometheus（数据源①）
curl -fsS -o /dev/null -m3 "http://127.0.0.1:${LOCAL_PROM_PORT}/-/ready" 2>/dev/null && echo "==> 本机 Prometheus(${LOCAL_PROM_PORT}) OK" || echo "!! 警告：本机 Prometheus(${LOCAL_PROM_PORT}) 不可达（数据源①会空）"

# 5) 起统一 Grafana
echo "==> docker compose up -d"; docker compose up -d
echo ""; echo "==> 完成。访问 http://<本机内网IP>:${GRAFANA_PORT}（在 5090 即 http://192.168.100.50:${GRAFANA_PORT}）"
echo "    看板：/d/g4-01、/d/locateanything"
