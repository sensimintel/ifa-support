# grafana-gcp — 双服务器统一监控看板（一步拉起）

在 **5090（主服务器）** 上一步拉起两样东西，得到一个统一的 Grafana 监控入口：

1. **frp STCP 隧道**：把 GCP `gpu-g4-01`（算力服务器）上 loopback 的 **Prometheus** 点对点拉到本机 `127.0.0.1:${G4_PROM_LOCAL_PORT}`（默认 29090）；
2. **本机统一 Grafana**（docker，绑 `0.0.0.0:${GRAFANA_PORT}`，默认 3001）：两个数据源分别展示两台服务器。

访问：**`http://192.168.0.50:3001`**（office 局域网直连，不走公网、不用 SSH 隧道）。

## 架构（方案二 · 算力/主分离 + 统一 Grafana）

```text
  局域网用户 ──http──> 本机统一 Grafana(:3001, 绑 0.0.0.0)
                         ├─ 数据源① 5090 本机 Prometheus(127.0.0.1:9091)  → LocateAnything 看板
                         └─ 数据源② g4-01 Prometheus(127.0.0.1:29090)     → g4-01 vLLM&GPU 看板
                                      ▲
                                 frp STCP 隧道(frpc visitor, 只用 frp 服务器控制口 7000)
                                      ▲
                              g4-01: 本地 Prometheus(:9090, loopback) 抓 VLM(:8000)+gpu-exporter(:9835)
                                     frpc 以 STCP 暴露 9090
```

## 为什么用 frp STCP，而不是 gcloud IAP

- g4-01 无公网 SSH，观测栈全绑 `127.0.0.1`；但 **5090 没装 gcloud**，且 5090 到 frp 服务器**只放行 7000/10022**（公网高位端口出不去）。
- frp STCP 只用 frp 服务器控制口 7000 做点对点隧道，不需要 gcloud/IAP，也不公网暴露 metrics——是这个网络下唯一可行且安全的方式。
- g4-01 端的 Prometheus 与 frpc（STCP server 侧）属 g4-01 的部署（见 `odyss-models/deploy/gcp-g4`），本套件只管 **5090 侧**：frpc visitor + 统一 Grafana。

## 用法

```bash
cd grafana-gcp
cp .env.example .env      # 首次：填 FRP_TOKEN / STCP_SECRET / GRAFANA_ADMIN_PASSWORD
bash up.sh                # 一步拉起：frp 隧道 + 统一 Grafana
# 访问 http://192.168.0.50:3001
bash down.sh              # 停
```

`.env` 关键项：
- `FRP_TOKEN`：与 5090 现有 frpc（`/usr/local/frp/frpc/frpc.toml`）的 `auth.token` 一致。
- `STCP_SECRET`：与 g4-01 端 frpc 的 `g4-prometheus` proxy 的 `secretKey` 一致。
- `LOCAL_PROM_PORT=9091`：5090 本机 Prometheus（9090 被 mihomo 占）。
- `G4_PROM_LOCAL_PORT=29090`：g4-01 Prometheus 隧道落地端口。

## 前置

- 5090 iptables INPUT 默认 DROP：需放行 Grafana 端口（`iptables -I INPUT -s 192.168.0.0/24 -p tcp --dport 3001 -j ACCEPT`）。
- g4-01 端已跑 `prometheus` + `frpc`（STCP 暴露 `g4-prometheus`=本地 9090）。

## 文件

- `docker-compose.yml`：统一 Grafana（host 网络，0.0.0.0:3001）。
- `grafana/provisioning/datasources/prometheus.yml`：两个数据源①②。
- `grafana/dashboards/`：`locateanything.json`（→数据源①）、`g4-vllm.json`（→数据源②）。
- `tunnel/frp-tunnel.sh` + `frpc.toml.tmpl`：frp STCP visitor（下载 frpc + 渲染配置 + 常驻）。
- `up.sh` / `down.sh`：一步拉起 / 停。
