# grafana-gcp — 统一监控看板（一步拉起）

在 **5090（主服务器）** 上一步拉起 **本机统一 Grafana**（docker，绑 `0.0.0.0:${GRAFANA_PORT}`，默认 3001），唯一数据源为 **本机 Prometheus（`127.0.0.1:9091`）**。

访问：**`http://192.168.100.50:3001`**（office 局域网直连，不走公网、不用 SSH 隧道）。

访问：**`http://192.168.100.50:3001`**（office 局域网直连，不走公网、不用 SSH 隧道）。

## 架构（本机单数据源）

```text
  局域网用户 ──http──> 本机统一 Grafana(:3001, 绑 0.0.0.0)
                         └─ 唯一数据源：本机 Prometheus(127.0.0.1:9091)
                              抓取：宿主 vLLM(:8000) / node_exporter / gpu-exporter / SAM3 等
```

- 本机 9090 被 mihomo 代理占用，Prometheus 固定跑在 **9091**。
- 历史上的 g4 联邦观测（frp STCP 隧道 29090、g4-01 Prometheus 数据源、systemd 服务
  `ifa-grafana-tunnel`）**已整体废弃**：VLM 已本机化（宿主 `vllm serve :8000`），不再有
  第二台算力服务器要联邦。历史机器上若残留隧道服务，手动
  `sudo systemctl disable --now ifa-grafana-tunnel` 清理即可。

## 用法

```bash
cd grafana-gcp
cp .env.example .env      # 首次：填 GRAFANA_ADMIN_PASSWORD
bash up.sh                # 一步拉起统一 Grafana
# 访问 http://192.168.100.50:3001
bash down.sh              # 停
```

`.env` 关键项：
- `GRAFANA_PORT=3001`：Grafana 端口（host 网络直绑）。
- `LOCAL_PROM_PORT=9091`：本机 Prometheus（9090 被 mihomo 占）。

## 前置

- 5090 iptables INPUT 默认 DROP：需放行 Grafana 端口（`iptables -I INPUT -s 192.168.100.0/24 -p tcp --dport 3001 -j ACCEPT`）。
- 本机 Prometheus 已在 9091 运行（部署正源见 odyss-models 仓 `deploy/gpu5090`）。

## 文件

- `docker-compose.yml`：统一 Grafana（host 网络，0.0.0.0:3001）。
- `grafana/provisioning/datasources/prometheus.yml`：唯一数据源（本机 Prometheus 9091），并幂等清理历史 g4-01 数据源。
- `grafana/dashboards/`：
  - `g4-vllm.json`：**本机 Qwen vLLM** 看板（宿主 `vllm serve :8000`，vLLM 指标经本机 Prometheus）。
  - `gpu5090-server.json`：5090 服务器性能（CPU/内存/GPU/温度/磁盘/网络；GPU 现为双卡 **RTX Pro 6000 Blackwell 96GB（vLLM）+ RTX 5090 32GB（SAM3/DA3）**，面板会出两张卡的曲线），依赖 odyss-models `deploy/gpu5090` 栈里的 node_exporter + gpu-exporter。
  - `sam3.json`：SAM3 服务看板。
- `up.sh` / `down.sh`：一步拉起 / 停。
