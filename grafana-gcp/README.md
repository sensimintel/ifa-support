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
- `grafana/dashboards/`（2026-08-24 起服务器为**双卡**：GPU0 = RTX Pro 6000 Blackwell 96GB 跑宿主 Qwen vLLM:8000，GPU1 = RTX 5090 32GB 跑 SAM3:8013。GPU 指标按「服务器整体 + 每服务锁自己那张卡」两个层面呈现）：
  - `g4-vllm.json`：**Qwen vLLM · RTX Pro 6000** 看板（宿主 `vllm serve :8000` 的吞吐/延时/KV cache + GPU 面板经隐藏变量 `gpu_uuid` 锁定 GPU0=RTX Pro 6000，卡按名字匹配、换卡不用改看板）。
  - `sam3.json`：**SAM3 · RTX 5090** 服务看板（QPS/延时/错误率 + GPU 面板同机制锁定 GPU1=RTX 5090）。
  - `gpu5090-server.json`：5090 服务器性能（CPU/内存/温度/磁盘/网络 + **双卡 GPU**——所有 GPU 面板 join `nvidia_smi_gpu_info` 按卡名出双 series），依赖 odyss-models `deploy/gpu5090` 栈里的 node_exporter + gpu-exporter。
  - ⚠️ gpu-exporter 容器在换卡/驱动变更后可能报 `Failed to initialize NVML`（`nvidia_smi_command_exit_code 255`、GPU 指标整体消失），`docker compose -f compose.gpu.yml up -d --force-recreate gpu-exporter`（odyss-models `deploy/gpu5090/`）重建即恢复。
- `up.sh` / `down.sh`：一步拉起 / 停。
