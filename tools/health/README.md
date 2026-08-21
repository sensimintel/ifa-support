# tools/health · 5090 全栈健康体检（三态期望清单 + 单元修复入口）

一步体检 .50 上 ifa 演示栈的全部服务。**健康的定义是「能力可用」而不是「容器在跑」**——探测打到业务语义层（HTTP 语义码、隧道后 `/v1/models`），而非只看 `docker ps`。配套 Claude skill：`ifa-health-check`。

```bash
# 开发机一步执行（只读，不改现场）：
ssh odyss-server-frpc 'bash -s' < tools/health/check.sh
```

输出每行 `状态|组件|详情`（OK / FAIL / WARN / INFO），有 FAIL 时退出码 1。

## 三态期望清单（本表是唯一正源，改期望先改这里）

### ① 应运行——按四种能力分组

| 能力 | 组件 | 探测 | 所属单元 |
|---|---|---|---|
| A 业务闭环 | postgres / redis / minio | 容器 healthy | local-stack |
| A | llm-mock / llm-tunnel | 容器 running | local-stack |
| A | services | 容器 + 18090 空参 send-code 返回 **400** | local-stack |
| A | superadmin | 18091=200 + `/admin-api` 非 000/502/504 | local-stack |
| A | VLM 深链 | 栈内经 llm-tunnel 探远端 vLLM `/v1/models` 有应答 | local-stack + GCP |
| B 演示 | da3-web | systemd active + 8060=200 | ifa-support 根目录 |
| B | SAM3 | systemd active + 8013 监听 | model/sam3 |
| B | LA vLLM / gateway-1 / LB | 容器 healthy + 8001 `/v1/models` + 8010 监听 + 8000 非 5xx | odyss-models gpu5090 |
| C 观测 | 统一 Grafana | 容器 + 3001 `/api/health`=200 | grafana-gcp |
| C | g4 联邦隧道 | systemd active + 29090 `/-/ready`=200 | grafana-gcp |
| C | 本机 Prometheus + 双 exporter + LA-Grafana | 容器 + 9091 `/-/ready`=200 | odyss-models gpu5090 |
| D 底座 | docker / GPU 驱动 / frp 公网入口 / 磁盘 | daemon、nvidia-smi、systemd frpc、根分区 <90% | 系统 |

### ② 预期停止——在跑反而是异常（显存风险，报告用户，勿当健康）

| 组件 | 停止原因 |
|---|---|
| locateanything-server-2（gateway-2, 8020） | 给 SAM3 腾显存（见 `model/README.md` 显存预算） |
| clip-score-server（SigLIP, 7861） | 同上 |

### ③ 一次性完成——Exited(0) 即健康

`odyss-ifa-migrate`、`odyss-ifa-minio-init`。

### 范围外（体检不管，也不要动）

odyss-gitea、cpa-preview 两容器、comfyui / food-image-search / milvus / netalertx 等历史停用容器。

## 单元修复入口（每个单元只有一个合法拉起方式）

| 异常组件 | 唯一拉起入口 |
|---|---|
| local-stack 任何容器 | `cd ~/odyss-services-ifa && docker compose up -d`（幂等，一次拉齐全栈） |
| da3-web | `sudo systemctl restart da3-web`；要更新代码用 `~/da3-web/deploy.sh` |
| SAM3 | `sudo systemctl restart sam3` |
| LA / 本机观测容器 | `cd ~/odyss-models/deploy/gpu5090 && docker compose -f compose.gpu.yml up -d locateanything-vllm locateanything-gateway-1 locateanything-lb prometheus grafana gpu-exporter node-exporter` ——**必须点名服务，严禁裸 `up -d`**（会拉起 gateway-2 与 siglip-score 抢显存） |
| 统一 Grafana | `cd ~/da3-web/grafana-gcp && ./up.sh`（含隧道自检） |
| g4 联邦隧道 | `sudo systemctl restart ifa-grafana-tunnel` |
| frp 公网入口 | **不自动重启**：你若正经 frpc 连着，它必然活着；真挂了走局域网入口人工处理 |
| VLM 深链失败 | 根因大概率在 GCP g4-01 侧（用 ssh-gcp-gpu 排查）；本机最多 `docker restart odyss-ifa-llm-tunnel` |
| 预期停止组件在跑 | 不自动停，报告用户决策（可能有人临时在用） |

修复后**必须重跑 check.sh 复验**，以复验结果为准出报告。

## LAN 可达性（服务器侧 OK ≠ 用户打得开）

5090 开着 ufw 且 INPUT 默认 DROP。**docker 端口映射的服务（18090/18091）走 FORWARD 链绕过 ufw**；**host 网络的服务（3001/3000/8000/9091 等）进 INPUT 链，必须逐端口放行**。因此体检除了在服务器上跑 check.sh，还要**在开发机（Mac）上**对用户入口做连通性探测：

```bash
for p in 18090 18091 8060 3001; do nc -z -G 3 192.168.100.50 $p && echo "$p 通" || echo "$p 不通"; done
```

不通且服务器侧 OK → 补 ufw 放行（仅限局域网网段，照现有模式）：

```bash
sudo ufw allow from 192.168.100.0/24 to any port <端口> proto tcp comment "<用途> LAN"
```

已放行：8060（da3-web）、3001（统一 Grafana，2026-07-31 补）。8000/3000 有意未放行（LA 由 da3-web 经 127.0.0.1 内部调用；旧 LA-Grafana 走公网 file.odyss.life/grafana）。

## 已知坑：daemon-reload 吊销容器 NVML

宿主执行 `systemctl daemon-reload`（装 systemd 单元、up.sh 都会触发）后，NVIDIA 容器内新起的进程会报 `Failed to initialize NVML: Unknown Error`：gpu-exporter 每次抓取新起 nvidia-smi → GPU 指标消失（但 exporter 进程与 Prometheus target 都显示正常——**"假 up"**，2026-07-31~08-07 因此静默断采一周才被发现）；vllm/gateway 等服务容器启动时已持有 GPU 句柄，**推理不受影响、无需重启**。修复：`cd ~/odyss-models/deploy/gpu5090 && docker compose -f compose.gpu.yml up -d --force-recreate gpu-exporter`——**不要只 `docker restart`**：2026-08-07 实测该次故障中 restart 无效（07-31 重启过一次仍持续报错），只有 force-recreate 重走 nvidia runtime 设备注入才恢复。修复后跑本 check.sh 复验「观测/GPU指标」为 OK。2026-07-03 起反复出现，根治要动 nvidia-container-toolkit 配置（odyss-models 仓的部署层欠账）。

## 纪律（继承 MANAGEMENT.md §1/§6）

- 部署机上只探测 + 跑上表入口，**严禁改配置孤本、git commit/push、裸 docker 命令做部署级变更**。
- 期望状态变了（如 gateway-2 恢复常驻）→ 先改本 README 与 check.sh 再执行，不允许现场口头例外。
