# tools/health · 5090 全栈健康体检（两态期望清单 + 单元修复入口）

一步体检 .50 上 ifa 演示栈的全部服务。**健康的定义是「能力可用」而不是「容器在跑」**——探测打到业务语义层（HTTP 语义码、本机 vLLM `/v1/models`），而非只看 `docker ps`。配套 Claude skill：`ifa-health-check`。

```bash
# 开发机一步执行（只读，不改现场）：
ssh odyss-server-frpc 'bash -s' < tools/health/check.sh
```

输出每行 `状态|组件|详情`（OK / FAIL / WARN / INFO），有 FAIL 时退出码 1。

## 两态期望清单（本表是唯一正源，改期望先改这里）

### ① 应运行——按四种能力分组

| 能力 | 组件 | 探测 | 所属单元 |
|---|---|---|---|
| A 业务闭环 | postgres / redis / minio | 容器 healthy | local-stack |
| A | llm-mock / llm-switch | 容器 running（llm-switch 内部 8095，qwen/gemini 切换器） | local-stack |
| A | services | 容器 + 18090 空参 send-code 返回 **400** | local-stack |
| A | superadmin | 18091=200 + `/admin-api` 非 000/502/504 | local-stack |
| A | 本机 vLLM | 8000 `/v1/models` 返回 **200/401**（带 api-key，未带 key 的 401 也算活） | 宿主裸进程（/home/odyss/vllm-env，随开机启动） |
| A | VLM 深链 | 栈内经 docker 网关 `172.23.0.1:8000` 探宿主 vLLM 有应答 | local-stack + 宿主 vLLM |
| B 演示 | da3-web | systemd active + 8060=200/30x（根路径 307 → /experience） | ifa-support 根目录 |
| B | SAM3 | systemd active + 8013 监听 | model/sam3 |
| C 观测 | 统一 Grafana | 容器 + 3001 `/api/health`=200 | grafana-gcp |
| C | 本机 Prometheus + 双 exporter | 容器 + 9091 `/-/ready`=200 + 9835 GPU 指标本体 | odyss-models gpu5090 |
| D 底座 | docker / GPU 驱动 / frp 公网入口 / 磁盘 | daemon、nvidia-smi、systemd frpc、根分区 <90% | 系统 |

补充事实：

- 本机 vLLM 模型为 Qwen3.6-35B-A3B-FP8（served name 含别名 gemini-3.1-pro-preview），监听 `0.0.0.0:8000`；services 调 VLM 走 `http://172.23.0.1:8000/v1`（docker 网关直达宿主）。
- 显卡为 RTX Pro 6000 Blackwell 96GB（vLLM 约 64G + SAM3 约 9G），显存充裕，**「预期停止腾显存」概念已随 LocateAnything 下线一并消失**。
- 观测容器名仍带 `locateanything-` 前缀（历史名，职能已是全机观测）：保留容器名，探测标签一律用中性名（「观测/本机Prometheus」「观测/gpu-exporter」等）。

### ② 一次性完成——Exited(0) 即健康

`odyss-ifa-migrate`、`odyss-ifa-minio-init`。

### 范围外（体检不管，也不要动）

- **已下线，不再探测**：LocateAnything 全家桶（LA vLLM 8001、gateway-1 8010、gateway-2 8020、LB 8000-nginx、SigLIP/clip-score-server 7861、旧 LA-Grafana 127.0.0.1:3000）；llm-tunnel（28000，→GCP g4 链路已废弃，容器将从 compose 移除）；g4 联邦观测（ifa-grafana-tunnel systemd 与 127.0.0.1:29090）。注意 **8000 端口现属宿主 vLLM**，与旧 LA-LB 无关。
- 历史停用：odyss-gitea、cpa-preview 两容器、comfyui / food-image-search / milvus / netalertx 等。

## 单元修复入口（每个单元只有一个合法拉起方式）

| 异常组件 | 唯一拉起入口 |
|---|---|
| local-stack 任何容器 | `cd ~/odyss-services-ifa && docker compose up -d`（幂等，一次拉齐全栈） |
| da3-web | `sudo systemctl restart da3-web`；要更新代码用 `~/da3-web/deploy.sh` |
| SAM3 | `sudo systemctl restart sam3` |
| 本机观测容器 | `cd ~/odyss-models/deploy/gpu5090 && docker compose -f compose.gpu.yml up -d prometheus grafana gpu-exporter node-exporter` ——**必须点名服务，严禁裸 `up -d`** |
| 统一 Grafana | `cd ~/da3-web/grafana-gcp && ./up.sh` |
| 本机 vLLM（8000） | 宿主侧排查裸进程：`pgrep -af vllm` + 8000 监听 + 其启动日志（/home/odyss/vllm-env，随开机启动）。**不再指向 GCP g4/ssh-gcp-gpu，该链路已废弃** |
| VLM 深链失败而本机 vLLM OK | 查 docker 网络/防火墙（栈内走 172.23.0.1 直达宿主）；本机最多 `cd ~/odyss-services-ifa && docker compose up -d` 复位栈 |
| frp 公网入口 | FAIL 先判断是否**整机断外网**（如 `curl -m 5 -sI https://www.baidu.com`）：断网则 frpc 必然 activating，属网络侧待办（VLAN40 transit 未通，2026-08-21 迁办公网后的已知现状），**勿反复重启 frpc**；外网正常仍挂再走局域网入口人工处理 |

修复后**必须重跑 check.sh 复验**，以复验结果为准出报告。

## LAN 可达性（服务器侧 OK ≠ 用户打得开）

5090 开着 ufw 且 INPUT 默认 DROP。**docker 端口映射的服务（18090/18091）走 FORWARD 链绕过 ufw**；**host 网络/宿主进程的服务（3001/8000/9091 等）进 INPUT 链，必须逐端口放行**。因此体检除了在服务器上跑 check.sh，还要**在开发机（Mac）上**对用户入口做连通性探测：

```bash
for p in 18090 18091 8060 3001; do nc -z -G 3 192.168.100.50 $p && echo "$p 通" || echo "$p 不通"; done
```

不通且服务器侧 OK → 补 ufw 放行（仅限所在局域网网段，办公网 192.168.100.0/24 / 现场 192.168.0.0/24，照现有模式）：

```bash
sudo ufw allow from 192.168.100.0/24 to any port <端口> proto tcp comment "<用途> LAN"
```

已放行：8060（da3-web）、3001（统一 Grafana，2026-07-31 补）。8000 是宿主 vLLM：虽监听 `0.0.0.0:8000`，但 ufw 未对 LAN 放行——栈内容器经 docker 网关 172.23.0.1 访问、不依赖 LAN 放行；是否对 LAN 开放按需另定，默认不放行。

## 已知坑：daemon-reload 吊销容器 NVML

宿主执行 `systemctl daemon-reload`（装 systemd 单元、up.sh 都会触发）后，NVIDIA 容器内新起的进程会报 `Failed to initialize NVML: Unknown Error`：gpu-exporter 每次抓取新起 nvidia-smi → GPU 指标消失（但 exporter 进程与 Prometheus target 都显示正常——**"假 up"**，2026-07-31~08-07 因此静默断采一周才被发现）；长驻且启动时已持有 GPU 句柄的容器不受影响；宿主裸进程 vLLM 不经 nvidia-container-toolkit，无此坑。修复：`cd ~/odyss-models/deploy/gpu5090 && docker compose -f compose.gpu.yml up -d --force-recreate gpu-exporter`——**不要只 `docker restart`**：2026-08-07 实测该次故障中 restart 无效（07-31 重启过一次仍持续报错），只有 force-recreate 重走 nvidia runtime 设备注入才恢复。修复后跑本 check.sh 复验「观测/GPU指标」为 OK。2026-07-03 起反复出现，根治要动 nvidia-container-toolkit 配置（odyss-models 仓的部署层欠账）。

## 纪律（继承 MANAGEMENT.md §1/§6）

- 部署机上只探测 + 跑上表入口，**严禁改配置孤本、git commit/push、裸 docker 命令做部署级变更**。
- 期望状态变了（如某组件上/下线、端口变更）→ 先改本 README 与 check.sh 再执行，不允许现场口头例外。
