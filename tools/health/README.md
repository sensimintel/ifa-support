# tools/health · 5090 全栈健康体检（两态期望清单 + 单元修复入口）

一步体检 .50 上 ifa 演示栈的全部服务。**健康的定义是「能力可用」而不是「容器在跑」**——探测打到业务语义层（HTTP 语义码、本机 vLLM `/v1/models`），而非只看 `docker ps`。配套 Claude skill：`ifa-health-check`。

```bash
# 开发机一步执行（只读，不改现场）：
ssh odyss-server-frpc 'bash -s' < tools/health/check.sh
# 整机无外网期间 frp 入口不可用，改用局域网别名：
ssh odyss-server-local 'bash -s' < tools/health/check.sh
```

输出每行 `状态|组件|详情`（OK / FAIL / WARN / INFO），有 FAIL 时退出码 1。

## 两态期望清单（本表是唯一正源，改期望先改这里）

### ① 应运行——按四种能力分组

| 能力 | 组件 | 探测 | 所属单元 |
|---|---|---|---|
| A 业务闭环 | postgres / redis / minio | 容器 healthy | local-stack |
| A | llm-mock / llm-switch | 容器 running（llm-switch 内部 8095，qwen/gemini 切换器） | local-stack |
| A | services | 容器 + 18090 空参 send-code 返回 **400** | local-stack |
| A | 登录密钥 | 18090 `/api/v1/auth/password-encryption-key` 返回 **200**（503=孤本未配 encryption key，App 登录请求根本发不出——真实故障史） | local-stack |
| A | superadmin | 18091=200 + `/admin-api` 非 000/502/504 | local-stack |
| A | llm-switch 控制面 | 18091 `/ops-api/` 非 000/502/504（superadmin 顶栏模型切换走此反代，容器 running ≠ 反代通） | local-stack |
| A | lumen 观测栈 | lumen-postgres(healthy)/collector/observation 容器 + 18091 `/lumen/api/health`=200（支撑 superadmin 的 Lumen 数据板块） | 独立 compose 项目 `/home/odyss/odyss-ifa-lumen`，**不被 odyss-services-ifa 的 up -d 覆盖** |
| A | 本机 vLLM | 8000 `/v1/models` 返回 **200/401**（带 api-key，未带 key 的 401 也算活） | 宿主裸进程（/home/odyss/vllm-env，随开机启动） |
| A | VLM 深链 | 栈内经 docker 网关 `172.23.0.1:8000` 探宿主 vLLM 有应答 | local-stack + 宿主 vLLM |
| B 演示 | da3-web | systemd active + 8060=200/30x（根路径 307 → /experience）+ **监听 PID == MainPID**（漂移判据，防游离/孤儿进程假绿） | ifa-support 根目录 |
| B | SAM3 | systemd active + 8013 监听 + **监听 PID == MainPID** | model/sam3 |
| B | mac-mini 帧链路 | 8060 `/api/frame/status` 存在 `macmini-*` 设备且帧龄 ≤30s（断推 = 浅体验区无画面） | mac-mini（192.168.100.3）cam-pusher LaunchDaemon |
| B | dx-backend | systemd active + 8070 `/api/health`=200 + 18091 `/dx-api/` 反代 200（深体验区四通道秤 + 分组绑定） | ifa-support 根目录（dx_backend.py，宿主 :8070） |
| B | 秤链路 | 8070 `/api/food-scales` 的最大 `age_s` ≤30s（演示期口径：过期/无读数即 FAIL） | 秤 SJ101T2（192.168.100.80:502，Modbus TCP） |
| C 观测 | 统一 Grafana | 容器 + 3001 `/api/health`=200 | grafana-gcp |
| C | 本机 Prometheus + 双 exporter | 容器 + 9091 `/-/ready`=200 + 9835 GPU 指标本体 | odyss-models gpu5090 |
| D 底座 | docker / GPU 驱动 / frp 公网入口 / 磁盘 | daemon、nvidia-smi、systemd frpc、根分区 <90% | 系统 |

补充事实：

- 本机 vLLM 模型为 Qwen3.6-35B-A3B-FP8（served name 含别名 gemini-3.1-pro-preview），监听 `0.0.0.0:8000`；services 调 VLM 走 `http://172.23.0.1:8000/v1`（docker 网关直达宿主）。
- 显卡为双卡（2026-08-24 起）：GPU0 = RTX Pro 6000 Blackwell 96GB **独占给 vLLM**（`gpu-memory-utilization 0.90`）；GPU1 = RTX 5090 32GB 归 SAM3 独享（`SAM3_MEM_FRACTION=0.9`；DA3 已于 2026-08-25 退役）。三个 unit 均在 systemd 钉卡（sam3/da3-web 按 GPU UUID；vllm 按 `PCI_BUS_ID`+序号 0——vLLM 不认 UUID 形式），check.sh 有「落卡」检查项兜底；**「预期停止腾显存」概念已随 LocateAnything 下线一并消失**。
- 观测容器名仍带 `locateanything-` 前缀（历史名，职能已是全机观测）：保留容器名，探测标签一律用中性名（「观测/本机Prometheus」「观测/gpu-exporter」等）。

### ② 一次性完成——Exited(0) 即健康

`odyss-ifa-migrate`、`odyss-ifa-minio-init`、`odyss-ifa-lumen-migrate`。

### 范围外（体检不管，也不要动）

- **已下线，不再探测**：LocateAnything 全家桶（LA vLLM 8001、gateway-1 8010、gateway-2 8020、LB 8000-nginx、SigLIP/clip-score-server 7861、旧 LA-Grafana 127.0.0.1:3000）；llm-tunnel（28000，→GCP g4 链路已废弃，容器将从 compose 移除）；g4 联邦观测（ifa-grafana-tunnel systemd 与 127.0.0.1:29090）。注意 **8000 端口现属宿主 vLLM**，与旧 LA-LB 无关。
- 历史停用：odyss-gitea、cpa-preview 两容器、comfyui / food-image-search / milvus / netalertx 等。

## 单元修复入口（每个单元只有一个合法拉起方式）

| 异常组件 | 唯一拉起入口 |
|---|---|
| local-stack 任何容器 | `cd ~/odyss-services-ifa && docker compose up -d`（幂等，一次拉齐全栈） |
| 登录密钥 503 | 孤本缺 encryption key 配置：查 `~/odyss-services-ifa` 的 .env/配置补 key 后 `docker compose up -d`（按 local-stack SOP，勿在 5090 改入仓文件） |
| lumen 栈任何容器 | `cd /home/odyss/odyss-ifa-lumen && docker compose up -d`（独立项目，**不被 odyss-services-ifa 的 up -d 覆盖**） |
| da3-web | `sudo systemctl restart da3-web`；要更新代码用 `~/da3-web/deploy.sh` |
| da3-web/SAM3 漂移（监听 PID ≠ MainPID） | da3-web 用 `~/da3-web/deploy.sh`（自动清游离进程再走 systemd）；SAM3 先 `pkill -f "sam3[_]server.py"` 再 `sudo systemctl restart sam3`。**勿裸 restart**（端口被占会 bind 失败） |
| SAM3 | `sudo systemctl restart sam3` |
| dx-backend | `sudo systemctl restart dx-backend`；要更新代码用 `~/da3-web/deploy.sh`（da3-web 与 dx-backend 一并部署，内含 8070 就绪检查）。反代 504 时查 ufw 是否放行 docker 子网 `172.23.0.0/16 → 8070` |
| 秤链路 | 服务器侧无拉起动作：查秤上电/网线/IP=192.168.100.80，dx-backend 持久连接会自动重连。**勿裸探 Modbus**：新 TCP 连接首个请求约 6.4s 才应答，短超时会误判"固件卡死"——一律看 `8070/api/food-scales` 的 `age_s` |
| mac-mini 帧链路 | cam-pusher 自带 LaunchDaemon KeepAlive + 5 级自愈，多数断推会自恢复；仍断则开发机执行 `ssh mac-mini 'MINI_SUDO_PASS=… ~/ifa-support/mac-mini/deploy.sh'`（或 mini 上 `cd ~/ifa-support/mac-mini && ./deploy.sh`） |
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

不通且服务器侧 OK → 补 ufw 放行（演示网网段 192.168.100.0/24，照现有模式）：

```bash
sudo ufw allow from 192.168.100.0/24 to any port <端口> proto tcp comment "<用途> LAN"
```

已放行：8060（da3-web）、3001（统一 Grafana，2026-07-31 补）。8000 是宿主 vLLM：虽监听 `0.0.0.0:8000`，但 ufw 未对 LAN 放行——栈内容器经 docker 网关 172.23.0.1 访问、不依赖 LAN 放行；是否对 LAN 开放按需另定，默认不放行。

## 已知坑：daemon-reload 吊销容器 NVML

宿主执行 `systemctl daemon-reload`（装 systemd 单元、up.sh 都会触发）后，NVIDIA 容器内新起的进程会报 `Failed to initialize NVML: Unknown Error`：gpu-exporter 每次抓取新起 nvidia-smi → GPU 指标消失（但 exporter 进程与 Prometheus target 都显示正常——**"假 up"**，2026-07-31~08-07 因此静默断采一周才被发现）；长驻且启动时已持有 GPU 句柄的容器不受影响；宿主裸进程 vLLM 不经 nvidia-container-toolkit，无此坑。修复：`cd ~/odyss-models/deploy/gpu5090 && docker compose -f compose.gpu.yml up -d --force-recreate gpu-exporter`——**不要只 `docker restart`**：2026-08-07 实测该次故障中 restart 无效（07-31 重启过一次仍持续报错），只有 force-recreate 重走 nvidia runtime 设备注入才恢复。修复后跑本 check.sh 复验「观测/GPU指标」为 OK。2026-07-03 起反复出现，根治要动 nvidia-container-toolkit 配置（odyss-models 仓的部署层欠账）。

**自愈 cron**（odyss-models PR#30）：`gpu_exporter_selfheal.sh` 每 5 分钟探 9835，连续缺失自动 force-recreate；check.sh 以 INFO 报告其安装状态。人工处置前先 `crontab -l | grep gpu_exporter_selfheal` 并看 `~/odyss-models/deploy/gpu5090/cache/gpu_exporter_selfheal.log`——cron 已装时可能已自愈/即将自愈，避免重复动作、误报"人工修好的"。

## 纪律（继承 MANAGEMENT.md §1/§6）

- 部署机上只探测 + 跑上表入口，**严禁改配置孤本、git commit/push、裸 docker 命令做部署级变更**。
- 期望状态变了（如某组件上/下线、端口变更）→ 先改本 README 与 check.sh 再执行，不允许现场口头例外。
