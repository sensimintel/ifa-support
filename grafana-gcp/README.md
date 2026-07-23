# grafana-gcp — GCP gpu-g4-01 监控看板（一步拉起）

在**任意一台 Linux 服务器**（默认在 5090）上，一步同时拉起两样东西：

1. **持久 IAP 隧道**：把 GCP `gpu-g4-01` 上 loopback 的 **Prometheus** 转发到本机；
2. **本机 Grafana**（docker，绑 `0.0.0.0`）：以隧道端口为数据源，展示现成的「gpu-g4-01 vLLM 观测」看板。

于是你就能用 **`http://<本机内网IP>:<GRAFANA_PORT>`**（在 5090 上即 `http://192.168.0.50:3001`）访问 GCP 那台机器的 vLLM/GPU 监控。

## 为什么是「隧道 Prometheus + 本机 Grafana」

`gpu-g4-01` 无公网 SSH 入口，观测栈（vLLM `:8000` / Prometheus `:9090` / gpu_exporter `:9835` / 它自己的 Grafana `:3000`）**全绑 `127.0.0.1`**，唯一入口是 **gcloud IAP 隧道**（见 `odyss-models/deploy/gcp-g4`）。而 GCP 端 Grafana 也只 loopback，无法直接给局域网看。

所以这里在**本机**另起一份 Grafana（绑 `0.0.0.0`，局域网可达），数据经 IAP 隧道只转发 **Prometheus 一个端口**过来。看板 `grafana/dashboards/g4-vllm.json` 与 GCP 端是同一份（数据源 uid 都是 `prometheus`），指标全在 GCP 的 Prometheus 里（15 天保留）。

```text
  局域网用户 ──http──> 本机 Grafana(:3001, 绑0.0.0.0)
                         │ 数据源 = 127.0.0.1:19090
                         ▼
                    IAP 隧道(gcloud, 常驻自愈)
                         │  -L 127.0.0.1:19090 : 127.0.0.1:9090
                         ▼
              GCP gpu-g4-01  Prometheus 127.0.0.1:9090（loopback）
```

## 端口（可在 .env 改）

| 变量 | 默认 | 说明 |
|---|---|---|
| `GRAFANA_PORT` | `3001` | 本机 Grafana 端口（访问入口）。避开 5090 上 LocateAnything 已占的 3000。 |
| `LOCAL_PROM_PORT` | `19090` | 隧道在本机的落地口 = Grafana 数据源。避开 5090 已占的 9090。 |
| `REMOTE_PROM_PORT` | `9090` | GCP 端 Prometheus 的 loopback 端口（一般不改）。 |

## 前置

1. **docker + docker compose**（5090 已有）。
2. **gcloud SDK + 已登录 GCP Workforce 身份**（见下节）。这是隧道能建立的关键。

## 认证与持久化（务必读）

隧道走 GCP **Workforce Identity Federation**（不是 Google 账号、不是 SSH 密钥），凭证由本机 `gcloud` 持有。首次/过期时需**本人在浏览器**完成一次 SSO 登录：

```bash
# 若本机还没有登录配置文件，先生成一份（本地操作、无密钥、不改云资源）：
gcloud iam workforce-pools create-login-config \
  locations/global/workforcePools/odyss-workforce/providers/cloudflare-access \
  --output-file=./odyss-gcp-login.json

# 浏览器 SSO 登录（会经 Cloudflare Access → GitHub sensimintel 组织验证）：
gcloud auth login --login-config=./odyss-gcp-login.json
gcloud config set project pelagic-pod-489307-g3
```

> 服务器无桌面浏览器时，`gcloud auth login` 会打印一个 URL，在自己电脑打开完成授权即可（headless 流程）。

**持久化的边界（重要）**：凭证在有效期内，隧道**全自动、断线自愈**，无需人工。但 Workforce 联邦凭证**会过期**；过期后隧道会持续重连失败（`tunnel/tunnel.log` 里会提示「无 active 凭证」），此时需**重跑一次上面的 `gcloud auth login`**，隧道即自动恢复。

> 想要真正无人值守（不受凭证过期影响）：需改用具备 IAP 访问权限的 **service account 长期凭证**（`gcloud auth activate-service-account`）。这需要 GCP 侧配 IAM，属云资源变更、需你决策，本目录未内置。

## 一步拉起

```bash
cd grafana-gcp
cp .env.example .env      # 首次；按需改端口/密码（up.sh 没有 .env 时也会自动拷）
./up.sh
```

`up.sh` 会：① 幂等拉起隧道（nohup，日志 `tunnel/tunnel.log`）→ ② 等隧道就绪 → ③ `docker compose up -d` 起 Grafana → ④ 健康检查 → ⑤ 打印访问地址。

访问 `http://<本机内网IP>:3001`，登录用 `.env` 里的 admin 账号，打开看板「gpu-g4-01 vLLM 观测」。

停止：

```bash
./down.sh                 # 停 Grafana + 停隧道
```

## 开机自启（可选，正规化）

`up.sh` 默认用 nohup 常驻隧道，重启机器后需再跑一次 `./up.sh`。想开机自启：

- **Grafana**：`docker-compose.yml` 已设 `restart: unless-stopped`，docker 服务在则自动起。
- **隧道**：装 `tunnel/gcp-tunnel.service`（把里面 `__USER__`/`__DIR__` 换成实际值），见该文件顶部注释。

## 部署到 5090

与本仓其它部分同一「git 部署源」纪律：**本地改 → push → 5090 `git pull`**，5090 只 pull 不 commit。

```
# 本地：改完 push
# 5090：
cd ~/da3-web && git pull --ff-only && cd grafana-gcp && ./up.sh
```

> 首次在 5090 上用，需先在 5090 装好 gcloud 并完成一次 Workforce SSO 登录（见「认证与持久化」）。

## 排障

- **`up.sh` 卡在「等待隧道就绪」失败**：看 `tunnel/tunnel.log`。多为 gcloud 未登录/凭证过期 → 重跑 `gcloud auth login`。
- **Grafana 起来但看板无数据**：确认隧道进程在（`pgrep -f grafana-gcp/tunnel/gcp-tunnel.sh`）、`curl 127.0.0.1:${LOCAL_PROM_PORT}/-/ready` 返回 200；再确认 GCP 那台的 Prometheus/vLLM 在跑。
- **端口冲突**：改 `.env` 的 `GRAFANA_PORT` / `LOCAL_PROM_PORT` 后重跑 `./up.sh`。
- **看板里指标口径**：抓取间隔等设计说明见 `odyss-models/deploy/gcp-g4/`（本目录看板 JSON 即源自那里）。
