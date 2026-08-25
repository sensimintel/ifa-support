# Odyss 本地化组件管理规范（管理仓）

本文是「frontend / services / superadmin / VLM / ifa-support 一整套本地化拉起」的管理总纲。目标只有两个：**开发与部署一致**、**一套拉起不靠记忆**。

## 1. 核心原则：编排收敛、代码不搬

- **每个组件只有一个代码正源**（见 §2 表），任何仓的代码**永不复制**到别的仓——需要别的仓的能力，用「构建产物 / 版本引用 / 交叉链接」，不用 copy。
- **ifa-support 是编排正源**：怎么构建、怎么配置、怎么拉起、版本 pin 在哪，全部以本仓 `local-stack/` 为准，且必须是**可执行的脚本与模板**，不是散文档。
- **版本锚**：`local-stack/manifest.env` 是唯一版本清单（组件仓分支 + base 镜像 tag + VLM 模型名）。换版本改 manifest，不改脚本。
- **密钥纪律**：密钥只存在于两处——目标机 gitignored 的 `stack.env` / 渲染产物，或服务器本地文件（如 GCP 的 `/etc/vllm/vllm.env`）。仓里只有 `*.example` / `*.template.*`（键名契约，值为假）。
- **部署机纪律**：部署机（5090 / GCP / L20）上**严禁手改配置孤本、严禁 git commit/push**。只做两类事：只读排查；跑本仓/正源仓的部署脚本。发现现场配置需要改 → 回正源仓改模板 → 重新同步部署。现场手改 = 漂移之源（2026-07 的 502 与 meal 404 均由此来）。

## 2. 组件正源与分支规则

| 组件 | 代码正源 | 分支规则（至 2026-08-21 窗口） | 发布方式 |
|---|---|---|---|
| 后端 API | odyss-services | PR base **ifa**，走 OLI | `./stack build` 编译进离线包 |
| 设备观测前端 | odyss-superadmin | PR base **ifa**，走 OLI | `./stack build` 出 dist 进离线包 |
| 手机 App | odyss-frontend | **直推 ifa**，不走 OLI | ios-build 装机，App 内 base_url 指向栈地址 |
| 传输内核 | react-native-odyss-capability_hub | **直推 ifa**，不走 OLI | 随 frontend 构建 |
| 真实 VLM 服务端 | 5090 本机 vLLM 裸进程（`vllm serve :8000`，Qwen3.6-35B-A3B-FP8，别名 `gemini-3.1-pro-preview`） | —（宿主进程，非仓内组件） | 本机启动配置即模型名正源；odyss-models `deploy/gcp-g4/` 单元保留但已非 IFA 链路依赖 |
| 观测平台（lumen） | odyss-lumen | main，走 OLI（不在 ifa 窗口规则内） | `local-stack/lumen/` overlay：`build-lumen-artifacts.sh` + `deploy-5090.sh` |
| 编排 / SOP（本仓） | ifa-support `local-stack/` | **直推 main**，不发 PR | 目标机解离线包 / git pull 后 `./stack bootstrap` |

窗口过期后分支规则回归各仓默认（main），届时更新本表。

> **窗口内合进 ifa 的改动不会进 test 环境，这不是 CI 故障。** services 与 superadmin 的 CI 都只在 push `main` 时触发（前者构建镜像，后者部署 Cloudflare Pages 的 test 站点），而窗口内两仓的 PR 一律合 `ifa`，因此 ifa 上的改动不会出现在 `superadmin-test.odyss.life`。这是有意的隔离：ifa 的产物只经 `./stack build` 发往 5090，**窗口内的验证一律在 5090 做**。
>
> 两点补充：
>
> 1. **`ifa` 与 `main` 分叉是预期的，不必追平。** 窗口内 superadmin 的改动以 `ifa` 为 base，但 `main` 上仍会有其他人的提交（2026-07-27 切换后两小时内就合入了 #343、#344）。**5090 只跟 `ifa`、按自己的节奏走，不要求跟上 main 的每次改动**——这正是把 5090 与主干隔离开的目的。
> 2. **`ifa` 是否从 `main` 快进由人工决定。** `./stack build` 只认 `manifest.env` 的 `SUPERADMIN_REF`，不会自动跟随 main；确实想让 5090 拿到 main 上的新功能时，才显式快进到目标点再构建、重发 dist。

## 3. 一套拉起的形态

- **唯一入口**：`local-stack/stack`。开发机 `./stack build && ./stack bundle`；目标机 `./stack bootstrap`；日常 `./stack up|down|restart|ps|logs|backup|restore|mode`。README 之外不应存在需要背的命令。
- **开发一致性靠源指针，不靠自觉**：`stack.env` 里 `SERVICES_REPO=...` 可把构建源指向本地工作区任意 checkout——开发与部署用**同一套编排与模板**，只有源指针不同。
- **模式即配置**：`LLM_MODE=mock|real` 决定 LLM 链路（mock 全离线 / real 直连本机 vLLM `172.23.0.1:8000`）。切模式 = 改 `stack.env` + 重跑 bootstrap；密钥与数据不动。
- **手机 App 是接入方不是栈内组件**：App 不进 compose，靠「App 内自定义后端地址」指向栈的 18090 完成接入。
- **健康体检**：`tools/health/check.sh`（只读探活，三态期望清单与单元修复入口见 `tools/health/README.md`），配套 Claude skill `ifa-health-check`：体检 → 按单元唯一入口拉起异常 → 复验 → 报告。

## 4. 已知契约（跨仓约束，改动需过脑）

- **meal 模型名对齐契约**（2026-07-31 起取代旧「别名契约」）：odyss-services（ifa 分支）workflow YAML 的 meal 节点直接写 `VLM_MODEL` 真名（如 `Qwen3.6-35B-A3B-FP8`），须与真实 VLM 端 `--served-model-name` 一致（正源：**5090 本机 vLLM 启动配置**；旧正源 odyss-models `deploy/gcp-g4/serve_model.sh` 对应的 gcp-g4 单元保留但已非 IFA 链路依赖），切模型时 services YAML 与 serve 侧同步改（注意 services YAML 变更会使 checkpoint 全量失效）。serve 侧的 `gemini-3.1-pro-preview` 别名仅为旧二进制回滚兼容保留。
- **nginx 反代必须动态解析**：容器上游一律「resolver 127.0.0.11 + 变量 proxy_pass + compose 服务名」，静态 proxy_pass 会在上游容器重建后 502（已在 `superadmin/superadmin.conf` 固化）。
- **跨机隧道做成栈内容器**：宿主防火墙（如 5090 INPUT policy DROP）不影响容器互访；不要 bind 宿主端口。

## 5. 现场对齐状态（2026-08-21）

> 文中「5090」沿用主机称谓；2026-08-21 换卡 **RTX Pro 6000 Blackwell 96GB**，2026-08-24 起为**双卡**：GPU0 = RTX Pro 6000 96GB（独占给 vLLM），GPU1 = RTX 5090 32GB（SAM3 独享；DA3 已于 2026-08-25 退役）。三个 unit（vllm / sam3 / da3-web）都在 systemd 里钉卡，重启后不会落错卡——sam3/da3-web 用 GPU UUID；vllm 用 `CUDA_DEVICE_ORDER=PCI_BUS_ID` + `CUDA_VISIBLE_DEVICES=0`（**vLLM 把该变量按整数解析、不认 UUID**，UUID 会 ValidationError 崩溃循环，2026-08-24 实测）；不做全文改名。

| 部署点 | 状态 | 待办 |
|---|---|---|
| 5090 本机 VLM（RTX Pro 6000 96GB 独占） | **VLM 已本机化承接**：宿主裸进程 `vllm serve :8000`（Qwen3.6-35B-A3B-FP8，别名 `gemini-3.1-pro-preview`）；services 的 `chunk_inhouse.base_url` 已切 `http://172.23.0.1:8000/v1`（2026-08-21 服务器手工生效，repo 配置另行同步）。2026-08-24 双卡布局后 vLLM **独占 GPU0=RTX Pro 6000**（unit 里 `CUDA_DEVICE_ORDER=PCI_BUS_ID`+`CUDA_VISIBLE_DEVICES=0` 钉卡——vLLM 不认 UUID 形式；`gpu-memory-utilization 0.90` 约 86G）；SAM3 挪到 GPU1=RTX 5090 32GB（DA3 已于 2026-08-25 整体退役，权重/源码从服务器删除）。vllm.service 正源即 `/etc/systemd/system/vllm.service`（服务器本地运维配置，api-key 在 `/etc/vllm/vllm.env`，均不入库） | repo 侧配置同步落库后复核一致 |
| GCP gpu-g4-01 | **已非 IFA 链路依赖**：llm-tunnel(:28000 frp) 与 8011 反向 SSH keeper 隧道均废弃；odyss-models `deploy/gcp-g4/` 单元保留但不再是本链路的 VLM 正源 | — |
| LocateAnything（原 5090 生产链） | **已从 IFA 链路完全剔除**（LA vLLM / gateway / LB / SigLIP 全下线）；名为 locateanything-prometheus / gpu-exporter / node-exporter 的容器保留（历史名，承担全机观测） | — |
| 观测入口 | 统一 Grafana `:3001` + 本机 Prometheus `:9091` 为唯一观测入口；g4 联邦数据源 / 29090 隧道废弃 | — |
| 5090 `~/odyss-services-ifa` | **手工演化的孤本**（容器名 odyss-ifa-*、pg/minio **无命名卷**、配置手改）；业务版本更新已收敛到 `./stack deploy-5090`（2026-08-18 起）。**LLM 形态与标准栈不同**：`infra.llm` 指向远端 gemini 网关（macaron），`chunk_inhouse` 指向本机 vLLM（`172.23.0.1:8000`），于是 chunk 判定能在 inhouse 故障时按 `fallback_model: gemini-3.1-pro-preview` 真正回退到另一个 provider——标准栈两者配置同源，回退无隔离价值，故模板刻意不设 fallback（2026-08-18 实时链切 Qwen 时确认）。**孤本 compose 的 services/llm-mock 块漏写 restart 策略**（默认 no，宿主机重启后不自愈；2026-08-21 秤读数 18090 拒连即此因）：孤本禁止手改，由 `deploy-5090.sh` 在重建后 `docker update --restart unless-stopped` 兜底，重拉标准栈（模板已写 unless-stopped）后自然消除 | 下次维护窗口用本仓 SOP 重新拉起：`./stack backup` 思路导出数据 → 按 bundle 重部署 → restore。在此之前，业务代码更新一律走 `./stack deploy-5090`（构建 → 建镜像 → migrate → **只**重建 services/llm-mock → 发 dist），**严禁手敲 `docker compose up -d`**：pg/minio 无命名卷，按 depends_on 连带重建等于清空演示库 |
| 5090 `~/odyss-ifa-lumen`（lumen overlay） | 由本仓 `local-stack/lumen/` 管理（独立 compose 项目挂 `odyss-ifa-network`，collector 以别名 `lumen-collector.odyss.internal:80` 接住 services 默认导出；lumen 数据在命名卷 `odyss-ifa-lumen-pg-data`） | 更新一律 `deploy-5090.sh` 重放；业务栈重拉为标准 local-stack 后把 `LUMEN_ATTACH_NETWORK` 换成 `odyss-local-network` 重挂 |

## 6. 违规即失败

在部署机上改代码/配置孤本、复制仓代码、密钥入库、绕过 `./stack` 手敲 docker 命令做部署级变更——都视同任务失败，参照全局 CLAUDE.md 同名纪律。
