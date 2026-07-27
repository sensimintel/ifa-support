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
| 真实 VLM 服务端 | odyss-models `deploy/gcp-g4/` | main，走 OLI | 该仓 `deploy.sh`（IAP scp + 远程 setup） |
| 编排 / SOP（本仓） | ifa-support `local-stack/` | **直推 main**，不发 PR | 目标机解离线包 / git pull 后 `./stack bootstrap` |

窗口过期后分支规则回归各仓默认（main），届时更新本表。

> **窗口内 test 环境冻结属预期，不是 CI 故障。** services 与 superadmin 的 CI 都只在 push `main` 时触发（前者构建镜像，后者部署 Cloudflare Pages 的 test 站点），而窗口内两仓的 PR 一律合 `ifa` 不合 `main`，因此 `superadmin-test.odyss.life` 会停在窗口前最后一次 main（2026-07-27 的 `41fc726`，已含 App 活性判定修复）。这是有意的隔离：ifa 的产物只经 `./stack build` 发往 5090，**窗口内的验证一律在 5090 做**。窗口结束合回 main 时 test 会自动追上。

## 3. 一套拉起的形态

- **唯一入口**：`local-stack/stack`。开发机 `./stack build && ./stack bundle`；目标机 `./stack bootstrap`；日常 `./stack up|down|restart|ps|logs|backup|restore|mode`。README 之外不应存在需要背的命令。
- **开发一致性靠源指针，不靠自觉**：`stack.env` 里 `SERVICES_REPO=...` 可把构建源指向本地工作区任意 checkout——开发与部署用**同一套编排与模板**，只有源指针不同。
- **模式即配置**：`LLM_MODE=mock|real` 决定 LLM 链路（mock 全离线 / real 经 frp 隧道接真实 VLM）。切模式 = 改 `stack.env` + 重跑 bootstrap；密钥与数据不动。
- **手机 App 是接入方不是栈内组件**：App 不进 compose，靠「App 内自定义后端地址」指向栈的 18090 完成接入。

## 4. 已知契约（跨仓约束，改动需过脑）

- **meal 模型别名契约**：odyss-services 内置 workflow 的 meal 节点硬编码 `gemini-3.1-pro-preview`（runtime-config 的 `meal_model` 对字面量无效，勿改 services 的 workflow YAML——生产走网关依赖该名）。因此真实 VLM 端必须以 `SERVED_ALIASES` 同时应答该名（正源：odyss-models `deploy/gcp-g4/serve_model.sh`），`manifest.env` 的 `VLM_REQUIRED_ALIAS` 记录此契约。
- **nginx 反代必须动态解析**：容器上游一律「resolver 127.0.0.11 + 变量 proxy_pass + compose 服务名」，静态 proxy_pass 会在上游容器重建后 502（已在 `superadmin/superadmin.conf` 固化）。
- **跨机隧道做成栈内容器**：宿主防火墙（如 5090 INPUT policy DROP）不影响容器互访；不要 bind 宿主端口。

## 5. 现场对齐状态（2026-07-27）

| 部署点 | 状态 | 待办 |
|---|---|---|
| GCP gpu-g4-01（VLM） | 与 odyss-models 正源一致 | SERVED_ALIASES 别名随 odyss-models PR 落地 |
| 5090 `~/odyss-services-ifa` | **手工演化的孤本**（容器名 odyss-ifa-*、pg/minio **无命名卷**、配置手改） | 下次维护窗口用本仓 SOP 重新拉起：`./stack backup` 思路导出数据 → 按 bundle 重部署 → restore；在此之前只允许对齐 nginx conf、superadmin `dist` 等无状态文件（`superadmin/dist` 是 bind mount，换文件即生效，不需重建容器） |

## 6. 违规即失败

在部署机上改代码/配置孤本、复制仓代码、密钥入库、绕过 `./stack` 手敲 docker 命令做部署级变更——都视同任务失败，参照全局 CLAUDE.md 同名纪律。
