# local-stack：Odyss 全栈本地化 SOP（局域网闭环）

把 **odyss-services（后端 API，workers 全开）+ superadmin（设备观测控制台）+ 全部存储（postgres / minio / redis）** 完整拉起在一台服务器上，拉起之后**不依赖公网**：镜像离线导入、验证码免邮件、数据全部落本机命名卷。手机 App 与浏览器在同一局域网内即可使用全部功能（含实时分析链路）。

两条第一性原则：

1. **「联网准备一次，离线运行永久」**——所有需要公网的动作（拉镜像、编译、前端构建）都收敛在准备阶段并产出自包含离线包；目标机只消费离线包。
2. **「编排收敛、代码不搬」**——本目录是整套栈的唯一编排正源：版本锚在 `manifest.env`，本机差异与密钥在 gitignored 的 `stack.env`，一切动作走 `./stack` 唯一入口。**部署机上严禁手改配置孤本**；要改，回本仓改模板/清单，重新同步。全套管理规范见仓根 [`MANAGEMENT.md`](../MANAGEMENT.md)。

## 总览

```
阶段 A（开发机，需公网+代码仓）        阶段 B（目标机，拉起后零公网）
┌───────────────────────────┐        ┌────────────────────────────────┐
│ ./stack build             │        │ ./stack bootstrap              │
│  Go 二进制 + 前端 dist     │  tar   │  load 镜像 → 密钥/配置渲染      │
│ ./stack bundle            │ ─────▶ │  → build 镜像 → up -d          │
│  按 manifest 拉镜像打离线包 │        │  → migrate → 建 admin 账号     │
└───────────────────────────┘        │ ./stack backup|restore 数据     │
                                     └────────────────────────────────┘
```

栈内组件（详见 `docker-compose.yml`，全部命名卷，容器重建不丢数据）：

| 组件 | 端口 | 说明 |
|---|---|---|
| odyss-services | 18090 | 后端 API，**workers 默认开启**（outbox/realtime/delivery 全量） |
| superadmin (nginx) | 18091 | 设备观测控制台，`/admin-api` 同域反代到 services（动态 DNS 解析，services 重建不 502） |
| postgres | 内部 | 业务库，卷 `odyss-local-pg-data` |
| minio | 内部 | 对象存储（App 上传的图片/音频），卷 `odyss-local-minio-data` |
| valkey (redis) | 内部 | 缓存，卷 `odyss-local-redis-data` |
| llm-mock | 内部 | 栈内 mock LLM（mock 模式的全部 LLM 调用；real 模式下作为兜底常驻） |
| llm-tunnel | 内部 | 仅 real 模式（compose profile `real-vlm`）：frp stcp visitor 接远端真实 VLM |

## LLM 两种模式

模式由 `stack.env` 的 `LLM_MODE` 决定（缺省 `mock`），**切换 = 改 stack.env 后重跑 `./stack bootstrap`**（幂等，密钥与数据不动）：

- **mock（默认）**：chunk / 整餐分析全走栈内 llm-mock，完全离线。
- **real**：chunk（`chunk_inhouse`）与整餐都直连 `llm-tunnel` 背后的远端 vLLM（模型见 `manifest.env` 的 `VLM_MODEL`）。需在 `stack.env` 配齐 `VLM_API_KEY` 与 `FRP_*`（见 `stack.env.example`）。
  - VLM 服务端正源在 **odyss-models 仓 `deploy/gcp-g4/`**（勿复制，改动走该仓 PR + 其 `deploy.sh` 部署）。
  - ⚠️ **模型名对齐契约**（2026-07-31 起取代旧「别名契约」）：odyss-services（ifa）workflow YAML 的 chunk 与 meal 节点均直接写 `VLM_MODEL` 真名，须与 vLLM 的 `--served-model-name` 一致，否则分析 404；切模型时 services YAML 与 serve 侧同步改。serve 侧的 `gemini-3.1-pro-preview` 别名仅为旧二进制回滚兼容保留。

## 阶段 A：制作离线包（有公网的开发机）

前置：Go 工具链、Node、docker；本机 clone 了两个业务仓（分支以 `manifest.env` 为准：services=ifa、superadmin=ifa；路径不同时在 `stack.env` 里配 `SERVICES_REPO`/`SUPERADMIN_REPO`）。

```bash
cd local-stack
./stack build    # 构建二进制 + 前端 dist → artifacts/（分支与 manifest 不符会告警）
./stack bundle   # 按 manifest 拉齐 base 镜像，产出 odyss-local-stack-bundle-<日期>.tar
```

把产出的 bundle tar 拷到目标机（U 盘 / scp / 内网传输均可）。

## 阶段 B：目标机拉起（此后零公网）

前置：目标机已装 docker 与 docker compose 插件、openssl。

```bash
tar xf odyss-local-stack-bundle-<日期>.tar
cd local-stack
# 需要 real 模式或自定义账号时：cp stack.env.example stack.env && 编辑
./stack bootstrap
```

脚本幂等，可重复执行。完成后输出访问地址：

- **App 后端**：App 登录页切自定义后端，填 `http://<目标机IP>:18090`
- **Superadmin**：浏览器打开 `http://<目标机IP>:18091`，用脚本输出的 admin 账号登录

## 日常运维（全部离线可做）

```bash
./stack ps                 # 容器状态
./stack logs               # services 日志（跟随）
./stack mode               # 当前 LLM 模式与生效配置摘要
./stack restart            # 重启业务容器（services + superadmin）
./stack backup             # 数据备份（滚动保留 7 份；建议 cron 每日 3 点）
./stack restore backups/<时间戳>   # 恢复（破坏性，按提示确认）
```

### 自助造测试账号（无邮件通道，验证码从接口响应拿）

```bash
API=http://127.0.0.1:18090/api/v1
curl -s -X POST $API/auth/send-code -H 'Content-Type: application/json' \
  -d '{"email":"test2@odyss.dev","type":"login"}'        # 响应里的 data.code 即验证码
curl -s -X POST $API/auth/register -H 'Content-Type: application/json' \
  -d '{"email":"test2@odyss.dev","code":"<验证码>","password":"Odyss@2026"}'
```

密码需含大小写、数字、特殊字符。要让账号能登 superadmin，再执行：

```bash
docker exec odyss-local-postgres psql -U odyss -d odyss_services \
  -c "UPDATE users SET is_superuser=true WHERE email='test2@odyss.dev'"
```

## 更新业务版本

服务代码或前端有更新时，回到阶段 A 重跑 `./stack build && ./stack bundle` 得到新 bundle，目标机解包后重跑 `./stack bootstrap` 即可（配置重渲染 + 镜像重建 + 增量 migrate，数据在命名卷里不受影响）。

### 5090 演示机（孤本栈）的增量更新

5090 的 `~/odyss-services-ifa` 是标准化之前的手工孤本（见 `MANAGEMENT.md` §5），走不了上面的 bundle 流程。ifa 分支有新代码时用：

```bash
./stack deploy-5090     # 校验两仓在 ifa tip → 本机构建 → 建镜像 → migrate → 只重建 services/llm-mock → 发 dist → 探活
```

不在局域网时走 frp：`DEPLOY_TARGET=odyss-server-frpc ./stack deploy-5090`。

⚠️ 该栈的 pg/minio **没有命名卷**，数据在容器可写层里。所以更新只能用这个脚本（它对 compose 的每次调用都点名服务并带 `--no-deps`），**不要手敲 `docker compose up -d`**——按 depends_on 连带重建 postgres 就是清库。脚本每次部署前自动 `pg_dump -Fc` 留一份退路（滚动保留 3 份），dist 同样滚动备份 3 份。

## 边界与注意事项

- **安全边界**：18090/18091 仅应在局域网可达，不要做公网暴露；账号密码是唯一防线。`config/` 下渲染出的运行配置与 `stack.env` 含密钥，随备份妥善保管，均不入 git。
- **FCM 推送、邮件、地理编码在闭环内均不可用**（配置已显式关闭/留空），对应功能静默降级。
- 镜像版本 pin 在 `manifest.env`，离线环境不自动升级；升级 base 镜像改 manifest 后重做离线包。
- real 模式断网时：chunk/meal 分析报错但设备连接、上传、历史数据不受影响；可随时切回 mock 模式。
