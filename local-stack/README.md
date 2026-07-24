# local-stack：Odyss 全栈本地化 SOP（局域网闭环）

把 **odyss-services（后端 API）+ superadmin（设备观测控制台）+ 全部存储（postgres / minio / redis）** 完整拉起在一台服务器上，拉起之后**完全不依赖公网**：镜像离线导入、LLM 走本地 mock、验证码免邮件、数据全部落本机命名卷。手机 App 与浏览器在同一局域网内即可使用全部功能。

第一性原则：**「联网准备一次，离线运行永久」**。所有需要公网的动作（拉镜像、编译、前端构建）都收敛在准备阶段并产出一个自包含离线包；目标机只消费离线包，之后断网、换网、重启都不影响运行。

## 总览

```
阶段 A（开发机，需公网+代码仓）      阶段 B（目标机，拉起后零公网）
┌─────────────────────────┐        ┌──────────────────────────────┐
│ build-artifacts.sh      │        │ bootstrap.sh                 │
│  Go 二进制 + 前端 dist   │  tar   │  load 镜像 → 生成密钥/配置    │
│ make-offline-bundle.sh  │ ─────▶ │  → build 镜像 → up -d        │
│  拉 base 镜像 docker save│        │  → migrate → 建 admin 账号   │
└─────────────────────────┘        │ backup.sh / restore.sh 数据   │
                                   └──────────────────────────────┘
```

栈内组件（详见 `docker-compose.yml`，全部命名卷，容器重建不丢数据）：

| 组件 | 端口 | 说明 |
|---|---|---|
| odyss-services | 18090 | 后端 API（`-workers=false` 只跑 HTTP） |
| superadmin (nginx) | 18091 | 设备观测控制台，`/admin-api` 同域反代到 services |
| postgres 18.4 | 内部 | 业务库，卷 `odyss-local-pg-data` |
| minio | 内部 | 对象存储（App 上传的图片/音频），卷 `odyss-local-minio-data` |
| valkey (redis) | 内部 | 缓存，卷 `odyss-local-redis-data` |
| llm-mock | 内部 | 本地 mock LLM，闭环内所有 LLM 调用的兜底 |

## 阶段 A：制作离线包（有公网的开发机）

前置：Go 工具链、Node、docker；本机 clone 了 `odyss-services`（**ifa 分支**）与 `odyss-superadmin`（main 分支）。

```bash
cd local-stack
# 1. 构建业务产物（二进制 + 前端 dist）→ artifacts/
#    代码仓不在默认相对路径时用环境变量指定：
#    SERVICES_REPO=/path/to/odyss-services SUPERADMIN_REPO=/path/to/odyss-superadmin ./scripts/build-artifacts.sh
./scripts/build-artifacts.sh

# 2. 拉齐 base 镜像并打离线包（产出 odyss-local-stack-bundle-<日期>.tar）
./scripts/make-offline-bundle.sh
```

把产出的 bundle tar 拷到目标机（U 盘 / scp / 内网传输均可）。

## 阶段 B：目标机拉起（此后零公网）

前置：目标机已装 docker 与 docker compose 插件、openssl（常规服务器装机自带）。

```bash
tar xf odyss-local-stack-bundle-<日期>.tar
cd local-stack
# 可用环境变量覆盖默认 admin 账号：ADMIN_EMAIL=xx ADMIN_PASSWORD='Xx@12345' ./scripts/bootstrap.sh
./scripts/bootstrap.sh
```

脚本幂等，可重复执行。完成后输出访问地址：

- **App 后端**：App 登录页切自定义后端，填 `http://<目标机IP>:18090`
- **Superadmin**：浏览器打开 `http://<目标机IP>:18091`，用脚本输出的 admin 账号登录（账号密码登录，不走零信任）

## 日常运维（全部离线可做）

```bash
# 数据备份（postgres dump + minio 卷 + 运行配置；在线执行，滚动保留 7 份）
./scripts/backup.sh
# 建议 cron 每日 3 点：0 3 * * * /path/to/local-stack/scripts/backup.sh >> /path/to/local-stack/backups/backup.log 2>&1

# 数据恢复（破坏性覆盖，按提示确认）
./scripts/restore.sh backups/<时间戳>

# 常用排查
docker compose ps
docker logs odyss-local-services --since 30m
docker exec odyss-local-postgres psql -U odyss -d odyss_services -c '\dt'
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

服务代码或前端有更新时，回到阶段 A 重跑两个脚本得到新 bundle，目标机解包后：

```bash
docker build -t odyss-services:local -f services.Dockerfile . && docker compose up -d
```

数据在命名卷里，更新镜像不影响；migrate 容器会在 up 时自动执行增量迁移。

## 边界与注意事项

- **实时分析链路为空属预期**：services 以 `-workers=false` 运行，outbox / realtime / delivery worker 不启动，「实时进餐状态」等 worker 加工数据不会产生；设备连接、上传、绑定、历史数据全部正常。若需完整实时链路，需单独评估开 workers（chunk 分析可由栈内 llm-mock 承接，其余依赖逐项确认）。
- **安全边界**：18090/18091 仅应在局域网可达，不要做公网暴露；账号密码是唯一防线。`config/runtime-config.yaml` 含私钥与签发密钥，随备份一起妥善保管，不入 git。
- **FCM 推送、邮件、地理编码在闭环内均不可用**（配置已显式关闭/留空），对应功能静默降级。
- 镜像版本已 pin 死（postgres 18.4 / valkey 9.1.0 等），离线环境不自动升级；升级 base 镜像需重做离线包。
