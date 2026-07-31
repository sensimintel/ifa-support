# lumen overlay：把观测平台挂进本地栈

把 **odyss-lumen（trace collector + observation 后端 + 自有 Postgres）**作为独立 compose 项目
挂进既有业务栈网络，让栈内 odyss-services 的 span 有处可去、superadmin 的 Lumen（数据资产）
板块在本地栈可用——与 test 环境能力对齐。

## 设计要点（为什么业务栈 compose 零改动）

- odyss-services 的 probe **默认**把 span 导出到 `http://lumen-collector.odyss.internal/v1/spans:batch`
  （AWS VPC 内网域名，端口 80）。overlay 给 collector 容器挂**网络别名**
  `lumen-collector.odyss.internal` 并监听 `:80`——Docker 内置 DNS 逐查询解析，
  **正在运行的 services 容器无需重启**，overlay 一起来 span 即开始入库。
- observation 以别名 `lumen-observation:8090` 供 superadmin nginx 反代 `/lumen/`
  （见 `../superadmin/superadmin.conf`，前缀剥除规则与 superadmin 仓 vite dev 代理一致）。
- lumen 数据独立成库（overlay 自带 postgres + 命名卷 `odyss-ifa-lumen-pg-data`），
  不与业务库互相影响；迁移由 `lumen-migrate` 一次性容器按 `db/migrations` 顺序应用、记账幂等。
- 挂接网络用 `LUMEN_ATTACH_NETWORK` 覆盖：5090 孤本栈 = `odyss-ifa-network`（默认）；
  标准 local-stack = `odyss-local-network`。

## 用法

```bash
# 开发机（需公网 + 本机 clone 了 odyss-lumen，分支以 manifest.env 的 LUMEN_REF 为准）
./scripts/build-lumen-artifacts.sh   # 出二进制 + 迁移 SQL 到 artifacts/
./scripts/deploy-5090.sh             # rsync 到 5090 → 构建镜像 → compose up → 同步 superadmin 无状态文件
```

目标机手动操作（等价于 deploy 脚本第 2 步，排障用）：

```bash
cd ~/odyss-ifa-lumen
docker build -t odyss-lumen:ifa-stack -f lumen.Dockerfile .
docker compose up -d
docker logs odyss-ifa-lumen-observation --tail 20
```

## 已知降级（预期内，不是故障）

- **session-frames 帧轨道**（`/api/media/session-frames`）：s3blob 走 AWS 默认凭证链、
  不支持 minio 端点，本地栈无 AWS 凭证 → 该路由降级不挂（`/api/ready` 列 degraded，
  `/api/health` 仍 ok）。设备面板的帧图走 services admin API，不受影响。
- **DataHub / eval exec 控制面**：本地栈不部署 DataHub 与 exec 基座，对应控制面按设计降级；
  superadmin 的 DataHub 页在本地栈不可用（登录页会显示 DataHub 断）。
- **observation 自身的 root span**：上报到本 overlay 的 collector（observation.yaml 的
  `collector_url`），与业务 span 同库。

## 验证清单

1. `docker logs odyss-ifa-services | grep "tracing 导出异常"`——overlay 起来后不再新增。
2. `curl -s http://127.0.0.1:18091/lumen/api/health` → `{"status":"ok"}`（经 superadmin 反代）。
3. 设备上传中时：`curl -s "http://127.0.0.1:18091/lumen/api/records/chunks?scope=session:<session_id>&limit=3"`
   能看到 analysis_chunk records。
4. superadmin 页面出现「数据资产 → Lumen」板块，设备状态页「选中 chunk 投票」的
   「LLM 原始请求 / 响应」折叠区可展开出数据（新 chunk 才有请求留痕）。
