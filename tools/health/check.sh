#!/usr/bin/env bash
# =============================================================================
# ifa-health-check 探活脚本（只读，在 5090 上执行）。
# 用法（开发机）：ssh odyss-server-frpc 'bash -s' < tools/health/check.sh
#（整机无外网期间 frp 入口不可用，改用局域网别名：ssh odyss-server-local 'bash -s' < ...）
# 输出：每行「状态|组件|详情」，状态 ∈ OK / FAIL / WARN / INFO；存在 FAIL 时退出码 1。
# 铁律：本脚本只探测、不修复；修复按 tools/health/README.md 的「单元唯一入口」执行。
# 两态期望清单（应运行 / 一次性完成）以 README.md 为准，改期望先改仓。
# 设计出发点：健康 = 能力可用（业务闭环 / 演示 / 观测 / 底座），不是容器在跑，
# 因此探测尽量打到业务语义层（HTTP 语义码、本机 vLLM /v1/models），而非只看 docker ps。
# =============================================================================
set -uo pipefail

FAIL=0
say() { printf '%s|%s|%s\n' "$1" "$2" "$3"; [ "$1" = FAIL ] && FAIL=1; return 0; }

# 容器状态：status|exitcode|health（无健康检查时第三段为空）
ctr() { docker inspect -f '{{.State.Status}}|{{.State.ExitCode}}|{{if .State.Health}}{{.State.Health.Status}}{{end}}' "$1" 2>/dev/null; }

need_running() { # $1=容器名 $2=组件标签
  local s st ec hl
  s=$(ctr "$1") || { say FAIL "$2" "容器不存在"; return 0; }
  IFS='|' read -r st ec hl <<<"$s"
  if [ "$st" = running ] && { [ -z "$hl" ] || [ "$hl" = healthy ]; }; then
    say OK "$2" "running${hl:+ ($hl)}"
  elif [ "$st" = running ] && [ "$hl" = starting ]; then
    say WARN "$2" "running（健康检查 starting 中，稍后复验）"
  else
    say FAIL "$2" "$st${hl:+ ($hl)} exit=$ec"
  fi
}

need_oneshot_ok() { # 一次性任务：Exited(0) 才是健康
  local s st ec hl
  s=$(ctr "$1") || { say FAIL "$2" "容器不存在"; return 0; }
  IFS='|' read -r st ec hl <<<"$s"
  if [ "$st" = exited ] && [ "$ec" = 0 ]; then say OK "$2" "一次性任务已完成 Exited(0)"
  elif [ "$st" = running ]; then say WARN "$2" "一次性任务仍在运行"
  else say FAIL "$2" "$st exit=$ec"; fi
}

need_systemd() { # $1=unit $2=组件标签
  if systemctl is-active --quiet "$1"; then say OK "$2" "systemd active"
  else say FAIL "$2" "systemd $(systemctl is-active "$1" 2>/dev/null || echo unknown)"; fi
}

http_code() { curl -sS -o /dev/null -m "${3:-8}" ${2:+-X "$2"} -w '%{http_code}' "$1" 2>/dev/null || echo 000; }

need_http() { # $1=组件标签 $2=URL $3=期望码正则
  local c; c=$(http_code "$2")
  if [[ "$c" =~ $3 ]]; then say OK "$1" "HTTP $c $2"
  else say FAIL "$1" "HTTP $c $2（期望 $3）"; fi
}

# 漂移判据：监听 PID 必须 == systemd MainPID（允许差一级父子，防 fork 型启动）。
# 覆盖两种假绿：systemd inactive 但游离进程占端口；systemd active 但孤儿旧进程仍在应答
#（deploy.sh 假绿事故，2026-08 FAQ：健康检查探到的是旧进程的 200）。
need_pid_match() { # $1=unit $2=端口 $3=组件标签
  local main lp pp
  main=$(systemctl show -p MainPID --value "$1" 2>/dev/null)
  lp=$(ss -ltnp 2>/dev/null | awk -v p=":$2\$" '$4 ~ p {print; exit}' | grep -o 'pid=[0-9]*' | head -1 | tr -dc 0-9)
  if [ -z "$lp" ]; then
    say FAIL "$3" "端口 $2 未监听"
  elif [ -z "$main" ] || [ "$main" = 0 ]; then
    say FAIL "$3" "端口 $2 被 PID $lp 占用但 systemd 无主进程——游离进程接管（漂移）"
  elif [ "$lp" = "$main" ] || [ "$(ps -o ppid= -p "$lp" 2>/dev/null | tr -dc 0-9)" = "$main" ]; then
    say OK "$3" "监听 PID $lp 与 systemd MainPID 一致"
  else
    say FAIL "$3" "监听 PID $lp ≠ MainPID $main——孤儿进程仍在应答（漂移，deploy.sh 假绿形态）"
  fi
}

# ---- 双卡落位：GPU0=RTX PRO 6000 96G 独占 vLLM；GPU1=RTX 5090 32G 归 SAM3（DA3 已于 2026-08-25 退役） ----
GPU_PRO6000="GPU-5938e8ce-154b-31e1-f5c6-2aa6cbe73fc6"   # RTX PRO 6000 Blackwell 96G
GPU_RTX5090="GPU-565f6e8d-550f-38cf-56d1-fe49fb280b05"   # RTX 5090 32G
# unit 的 GPU 计算进程必须落在期望卡上（拿 nvidia-smi compute-apps 的 pid 反查所属 systemd unit）。
# $4=lazy 表示该 unit 的模型是懒加载：暂无 GPU 进程不算异常
need_gpu_place() { # $1=unit $2=期望UUID $3=组件标签 $4=可选lazy
  local found=0 wrong="" pid uuid u
  while IFS=, read -r pid uuid; do
    pid=$(tr -dc 0-9 <<<"$pid"); uuid=$(tr -d ' ' <<<"$uuid")
    [ -n "$pid" ] || continue
    u=$(ps -o unit= -p "$pid" 2>/dev/null | tr -d ' ')
    [ "$u" = "$1.service" ] || continue
    if [ "$uuid" = "$2" ]; then found=1; else wrong="$uuid"; fi
  done < <(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>/dev/null)
  if [ -n "$wrong" ]; then say FAIL "$3" "GPU 进程落在 $wrong，期望 $2——查该 unit 的 CUDA_VISIBLE_DEVICES 后 daemon-reload + restart"
  elif [ "$found" = 1 ]; then say OK "$3" "已落在期望卡（$2）"
  elif [ "${4:-}" = lazy ]; then say INFO "$3" "该 unit 暂无 GPU 计算进程（懒加载未触发，属正常）"
  else say FAIL "$3" "该 unit 无 GPU 计算进程——模型没真正加载上卡？"; fi
}

# ---------------- D. 底座 ----------------
if docker info >/dev/null 2>&1; then say OK "底座/docker" "daemon 可用"
else say FAIL "底座/docker" "daemon 不可用——后续容器检查全部失效"; fi
nvidia-smi >/dev/null 2>&1 && say OK "底座/GPU驱动" "nvidia-smi 正常" || say FAIL "底座/GPU驱动" "nvidia-smi 失败"
# frpc FAIL 时先判断是否整机断外网（如 curl -m 5 一个外网地址）：断网属网络侧待办
# （VLAN40 transit 未通），勿反复重启 frpc；处置细节见 README 修复入口表。
need_systemd frpc "底座/frp公网入口"
disk=$(df --output=pcent / 2>/dev/null | tail -1 | tr -dc 0-9)
if [ -n "$disk" ] && [ "$disk" -ge 90 ]; then say WARN "底座/磁盘" "根分区已用 ${disk}%"
else say OK "底座/磁盘" "根分区已用 ${disk:-?}%"; fi
say INFO "底座/显存" "$(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null | paste -sd '；' - || echo 未知)"
say INFO "底座/负载" "$(uptime 2>/dev/null | sed 's/.*load average/load average/' || echo 未知)"

# ---------------- A. 业务闭环（local-stack 孤本 ~/odyss-services-ifa） ----------------
need_running odyss-ifa-postgres "业务/postgres"
need_running odyss-ifa-redis "业务/redis"
need_running odyss-ifa-minio "业务/minio"
need_oneshot_ok odyss-ifa-migrate "业务/migrate"
need_oneshot_ok odyss-ifa-minio-init "业务/minio-init"
need_running odyss-ifa-llm-mock "业务/llm-mock"
need_running odyss-ifa-llm-switch "业务/llm-switch"
need_running odyss-ifa-services "业务/services"
# 语义探活：空 JSON 打验证码接口应返回 400（说明 API 路由与校验层活着，而非仅端口在听）
code=$(curl -sS -o /dev/null -m 8 -X POST -H 'Content-Type: application/json' -d '{}' \
  -w '%{http_code}' "http://127.0.0.1:18090/api/v1/auth/send-code" 2>/dev/null || echo 000)
if [ "$code" = 400 ]; then say OK "业务/API语义" "18090 空参返回 400（校验层应答）"
else say FAIL "业务/API语义" "18090 返回 $code（期望 400）"; fi
# 登录关键依赖：加密公钥必须 200（503=孤本未配 encryption key，App 登录请求根本发不出——真实故障史）
need_http "业务/登录密钥" "http://127.0.0.1:18090/api/v1/auth/password-encryption-key" '^200$'
need_http "业务/superadmin页面" "http://127.0.0.1:18091/" '^200$'
code=$(http_code "http://127.0.0.1:18091/admin-api/ping")
case "$code" in
  000|502|504) say FAIL "业务/反代" "admin-api 返回 $code（nginx→services 断了）" ;;
  *) say OK "业务/反代" "admin-api 返回 $code（后端应答，反代通）" ;;
esac
# llm-switch 控制面：superadmin 顶栏模型切换走 18091 /ops-api/ 反代（容器 running ≠ 反代通）
code=$(http_code "http://127.0.0.1:18091/ops-api/")
case "$code" in
  000|502|504) say FAIL "业务/llm-switch控制面" "ops-api 返回 $code（nginx→llm-switch 断了）" ;;
  *) say OK "业务/llm-switch控制面" "ops-api 返回 $code（后端应答，反代通）" ;;
esac
# lumen 观测栈：独立 compose 项目（/home/odyss/odyss-ifa-lumen），支撑 superadmin 的 Lumen 数据板块；
# 不被 ~/odyss-services-ifa 的 up -d 覆盖，须单独探测
need_running odyss-ifa-lumen-postgres "业务/lumen-postgres"
need_running odyss-ifa-lumen-collector "业务/lumen-collector"
need_running odyss-ifa-lumen-observation "业务/lumen-observation"
need_oneshot_ok odyss-ifa-lumen-migrate "业务/lumen-migrate"
need_http "业务/lumen接口" "http://127.0.0.1:18091/lumen/api/health" '^200$'
# 本机 VLM：宿主裸进程 vllm serve（/home/odyss/vllm-env，0.0.0.0:8000，随开机启动）。
# 带 api-key：未带 key 探 /v1/models 可能 401，200/401 都算活。
need_http "业务/本机vLLM" "http://127.0.0.1:8000/v1/models" '^(200|401)$'
need_gpu_place vllm "$GPU_PRO6000" "业务/vLLM落卡"
# VLM 深链：从栈内网络经 docker 网关 172.23.0.1 直探宿主 vLLM（HTTP 4xx 也算活：仅说明未带 key）
NET=$(docker inspect odyss-ifa-services -f '{{range $k,$v := .NetworkSettings.Networks}}{{$k}}{{end}}' 2>/dev/null || true)
if [ -n "${NET:-}" ]; then
  out=$(docker run --rm --network "$NET" alpine:3.20 sh -c 'wget -qO- -T 10 http://172.23.0.1:8000/v1/models 2>&1' || true)
  if echo "$out" | grep -qE '"id"|"object"|HTTP/[0-9.]+ 4'; then
    say OK "业务/VLM链路" "栈内经 172.23.0.1:8000 探到宿主 vLLM 应答"
  else
    say FAIL "业务/VLM链路" "172.23.0.1:8000 栈内不可达（${out:0:80}）——先看「业务/本机vLLM」：本机也挂查宿主 vllm 进程；本机 OK 则查 docker 网络/防火墙"
  fi
else
  say WARN "业务/VLM链路" "services 容器不在，跳过深链探测"
fi

# ---------------- B. 演示 ----------------
need_systemd da3-web "演示/da3-web"
# 根路径现为 307 → /experience，200 与 30x 都算活
need_http "演示/da3-web页面" "http://127.0.0.1:8060/" '^(200|30[0-9])$'
need_pid_match da3-web 8060 "演示/da3-web漂移"
need_systemd sam3 "演示/SAM3"
ss -tln 2>/dev/null | grep -q ':8013 ' && say OK "演示/SAM3端口" "8013 在监听" || say FAIL "演示/SAM3端口" "8013 未监听"
need_pid_match sam3 8013 "演示/SAM3漂移"
need_gpu_place sam3 "$GPU_RTX5090" "演示/SAM3落卡"
# mac-mini 帧链路：cam-pusher（192.168.100.3，LaunchDaemon）持续推帧到 8060，帧桶 60s 过期。
# 判据：/api/frame/status 存在 macmini-* 设备且帧龄 ≤30s（演示链路第一环，断推 = 浅体验区无画面）
fs=$(curl -sS -m 8 http://127.0.0.1:8060/api/frame/status 2>/dev/null)
verdict=$(printf '%s' "$fs" | python3 -c '
import json,sys
try:
    d = json.load(sys.stdin)
except Exception:
    print("PARSE"); raise SystemExit
devs = [x for x in d.get("devices", []) if str(x.get("device_id", "")).startswith("macmini")]
if not devs:
    print("NONE")
else:
    a = min(float(x.get("age", 1e9)) for x in devs)
    print(("OK|%.1f" % a) if a <= 30 else ("STALE|%.1f" % a))' 2>/dev/null)
case "$verdict" in
  OK\|*)    say OK "演示/mac-mini帧链路" "macmini 设备帧龄 ${verdict#OK|}s" ;;
  STALE\|*) say FAIL "演示/mac-mini帧链路" "macmini 帧龄 ${verdict#STALE|}s（>30s：cam-pusher 断推/断网？自带 KeepAlive 自愈，仍断走 mini deploy.sh）" ;;
  NONE)     say FAIL "演示/mac-mini帧链路" "8060 无 macmini-* 设备帧（mac-mini 断推/断网/未上电？）" ;;
  *)        say FAIL "演示/mac-mini帧链路" "8060 /api/frame/status 不可用或返回异常" ;;
esac
# 深体验区 dx-backend（宿主 :8070，systemd）：四通道秤读数 + 手机/项链/秤分组绑定
need_systemd dx-backend "演示/dx-backend"
need_http "演示/dx-backend接口" "http://127.0.0.1:8070/api/health" '^200$'
need_http "演示/dx-backend反代" "http://127.0.0.1:18091/dx-api/api/health" '^200$'
# 秤链路：一律经 dx-backend 持久连接看数据新鲜度（age_s），勿裸探 Modbus——
# 新 TCP 连接首个请求约 6.4s 才应答，短超时会误判"固件卡死"。秤断电/断网 → FAIL（演示期口径）。
sc=$(curl -sS -m 15 http://127.0.0.1:8070/api/food-scales 2>/dev/null)
verdict=$(printf '%s' "$sc" | python3 -c '
import json,sys
try:
    d = json.load(sys.stdin)
except Exception:
    print("PARSE"); raise SystemExit
ages = [s.get("age_s") for s in d.get("scales", []) if s.get("age_s") is not None]
if not ages:
    print("NONE")
else:
    a = max(ages)
    print(("OK|%.1f" % a) if a <= 30 else ("STALE|%.1f" % a))' 2>/dev/null)
case "$verdict" in
  OK\|*)    say OK "演示/秤链路" "四通道读数新鲜（最大 age_s=${verdict#OK|}）" ;;
  STALE\|*) say FAIL "演示/秤链路" "读数过期 age_s=${verdict#STALE|}（>30s：查秤上电/网线/IP=192.168.100.80，dx-backend 会自动重连）" ;;
  NONE)     say FAIL "演示/秤链路" "food-scales 无读数（dx-backend 未连上秤 192.168.100.80:502？）" ;;
  *)        say FAIL "演示/秤链路" "8070 /api/food-scales 不可用或返回异常" ;;
esac

# ---------------- C. 观测 ----------------
need_running ifa-grafana-gcp "观测/统一Grafana"
need_http "观测/统一Grafana接口" "http://127.0.0.1:3001/api/health" '^200$'
# 下列容器名仍带 locateanything- 前缀（历史名，职能已是全机观测）：保留容器名，标签用中性名
need_running locateanything-prometheus "观测/本机Prometheus"
need_http "观测/本机Prometheus接口" "http://127.0.0.1:9091/-/ready" '^200$'
need_running locateanything-node-exporter "观测/node-exporter"
need_running locateanything-gpu-exporter "观测/gpu-exporter"
# 语义探活：exporter 进程活着 ≠ 有 GPU 指标。容器内 NVML 会被宿主 systemctl daemon-reload
# 吊销（现象：nvidia-smi "Failed to initialize NVML"，看板 GPU 面板 No data，而 Prometheus
# target 仍显示 up——"假 up"，2026-07-31~08-07 因此静默断采一周），必须查指标本体与失败计数。
gpum=$(curl -sS -m 8 http://127.0.0.1:9835/metrics 2>/dev/null)
gpufs=$(printf '%s' "$gpum" | awk '/^nvidia_smi_failed_scrapes_total/{print int($2)}')
if printf '%s' "$gpum" | grep -q nvidia_smi_memory_used_bytes; then
  if [ "${gpufs:-0}" -gt 0 ]; then
    say WARN "观测/GPU指标" "9835 有指标但容器本次生命周期内已累计 ${gpufs} 次抓取失败——NVML 间歇失效前兆，留意复发"
  else
    say OK "观测/GPU指标" "9835 有 nvidia_smi 指标（failed_scrapes=0）"
  fi
else
  say FAIL "观测/GPU指标" "9835 无 GPU 指标（failed_scrapes=${gpufs:-?}）——NVML 被吊销；docker restart 可能无效（2026-08-07 实测），用 cd ~/odyss-models/deploy/gpu5090 && docker compose -f compose.gpu.yml up -d --force-recreate gpu-exporter"
fi
# 自愈 cron 知会：odyss-models 的 gpu_exporter_selfheal 每 5 分钟探 9835，连续缺失自动 force-recreate。
# 人工处置 GPU 指标前先看这里——cron 已装时可能已自愈/即将自愈，避免重复动作
if crontab -l 2>/dev/null | grep -q gpu_exporter_selfheal; then
  say INFO "观测/GPU指标自愈cron" "已安装（人工 force-recreate 前先看 selfheal 日志，避免与 cron 重复处置）"
else
  say INFO "观测/GPU指标自愈cron" "未安装（NVML 吊销需人工 force-recreate）"
fi

# ---------------- 汇总 ----------------
echo "===="
if [ "$FAIL" = 1 ]; then
  echo "RESULT|FAIL|存在异常项：按 tools/health/README.md 的单元入口拉起后必须复验"
  exit 1
else
  echo "RESULT|OK|全部健康（WARN/INFO 项见上，仅需知会）"
fi
