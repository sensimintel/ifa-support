#!/usr/bin/env bash
# =============================================================================
# ifa-health-check 探活脚本（只读，在 5090 上执行）。
# 用法（开发机）：ssh odyss-server-frpc 'bash -s' < tools/health/check.sh
# 输出：每行「状态|组件|详情」，状态 ∈ OK / FAIL / WARN / INFO；存在 FAIL 时退出码 1。
# 铁律：本脚本只探测、不修复；修复按 tools/health/README.md 的「单元唯一入口」执行。
# 三态期望清单（应运行 / 预期停止 / 一次性完成）以 README.md 为准，改期望先改仓。
# 设计出发点：健康 = 能力可用（业务闭环 / 演示 / 观测 / 底座），不是容器在跑，
# 因此探测尽量打到业务语义层（HTTP 语义码、隧道后 /v1/models），而非只看 docker ps。
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

expect_stopped() { # 预期停止（腾显存）：在跑反而是异常
  local s st ec hl
  s=$(ctr "$1") || { say OK "$2" "容器不存在（等价停止）"; return 0; }
  IFS='|' read -r st ec hl <<<"$s"
  if [ "$st" = running ]; then say WARN "$2" "预期停止却在运行——显存风险，报告用户，勿当健康"
  else say OK "$2" "按预期停止（$st）"; fi
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

# ---------------- D. 底座 ----------------
if docker info >/dev/null 2>&1; then say OK "底座/docker" "daemon 可用"
else say FAIL "底座/docker" "daemon 不可用——后续容器检查全部失效"; fi
nvidia-smi >/dev/null 2>&1 && say OK "底座/GPU驱动" "nvidia-smi 正常" || say FAIL "底座/GPU驱动" "nvidia-smi 失败"
need_systemd frpc "底座/frp公网入口"
disk=$(df --output=pcent / 2>/dev/null | tail -1 | tr -dc 0-9)
if [ -n "$disk" ] && [ "$disk" -ge 90 ]; then say WARN "底座/磁盘" "根分区已用 ${disk}%"
else say OK "底座/磁盘" "根分区已用 ${disk:-?}%"; fi
say INFO "底座/显存" "$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo 未知)"
say INFO "底座/负载" "$(uptime 2>/dev/null | sed 's/.*load average/load average/' || echo 未知)"

# ---------------- A. 业务闭环（local-stack 孤本 ~/odyss-services-ifa） ----------------
need_running odyss-ifa-postgres "业务/postgres"
need_running odyss-ifa-redis "业务/redis"
need_running odyss-ifa-minio "业务/minio"
need_oneshot_ok odyss-ifa-migrate "业务/migrate"
need_oneshot_ok odyss-ifa-minio-init "业务/minio-init"
need_running odyss-ifa-llm-mock "业务/llm-mock"
need_running odyss-ifa-llm-tunnel "业务/llm-tunnel"
need_running odyss-ifa-services "业务/services"
# 语义探活：空 JSON 打验证码接口应返回 400（说明 API 路由与校验层活着，而非仅端口在听）
code=$(curl -sS -o /dev/null -m 8 -X POST -H 'Content-Type: application/json' -d '{}' \
  -w '%{http_code}' "http://127.0.0.1:18090/api/v1/auth/send-code" 2>/dev/null || echo 000)
if [ "$code" = 400 ]; then say OK "业务/API语义" "18090 空参返回 400（校验层应答）"
else say FAIL "业务/API语义" "18090 返回 $code（期望 400）"; fi
need_http "业务/superadmin页面" "http://127.0.0.1:18091/" '^200$'
code=$(http_code "http://127.0.0.1:18091/admin-api/ping")
case "$code" in
  000|502|504) say FAIL "业务/反代" "admin-api 返回 $code（nginx→services 断了）" ;;
  *) say OK "业务/反代" "admin-api 返回 $code（后端应答，反代通）" ;;
esac
# VLM 深链：从栈内网络经 llm-tunnel 直探远端 vLLM（401/403 也算活：仅说明未带 key）
NET=$(docker inspect odyss-ifa-services -f '{{range $k,$v := .NetworkSettings.Networks}}{{$k}}{{end}}' 2>/dev/null || true)
if [ -n "${NET:-}" ]; then
  out=$(docker run --rm --network "$NET" alpine:3.20 sh -c 'wget -qO- -T 10 http://llm-tunnel:28000/v1/models 2>&1' || true)
  if echo "$out" | grep -qE '"id"|"object"|HTTP/[0-9.]+ 4'; then
    say OK "业务/VLM链路" "经 llm-tunnel 探到远端 vLLM 应答"
  else
    say FAIL "业务/VLM链路" "llm-tunnel:28000 不可达（${out:0:80}）——先查 GCP g4-01 侧再动本机"
  fi
else
  say WARN "业务/VLM链路" "services 容器不在，跳过深链探测"
fi

# ---------------- B. 演示 ----------------
need_systemd da3-web "演示/da3-web"
need_http "演示/da3-web页面" "http://127.0.0.1:8060/" '^200$'
need_systemd sam3 "演示/SAM3"
ss -tln 2>/dev/null | grep -q ':8013 ' && say OK "演示/SAM3端口" "8013 在监听" || say FAIL "演示/SAM3端口" "8013 未监听"
need_running locateanything-vllm "演示/LA-vLLM"
need_http "演示/LA-vLLM接口" "http://127.0.0.1:8001/v1/models" '^(200|401|403)$'
need_running locateanything-server-1 "演示/LA-gateway1"
ss -tln 2>/dev/null | grep -q ':8010 ' && say OK "演示/LA-gateway1端口" "8010 在监听" || say FAIL "演示/LA-gateway1端口" "8010 未监听"
need_running locateanything-lb "演示/LA-LB"
code=$(http_code "http://127.0.0.1:8000/")
case "$code" in
  000|5??) say FAIL "演示/LA入口" "8000 返回 $code" ;;
  *) say OK "演示/LA入口" "8000 返回 $code" ;;
esac

# ---------------- C. 观测 ----------------
need_running ifa-grafana-gcp "观测/统一Grafana"
need_http "观测/统一Grafana接口" "http://127.0.0.1:3001/api/health" '^200$'
need_systemd ifa-grafana-tunnel "观测/g4隧道"
need_http "观测/g4联邦数据源" "http://127.0.0.1:29090/-/ready" '^200$'
need_running locateanything-prometheus "观测/本机Prometheus"
need_http "观测/本机Prometheus接口" "http://127.0.0.1:9091/-/ready" '^200$'
need_running locateanything-grafana "观测/LA-Grafana"
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

# ---------------- 预期停止（反向异常检测：在跑才是问题） ----------------
expect_stopped locateanything-server-2 "预期停止/LA-gateway2"
expect_stopped clip-score-server "预期停止/SigLIP"

# ---------------- 汇总 ----------------
echo "===="
if [ "$FAIL" = 1 ]; then
  echo "RESULT|FAIL|存在异常项：按 tools/health/README.md 的单元入口拉起后必须复验"
  exit 1
else
  echo "RESULT|OK|全部健康（WARN/INFO 项见上，仅需知会）"
fi
