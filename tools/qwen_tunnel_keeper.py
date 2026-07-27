#!/usr/bin/env python3
"""Qwen 识别隧道守护（跑在 Mac 上，launchd 常驻）。

背景：识别用的 Qwen3-VL 在 GCP gpu-g4-01（无公网入口，仅 IAP 可达），而 GCP
Workforce 凭证只在这台 Mac 上，所以隧道必须由 Mac 维持，链路为：
    5090:8011 ←反向SSH← Mac:18011 ←IAP← gpu-g4-01:8000(vllm)

本守护做三件事：
  1. 拉起并持有两段隧道子进程（IAP 段 + 反向段），任一退出自动重建；
  2. 每 5s 向 5090 的 da3-web 心跳（POST /api/tunnel/keeper）：上报状态文案、
     领取网页「一键重建」指令、拿回 5090 侧的全链路探测结果（连续 3 次不通
     也触发重建，覆盖“进程活着但转发已坏”的情况）；
  3. 识别 gcloud 凭证过期：IAP 段起不来且日志有过期特征时，把“需在 Mac
     浏览器重登 gcloud”上报到网页，并退避 60s 防止热循环。

安装（launchd 常驻，开机自启、崩溃自拉）：
    cp tools/life.odyss.qwen-tunnel-keeper.plist ~/Library/LaunchAgents/
    launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/life.odyss.qwen-tunnel-keeper.plist
卸载：
    launchctl bootout gui/$(id -u)/life.odyss.qwen-tunnel-keeper
日志：~/Library/Logs/qwen-tunnel-keeper.log（守护）/ qwen-tunnel-keeper-iap.log（IAP 段）

凭证过期后：用户在 Mac 浏览器重跑
    gcloud auth login --login-config=~/.claude/skills/ssh-gcp-gpu/odyss-gcp-login.json
守护会在下一轮重建时自动恢复，无需重启守护。
"""
import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request

APP = os.environ.get("TUNNEL_APP", "http://192.168.0.50:8060")   # 5090 的 da3-web
LOCAL_PORT = 18011        # Mac 本机中转端口
POLL = 5.0                # 心跳/巡检周期(秒)
IAP_LOG = os.path.expanduser("~/Library/Logs/qwen-tunnel-keeper-iap.log")

# IAP 段：Mac:18011 → gpu-g4-01:8000（vllm Qwen）
IAP_CMD = ["gcloud", "compute", "ssh", "gpu-g4-01",
           "--project=pelagic-pod-489307-g3", "--zone=asia-southeast1-b",
           "--tunnel-through-iap", "--quiet", "--",
           "-N", "-T", "-o", "ExitOnForwardFailure=yes",
           "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3",
           "-L", f"127.0.0.1:{LOCAL_PORT}:127.0.0.1:8000"]
# 反向段：5090:8011 → Mac:18011
REV_CMD = ["ssh", "-N", "-o", "ExitOnForwardFailure=yes",
           "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3",
           "-o", "ConnectTimeout=10",
           "-R", f"127.0.0.1:8011:127.0.0.1:{LOCAL_PORT}", "odyss@192.168.0.50"]

iap_proc = None
rev_proc = None
msg = ""                  # 上报给网页的状态文案（空=正常）


def log(s):
    print(time.strftime("%m-%d %H:%M:%S"), s, flush=True)


def local_up():
    """探本机 IAP 段：有 HTTP 应答（含未带 key 的 401）即通。"""
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{LOCAL_PORT}/v1/models", timeout=3)
        return True
    except urllib.error.HTTPError:
        return True
    except Exception:
        return False


def kill_stray():
    """按端口特征精确清理游离隧道进程（含历史 nohup / ssh -f 残留），避免端口占用。"""
    for pat in (f"{LOCAL_PORT}:127.0.0.1:8000", f"8011:127.0.0.1:{LOCAL_PORT}"):
        subprocess.run(["pkill", "-f", pat], check=False)


def port_busy():
    """本机 18011 是否仍有监听（旧隧道未放口时新 IAP 段会 ExitOnForwardFailure 起不来）。"""
    s = socket.socket()
    s.settimeout(0.5)
    try:
        s.connect(("127.0.0.1", LOCAL_PORT))
        return True
    except Exception:
        return False
    finally:
        s.close()


def creds_expired():
    """看 IAP 段日志尾部有没有 gcloud 凭证过期特征。"""
    try:
        with open(IAP_LOG, "rb") as f:
            f.seek(max(0, os.path.getsize(IAP_LOG) - 4000))
            tail = f.read().decode("utf-8", "ignore")
    except Exception:
        return False
    return any(k in tail for k in ("Reauthentication", "invalid_grant",
                                   "refreshing your current auth"))


def start_iap():
    global iap_proc
    logf = open(IAP_LOG, "ab")
    iap_proc = subprocess.Popen(IAP_CMD, stdout=logf, stderr=logf)


def start_rev():
    global rev_proc
    rev_proc = subprocess.Popen(REV_CMD, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)


def rebuild(reason):
    """全量重建两段隧道；失败时把原因写进 msg 供网页展示。"""
    global msg
    log(f"重建隧道：{reason}")
    msg = "重建中…"
    for p in (iap_proc, rev_proc):
        if p and p.poll() is None:
            p.terminate()
    kill_stray()
    # 等旧隧道真正放掉 18011 再起新 IAP 段（否则 ExitOnForwardFailure 必败），最长 12s
    for _ in range(6):
        if not port_busy():
            break
        kill_stray()
        time.sleep(2)
    start_iap()
    for i in range(40):          # IAP 段最长等 40s
        if local_up():
            break
        if iap_proc.poll() is not None:   # IAP 段进程已死，等下去没意义
            break
        if i % 5 == 4:
            _try_heartbeat("重建中…")     # 重建期间保持心跳，网页别误判守护掉线
        time.sleep(1)
    else:
        pass
    if not local_up():
        msg = ("gcloud 凭证过期，需在 Mac 浏览器重登" if creds_expired()
               else "IAP 段起不来，看 Mac 上的 keeper 日志")
        log(f"重建失败：{msg}")
        return
    start_rev()
    msg = ""
    log("隧道已重建")


def _try_heartbeat(text):
    """尽力而为的心跳（重建过程内用），失败忽略。"""
    try:
        heartbeat(text)
    except Exception:
        pass


def heartbeat(cur_msg):
    """心跳：上报状态、领取网页重建指令、拿回 5090 侧探测结果。"""
    body = json.dumps({"msg": cur_msg}).encode()
    req = urllib.request.Request(APP + "/api/tunnel/keeper", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as r:
        d = json.load(r)
    return bool(d.get("rebuild")), d.get("up")


def main():
    global msg
    log("守护启动，接管隧道")
    rebuild("守护启动，接管现有/残留隧道")
    down_n = 0                   # 5090 侧连续探测不通的次数
    backoff_until = 0.0          # 凭证过期时的重建退避
    while True:
        time.sleep(POLL)
        dead = (iap_proc is None or iap_proc.poll() is not None
                or rev_proc is None or rev_proc.poll() is not None)
        cur_msg = msg or ("隧道进程退出，准备自动重建" if dead else "")
        req = False
        server_up = None
        try:
            req, server_up = heartbeat(cur_msg)
        except Exception as e:
            log(f"心跳失败（5090 不可达？）：{type(e).__name__}: {e}")
        down_n = down_n + 1 if server_up is False else 0
        now = time.time()
        if req:
            rebuild("网页一键重建指令")
            down_n = 0
        elif (dead or down_n >= 3) and now >= backoff_until:
            rebuild("隧道进程退出" if dead else "5090 侧连续探测不通")
            down_n = 0
            if "凭证过期" in msg:
                backoff_until = now + 60   # 凭证问题重建必败，退避防热循环


if __name__ == "__main__":
    main()
