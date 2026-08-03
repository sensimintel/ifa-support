"""llm-switch actuator：5090 本地栈的识别模型一键切换。

只做两件事：
1. GET /llm-mode  -> {"mode": "qwen"|"gemini"|"unknown", "switching": bool}
2. POST /llm-mode {"target": "qwen"|"gemini"} -> 用对应 runtime-config 变体覆盖
   /config/runtime-config.yaml，然后 docker restart 目标 services 容器。

模式判定依据 runtime-config 里 chunk_inhouse.enabled：true=qwen（自建隧道）、
false=gemini（远端网关）。变体文件缺失时拒绝切换，绝不写半截配置。
仅监听栈内网络，不暴露宿主端口；无鉴权（局域网演示栈口径，与 18091 一致）。
"""
import json
import os
import re
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

CONFIG_DIR = "/config"
ACTIVE = os.path.join(CONFIG_DIR, "runtime-config.yaml")
VARIANTS = {
    "qwen": os.path.join(CONFIG_DIR, "runtime-config.qwen.yaml"),
    "gemini": os.path.join(CONFIG_DIR, "runtime-config.gemini.yaml"),
}
TARGET_CONTAINER = os.environ.get("TARGET_CONTAINER", "odyss-ifa-services")

_lock = threading.Lock()
_switching = False


def current_mode():
    try:
        with open(ACTIVE, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return "unknown"
    m = re.search(r"chunk_inhouse:\s*\n\s*enabled:\s*(true|false)", text)
    if not m:
        return "unknown"
    return "qwen" if m.group(1) == "true" else "gemini"


def do_switch(target):
    global _switching
    try:
        shutil.copyfile(VARIANTS[target], ACTIVE)
        subprocess.run(["docker", "restart", TARGET_CONTAINER], check=True, timeout=120)
    finally:
        with _lock:
            _switching = False


class Handler(BaseHTTPRequestHandler):
    def _json(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") != "/llm-mode":
            return self._json(404, {"error": "not found"})
        with _lock:
            switching = _switching
        self._json(200, {"mode": current_mode(), "switching": switching})

    def do_POST(self):
        global _switching
        if self.path.rstrip("/") != "/llm-mode":
            return self._json(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            target = json.loads(self.rfile.read(length) or b"{}").get("target")
        except (ValueError, json.JSONDecodeError):
            return self._json(400, {"error": "请求体必须是 JSON"})
        if target not in VARIANTS:
            return self._json(400, {"error": "target 只支持 qwen|gemini"})
        if not os.path.exists(VARIANTS[target]):
            return self._json(500, {"error": f"缺少配置变体 {VARIANTS[target]}"})
        with _lock:
            if _switching:
                return self._json(409, {"error": "已有切换在进行中"})
            _switching = True
        threading.Thread(target=do_switch, args=(target,), daemon=True).start()
        self._json(202, {"ok": True, "target": target})

    def log_message(self, fmt, *args):
        print("[llm-switch]", fmt % args)


if __name__ == "__main__":
    HTTPServer(("0.0.0.0", 8095), Handler).serve_forever()
