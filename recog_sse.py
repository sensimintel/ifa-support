# -*- coding: utf-8 -*-
"""OpenAI 兼容流式响应（SSE）的拼装（纯逻辑，无 cv2/torch/网络依赖，可单测）。

识别调用改流式后，服务端不再一次性给完整 JSON，而是逐块推 `data: {...}`。
这里只做三件事，不碰网络也不碰图像：
  · 把每块 delta.content 按到达顺序拼回完整字符串；
  · 记下**首个内容块**到达的耗时 ttft_ms——它把 http 段切成「上行传图+排队+prefill」
    与「decode」两截，之前这两截混在一个 http_ms 里，「慢在传图还是慢在生成」只能靠猜；
  · 捡走末尾 usage-only 块里的 token 数（流式下 usage 只在那一块给）。

容错取向：脏块跳过而不是整轮报废（半行 JSON、心跳注释、空 delta 都可能出现），
但总时长超限必须抛错——urlopen 的 timeout 只管单次 read，流式下总时长不受它约束。
"""
import json


def read_sse_completion(lines, t_start, timeout, now):
    """把 SSE 行迭代器拼成 (content, usage, ttft_ms)。

    lines：可迭代的原始行（bytes 或 str 均可，逐行给），通常就是 HTTPResponse 本身；
    t_start：请求发出时刻；timeout：总时长上限(秒)，超限抛 TimeoutError；
    now：取当前时刻的函数（注入以便测试）。
    ttft_ms 在整轮无任何内容块时为 None（例如模型只回了空 delta）。"""
    parts, usage, ttft_ms = [], {}, None
    for line in lines:
        if now() - t_start > timeout:
            raise TimeoutError("流式读取超过 %.0fs" % timeout)
        if isinstance(line, bytes):
            line = line.decode("utf-8", "ignore")
        line = line.strip()
        if not line.startswith("data:"):
            continue                      # 注释行 / 心跳 / 空行
        chunk = line[5:].strip()
        if chunk == "[DONE]":
            break
        try:
            obj = json.loads(chunk)
        except ValueError:
            continue                      # 半行或脏块：跳过，别让一块毁掉整轮
        if not isinstance(obj, dict):
            continue
        for ch in obj.get("choices") or []:
            piece = (ch.get("delta") or {}).get("content")
            if piece:
                if ttft_ms is None:
                    ttft_ms = (now() - t_start) * 1000.0
                parts.append(piece)
        if obj.get("usage"):              # usage-only 尾块
            usage = obj["usage"]
    return "".join(parts), usage, ttft_ms
