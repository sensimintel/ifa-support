# -*- coding: utf-8 -*-
"""去重合并的证据码校验与解码（纯逻辑，无 cv2/torch 依赖，可单测）。

模型在 match_evidence 里给的不再是自然语言对照文本，而是一个 8 字符证据码
`BxCxSxVx`（x ∈ 1 一致 / 0 不一致 / ? 看不清），无相似候选时给 `NONE`。

为什么换成码：
  · 省输出——自然语言那版每个 item 约 30 token，多物品帧里它和 match_reason
    加起来占了输出的一半，而 decode 是识别延时里最大的一块；
  · 判定更硬——旧版闸门是「文本里含『不一致』三个字」，模型换个措辞就绕过去了；
    码是可校验的，格式不合法本身就构成拒合并的理由（宁拒勿并）。

控制面那侧仍要给人看，所以这里解码回中文，写进观测日志与合并史——
superadmin 的 RecogLogPanel 不需要跟着改。
"""
import re

EV_NONE = "NONE"          # 无相似候选（此时不该有 match）
_EV_FIELDS = ("品牌与包装", "颜色与外观", "形状与份量", "容器与摆放")
_EV_STATE = {"1": "一致", "0": "不一致", "?": "看不清"}
_EV_RE = re.compile(r"B([10?])C([10?])S([10?])V([10?])")


def check_evidence(code):
    """校验并解码证据码，返回 (verdict, text)。

    verdict：
      "ok"        四项无 0，允许继续走后面的闸门；
      "mismatch"  至少一项为 0 —— prompt 明令任一项不一致禁并，模型自相矛盾；
      "none"      模型自称无相似候选，却仍给了 match，同样矛盾；
      "malformed" 码不合法（漏项/乱写/空）——证据不可信，按拒合并处理。
    text：人类可读的逐项对照，供日志与控制面展示。"""
    code = (code or "").strip().upper()
    if code == EV_NONE:
        return "none", "无相似候选"
    m = _EV_RE.fullmatch(code)
    if not m:
        return "malformed", "证据码非法：%s" % (code or "空")
    text = "；".join("%s%s" % (f, _EV_STATE[s]) for f, s in zip(_EV_FIELDS, m.groups()))
    return ("mismatch" if "0" in m.groups() else "ok"), text
