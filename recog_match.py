# -*- coding: utf-8 -*-
"""去重合并相关的纯逻辑（证据码校验与解码 + 参考图取框），无 cv2/torch 依赖，可单测。

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

# diff 字段的逃逸话术：写了这些等于没给否定证据。旧版证据码之所以退化成清一色
# B1C1S1V1，就是因为「说不出差别」零成本——现在说不出就不许合并。
# prompt 那侧的文案也从这里取，避免两处各写一份、改一处漏一处。
# 2026-08-25 模型产出改英文后，套话清单跟着换英文；匹配一律先转小写，
# 免得 "No difference" 这种首字母大写的写法从闸门缝里漏过去。
DIFF_BLANKS = ("no difference", "no differences", "exactly the same", "identical",
               "looks the same", "nothing different", "indistinguishable")
DIFF_MIN_LEN = 15         # diff 至少这么长才算写了东西（英文 15 字符 ≈ 三四个词）
_NO_TEXT = ("", "none", "null", "无", "无文字", "n/a", "-")   # cur_text 视同「没读到文字」


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


def derive_confidence(code) -> str:
    """由证据码推导合并置信度（match_confidence 字段已从输出契约移除，2026-08-24）。

    旧 prompt 里 high 的定义是「B=1，或 C、S、V 三项全 1 且参考图清晰可辨」——
    前两个条件本来就是证据码的确定性函数，让模型再报一遍 confidence 纯属可派生
    冗余（每轮 ~7 token），还留了「码很硬却自报 high 造假」的口子。改成服务端推导：
    语义与旧闸门四完全一致，「参考图清晰可辨」的主观项由 ? 状态吸收（看不清该填 ?，
    填了 ? 自然推不出 high）。码非法/NONE 一律 low——宁拒勿并。"""
    m = _EV_RE.fullmatch((code or "").strip().upper())
    if not m:
        return "low"
    b, c, s, v = m.groups()
    return "high" if b == "1" or (c == "1" and s == "1" and v == "1") else "low"


def check_self_evidence(item):
    """闸门六·当前画面自证 + 闸门七·B 位抵押（纯格式判定，零语义、零误伤空间）。

    背景：线上出现过一张卡自我确认 252 次，合并史里清一色「四项全一致 + high」。
    根因之一是填 `1` 零成本——模型不需要为「一致」这个主张付出任何额外输出。
    这两道闸把成本加回去：
      · 说不出【当前画面】里这一样东西长什么样（seen 空）→ 多半是照着参考图编的；
      · 说不出与候选的任何一处差别（diff 空/太短/是套话）→ 不许合并；
      · 证据码 B 位填 1，却没在 cur_text 里照抄出任何包装文字 → 这个 1 没有抵押物。

    返回 (ok, reason)：ok=False 时 reason 是给日志与控制面看的中文拒合并理由。"""
    seen = str((item or {}).get("seen") or "").strip()
    if not seen:
        return False, "没写 seen：说不出当前画面里它长什么样"
    diff = str((item or {}).get("diff") or "").strip()
    if not diff:
        return False, "没写 diff：说不出与候选的任何差别"
    # 先判套话再判长度：套话往往本身就短，先报长度会把真正的原因盖掉，
    # 而拒合并理由是要给人看的（控制面「VLM 识别日志」直接展示这句）
    if any(w in diff.lower() for w in DIFF_BLANKS):
        return False, "diff 是套话：%s" % diff
    if len(diff) < DIFF_MIN_LEN:
        return False, "diff 太短（%d 字符）：%s" % (len(diff), diff)
    code = str((item or {}).get("match_evidence") or "").strip().upper()
    m = _EV_RE.fullmatch(code)
    if m and m.group(1) == "1":
        cur_text = str((item or {}).get("cur_text") or "").strip().lower()
        if cur_text in _NO_TEXT:
            return False, "证据码 B=1 却没读到任何包装文字（cur_text=%s）" % (cur_text or "空")
    return True, ""


# 参考图取框：外扩比例与最小边长（归一化坐标）。外扩是因为检测框常常贴着物体边缘，
# 裁太紧会把包装文字的边角切掉，而那正是判同最有价值的信息。
REF_CROP_EXPAND = 1.25
REF_CROP_MIN = 0.12


def pick_ref_box(detections, expand=REF_CROP_EXPAND):
    """从本轮检测框里挑一个用来裁参考图，返回归一化 (x1,y1,x2,y2)；没有框返回 None。

    为什么要裁：参考图过去存的是**整帧**，而当前画面也是同一台相机、同一张桌子——
    两张图九成像素相同，于是「品牌/颜色/形状/容器四项一致」成了客观事实，
    证据码必然全 1，去重形同虚设（线上一张卡因此自我确认了 252 次）。
    裁成物体特写之后，四项对照才重新有区分度。

    挑最大的那个框：一轮只放行一个 item，最大框最可能就是它。"""
    best, best_area = None, 0.0
    for det in detections or []:
        try:
            _, x1, y1, x2, y2 = det
        except (TypeError, ValueError):
            continue
        x1, y1, x2, y2 = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area > best_area:
            best, best_area = (x1, y1, x2, y2), area
    if best is None or best_area <= 0:
        return None
    x1, y1, x2, y2 = best
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    hw = max(REF_CROP_MIN, (x2 - x1) * expand) / 2.0
    hh = max(REF_CROP_MIN, (y2 - y1) * expand) / 2.0
    return (max(0.0, cx - hw), max(0.0, cy - hh),
            min(1.0, cx + hw), min(1.0, cy + hh))
