# -*- coding: utf-8 -*-
"""内联 JS 语法闸门：页面常量里的 <script> 必须能被解析。

app.py 里的页面是 Python 三引号字符串，JS 又是一层字符串——两层转义叠在一起，
`'\\n'` 少写一个反斜杠就会被 Python 提前解析成真换行，把 JS 字符串截断。后果是整个
script 块 SyntaxError、页面所有 JS 一起哑掉（2026-08-27 实锤：浅体验区没了点云，
服务端与帧链路全正常，纯前端脚本没跑）。这种错 Python 侧语法检查看不出来，页面也
照样返回 200，只有真机打开才发现——故在这里用 node 逐个 parse。
"""
import ast
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "app.py").read_text(encoding="utf-8")

# 页面常量 → 其内联 script。带 src= 的外链标签没有正文，跳过
PAGE_CONSTS = ("EXPERIENCE_PAGE", "RECOG_PAGE", "SAM3TUNE_PAGE")

# 必须取 ast 解析后的**常量值**，不能拿源码文本去正则——源码里 `\n` 还是两个字符、
# 送进 node 是合法字符串，正是要抓的那个错在源码层面看不出来。这里取到的才是浏览器
# 真正收到的字节
_VALUES = {
    node.targets[0].id: node.value.value
    for node in ast.parse(SRC).body
    if isinstance(node, ast.Assign)
    and isinstance(node.targets[0], ast.Name)
    and isinstance(node.value, ast.Constant)
    and isinstance(node.value.value, str)
}


def _page(name: str) -> str:
    assert name in _VALUES, f"未找到页面常量 {name}"
    return _VALUES[name]


def _inline_scripts(html: str):
    return [
        body
        for tag, body in re.findall(r"<script([^>]*)>(.*?)</script>", html, re.S)
        if "src=" not in tag and body.strip()
    ]


@unittest.skipUnless(shutil.which("node"), "本机没有 node，跳过 JS 语法检查")
class InlineJsSyntaxTest(unittest.TestCase):
    def test_inline_scripts_parse(self):
        for name in PAGE_CONSTS:
            scripts = _inline_scripts(_page(name))
            self.assertTrue(scripts, f"{name} 里没抽到内联 script，正则该跟着页面改")
            for i, js in enumerate(scripts):
                with tempfile.NamedTemporaryFile("w", suffix=".js", encoding="utf-8") as f:
                    f.write(js)
                    f.flush()
                    r = subprocess.run(["node", "--check", f.name],
                                       capture_output=True, text=True)
                self.assertEqual(r.returncode, 0,
                                 f"{name} 第 {i + 1} 段内联 JS 语法错误：\n{r.stderr}")


if __name__ == "__main__":
    unittest.main()
