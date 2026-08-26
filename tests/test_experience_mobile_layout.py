# -*- coding: utf-8 -*-
"""/experience 手机竖屏版式锁（Figma「IFA 专项 · 浅体验区」1616-4400 / 4397 / 4484 标注）。

这几个值是在真机上逐版对出来的，纯 CSS 常量、没有运行时兜底，改错了页面照样能跑，
只有到展台上才会看出来——故用源码断言钉住，防后续改版顺手改掉。

一个容易踩的坑记在这里：标注量自「未全屏」的旧截图（底部还留着浏览器工具栏那条），
所以落地取的是标注反推的**目标距屏底绝对值**，不是标注上写的位移量（照位移量减会得
负值、元素直接掉出屏幕）。PWA 全屏那几个 meta 是这套值成立的前提，一并锁住。
"""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _src() -> str:
    return (ROOT / "app.py").read_text(encoding="utf-8")


class ExperienceMobileLayoutTest(unittest.TestCase):
    def test_portrait_anchors_match_figma(self):
        # logo 上提 12（.25→.13）；待机文案与营养卡按距屏底 87 / 48 贴底（各自再加 safe-area）
        src = _src()
        for needle in (
            "#logo{left:50%;top:calc(env(safe-area-inset-top,0px) + .13rem)",
            "#idle{width:auto;margin:0 0 calc(env(safe-area-inset-bottom,0px) + .53rem)}",
            "margin:0 0 calc(env(safe-area-inset-bottom,0px) + .14rem);padding:.16rem}",
        ):
            self.assertIn(needle, src, needle)

    def test_macros_row_gap_is_two_more(self):
        # 三宫格与其上方分隔线的间距比通用区块间距多 2（.12 → .14）
        src = _src()
        self.assertIn("#card .rvw:has(#macros){margin-top:.14rem}", src)
        self.assertIn("#card .rvw+.rvw{margin-top:.12rem}", src)

    def test_portrait_shade_is_tunable_gradient(self):
        # 顶底压暗改纵向渐变（原贴图顶底 alpha 0.96、20% 内急落，观感死黑），且可 URL 调参
        src = _src()
        self.assertIn("--sh-top:.62;--sh-bot:.62;--sh-span:24", src)
        self.assertIn("#shade{background:linear-gradient(to bottom,", src)
        for key in ("shtop", "shbot", "shspan"):
            self.assertIn("'%s'" % key, src, key)
        # 竖屏不再挂那张贴图（文件本身留着备回退，但不应再被引用）
        self.assertNotIn("bg-shade-portrait.png')", src)

    def test_pwa_fullscreen_meta_present(self):
        # 没有这几条，主屏图标打开时顶底会各留一条黑带，上面那套贴底值全部对不上
        src = _src()
        for needle in (
            '<meta name="apple-mobile-web-app-capable" content="yes">',
            '<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">',
            '<link rel="manifest" href="/static/experience-manifest.json">',
        ):
            self.assertIn(needle, src, needle)


if __name__ == "__main__":
    unittest.main()
