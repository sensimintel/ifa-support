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
        # logo 上提 12（.25→.13）；待机文案 80 / 营养卡 54，按「距屏幕最底」算——
        # 不叠加 safe-area（叠加会凭空多抬 34），再扣掉视口缺口 --vpgap
        src = _src()
        for needle in (
            "#logo{left:50%;top:calc(env(safe-area-inset-top,0px) + .13rem)",
            "#idle{width:auto;margin:0 0 max(0px,calc(.8rem - var(--vpgap)))}",
            "margin:0 0 max(0px,calc(.54rem - var(--vpgap)));padding:.16rem}",
        ):
            self.assertIn(needle, src, needle)

    def test_viewport_gap_is_measured_and_portrait_only(self):
        # 视口短于屏幕时（iOS standalone 画不到最底那一条），贴底值必须扣掉缺口
        src = _src()
        self.assertIn("--vpgap:0px}", src)
        self.assertIn("function setVpGap()", src)
        self.assertIn("(screen.height||0)-innerHeight-off", src)
        # 必须按 env(top) 判向：视口短的那截在顶部（状态栏 opaque）时底部本就贴屏底，
        # 再补偿会把内容顶起来
        self.assertIn("const g=safeTop>0?", src)
        self.assertIn("innerHeight>innerWidth", src)   # 只在竖屏校正
        # 只在主屏图标打开的 web app 里补偿：浏览器标签页里 screen 与视口不同口径，
        # 窗口没最大化就会算出假缺口把内容顶飞
        self.assertIn("navigator.standalone===true||matchMedia('(display-mode:standalone)')", src)

    def test_geometry_selfcheck_available(self):
        # ?geom=1 自检面板：真机上「看着不对」时一屏截图就能分清是版式偏了还是视口短了
        src = _src()
        self.assertIn("get('geom')", src)
        self.assertIn("视口底距屏底", src)
        # 主屏图标打开时没有地址栏，必须留手势入口，否则真机上根本调不出来
        self.assertIn("if(n>=5){n=0;showGeom();}", src)

    def test_macros_row_gap(self):
        # 三宫格与其上方分隔线的间距：Figma 标注 +2（.12→.14），真机走查再收窄 5（→.09）；
        # 名称那块下移同样的 5，卡片总高不变
        src = _src()
        self.assertIn("#card .rvw:has(#macros){margin-top:.09rem}", src)
        self.assertIn("#card .rvw.nom{margin-top:.05rem}", src)
        self.assertIn("#card .rvw+.rvw{margin-top:.12rem}", src)

    def test_portrait_shade_is_tunable_gradient(self):
        # 顶底压暗改纵向渐变（原贴图顶底 alpha 0.96、20% 内急落，观感死黑），且可 URL 调参
        src = _src()
        self.assertIn("--sh-top:.8;--sh-bot:.8;--sh-span:22", src)
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
            # 取 black 而非 black-translucent：后者视口整体上移状态栏高度、高度不加高，
            # 屏幕最底同样多少像素页面画不到（iPhone 17 实测底部黑 62）
            '<meta name="apple-mobile-web-app-status-bar-style" content="black">',
            '<link rel="manifest" href="/static/experience-manifest.json">',
        ):
            self.assertIn(needle, src, needle)


if __name__ == "__main__":
    unittest.main()
