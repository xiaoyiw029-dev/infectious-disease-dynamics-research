"""Render the canonical Markdown record; leave the source unchanged."""

from pathlib import Path

import mistune


def render(source: Path, target: Path) -> None:
    markdown = mistune.create_markdown(escape=True, plugins=["table", "math"])
    body = markdown(source.read_text(encoding="utf-8-sig"))
    template = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>研究决策总档案</title>
<script>window.MathJax={tex:{inlineMath:[['\\(','\\)']],displayMath:[['$$','$$'],['\\[','\\]']]},svg:{fontCache:'global'}};</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
<style>
body{max-width:1000px;margin:32px auto;padding:0 24px 60px;font-family:'Microsoft YaHei',sans-serif;line-height:1.8;color:#222;background:#fff}
h1,h2{line-height:1.4}h2{margin-top:32px;border-bottom:1px solid #ddd}
table{border-collapse:collapse;display:block;overflow:auto;margin:16px 0;width:100%}
th,td{border:1px solid #ddd;padding:8px;text-align:left}th{background:#f1f5f3}
a{color:#126250}code{overflow-wrap:anywhere}.math{overflow-x:auto}p{overflow-wrap:break-word}
@media print{body{margin:0;padding:0}h2{break-after:avoid}}
</style></head><body>
<p>公式首次显示需加载MathJax；编辑源为同目录Markdown文件，更新日期见正文。</p>
__BODY__
</body></html>'''
    target.write_text(template.replace("__BODY__", body), encoding="utf-8")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    directory = root / "00_项目总览"
    render(directory / "研究决策总档案.md", directory / "研究决策总档案_阅读版.html")
