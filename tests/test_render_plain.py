"""render.py plain= extension (no bundle): plain=True drops markdown markers and keeps
the line structure; plain=False is the frozen model-input render."""
from pathlib import Path

import pytest

from htmlsift.core.render import parse, render_tree

DATA = Path(__file__).parent / "data"


def _plain(body):
    lines, _ = render_tree(parse(f"<html><body>{body}</body></html>"), plain=True)
    return [ln for ln in lines if ln.strip()]


CASES = [
    ("<h1>Title</h1>", ["Title"]),
    ("<h3>\n  Indented title\n</h3>", ["Indented title"]),
    ('<h3> <a href="/x"> Linked title</a></h3>', ["Linked title"]),
    ("<h2>A <b>bold</b> title</h2>", ["A bold title"]),
    ("<p>x <b>b</b> <strong>s</strong> <i>i</i> <em>e</em> <code>c</code> <del>d</del></p>",
     ["x b s i e c d"]),
    ("<blockquote><p>one</p><p>two</p></blockquote>", ["one", "two"]),
    ("<pre>def f():\n    return 1</pre>", ["def f():", "    return 1"]),
    ("<p>above</p><hr><p>below</p>", ["above", "below"]),
    ("<table><tr><th>k</th><th>v</th></tr><tr><td>a|b</td><td>1</td></tr></table>",
     ["| k | v |", "| --- | --- |", "| a|b | 1 |"]),
    ("<ul><li>one</li><li>two</li></ul>", ["- one", "- two"]),
    ('<ol start="3"><li>three</li><li>four</li></ol>', ["3. three", "4. four"]),
    ("<ul><li>outer<ul><li>inner</li></ul></li></ul>", ["- outer", "  - inner"]),
    ("<dl><dt>term</dt><dd>definition</dd></dl>", ["term", "    definition"]),
    ("<p>C# and #tag, 5 * 3 = 15, a|b and _x_</p>", ["C# and #tag, 5 * 3 = 15, a|b and _x_"]),
]


@pytest.mark.parametrize("body, expected", CASES)
def test_plain_lines(body, expected):
    assert _plain(body) == expected


MARKER_ONLY_WRAPS = ["<p><b><hr></b></p><p>after</p>",
                     "<p><strong><pre>x</pre></strong></p><p>after</p>",
                     "<p><b><blockquote> </blockquote></b></p><p>after</p>"]


@pytest.mark.parametrize("body", MARKER_ONLY_WRAPS)
def test_marker_only_wrap(body):
    assert _plain(body)[-1] == "after"


@pytest.mark.parametrize("body", [c[0] for c in CASES] + MARKER_ONLY_WRAPS
                         + ["<h2> </h2>", "<p><b></b> x</p>"])
def test_plain_same_line_count(body):
    root = parse(f"<html><body>{body}</body></html>")
    assert len(render_tree(root, plain=True)[0]) == len(render_tree(root)[0])


@pytest.mark.parametrize("page", sorted(DATA.glob("*.html")), ids=lambda p: p.name)
def test_plain_same_line_count_on_pages(page):
    root = parse(page.read_text(encoding="utf-8"))
    assert len(render_tree(root, plain=True)[0]) == len(render_tree(root)[0])


def test_plain_false_is_default():
    root = parse((DATA / "sample.html").read_text(encoding="utf-8"))
    assert render_tree(root, plain=False) == render_tree(root)


def test_markdown_render_keeps_markers():
    lines, _ = render_tree(parse("<html><body><h1>T</h1><p><b>x</b></p></body></html>"))
    assert "# T" in lines and "**x**" in lines
