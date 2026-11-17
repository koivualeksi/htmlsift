"""render.py (no bundle): the render_hidden toggle, hidden-subtree tails, line_src
provenance and empty input."""
from htmlsift.core.render import parse, render, render_tree


def test_render_hidden_toggle():
    for decl in ("display:none", "display: none"):
        html = f'<html><body><p>Visible.</p><p style="{decl}">Hidden.</p></body></html>'
        assert "Hidden." not in render(html)
        assert "Visible." in render(html)
        assert "Hidden." in render(html, render_hidden=True)


def test_hidden_subtree_skipped_but_tail_renders():
    html = ('<html><body><div>'
            '<p style="display:none">Hidden.</p>Tail text.'
            "</div></body></html>")
    out = render(html)
    assert "Hidden." not in out
    assert "Tail text." in out


def test_line_src_provenance():
    html = "<html><body><h1>Title</h1><p>Para <a href='/x'>link</a>.</p></body></html>"
    tree = parse(html)
    h1 = tree.xpath("//h1")[0]
    p = tree.xpath("//p")[0]
    a = tree.xpath("//a")[0]
    lines, src = render_tree(tree)
    assert len(lines) == len(src)
    assert src[lines.index("# Title")] == {h1}
    assert src[lines.index("Para link.")] == {p, a}
    for line, s in zip(lines, src, strict=True):
        assert bool(s) == bool(line.strip())     # separators carry no source


def test_render_empty_input():
    assert render("") == ""
    assert render("   ") == ""
