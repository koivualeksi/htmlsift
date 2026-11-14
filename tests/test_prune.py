"""Pruner tests (no bundle): select rendered lines by text, prune, and check that the
pruned tree holds exactly the text of the selected lines."""
import random
import re
from pathlib import Path

import numpy as np
import pytest
from lxml import html as lxml_html

from htmlsift.core.prune import _is_code_block, prune
from htmlsift.core.render import parse, render_tree

DATA = Path(__file__).parent / "data"


def _prune(body, keep):
    """Prune a page to the rendered lines whose text is in keep -> (body html, plain lines)."""
    root = parse(f"<html><body>{body}</body></html>")
    lines, _, pieces = render_tree(root, pieces=True)
    blocks = [i for i, ln in enumerate(lines) if ln.strip()]
    selected = {k for k, i in enumerate(blocks) if lines[i].strip() in keep}
    prune(root, [pieces[i] for i in blocks], selected)
    body_el = root.find("body")
    out = "" if body_el is None else lxml_html.tostring(body_el, encoding="unicode")
    plain = [ln.rstrip() for ln in render_tree(root, plain=True)[0] if ln.strip()]
    return out, plain


def test_parent_text_dropped():
    out, _ = _prune("<div>Share | Login<p>Article</p></div>", {"Article"})
    assert out == "<body><div><p>Article</p></div></body>"


def test_tail_of_kept_element_dropped():
    out, _ = _prune("<div><p>Article</p>Share | Login</div>", {"Article"})
    assert out == "<body><div><p>Article</p></div></body>"


def test_children_of_kept_element_dropped():
    out, _ = _prune("<div>Intro.<p>Related: Top 10</p></div>", {"Intro."})
    assert out == "<body><div>Intro.</div></body>"


def test_br_lines_kept_and_dropped_one_by_one():
    out, _ = _prune("<p>Line one<br>Line two<br>Line three</p>", {"Line one", "Line three"})
    assert out == "<body><p>Line one<br>Line three</p></body>"


def test_tail_kept_without_its_element():
    out, _ = _prune("<div><h2>Related</h2>Body text.</div>", {"Body text."})
    assert out == "<body><div>Body text.</div></body>"


def test_tail_kept_in_inline_parent_keeps_its_line_break():
    _, plain = _prune("<div>Call us <a><div>Card</div>Join now</a></div>", {"Call us", "Join now"})
    assert plain == ["Call us", "Join now"]


def test_br_before_kept_tail_in_inline_parent_stays():
    _, plain = _prune("<div>Morning <a><br>12pm</a></div>", {"Morning", "12pm"})
    assert plain == ["Morning", "12pm"]


def test_bare_br_between_kept_lines_kept():
    out, plain = _prune("<div><a>Personal income</a><br><a>Population</a><br><a>Ad</a></div>",
                        {"Personal income", "Population"})
    assert plain == ["Personal income", "Population"]
    assert "<a>Personal income</a><br><a>Population</a>" in out and "Ad" not in out


def test_removed_block_between_kept_inline_text_leaves_a_break():
    out, plain = _prune("<div><span>Gender</span><div>Pick one</div><span>Required</span></div>",
                        {"Gender", "Required"})
    assert plain == ["Gender", "Required"]
    assert out == "<body><div><span>Gender</span><br><span>Required</span></div></body>"


def test_inline_elements_on_one_line_stay_joined():
    out, plain = _prune("<p><b>Hel</b><i>lo</i> world</p>", {"**Hel**_lo_ world"})
    assert out == "<body><p><b>Hel</b><i>lo</i> world</p></body>"


def test_page_br_spacing_kept():
    out, plain = _prune("<p>Para one<br><br>Para two</p>", {"Para one", "Para two"})
    assert plain == ["Para one", "Para two"]
    assert out == "<body><p>Para one<br><br>Para two</p></body>"


def test_inline_markup_and_image_kept_with_their_line():
    out, _ = _prune('<p>Read <b>this <i>now</i></b> please <img src="a.png"></p><p>Ad</p>',
                    {"Read **this _now_** please"})
    assert out == '<body><p>Read <b>this <i>now</i></b> please <img src="a.png"></p></body>'


def test_table_rows_kept_one_by_one():
    out, _ = _prune("<table><tr><th>Name</th><th>Info</th></tr><tr><td>A</td><td>1</td></tr>"
                    "<tr><td>B</td><td>2</td></tr></table>", {"| A | 1 |"})
    assert "<td>A</td>" in out
    assert "Name" not in out and "<td>B</td>" not in out


def test_table_separator_line_keeps_nothing():
    out, _ = _prune("<table><tr><th>Name</th></tr><tr><td>A</td></tr></table>", {"| --- |"})
    assert out == ""


def test_hr_kept_with_its_line():
    out, _ = _prune("<p>A</p><hr><p>B</p>", {"A", "* * *"})
    assert out == "<body><p>A</p><hr></body>"


def test_code_block_partly_selected_stays_whole():
    out, _ = _prune("<pre><span>$ pip install htmlsift</span>\nCopy</pre>",
                    {"$ pip install htmlsift"})
    assert out == "<body><pre><span>$ pip install htmlsift</span>\nCopy</pre></body>"


def test_pre_with_page_structure_cut_to_selected_lines():
    out, plain = _prune("<pre>Menu | Home\n<div>Article line one\nline two</div>\nFooter</pre>",
                        {"Article line one", "line two"})
    assert plain == ["Article line one", "line two"]
    assert "Menu" not in out and "Footer" not in out and "<div>" in out


def test_unselected_pre_removed():
    out, _ = _prune("<div><p>Intro</p><pre>code\nmore</pre></div>", {"Intro"})
    assert out == "<body><div><p>Intro</p></div></body>"


def test_nothing_selected():
    out, _ = _prune("<div>Share<p>Article</p></div>", set())
    assert out == ""


_SEPARATOR = re.compile(r"^\|( ?-+ ?\|)+$")


def _lines(lines):
    """Non-empty lines without list markers (they renumber) and table separator rows."""
    out = [re.sub(r"^(\s*(\d+\.|-)\s+)+", "", ln).strip() for ln in lines if ln.strip()]
    return [ln for ln in out if not _SEPARATOR.match(ln)]


@pytest.mark.parametrize("name", ["sample.html", "article_fi.html", "article_ja.html"])
def test_pruned_tree_holds_the_selected_lines(name):
    html = (DATA / name).read_text(encoding="utf-8")
    rng = random.Random(0)
    for _ in range(100):
        root = parse(html)
        lines, _, pieces = render_tree(root, pieces=True)
        plain, _ = render_tree(root, plain=True)
        blocks = [i for i, ln in enumerate(lines) if ln.strip()]
        selected = {k for k in range(len(blocks)) if rng.random() < 0.5}
        code = {p[0] for k in selected for p in pieces[blocks[k]]
                if p[1] == "pre" and _is_code_block(p[0])}
        selected |= {k for k, i in enumerate(blocks)
                     if any(p[1] == "pre" and p[0] in code for p in pieces[i])}
        want = _lines([plain[blocks[k]] for k in sorted(selected)])
        prune(root, [pieces[i] for i in blocks], selected)
        assert _lines(render_tree(root, plain=True)[0]) == want


def test_extract_html_and_markdown_drop_rejected_lines(keep_all):
    keep_all._model.infer_fn = lambda blocks: np.array(
        [0.0 if "Share" in b else 1.0 for b in blocks], dtype=np.float32)
    page = "<html><body><div>Share | Login<p>Article</p>Share this</div></body></html>"
    assert keep_all.extract(page) == "Article"
    assert keep_all.extract(page, "markdown") == "Article"
    assert "Share" not in keep_all.extract(page, "html")


def test_extract_html_keeps_every_inline_image(keep_all):
    n = 50
    body = "".join(f'<p>Read <a href="/x{i}">this</a> <img src="i{i}.png"></p>'
                   for i in range(n))
    out = keep_all.extract(f"<html><body>{body}</body></html>", output="html")
    missing = [i for i in range(n) if f'src="i{i}.png"' not in out]
    assert missing == []
