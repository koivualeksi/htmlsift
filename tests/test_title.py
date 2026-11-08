import pytest

from htmlsift._title import (
    SIM,
    TITLE_SEP,
    find_title,
    norm,
    page_titles,
    title_candidates,
    title_match,
)
from htmlsift.core.render import parse, render_tree


def test_norm():
    assert norm("  Ｈｅｌｌｏ,   WORLD!_x ") == "hello world x"
    assert norm("Straße") == "strasse"
    assert norm("—|—") == ""


@pytest.mark.parametrize("title, parts", [
    ("A | B", ["A", "B"]),
    ("A|B", ["A", "B"]),
    ("A·B", ["A", "B"]),
    ("A • B", ["A", "B"]),
    ("A » B", ["A", "B"]),
    ("A - B", ["A", "B"]),
    ("A – B", ["A", "B"]),
    ("A — B", ["A", "B"]),
    ("A : B", ["A", "B"]),
    ("A :: B", ["A", "B"]),
    ("A // B", ["A", "B"]),
    ("A ~ B", ["A", "B"]),
    ("A -- B", ["A", "B"]),
])
def test_separator_splits(title, parts):
    assert TITLE_SEP.split(title) == parts


@pytest.mark.parametrize("title", [
    "e-mail", "05-08", "10:30", "Player Info: Jake Zard", "a/b", "A_B", "A｜B", "A：B",
    "A- B", "A -B",
])
def test_separator_does_not_split(title):
    assert TITLE_SEP.split(title) == [title]


def test_title_candidates():
    assert title_candidates("Post name | Site", None) == {"post name site", "post name",
                                                          "site"}
    assert title_candidates("X", "Y") == {"x", "y"}
    assert title_candidates(None, None) == set()
    assert title_candidates(" | ", None) == set()


def test_title_match():
    assert title_match("Post name", "Post name | Site", None) == 1.0
    assert title_match("POST  NAME!", "post name", None) == 1.0
    assert title_match("Post name", None, "Post name") == 1.0
    assert title_match("Post name", None, None) == 0.0
    assert title_match("Jake Zard", "Player Info: Jake Zard", None) < SIM


def test_title_match_threshold():
    # 8 of 10 chars shared -> ratio 2*8/(10+10) = 0.8, the boundary is inclusive
    assert title_match("abcdefghij", "abcdefghXY", None) == pytest.approx(SIM)
    assert title_match("abcdefghij", "abcdefgXYZ", None) < SIM


def test_page_titles():
    root = parse("<html><head><title> A \n  title </title><title>Second</title>"
                 "<meta property='og:title' content=' The   og '></head><body></body></html>")
    assert page_titles(root) == ("A title", "The og")
    root = parse("<html><head><meta name='og:title' content='Named'></head>"
                 "<body><p>x</p></body></html>")
    assert page_titles(root) == (None, "Named")
    assert page_titles(parse("<html><body><p>x</p></body></html>")) == (None, None)


def _blocks(html):
    root = parse(html)
    lines, line_src = render_tree(root)
    kept = [(ln.rstrip(), src) for ln, src in zip(lines, line_src, strict=True) if ln.strip()]
    return root, [ln for ln, _ in kept], [src for _, src in kept]


def _page(body, title="Post name | Site"):
    return f"<html><head><title>{title}</title></head><body>{body}</body></html>"


@pytest.mark.parametrize("tag", ["h1", "h3", "h6", "p"])
def test_find_title_tags(tag):
    root, blocks, src = _blocks(_page(f"<{tag}>Post <b>name</b></{tag}><p>Body text.</p>"))
    el = find_title(root, 0, "Post name", src)
    assert el is not None and el.tag == tag


@pytest.mark.parametrize("body", [
    "<div>Post name</div><p>Body.</p>",
    "<ul><li>Post name</li></ul><p>Body.</p>",
    "<span>Post name</span><p>Body.</p>",
])
def test_find_title_other_tags(body):
    root, _, src = _blocks(_page(body))
    assert find_title(root, 0, "Post name", src) is None


def test_find_title_no_match():
    root, _, src = _blocks(_page("<h1>Something else</h1><p>Body.</p>"))
    assert find_title(root, 0, "Something else", src) is None


def test_find_title_no_page_title():
    root, _, src = _blocks("<html><body><h1>Post name</h1><p>Body.</p></body></html>")
    assert find_title(root, 0, "Post name", src) is None


def test_find_title_multi_line_element():
    root, blocks, src = _blocks(_page("<p>Post name<br>Body text that follows.</p>"))
    assert len(blocks) == 2
    assert find_title(root, 0, "Post name", src) is None


def test_find_title_block_inside_heading():
    root, blocks, src = _blocks(_page("<h1>Post name<div>More</div></h1><p>Body.</p>"))
    assert blocks[0] == "# Post name"
    assert find_title(root, 0, "Post name", src) is None


def test_find_title_tail_text_is_not_inside():
    root, blocks, src = _blocks(_page("<div><h1>Post name</h1>Loose body text.</div>"))
    assert blocks == ["# Post name", "Loose body text."]
    assert find_title(root, 0, "Post name", src).tag == "h1"


def test_find_title_paragraph_in_list_item():
    root, blocks, src = _blocks(_page("<ul><li><p>Post name</p></li></ul><p>Body.</p>"))
    assert blocks[0] == "- Post name"
    assert find_title(root, 0, "- Post name", src).tag == "p"
