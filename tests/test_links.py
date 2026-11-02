"""Base URL resolution, link rewriting, and base_url through extract() (no bundle)."""
import pytest

from htmlsift._links import effective_base, make_absolute
from htmlsift.core.render import parse

PAGE = "https://example.com/blog/post"


def _root(head=""):
    return parse(f"<html><head>{head}</head><body><p>x</p></body></html>")


def test_no_base_url_no_base_tag():
    assert effective_base(_root()) is None


def test_base_url_only():
    assert effective_base(_root(), PAGE) == PAGE


def test_absolute_base_tag_overrides_base_url():
    root = _root('<base href="https://cdn.example.com/assets/">')
    assert effective_base(root, PAGE) == "https://cdn.example.com/assets/"


def test_relative_base_tag_resolves_against_base_url():
    assert effective_base(_root('<base href="/sub/">'), PAGE) == "https://example.com/sub/"


def test_first_base_tag_wins():
    root = _root('<base href="https://a.example/"><base href="https://b.example/">')
    assert effective_base(root, PAGE) == "https://a.example/"
    assert effective_base(root) == "https://a.example/"


def test_base_tag_in_body_counts():
    root = parse('<html><body><p>x</p><base href="https://a.example/"></body></html>')
    assert effective_base(root) == "https://a.example/"


@pytest.mark.parametrize("head", ['<base href="">', '<base href="  ">', "<base target=_blank>"])
def test_empty_or_missing_href_ignored(head):
    assert effective_base(_root(head), PAGE) == PAGE
    assert effective_base(_root(head)) is None


def test_href_whitespace_stripped():
    assert effective_base(_root('<base href=" https://a.example/ ">')) == "https://a.example/"


def test_absolute_base_tag_alone():
    assert effective_base(_root('<base href="https://a.example/x/">')) == "https://a.example/x/"


@pytest.mark.parametrize("href", ["/sub/", "//cdn.example.com/", "javascript:void(0)",
                                  "data:text/html,x", "http://[bad"])
def test_non_absolute_http_base_tag_alone_ignored(href):
    assert effective_base(_root(f'<base href="{href}">')) is None


@pytest.mark.parametrize("href", ["http://[bad", "javascript:void(0)", "data:text/html,x"])
def test_unusable_base_tag_falls_back_to_base_url(href):
    assert effective_base(_root(f'<base href="{href}">'), PAGE) == PAGE


def test_protocol_relative_base_tag_takes_base_url_scheme():
    root = _root('<base href="//cdn.example.com/">')
    assert effective_base(root, PAGE) == "https://cdn.example.com/"


def test_malformed_base_url():
    assert effective_base(_root(), "http://[bad") is None
    assert effective_base(_root('<base href="https://a.example/">'), "http://[bad") is None


def test_empty_base_url_is_none():
    assert effective_base(_root('<base href="/sub/">'), "") is None
    assert effective_base(_root('<base href="https://a.example/">'), "") == "https://a.example/"


def _hrefs(root):
    return {el.get("id"): el.get("href") or el.get("src") or el.get("action")
            for el in root.iter() if el.get("id")}


def test_make_absolute_rewrites_relative_links():
    root = parse('<p><a id="a" href="report">r</a><img id="i" src="/i.png">'
                 '<form id="f" action="send"></form></p>')
    make_absolute(root, PAGE)
    assert _hrefs(root) == {"a": "https://example.com/blog/report",
                            "i": "https://example.com/i.png",
                            "f": "https://example.com/blog/send"}


def test_make_absolute_skips_malformed_link():
    root = parse('<html><head><base href="https://cdn.example.com/"></head><body>'
                 '<a id="bad" href="http://[bad">b</a><a id="ok" href="x">x</a></body></html>')
    make_absolute(root, PAGE)
    assert _hrefs(root) == {"bad": "http://[bad", "ok": "https://example.com/blog/x"}


@pytest.mark.parametrize("href, expected", [
    ("#top", "https://example.com/blog/post#top"),
    ("mailto:a@example.com", "mailto:a@example.com"),
    ("javascript:void(0)", "javascript:void(0)"),
])
def test_make_absolute_special_links(href, expected):
    root = parse(f'<p><a id="a" href="{href}">a</a></p>')
    make_absolute(root, PAGE)
    assert _hrefs(root)["a"] == expected


def test_make_absolute_leaves_srcset():
    root = parse('<p><img src="a.png" srcset="a-2x.png 2x"></p>')
    make_absolute(root, PAGE)
    assert root.find(".//img").get("srcset") == "a-2x.png 2x"


def _page(head=""):
    return (f"<html><head>{head}</head><body><p>See <a href=\"report\">the report</a>.</p>"
            '<p>The chart: <img src="i.png" alt="chart"></p></body></html>')


def test_markdown_links_absolute(keep_all):
    md = keep_all.extract(_page(), output="markdown", base_url=PAGE)
    assert "[the report](https://example.com/blog/report)" in md
    assert "![chart](https://example.com/blog/i.png)" in md


def test_html_links_absolute(keep_all):
    out = keep_all.extract(_page(), output="html", base_url=PAGE)
    assert 'href="https://example.com/blog/report"' in out
    assert 'src="https://example.com/blog/i.png"' in out


def test_absolute_base_tag_without_base_url(keep_all):
    md = keep_all.extract(_page('<base href="https://cdn.example.com/x/">'), output="markdown")
    assert "[the report](https://cdn.example.com/x/report)" in md
    assert "![chart](https://cdn.example.com/x/i.png)" in md


def test_relative_base_tag_with_base_url(keep_all):
    md = keep_all.extract(_page('<base href="/sub/">'), output="markdown", base_url=PAGE)
    assert "[the report](https://example.com/sub/report)" in md


def test_links_unchanged_without_any_base(keep_all):
    md = keep_all.extract(_page(), output="markdown")
    assert "[the report](report)" in md and "![chart](i.png)" in md


def test_text_unchanged_by_base_url(keep_all):
    assert (keep_all.extract(_page(), base_url=PAGE) == keep_all.extract(_page())
            == "See the report.\nThe chart:")


@pytest.mark.parametrize("out", ["html", "markdown"])
def test_base_tag_with_malformed_link(keep_all, out):
    page = ('<html><head><base href="/sub/"></head><body>'
            '<p><a href="http://[bad">bad</a> and <a href="ok">ok</a></p></body></html>')
    result = keep_all.extract(page, output=out, base_url=PAGE)
    assert "https://example.com/sub/ok" in result and "http://[bad" in result
