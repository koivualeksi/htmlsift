"""html/markdown output sanitizing, unit and through extract() (no bundle)."""
import pytest
from lxml import html as lxml_html

from htmlsift._sanitize import _scheme, sanitize
from htmlsift.core.render import SKIP_TAGS, parse


def _clean(body):
    root = parse(f"<html><body>{body}</body></html>")
    sanitize(root)
    return lxml_html.tostring(root.find("body"), encoding="unicode")[6:-7]


@pytest.mark.parametrize("tag", ["script", "style"])
def test_skip_tags_dropped_tail_kept(tag):
    assert _clean(f"<p>a<{tag}>x</{tag}>b</p>") == "<p>ab</p>"


@pytest.mark.parametrize("tag", ["meta", "link", "base"])
def test_void_skip_tags_dropped(tag):
    assert _clean(f'<p>a<{tag} href="https://e.com" content="0;url=x">b</p>') == "<p>ab</p>"


def test_title_in_body_dropped():
    assert "x" not in _clean("<div>a<title>x</title>b</div>")


def test_drop_set_is_skip_tags():
    assert {"script", "style", "base", "meta", "link"} <= SKIP_TAGS


def test_script_text_gone():
    assert _clean("<div><p>a</p><script>steal()</script>tail</div>") == "<div><p>a</p>tail</div>"


def test_nested_script_in_kept_element():
    assert "steal" not in _clean("<p>a<span>b<script>steal()</script></span>c</p>")


def test_svg_kept_without_animation():
    out = _clean('<svg><a href="https://ok.example"><set attributeName="href" '
                 'to="javascript:steal()"/><animate attributeName="href" values="javascript:x"/>'
                 '<circle r="1"/></a><script>x()</script></svg>')
    assert "<svg>" in out and "<circle" in out and 'href="https://ok.example"' in out
    assert "set" not in out and "animate" not in out and "javascript" not in out
    assert "<script" not in out


@pytest.mark.parametrize("attr", ["onclick", "onload", "ONERROR", "onmouseover"])
def test_event_handlers_removed(attr):
    assert _clean(f'<p {attr}="steal()" class="c">x</p>') == '<p class="c">x</p>'


def test_srcdoc_removed_iframe_kept():
    out = _clean('<iframe src="https://www.youtube.com/embed/x" srcdoc="<script>x()</script>">'
                 "</iframe>")
    assert out == '<iframe src="https://www.youtube.com/embed/x"></iframe>'


def test_non_url_attributes_untouched():
    out = _clean('<a href="/b" title="javascript: the good parts" data-x="javascript:y">t</a>')
    assert 'title="javascript: the good parts"' in out and 'data-x="javascript:y"' in out


@pytest.mark.parametrize("href", [
    "javascript:steal()", "JaVaScRiPt:steal()", "  javascript:steal()", "java\tscript:x",
    "java\nscript:x", "\x01javascript:x", "&#x6A;avascript:x", "javascript&colon;x",
    "vbscript:x", "data:text/html,<script>x()</script>", "data:image/png;base64,AAAA",
    "file:///etc/passwd", "sms:+358401234567", "intent://x",
])
def test_disallowed_href_removed_text_kept(href):
    root = parse(f'<html><body><a href="{href}">t</a></body></html>')
    a = root.find(".//a")
    assert a.get("href") is not None
    sanitize(root)
    assert a.get("href") is None and a.text == "t"


@pytest.mark.parametrize("href", [
    "https://example.com/a", "http://example.com", "HTTPS://EXAMPLE.COM", "mailto:a@example.com",
    "tel:+358401234567", "ftp://example.com/f", "/about", "page.html", "#top", "?q=1",
    "//cdn.example.com/x", "",
])
def test_allowed_href_kept(href):
    root = parse("<html><body><a>t</a></body></html>")
    a = root.find(".//a")
    a.set("href", href)
    sanitize(root)
    assert a.get("href") == href


@pytest.mark.parametrize("src, kept", [
    ("data:image/png;base64,AAAA", True),
    ("DATA:image/gif;base64,R0lG", True),
    ("data:image/svg+xml,<svg/>", True),
    ("data:text/html,x", False),
    ("javascript:x", False),
    ("https://example.com/i.png", True),
])
def test_img_src(src, kept):
    root = parse("<html><body><p><img alt=a></p></body></html>")
    img = root.find(".//img")
    img.set("src", src)
    sanitize(root)
    assert (img.get("src") == src) is kept


def test_data_image_only_in_img_src():
    out = _clean('<iframe src="data:image/png;base64,AAAA"></iframe>'
                 '<a href="data:image/png;base64,AAAA">t</a>')
    assert "data:" not in out


@pytest.mark.parametrize("snippet", [
    '<form action="javascript:x"><button formaction="javascript:y">b</button></form>',
    '<video poster="javascript:x"></video>',
    '<blockquote cite="javascript:x">q</blockquote>',
    '<svg><a xlink:href="javascript:x">t</a></svg>',
    '<table background="javascript:x"><tr><td>c</td></tr></table>',
])
def test_other_url_attributes(snippet):
    out = _clean(snippet)
    assert "javascript" not in out


def test_root_never_dropped_and_comments_ignored():
    root = parse("<html><body><!-- c --><p>x</p></body></html>")
    sanitize(root)
    assert root.tag == "html" and root.find(".//p").text == "x"


def test_clean_tree_unchanged():
    body = ('<h1 id="t">T</h1><p class="c">a <a href="https://e.com/x">l</a> '
            '<img src="i.png" alt="i"></p><ul><li>x</li></ul>')
    assert _clean(body) == body


@pytest.mark.parametrize("url, scheme", [
    ("https://x", "https"), ("/a", None), ("a:b", "a"), ("1a:b", None), (" \x00JS:x", "js"),
])
def test_scheme(url, scheme):
    assert _scheme(url) == scheme


DIRTY = ("<html><head><title>T</title></head><body>"
         '<p onclick="steal()">Intro <a href="javascript:steal()">click</a> and '
         '<a href="https://e.com/ok">ok</a>.<script>steal()</script></p>'
         '<p>Pics <img src="javascript:x" alt="bad"><img src="data:image/png;base64,AAAA" '
         'alt="pix"></p><iframe srcdoc="<script>x()</script>" src="https://v.example/e">'
         "fallback</iframe></body></html>")


def test_extract_html_sanitized(keep_all):
    out = keep_all.extract(DIRTY, output="html")
    for bad in ("javascript", "onclick", "<script", "srcdoc", "steal"):
        assert bad not in out
    assert 'href="https://e.com/ok"' in out and 'src="https://v.example/e"' in out
    assert 'src="data:image/png;base64,AAAA"' in out and ">click</a>" in out


def test_extract_markdown_sanitized(keep_all):
    md = keep_all.extract(DIRTY, output="markdown")
    assert "javascript" not in md and "steal" not in md
    assert "click" in md and "[ok](https://e.com/ok)" in md
    assert "![pix](data:image/png;base64,AAAA)" in md


def test_extract_text_unaffected(keep_all):
    assert keep_all.extract(DIRTY) == "Intro click and ok.\nPics\nfallback"
