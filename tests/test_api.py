"""API-surface checks that run without a model bundle: empty HTML short-circuits before
the model loads, so these exercise the public contract with no network or weights.
"""
import pytest

from htmlsift import Extractor, extract
from htmlsift.extractor import _visible
from htmlsift.loader import load_model


def test_empty_html_with_blocks():
    blocks, probs = Extractor().extract("", with_blocks=True)
    assert blocks == [] and len(probs) == 0


@pytest.mark.parametrize("out", ["text", "html", "markdown"])
def test_empty_all_outputs(out):
    assert Extractor().extract("", output=out) == ""


def test_unknown_output_raises():
    with pytest.raises(ValueError):
        Extractor().extract("", output="bogus")


def test_module_level_extract_empty():
    assert extract("") == ""


def test_mini_rejects_gpu():
    # mini is CPU-only; the guard raises before any bundle fetch, so no bundle/GPU needed.
    with pytest.raises(ValueError):
        load_model("mini", device="cuda")


@pytest.mark.parametrize("absent", [{"torch", "transformers"}, {"transformers"}])
def test_base_without_deps_raises_before_download(monkeypatch, absent):
    monkeypatch.setattr("htmlsift.loader.find_spec",
                        lambda name: None if name in absent else object())

    def no_download(*a, **k):
        raise AssertionError("bundle fetched")
    monkeypatch.setattr("htmlsift.loader.resolve_bundle", no_download)
    with pytest.raises(ImportError) as exc:
        load_model("base")
    msg = str(exc.value)
    assert 'pip install "htmlsift[full]"' in msg
    assert all(m in msg for m in absent)
    assert ("torch," in msg) == ("torch" in absent)


@pytest.mark.parametrize("bad", [None, bytearray(b"<p>x</p>"), 42])
def test_non_str_bytes_rejected(bad):
    with pytest.raises(TypeError):
        extract(bad)


def test_str_with_encoding_rejected():
    with pytest.raises(TypeError):
        Extractor().extract("", encoding="utf-8")


@pytest.mark.parametrize("html", [b"", b"  \n", b"\xef\xbb\xbf"])
def test_empty_bytes(html):
    assert extract(html) == ""


def test_encoding_is_keyword_only():
    with pytest.raises(TypeError):
        Extractor().extract(b"", "text", False, "utf-8")


@pytest.mark.parametrize("bad", [b"https://x.org/", 42])
def test_non_str_base_url_rejected(bad):
    with pytest.raises(TypeError):
        extract("", base_url=bad)


def test_base_url_is_keyword_only():
    with pytest.raises(TypeError):
        Extractor().extract("", "text", False, None, "https://x.org/")


PLAIN_PAGE = ("<html><body><h1>Title</h1><p>Some <b>bold</b> text.</p>"
              "<blockquote><p>Quoted.</p></blockquote><ol><li>first</li></ol>"
              "<pre>x = 1</pre><hr></body></html>")


def test_text_is_plain(keep_all):
    assert keep_all.extract(PLAIN_PAGE) == "Title\nSome bold text.\nQuoted.\n1. first\nx = 1"


def test_text_drops_invisible_lines(keep_all):
    page = ("<p>Real text.</p><p>​</p><p>‍﻿</p><p>&shy;</p>"
            "<p>ㅤ</p><p>Second.</p>")
    assert keep_all.extract(page) == "Real text.\nSecond."


KEPT_AS_IS = [
    "\U0001f468‍\U0001f469‍\U0001f467",
    "می‌خواهم",
    "क्‍ष",
    "word​word",
]


@pytest.mark.parametrize("text", KEPT_AS_IS)
def test_text_keeps_zero_width_inside_content(keep_all, text):
    assert keep_all.extract(f"<p>{text}</p>") == text


def test_text_all_invisible_is_empty(keep_all):
    assert keep_all.extract("<p>​</p><p>⠀</p>") == ""


@pytest.mark.parametrize("line", [
    "​", "‌", "‍", "⁠", "﻿", "­", "‎", "\x81",
    " ", "　", "ᅟ", "ᅠ", "ㅤ", "ﾠ", "⠀",
    " ​\t‍ ", "",
])
def test_invisible(line):
    assert not _visible(line)


@pytest.mark.parametrize("line", ["a", "-", "1.", "|", "漢", "​x"])
def test_visible(line):
    assert _visible(line)


def test_with_blocks_keeps_markdown(keep_all):
    blocks, _ = keep_all.extract(PLAIN_PAGE, with_blocks=True)
    assert "# Title" in blocks and "Some **bold** text." in blocks


def _titled(body, title="Post name | Site"):
    return (f"<html><head><title>{title}</title></head>"
            f"<body><header>{body}</header><p>Body text.</p></body></html>")


TITLE_DROPPED = {
    "text": "Body text.",
    "markdown": "Body text.",
}


@pytest.mark.parametrize("tag", ["h1", "h3", "p"])
@pytest.mark.parametrize("output", ["text", "markdown"])
def test_drop_title(keep_all, tag, output):
    page = _titled(f"<{tag}>Post name</{tag}>")
    assert keep_all.extract(page, output, drop_title=True) == TITLE_DROPPED[output]


@pytest.mark.parametrize("tag", ["h1", "h3", "p"])
def test_drop_title_html(keep_all, tag):
    html = keep_all.extract(_titled(f"<{tag}>Post name</{tag}>"), "html", drop_title=True)
    assert "Post name" not in html.split("</head>")[-1]
    assert html.count(f"<{tag}>") == (1 if tag == "p" else 0)
    assert "<p>Body text.</p>" in html


@pytest.mark.parametrize("output", ["text", "markdown", "html"])
def test_title_kept_by_default(keep_all, output):
    assert "Post name" in keep_all.extract(_titled("<h1>Post name</h1>"), output)


@pytest.mark.parametrize("body", [
    "<h1>Another heading</h1>",             # no match
    "<div>Post name</div>",                 # not h1-h6 or p
    "<ul><li>Post name</li></ul>",
    "<p>Intro.</p><h1>Post name</h1>",      # not the first line
    "<p>Post name<br>More body.</p>",       # element holds more lines
])
@pytest.mark.parametrize("output", ["text", "markdown", "html"])
def test_drop_title_keeps(keep_all, body, output):
    page = _titled(body)
    assert (keep_all.extract(page, output, drop_title=True)
            == keep_all.extract(page, output))


def test_drop_title_skips_invisible_lead(keep_all):
    page = _titled("<p>​</p><h1>Post name</h1>")
    assert keep_all.extract(page, drop_title=True) == "Body text."


def test_drop_title_skips_marker_only_lead(keep_all):
    page = _titled("<h2> </h2><pre></pre><h1>Post name</h1>")
    assert keep_all.extract(page, drop_title=True) == "Body text."


def test_drop_title_og_title(keep_all):
    page = ("<html><head><meta property='og:title' content='Post name'></head>"
            "<body><h2>Post name</h2><p>Body text.</p></body></html>")
    assert keep_all.extract(page, drop_title=True) == "Body text."


@pytest.mark.parametrize("output", ["text", "markdown", "html"])
def test_drop_title_only_title(keep_all, output):
    page = "<html><head><title>Post name</title></head><body><h1>Post name</h1></body></html>"
    assert keep_all.extract(page, output, drop_title=True) == ""


def test_drop_title_kept_ancestor(keep_all):
    # the div's own text puts the div in the selection; its h1 child must still go
    page = ("<html><head><title>Post name</title></head><body>"
            "<div><h1>Post name</h1>Loose body text.</div></body></html>")
    html = keep_all.extract(page, "html", drop_title=True)
    assert "<h1>" not in html and "Loose body text." in html
    assert keep_all.extract(page, "markdown", drop_title=True) == "Loose body text."


def test_drop_title_ignored_with_blocks(keep_all):
    page = _titled("<h1>Post name</h1>")
    assert (keep_all.extract(page, with_blocks=True, drop_title=True)[0]
            == keep_all.extract(page, with_blocks=True)[0])


def test_drop_title_is_keyword_only():
    with pytest.raises(TypeError):
        Extractor().extract("", "text", False, None, None, True)


def test_module_extract_passes_drop_title():
    assert extract("", drop_title=True) == ""
