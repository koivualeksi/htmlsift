"""API-surface checks that run without a model bundle: empty HTML short-circuits before
the model loads, so these exercise the public contract with no network or weights.
"""
import pytest

from htmlsift import Extractor, extract
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


def test_with_blocks_keeps_markdown(keep_all):
    blocks, _ = keep_all.extract(PLAIN_PAGE, with_blocks=True)
    assert "# Title" in blocks and "Some **bold** text." in blocks
