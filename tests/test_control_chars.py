"""C0 control characters (no bundle): parse() removes them, so html/markdown output no
longer fails writing them back into the tree, and every platform gives the same output."""
import re
from types import SimpleNamespace

import numpy as np
import pytest

from htmlsift import Extractor
from htmlsift.core.render import parse

C0 = [chr(c) for c in range(1, 0x20) if c not in (0x09, 0x0A, 0x0D)]
BAD = re.compile("[\x01-\x08\x0b\x0c\x0e-\x1f]")


def _strings(root):
    for el in root.iter():
        yield el.text or ""
        yield el.tail or ""
        if isinstance(el.tag, str):
            yield from el.attrib.values()


def _clean(html):
    return not any(BAD.search(s) for s in _strings(parse(html)))


@pytest.mark.parametrize("c", C0)
@pytest.mark.parametrize("form", ["raw", "dec", "hex", "padded", "bare"])
def test_parse_removes_c0(c, form):
    n = ord(c)
    ref = {"raw": c, "dec": f"&#{n};", "hex": f"&#x{n:X};", "padded": f"&#000{n};",
           "bare": f"&#{n} "}[form]
    html = f'<html><body><p title="a{ref}b">x{ref}y<b>z</b>{ref}w</p></body></html>'
    assert _clean(html)
    assert _clean(html.encode("utf-8"))


@pytest.mark.parametrize("ref, kept", [("\t", "\t"), ("&#9;", "\t"), ("&#32;", " "),
                                       ("&#127;", "\x7f"), ("&#310;", "Ķ"),
                                       ("&#x1F600;", "\U0001f600")])
def test_parse_keeps_allowed(ref, kept):
    p = next(parse(f"<html><body><p>a{ref}b</p></body></html>").iter("p"))
    assert p.text == f"a{kept}b"


def test_parse_nul_unchanged():
    p = next(parse("<html><body><p>a\x00b&#0;c</p></body></html>").iter("p"))
    assert "\x00" not in p.text and p.text.startswith("a") and p.text.endswith("c")


@pytest.fixture
def drop_menu(monkeypatch):
    """An Extractor whose dummy model scores lines with "Menu" 0.0, the rest 1.0."""
    monkeypatch.setattr("htmlsift.extractor.prep_page", lambda tok, blocks, *a, **k: blocks)
    ex = Extractor()
    ex._model = SimpleNamespace(
        features=lambda lines, src: None, tok=None, window=0, cap=0,
        infer_fn=lambda blocks: np.array([0.0 if "Menu" in b else 1.0 for b in blocks],
                                         dtype=np.float32))
    return ex


PAGES = {
    "tail drop": "<ul><li><div>Menu</div>{c}Real content here</li></ul>",
    "script tail": "<p>Intro text<script>x</script>{c}after script</p>",
    "pre cut": "<pre><div>Menu</div>\nkeep{c} line\nMenu more</pre>",
    "link": '<p><a href="pa{c}ge.html">Link text here</a></p>',
}


@pytest.mark.parametrize("output", ["text", "html", "markdown"])
@pytest.mark.parametrize("c", ["\x03", "\x1f", "&#3;"])
@pytest.mark.parametrize("name", PAGES)
def test_extract_same_as_without(drop_menu, name, c, output):
    page = "<html><body>" + PAGES[name] + "</body></html>"
    kw = {"base_url": "https://example.org/"} if output != "text" else {}
    got = drop_menu.extract(page.format(c=c), output=output, **kw)
    assert got == drop_menu.extract(page.format(c=""), output=output, **kw)
    assert got and not BAD.search(got)
