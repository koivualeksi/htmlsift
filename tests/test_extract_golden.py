"""Full-pipeline goldens: extract() output pinned end to end for every output mode. Needs
a local mini bundle, so skipped unless HTMLSIFT_BUNDLE points at one (e.g. localbundle/mini).
Bless the references once, then every run compares:

    HTMLSIFT_BUNDLE=...\\localbundle\\mini HTMLSIFT_BLESS=1 pytest tests/test_extract_golden.py

The references track the model in the bundle (the fABC s1 keeper, the one release.py
publishes), so they also guard the shipped mini model.
"""
import os
from pathlib import Path

import pytest

from htmlsift import Extractor

DATA = Path(__file__).parent / "data"
BUNDLE = os.environ.get("HTMLSIFT_BUNDLE")
GOLDENS = {"text": "sample.extracted.txt", "html": "sample.extracted.html",
           "markdown": "sample.extracted.md"}


@pytest.fixture(scope="module")
def extractor():
    if not BUNDLE:
        pytest.skip("set HTMLSIFT_BUNDLE to a local mini bundle dir")
    return Extractor(bundle_dir=BUNDLE)


@pytest.mark.parametrize("output", list(GOLDENS))
def test_extract_golden(extractor, output):
    html = (DATA / "sample.html").read_text(encoding="utf-8")
    got = extractor.extract(html, output=output)
    golden = DATA / GOLDENS[output]
    if os.environ.get("HTMLSIFT_BLESS"):
        golden.write_text(got, encoding="utf-8")
    assert golden.exists(), \
        f"{golden.name} missing; bless with HTMLSIFT_BUNDLE=... HTMLSIFT_BLESS=1 pytest"
    assert got == golden.read_text(encoding="utf-8")


def test_with_blocks_aligned(extractor):
    html = (DATA / "sample.html").read_text(encoding="utf-8")
    blocks, probs = extractor.extract(html, with_blocks=True)
    assert len(blocks) == len(probs) > 0
    assert all(0.0 <= p <= 1.0 for p in probs)


ENCODED_PAGES = [("article_fi.html", "windows-1252", "cp1252", "Pohjois-Karjalassa"),
                 ("article_ja.html", "shift_jis", "cp932", "アオコ")]


def _declared(name, charset):
    html = (DATA / name).read_text(encoding="utf-8")
    return html.replace('<meta charset="utf-8">', f'<meta charset="{charset}">')


@pytest.mark.parametrize("name, charset, codec, marker", ENCODED_PAGES)
@pytest.mark.parametrize("output", list(GOLDENS))
def test_bytes_match_str(extractor, name, charset, codec, marker, output):
    html = _declared(name, charset)
    ref = extractor.extract(html, output=output)
    assert marker in ref
    assert extractor.extract(html.encode(codec), output=output) == ref
    no_meta = html.replace(f'<meta charset="{charset}">', "").encode(codec)
    assert extractor.extract(no_meta, output=output, encoding=charset) == ref


@pytest.mark.parametrize("name, charset, codec, marker", ENCODED_PAGES)
def test_bytes_match_str_blocks(extractor, name, charset, codec, marker):
    html = _declared(name, charset)
    ref_blocks, ref_probs = extractor.extract(html, with_blocks=True)
    blocks, probs = extractor.extract(html.encode(codec), with_blocks=True)
    assert blocks == ref_blocks
    assert (probs == ref_probs).all()
