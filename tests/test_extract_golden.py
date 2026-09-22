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
