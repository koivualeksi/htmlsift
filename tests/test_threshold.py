"""threshold=: lines with probability above it are kept, in every output."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

import htmlsift.extractor as extractor_mod
from htmlsift import Extractor, extract

PAGE = "<html><body><p>One.</p><p>Two.</p><p>Three.</p><p>Four.</p></body></html>"
PROBS = [0.9, 0.5, 0.3, 0.1]


@pytest.fixture
def ex(monkeypatch):
    """An Extractor whose dummy model scores the lines 0.9, 0.5, 0.3, 0.1."""
    monkeypatch.setattr("htmlsift.extractor.prep_page", lambda tok, blocks, *a, **k: blocks)
    ex = Extractor()
    ex._model = SimpleNamespace(
        features=lambda lines, src: None, tok=None, window=0, cap=0,
        infer_fn=lambda blocks: np.array(PROBS[:len(blocks)], dtype=np.float32))
    return ex


@pytest.mark.parametrize("threshold, kept", [
    (0.5, ["One."]), (0.2, ["One.", "Two.", "Three."]), (0.0, ["One.", "Two.", "Three.", "Four."]),
    (0.95, []), (1, [])])
def test_text(ex, threshold, kept):
    assert ex.extract(PAGE, threshold=threshold) == "\n".join(kept)


def test_default_is_half(ex):
    assert ex.extract(PAGE) == ex.extract(PAGE, threshold=0.5) == "One."


def test_equal_probability_dropped(ex):
    assert "Two." not in ex.extract(PAGE, threshold=0.5)


@pytest.mark.parametrize("output", ["html", "markdown"])
def test_html_and_markdown(ex, output):
    out = ex.extract(PAGE, output, threshold=0.2)
    assert all(w in out for w in ("One.", "Two.", "Three.")) and "Four." not in out


def test_numpy_scalar(ex):
    assert ex.extract(PAGE, threshold=np.float32(0.2)) == "One.\nTwo.\nThree."


def test_with_blocks_unthresholded(ex):
    blocks, probs = ex.extract(PAGE, with_blocks=True, threshold=0.95)
    assert blocks == ["One.", "Two.", "Three.", "Four."] and len(probs) == 4


def test_drop_title_uses_threshold(ex):
    page = ("<html><head><title>A page title</title></head><body>"
            "<h1>A page title</h1><p>Body.</p></body></html>")
    ex._model.infer_fn = lambda blocks: np.array([0.3, 0.9], dtype=np.float32)
    assert ex.extract(page, threshold=0.2) == "A page title\nBody."
    assert ex.extract(page, drop_title=True, threshold=0.2) == "Body."


def test_dedup_uses_threshold(ex):
    long = "A paragraph long enough to count as a repeat, " * 3
    page = f"<html><body><p>{long}</p><p>{long}</p></body></html>"
    ex._model.infer_fn = lambda blocks: np.array([0.3, 0.9], dtype=np.float32)
    assert ex.extract(page, dedup=True, threshold=0.5) == long.strip()
    assert ex.extract(page, dedup=True, threshold=0.2) == long.strip()
    assert ex.extract(page, threshold=0.2).count(long.strip()) == 2


@pytest.mark.parametrize("bad", [-0.1, 1.01, math.nan, math.inf])
def test_out_of_range(ex, bad):
    with pytest.raises(ValueError, match="threshold"):
        ex.extract(PAGE, threshold=bad)


@pytest.mark.parametrize("bad", [True, "0.5", None])
def test_not_a_number(ex, bad):
    with pytest.raises(TypeError, match="threshold"):
        ex.extract(PAGE, threshold=bad)


def test_module_extract_passes_it(monkeypatch, ex):
    monkeypatch.setitem(extractor_mod._CACHE, "mini", ex)
    assert extract(PAGE, threshold=0.2) == "One.\nTwo.\nThree."
