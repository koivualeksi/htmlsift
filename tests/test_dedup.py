"""dedup=True: later selected repeats of a long line are left out, in every output."""
from types import SimpleNamespace

import numpy as np
import pytest

from htmlsift import Extractor

LONG = "A paragraph long enough to count as a repeat, " * 3
OTHER = "Another paragraph that is long enough to be deduped as well, " * 2
JA = "これは重複を検出するのに十分な長さの日本語の段落です。" * 4


def _page(body):
    return f"<html><body>{body}</body></html>"


@pytest.fixture
def drop_first(monkeypatch):
    """An Extractor whose dummy model drops the first line, keeps the rest."""
    monkeypatch.setattr("htmlsift.extractor.prep_page", lambda tok, blocks, *a, **k: blocks)
    ex = Extractor()
    ex._model = SimpleNamespace(
        features=lambda lines, src: None, tok=None, window=0, cap=0,
        infer_fn=lambda blocks: np.array([0.0] + [1.0] * (len(blocks) - 1),
                                         dtype=np.float32))
    return ex


def test_keeps_first_copy(keep_all):
    page = _page(f"<p>{LONG}</p><p>Middle.</p><p>{LONG}</p><p>{OTHER}</p>")
    assert keep_all.extract(page, dedup=True).split("\n") == [
        LONG.strip(), "Middle.", OTHER.strip()]


@pytest.mark.parametrize("output", ["markdown", "html"])
def test_markers_case_and_punctuation_ignored(keep_all, output):
    page = _page(f"<h2>{LONG}</h2><ul><li>{LONG.upper()}!</li></ul><p>End.</p>")
    out = keep_all.extract(page, output, dedup=True)
    assert LONG.strip() in out and LONG.upper().strip() not in out and "End." in out


def test_non_latin(keep_all):
    assert keep_all.extract(_page(f"<p>{JA}</p><p>{JA}</p>"), dedup=True) == JA


def test_short_repeat_kept(keep_all):
    page = _page("<p>Read more.</p><p>Body.</p><p>Read more.</p>")
    assert keep_all.extract(page, dedup=True) == "Read more.\nBody.\nRead more."


def test_unselected_first_copy(drop_first):
    page = _page(f"<p>{LONG}</p><p>{LONG}</p><p>{LONG}</p>")
    assert drop_first.extract(page, dedup=True) == LONG.strip()


@pytest.mark.parametrize("output", ["text", "markdown", "html"])
def test_pre_left_alone(keep_all, output):
    page = _page(f"<pre>{LONG}\n{LONG}</pre><p>{LONG}</p>")
    assert keep_all.extract(page, output, dedup=True).count(LONG.strip()) == 3


@pytest.mark.parametrize("output", ["text", "markdown", "html"])
def test_off_by_default(keep_all, output):
    page = _page(f"<p>{LONG}</p><p>{LONG}</p>")
    assert keep_all.extract(page, output).count(LONG.strip()) == 2


def test_with_drop_title(keep_all):
    page = (f"<html><head><title>{LONG}</title></head><body>"
            f"<h1>{LONG}</h1><p>{LONG}</p><p>End.</p></body></html>")
    assert keep_all.extract(page, drop_title=True, dedup=True) == f"{LONG.strip()}\nEnd."


def test_ignored_with_blocks(keep_all):
    page = _page(f"<p>{LONG}</p><p>{LONG}</p>")
    blocks, probs = keep_all.extract(page, with_blocks=True, dedup=True)
    assert len(blocks) == 2 and (probs == 1.0).all()
