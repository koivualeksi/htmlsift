"""Render output and structural features (no bundle) on an authored page, against
hand-derived values. Locks the model input: drift in the renderer's walk or the
feature manifest fails here."""
import math
from pathlib import Path

import pytest

from htmlsift.core.features import FEATURE_NAMES, GROUPS, ZSCORE_COLS, collect_features
from htmlsift.core.render import parse, render_tree

DATA = Path(__file__).parent / "data"

# (block text, group-A tags that must be 1, mean DOM depth, has_link) for basic.html.
# log_chars is recomputed from the block text, which also checks row alignment.
EXPECTED = [
    ("# Main Title",                   {"main", "h1"},       4.0, 0),
    ("Intro with a link here inside.", {"main", "p"},        4.5, 1),
    ("Nested paragraph.",              {"main", "div", "p"}, 6.0, 0),
    ("- First item",                   {"main", "ul", "li"}, 5.0, 0),
    ("- Second item",                  {"main", "ul", "li"}, 5.0, 0),
    ("Site footer text",               {"footer"},           3.0, 0),
]

A_COLS = [n for n in FEATURE_NAMES if n.startswith("tag_")]


@pytest.fixture(scope="module")
def rendered():
    html = (DATA / "basic.html").read_text(encoding="utf-8")
    lines, line_src = render_tree(parse(html))
    feats = collect_features(lines, line_src, "ABC")
    blocks = [ln.rstrip() for ln in lines if ln.strip()]
    return blocks, feats


def test_render_blocks(rendered):
    blocks, _ = rendered
    assert blocks == [text for text, *_ in EXPECTED]


def test_render_deterministic():
    html = (DATA / "basic.html").read_text(encoding="utf-8")
    first, _ = render_tree(parse(html))
    second, _ = render_tree(parse(html))
    assert first == second


def test_feature_manifest():
    assert FEATURE_NAMES == GROUPS["A"] + GROUPS["B"] + GROUPS["C"]
    assert len(A_COLS) == 30 and A_COLS == GROUPS["A"]
    assert GROUPS["B"] == ["depth", "has_link"]
    assert GROUPS["C"] == ["log_chars"]
    assert ZSCORE_COLS == ["depth", "log_chars"]


def test_feature_values(rendered):
    blocks, feats = rendered
    assert feats.shape == (len(EXPECTED), len(FEATURE_NAMES))
    idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
    for bi, (text, a_tags, depth, has_link) in enumerate(EXPECTED):
        row = feats[bi]
        for col in A_COLS:
            expected = 1.0 if col[len("tag_"):] in a_tags else 0.0
            assert row[idx[col]] == expected, (bi, col)
        assert row[idx["depth"]] == pytest.approx(depth)
        assert row[idx["has_link"]] == has_link
        assert row[idx["log_chars"]] == pytest.approx(math.log1p(len(text)))
