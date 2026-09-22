"""features unit tests (no bundle): structural feature extraction from a rendered tree,
column-count/z-score index math, and z-score application."""
import numpy as np

from htmlsift.core.features import GROUPS, _zscore_idx, apply_zscore, collect_features, feature_dim
from htmlsift.core.render import parse, render_tree

HTML = "<main><p>Plain text here</p><p>See <a href='/x'>the link</a></p></main>"


def test_collect_features_shape_and_has_link():
    lines, line_src = render_tree(parse(HTML))
    fa = collect_features(lines, line_src, "ABC")
    cols = [c for g in ("A", "B", "C") for c in GROUPS[g]]
    assert fa.shape == (2, feature_dim("ABC"))          # two blocks x 33 columns
    hl, tp = cols.index("has_link"), cols.index("tag_p")
    assert fa[0, hl] == 0.0 and fa[1, hl] == 1.0        # link only in the second block
    assert fa[0, tp] == 1.0 and fa[1, tp] == 1.0        # both are <p>


def test_feature_dim():
    assert feature_dim("ABC") == 33
    assert feature_dim("BC") == 3
    assert feature_dim(None) == 0 and feature_dim("") == 0


def test_zscore_idx_matches_layout():
    cols = [c for g in ("A", "B", "C") for c in GROUPS[g]]
    assert _zscore_idx("ABC") == [cols.index("depth"), cols.index("log_chars")]
    assert _zscore_idx("C") == [0]


def test_apply_zscore_normalizes_only_named_columns_and_copies():
    fa = np.array([[1.0, 5.0, 2.0]], dtype=np.float32)
    stats = ([0, 2], np.array([1.0, 2.0], np.float32), np.array([2.0, 4.0], np.float32))
    out = apply_zscore(fa, stats)
    assert out.tolist() == [[0.0, 5.0, 0.0]]            # cols 0,2 normalized; col 1 untouched
    assert fa.tolist() == [[1.0, 5.0, 2.0]]             # input not mutated
