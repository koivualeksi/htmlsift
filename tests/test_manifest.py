"""Manifest / artifacts unit tests (no bundle): manifest parsing, z-score tuple assembly,
and local bundle path resolution."""
import json

import pytest

from htmlsift._artifacts import HEAD, TABLE, Manifest, resolve_bundle
from htmlsift.core.features import _zscore_idx
from htmlsift.loader import _zscore_from_manifest


def test_manifest_from_file(tmp_path):
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps({"mode": "mini", "feats": "ABC",
                             "zscore": {"mean": [1.0, 2.0], "std": [3.0, 4.0]}, "cap": 0}),
                 encoding="utf-8")
    m = Manifest.from_file(p)
    assert m.mode == "mini" and m.feats == "ABC"
    assert m.window == 8192 and m.cap == 0 and m.hidden is None     # defaults / absent
    assert m.zscore_mean == [1.0, 2.0] and m.zscore_std == [3.0, 4.0]


def test_zscore_from_manifest_mini():
    m = Manifest(mode="mini", feats="ABC", zscore_mean=[1.0, 2.0], zscore_std=[3.0, 4.0])
    idx, mean, std = _zscore_from_manifest(m)
    assert idx == _zscore_idx("ABC")
    assert list(mean) == [1.0, 2.0] and list(std) == [3.0, 4.0]


def test_zscore_from_manifest_text_only_is_none():
    assert _zscore_from_manifest(Manifest(mode="base", feats=None)) is None


def test_zscore_from_manifest_length_mismatch_raises():
    m = Manifest(mode="mini", feats="ABC", zscore_mean=[1.0], zscore_std=[3.0])
    with pytest.raises(ValueError):
        _zscore_from_manifest(m)


def test_resolve_bundle_local_dir(tmp_path):
    mini = resolve_bundle("mini", local_dir=tmp_path)
    assert mini["table"] == tmp_path / TABLE and mini["head"] == tmp_path / HEAD
    assert set(mini) == {"table", "head", "tokenizer", "manifest"}
    assert set(resolve_bundle("base", local_dir=tmp_path)) == {"config", "weights",
                                                               "tokenizer", "manifest"}
