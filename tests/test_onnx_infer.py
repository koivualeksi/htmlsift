"""The mini forward: embedding distinct ids once must pool to exactly what embedding every
token did, and only big pages shrink the arena. The real-model check needs a local bundle
(HTMLSIFT_BUNDLE)."""
import os
from pathlib import Path

import numpy as np
import pytest

from htmlsift.core import onnx_infer
from htmlsift.core.onnx_infer import arena_mb, load_mini, pool

DATA = Path(__file__).parent / "data"
BUNDLE = os.environ.get("HTMLSIFT_BUNDLE")
SHRINK_KEY = "memory.enable_memory_arena_shrinkage"


def _full_pool(emb, members):
    """The pre-dedup pooling: every token embedded, then a per-block mean."""
    out = np.zeros((len(members), emb.shape[1]), dtype=np.float32)
    for i, mem in enumerate(members):
        if mem:
            out[i] = emb[mem].mean(0)
    return out


def test_dedup_pool_matches_full_pool():
    rng = np.random.default_rng(0)
    table = rng.standard_normal((500, 16)).astype(np.float32)
    ids = rng.integers(0, 50, size=3000).astype(np.int64)        # heavy repetition
    members = [[], [0, 1, 2], [2, 3], list(range(4, 2500)), [], list(range(2500, 3000))]
    uniq, inv = np.unique(ids, return_inverse=True)
    assert np.array_equal(pool(table[uniq], inv, members), _full_pool(table[ids], members))


class _FakeSession:
    """Records each run's feed and run options; returns zeros of the right shape."""
    runs = []

    def __init__(self, path, *a, **k):
        self.kind = "table" if "table" in path else "head"

    def run(self, names, feed, run_options=None):
        _FakeSession.runs.append((self.kind, feed, run_options))
        if self.kind == "table":
            return [np.zeros((len(feed["ids"]), 4), dtype=np.float32)]
        return [np.zeros((1, feed["x"].shape[1]), dtype=np.float32)]


@pytest.fixture
def fake_infer(monkeypatch):
    monkeypatch.setattr(onnx_infer.ort, "InferenceSession", _FakeSession)
    _FakeSession.runs = []
    return load_mini("table.onnx", "head.onnx")


def _page(ids, n_blocks):
    ids = np.asarray(ids, dtype=np.int64)
    return {"ids": ids, "members": [[0]] * n_blocks, "feats": None}


def _shrunk():
    """Per run, whether arena shrinkage was requested."""
    return [ro is not None and ro.get_run_config_entry(SHRINK_KEY) == "cpu:0"
            for _, _, ro in _FakeSession.runs]


def test_table_runs_once_on_distinct_ids(fake_infer):
    fake_infer(_page([7, 3, 7, 7, 3, 9], 2))
    (kind, feed, _), _ = _FakeSession.runs
    assert kind == "table" and feed["ids"].tolist() == [3, 7, 9]
    assert feed["ids"].dtype == np.int64


def test_small_page_keeps_arena(fake_infer):
    n = 4800
    assert arena_mb(1, n) < onnx_infer.SHRINK_MB
    fake_infer(_page([0], n))
    assert _shrunk() == [False, False]


@pytest.mark.parametrize("n_unique, n_blocks", [(1, 5000), (9000, 1)])
def test_big_page_shrinks_arena(fake_infer, n_unique, n_blocks):
    assert arena_mb(n_unique, n_blocks) > onnx_infer.SHRINK_MB
    fake_infer(_page(np.arange(n_unique), n_blocks))
    assert _shrunk() == [True, True]


@pytest.mark.skipif(not BUNDLE, reason="set HTMLSIFT_BUNDLE to a local mini bundle dir")
def test_real_model_matches_full_table():
    import onnxruntime as ort

    from htmlsift import Extractor
    from htmlsift._artifacts import HEAD, TABLE
    from htmlsift.core.prep_page import prep_page
    from htmlsift.core.render import parse, render_tree

    ex = Extractor(bundle_dir=BUNDLE)
    html = (DATA / "sample.html").read_text(encoding="utf-8")
    blocks, probs = ex.extract(html, with_blocks=True)
    lines, src = render_tree(parse(html))
    m = ex.model
    page = prep_page(m.tok, blocks, m.window, cap=m.cap, feats=m.features(lines, src))
    table = ort.InferenceSession(str(Path(BUNDLE) / TABLE), providers=["CPUExecutionProvider"])
    head = ort.InferenceSession(str(Path(BUNDLE) / HEAD), providers=["CPUExecutionProvider"])
    emb = table.run(["out"], {"ids": page["ids"]})[0]
    x = np.concatenate([_full_pool(emb, page["members"]), page["feats"]], 1)
    logits = head.run(["logits"], {"x": x[None]})[0][0]
    ref = (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
    assert np.array_equal(probs, ref)
