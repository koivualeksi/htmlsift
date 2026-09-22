"""Assemble a mode into a ready-to-run Model: resolve the bundle, read its manifest,
build the tokenizer and the onnxruntime forward, and package the feature config. The
mini path is torch-free end to end; the base path uses torch.
"""
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ._artifacts import Manifest, resolve_bundle
from .core.features import _zscore_idx, apply_zscore, collect_features
from .core.tokenizer import FastTokenizer


@dataclass
class Model:
    """Everything one extract() needs, already loaded. Held by the Extractor so a
    process loads the model once and reuses it."""
    mode: str
    tok: Callable                 # prep_page-compatible tokenizer
    infer_fn: Callable            # page -> per-block probs (np.float32)
    feats: str | None             # feature group, e.g. "ABC"
    zscore: tuple | None          # (idx, mean, std) for apply_zscore
    cap: int
    window: int

    def features(self, lines, line_src):
        """z-scored structural features [n_blocks, K] for this page, or None for a
        text-only model. Feeds prep_page(..., feats=...)."""
        if not self.feats:
            return None
        fa = collect_features(lines, line_src, self.feats)
        if self.zscore is not None:
            fa = apply_zscore(fa, self.zscore)
        return fa


def load_model(mode="mini", bundle_dir=None, threads=None):
    """Load `mode` ("mini" or "base") into a Model. bundle_dir reads a local bundle
    (development); otherwise the bundle is fetched from the release repo and cached.
    threads maps to onnxruntime intra-op threads (mini) or torch threads (base).
    Backends are imported lazily, so a default install never imports torch."""
    if mode not in ("mini", "base"):
        raise ValueError(f"unknown mode {mode!r}; expected 'mini' or 'base'")
    paths = resolve_bundle(mode, local_dir=bundle_dir)
    manifest = Manifest.from_file(paths["manifest"])
    tok = FastTokenizer.from_file(paths["tokenizer"])
    if mode == "mini":
        from .core.onnx_infer import load_mini
        infer_fn = load_mini(paths["table"], paths["head"], threads)
    else:
        from .core.torch_infer import load_base
        infer_fn = load_base(paths["config"], paths["weights"], manifest.hidden,
                             feats=manifest.feats, threads=threads)
    return Model(mode=mode, tok=tok, infer_fn=infer_fn, feats=manifest.feats,
                 zscore=_zscore_from_manifest(manifest), cap=manifest.cap,
                 window=manifest.window)


def _zscore_from_manifest(m):
    """(idx, mean, std) for apply_zscore, or None. idx comes from the feats layout;
    the bundle's mean/std must be ordered to match it (ZSCORE_COLS in group order:
    depth, log_chars)."""
    if not m.feats:
        return None
    idx = _zscore_idx(m.feats)
    if not idx:
        return None
    mean = np.asarray(m.zscore_mean, dtype=np.float32)
    std = np.asarray(m.zscore_std, dtype=np.float32)
    if not (len(idx) == mean.size == std.size):
        raise ValueError(
            f"manifest z-score length mismatch: idx {len(idx)}, mean {mean.size}, std {std.size}")
    return (idx, mean, std)
