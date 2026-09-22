"""Torch backend for the base model: build the truncated granite encoder from a baked
config (no backbone download), pool block token states, run the BiGRU head. The
inference forward (pool / stitch / encode_window / pool_page / infer_page) is byte-
equivalent to the research original; construction uses AutoModel.from_config so the
package never pulls granite.

Entry point: load_base(...) -> infer_fn, the torch mirror of onnx_infer.load_mini.
"""
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from .features import feature_dim
from .prep_page import window_starts


def build_encoder(config, device="cpu", attn="sdpa"):
    """Build the base encoder from a baked (already layer-truncated) config and give it
    the fresh final LayerNorm the trained weights expect. from_config -- not
    from_pretrained -- so no granite backbone is downloaded; the weights are loaded by
    load_base."""
    model = AutoModel.from_config(config, attn_implementation=attn)
    model.final_norm = nn.LayerNorm(model.config.hidden_size, eps=1e-5)
    return model.to(device)


class BiGRUHead(nn.Module):
    """Per-block head: pooled block vectors [1, n, d_in] -> per-block logits [1, n]."""

    def __init__(self, d_in, hidden=256, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(d_in, hidden, num_layers=1, bidirectional=True, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(2 * hidden, 1)

    def forward(self, x, mask=None):
        h, _ = self.gru(x)
        return self.out(self.drop(h)).squeeze(-1)


def build_head(kind, d_in, hidden=256, dropout=0.2):
    """The base model ships the BiGRU head only."""
    if kind == "bigru":
        return BiGRUHead(d_in, hidden=hidden, dropout=dropout)
    raise ValueError(f"unknown head {kind!r}; the package ships 'bigru' only")


def pool(hidden, members, b_lo, b_hi, tok_off):
    """Mean-pool encoder states hidden [w, H] into per-block vectors [b_hi-b_lo, H].
    members hold page-level token indices; tok_off is the page index of hidden[0], so
    m - tok_off is the window-local row. A block with no tokens pools to a zero vector."""
    dev = hidden.device
    blocks = members[b_lo:b_hi]
    lengths = np.fromiter((len(m) for m in blocks), np.int64, len(blocks))
    counts = torch.tensor(np.maximum(lengths, 1), device=dev,
                          dtype=hidden.dtype).unsqueeze(1)
    pooled = hidden.new_zeros((b_hi - b_lo, hidden.shape[1]))
    if lengths.sum():
        mem_idx = np.concatenate([np.asarray(m, np.int64) for m in blocks if m]) - tok_off
        blk_idx = np.repeat(np.arange(len(blocks), dtype=np.int64), lengths)
        pooled = pooled.index_add(0, torch.from_numpy(blk_idx).to(dev),
                                  hidden[torch.from_numpy(mem_idx).to(dev)])
    return pooled / counts


def cat_feats(x, feats, b_lo, b_hi):
    """Concatenate per-block feature columns onto pooled vectors x [n, H] -> [n, H+K].
    No-op when feats is None (base is text-only)."""
    if feats is None:
        return x
    f = torch.as_tensor(feats[b_lo:b_hi], dtype=x.dtype, device=x.device)
    return torch.cat([x, f], dim=1)


def stitch_bounds(starts, window, k):
    """Window-local [lo, hi) slice window k owns under most-interior ownership: each
    token goes to the window whose centre is nearest. Used by the encoder stitch, so
    the ownership rule lives in one place."""
    s = starts[k]
    lo = 0 if k == 0 else (starts[k - 1] + window + s) // 2 - s
    hi = window if k == len(starts) - 1 else (s + window + starts[k + 1]) // 2 - s
    return lo, hi


def encode_window(encoder, ids_t, autocast):
    """Encode a 1-D token window -> hidden states [w, H] (fp32). bf16 autocast on the
    forward when enabled (CUDA); a no-op on CPU."""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        h = encoder(input_ids=ids_t[None],
                    attention_mask=torch.ones_like(ids_t)[None]).last_hidden_state[0]
    return h.float()


@torch.no_grad()
def pool_page(encoder, page, dev, autocast):
    """Full-page pooled block vectors [n_blocks, H]. A page within its window encodes in
    one pass; a longer page encodes in overlapping windows stitched by most-interior
    ownership, then the whole page is pooled. window/stride come from the page."""
    window, stride = page["window"], page.get("stride")
    ids = torch.from_numpy(page["ids"]).to(dev)
    n = len(ids)
    if n <= window:
        hidden = encode_window(encoder, ids, autocast)
    else:
        hidden = torch.full((n, encoder.config.hidden_size), float("nan"), device=dev)
        starts = window_starts(n, window, stride)
        for k, s in enumerate(starts):
            h = encode_window(encoder, ids[s:s + window], autocast)
            lo, hi = stitch_bounds(starts, window, k)
            hidden[s + lo:s + hi] = h[lo:hi]
        assert not torch.isnan(hidden).any(), f"stitch gap {page['tid']}"
    return pool(hidden, page["members"], 0, len(page["members"]), 0)


@torch.no_grad()
def infer_page(encoder, head, page, dev, autocast):
    """Full-page inference -> per-block probs (np.float32): pool the page, concat any
    features, run the head."""
    x = pool_page(encoder, page, dev, autocast)
    x = cat_feats(x, page.get("feats"), 0, len(page["members"]))
    probs = torch.sigmoid(head(x[None])[0])
    return probs.float().cpu().numpy()


def load_base(config_path, weights_path, hidden, feats=None, device=None,
              band=True, threads=None):
    """Load the base bundle into an infer_fn (page -> per-block probs) -- the torch
    mirror of onnx_infer.load_mini. Build the architecture from the baked config, load
    the trained encoder + head weights, optionally enable band attention, and close over
    the forward. `hidden` is the BiGRU head hidden size; `feats` the feature group (base
    is text-only -> None). No granite download.

    device=None auto-selects cuda when a GPU is present, else cpu -- and warns on the
    cpu fallback, so a silent CPU run can't pass for a GPU one. An explicit "cuda" with
    no GPU is a clear error rather than a cryptic failure deeper in torch."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            warnings.warn(
                "htmlsift: base defaulting to CPU (no CUDA device found); "
                "pass device='cpu' to silence, or device='cuda' to require GPU",
                stacklevel=2)
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"device={device!r} requested but no CUDA device is available")
    if threads and dev.type == "cpu":
        torch.set_num_threads(threads)
    config = AutoConfig.from_pretrained(str(Path(config_path).parent))
    enc = build_encoder(config, device=dev)
    head = build_head("bigru", d_in=enc.config.hidden_size + feature_dim(feats),
                      hidden=hidden).to(dev)
    state = torch.load(weights_path, map_location=dev, weights_only=True)
    enc.load_state_dict(state["encoder"])
    enc.eval()
    head.load_state_dict(state["head_state"])
    head.eval()
    if band:
        from . import band as band_mod
        band_mod.enable(enc)
    autocast = dev.type == "cuda"
    return lambda page: infer_page(enc, head, page, dev, autocast)
