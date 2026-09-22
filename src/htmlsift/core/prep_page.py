"""Page layout for inference: tokenize a page's blocks once, map every token back
to the block it came from, and (for the windowed encoder path) plan the token
windows. No gradients, no encoder -- deterministic shaping the forwards consume.
"""

import numpy as np


def block_spans(blocks):
    """Char span [start, end) of each block within "\\n".join(blocks)."""
    spans, pos = [], 0
    for b in blocks:
        spans.append((pos, pos + len(b)))
        pos += len(b) + 1                     # +1 for the joining newline
    return spans


def page_members(enc, blocks):
    """Per block, the indices of the tokens whose characters fall inside it.

    `enc` is a fast-tokenizer output carrying offset_mapping and
    special_tokens_mask. A single cursor advances through the non-special tokens as
    blocks are consumed left to right, so the whole map is one linear pass. A token
    joins a block when it overlaps the block's characters; the `end > bs` guard
    drops zero-width tokens (e.g. (0,0)) while keeping a subword that straddles the
    joining newline with the block it reaches into. A block with no tokens of its
    own gets [] and pools to a zero vector, so members stays aligned with blocks.
    """
    offsets = enc["offset_mapping"]
    special = enc["special_tokens_mask"]
    tok_idx = [i for i in range(len(offsets)) if not special[i]]
    members = []
    ti = 0
    for bs, be in block_spans(blocks):
        while ti < len(tok_idx) and offsets[tok_idx[ti]][1] <= bs:
            ti += 1
        j, mem = ti, []
        while j < len(tok_idx) and offsets[tok_idx[j]][0] < be:
            if offsets[tok_idx[j]][1] > bs:
                mem.append(tok_idx[j])
            j += 1
        members.append(mem)
    return members


def window_starts(n, window, stride=None):
    """Start offsets of the windows tiling n tokens (n > window). stride defaults to
    window // 2 (50% overlap); the last window snaps back to n - window so the tail
    is always fully covered. Shared with the encoder stitch (torch_infer.pool_page),
    so window boundaries are defined in one place."""
    stride = stride or window // 2
    starts = list(range(0, n - window + 1, stride))
    if not starts:
        starts = [0]
    if starts[-1] != n - window:
        starts.append(n - window)
    return starts


def apply_cap(ids, members, k):
    """Keep each block's first k member tokens, dropping the rest from the sequence
    -- a per-block width cap. Tokens belonging to no block (specials, the newline
    joins) are always kept. Members are renumbered onto the compacted ids."""
    keep = np.ones(len(ids), dtype=bool)
    for mem in members:
        if len(mem) > k:
            keep[np.asarray(mem[k:], dtype=np.int64)] = False
    newpos = np.cumsum(keep) - 1
    return ids[keep], [[int(newpos[t]) for t in mem[:k]] for mem in members]


def prep_page(tok, blocks, window, stride=None, cap=0, feats=None, tid=None):
    """Blocks -> the page dict the forwards consume: token ids, per-block members,
    optional structural features, and the window plan. The page carries its own
    window/stride so the encoder stitch cannot drift. truncation=False -- the page is
    windowed by the encoder, never truncated by the tokenizer. cap>0 applies the
    per-block width cap first.

    feats, when given, is an [n_blocks, K] array (already z-scored on the training
    stats, aligned with blocks) carried through to the head concat; feats=None runs
    text-only. The width cap drops tokens, not blocks, so feats stays aligned."""
    enc = tok("\n".join(blocks), add_special_tokens=True,
              return_offsets_mapping=True, return_special_tokens_mask=True,
              return_tensors=None, truncation=False)
    ids = np.array(enc["input_ids"], dtype=np.int64)
    members = page_members(enc, blocks)
    if cap:
        ids, members = apply_cap(ids, members, cap)
    if feats is not None:
        feats = np.asarray(feats, dtype=np.float32)
        assert feats.shape[0] == len(members), (feats.shape[0], len(members))
    return {"tid": tid, "ids": ids, "members": members, "feats": feats,
            "window": window, "stride": stride}
