"""Structural block features for the table-embedding arm (mini): the signal a
mean-pooled bag of token embeddings cannot carry, and nothing the rendered text
already shows.

Computed from render_tree's line_src (the DOM elements behind each block) -- no
second parse, no marker injection, no change to the renderer. Booleans for
presence; only the two genuine scalars (depth, log_chars) are z-scored, using the
stats shipped in the model bundle.

A column earns its place under one rule: it is absent from the tokenized block text
AND not recoverable by the BiGRU. Tag structure is flattened by markdown, and any
marker that survives is diluted by the per-block mean-pool; DOM depth and
link-ancestry never appear in the rendered text at all (the renderer prints an <a>
as bare text); block length is destroyed the moment the tokens are averaged.

This module is the feature manifest: column order and the group map live here and
nowhere else.
"""

import math

import numpy as np

# Group A -- ancestor-or-self tag presence, 30 booleans. Semantic containers that
# markdown flattens away, plus block tags whose surviving text marker the mean-pool
# dilutes.
TAG_VOCAB = [
    "main", "article", "section", "nav", "header", "footer", "aside",
    "h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "ul", "ol", "dl",
    "table", "tr", "td", "th", "blockquote", "pre", "code",
    "figure", "figcaption", "form", "button", "div",
]

GROUPS = {
    "A": [f"tag_{t}" for t in TAG_VOCAB],   # ancestor/self tag presence (boolean)
    "B": ["depth", "has_link"],             # DOM nesting depth (scalar); an <a> ancestor (boolean)
    "C": ["log_chars"],                     # block length -- what the mean-pool averages away
}

FEATURE_NAMES = [name for g in ("A", "B", "C") for name in GROUPS[g]]

# The only unbounded columns; z-scored with the bundle's stats. Booleans and
# already-bounded columns stay raw.
ZSCORE_COLS = ["depth", "log_chars"]

_ORDER = ("A", "B", "C")
_VOCAB = set(TAG_VOCAB)


def collect_features(lines, line_src, groups=_ORDER):
    """(lines, line_src) from render_tree -> [n_blocks, K] float32, one row per
    non-empty line (the block unit). Columns are the requested groups in manifest
    order; only those groups are computed, so a text-only arm passes groups=() and
    pays nothing. The ancestor chain of each element is cached and shared across
    blocks (below)."""
    groups = [g for g in _ORDER if g in groups]
    cols = [c for g in groups for c in GROUPS[g]]
    col = {c: i for i, c in enumerate(cols)}
    blocks = [(ln, src) for ln, src in zip(lines, line_src, strict=True) if ln.strip()]
    out = np.zeros((len(blocks), len(cols)), dtype=np.float32)
    need_a, need_b, need_c = "A" in groups, "B" in groups, "C" in groups

    # element -> (vocab tags, depth, link) for its ancestor-or-self chain, computed
    # once and reused. Every block re-reaches the same container chain to root, so
    # caching collapses that repetition: the walk stops at the first cached ancestor.
    # Keyed by the element object -- the cache holds the reference, which keeps the
    # lxml proxy (and thus its identity) stable for the call.
    chain = {}

    def chain_info(e):
        stack = []
        node = e
        while node is not None and isinstance(node.tag, str) and node not in chain:
            stack.append(node)
            node = node.getparent()
        if node is not None and isinstance(node.tag, str):
            tags, depth, link = chain[node]
        else:
            tags, depth, link = frozenset(), 0, False
        for nd in reversed(stack):                     # fold from cached base to e
            t = nd.tag
            if need_a and t in _VOCAB and t not in tags:
                tags = tags | {t}
            depth += 1
            link = link or t == "a"
            chain[nd] = (tags, depth, link)
        return tags, depth, link

    for bi, (text, src) in enumerate(blocks):
        if need_a or need_b:
            btags, depth_sum, n_src, has_link = set(), 0, 0, False
            for e in src:
                et, ed, el = chain_info(e)
                btags |= et
                depth_sum += ed
                n_src += 1
                has_link = has_link or el
            if need_a:
                for t in btags:
                    out[bi, col[f"tag_{t}"]] = 1.0
            if need_b:
                out[bi, col["depth"]] = depth_sum / n_src if n_src else 0.0
                out[bi, col["has_link"]] = float(has_link)
        if need_c:
            out[bi, col["log_chars"]] = math.log1p(len(text))
    return out


def feature_dim(groups):
    """Number of feature columns a group selection produces (None/"" -> 0). Sizes
    the head's d_in without hardcoding K."""
    return sum(len(GROUPS[g]) for g in _ORDER if groups and g in groups)


def _zscore_idx(groups):
    """Column positions of the unbounded scalars (ZSCORE_COLS) within a group
    selection's layout -- lets the loader pair the bundle's mean/std with the right
    columns for a given feats group."""
    cols = [c for g in _ORDER if g in groups for c in GROUPS[g]]
    return [i for i, c in enumerate(cols) if c in ZSCORE_COLS]


def apply_zscore(fa, stats):
    """A copy of a page's [n_blocks, K] array with the z-scored columns normalized by
    the bundle's (idx, mean, std) stats; booleans and bounded columns untouched."""
    idx, mean, std = stats
    out = fa.copy()
    out[:, idx] = (out[:, idx] - mean) / std
    return out
