"""The public extractor: HTML in, main content out. Holds a lazily-loaded Model and
runs one pipeline per call -- parse, render to lines, score each line, keep the lines
over threshold, serialize. Mirrors the research inference front-end (render_blocks +
prep_page + threshold); no sanitize step.
"""
import unicodedata

import numpy as np
from lxml import etree
from lxml import html as lxml_html

from ._decode import prepare_input
from ._links import effective_base, make_absolute
from ._title import find_title, norm
from .core.prep_page import prep_page
from .core.prune import prune
from .core.render import parse, render_tree
from .loader import load_model

THRESHOLD = 0.5
_OUTPUTS = ("text", "html", "markdown")
_INVISIBLE = frozenset({"Cc", "Cf", "Zs", "Zl", "Zp"})
_FILLERS = frozenset("ᅟᅠㅤﾠ⠀")


def _visible(line):
    """True if the line has a character that renders as a glyph."""
    return any(c not in _FILLERS and unicodedata.category(c) not in _INVISIBLE for c in line)


def _plain_text(root, block_line, n_lines, probs):
    """The selected blocks as plain text, from a marker-free render of the same tree."""
    sel = [i for i, p in zip(block_line, probs, strict=True) if p > THRESHOLD]
    if not sel:
        return ""
    plain, _ = render_tree(root, plain=True)
    if len(plain) != n_lines:
        raise RuntimeError(f"plain render has {len(plain)} lines, model render {n_lines}")
    return "\n".join(s for s in (plain[i].rstrip() for i in sel) if _visible(s))


def _drop_title(root, blocks, block_src, probs):
    """probs with the first selected block with text deselected if it is the page
    title, else probs unchanged."""
    for k in np.flatnonzero(probs > THRESHOLD):
        if not _visible(blocks[k]) or not norm(blocks[k]):
            continue
        if find_title(root, k, blocks[k], block_src) is not None:
            probs = probs.copy()
            probs[k] = 0.0
        return probs
    return probs


class Extractor:
    """Main-content extractor for one mode. The model loads on the first extract() call
    and is reused, so create one Extractor and call it many times."""

    def __init__(self, mode="mini", *, bundle_dir=None, threads=None, device=None):
        self.mode = mode
        self._bundle_dir = bundle_dir
        self._threads = threads
        self._device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = load_model(self.mode, bundle_dir=self._bundle_dir,
                                     threads=self._threads, device=self._device)
        return self._model

    def extract(self, html, output="text", with_blocks=False, *, encoding=None, base_url=None,
                drop_title=False):
        """Extract main content from HTML, given as str or bytes.

        bytes are decoded by BOM, then encoding= (e.g. an HTTP or WARC charset), then the
        page's <meta> declaration, then utf-8 / cp1252. encoding= with str raises TypeError.
        output="text" joins the selected lines as plain text: no heading, emphasis, code,
        quote, rule or fence markers; list bullets and ordered numbers stay, and a table
        row is its non-empty cells joined by " | ".
        output="html" prunes the DOM to the text of the selected lines (structure,
        links, images kept; a code block stays whole when any of its lines is selected) and
        serializes; output="markdown" renders that pruned DOM. with_blocks=True
        returns (blocks, probs) -- every block and its probability, unthresholded.
        base_url= makes links absolute in html/markdown output; the page's first
        <base href> is resolved against it, or used alone if absolute.
        drop_title=True leaves out the first selected line when it is an h1-h6 or p that
        matches the page's <title> or og:title, in every output (not with_blocks)."""
        if output not in _OUTPUTS:
            raise ValueError(f"unknown output {output!r}; expected 'text', 'html', or 'markdown'")
        if base_url is not None and not isinstance(base_url, str):
            raise TypeError(f"base_url must be str or None, not {type(base_url).__name__}")
        (root, blocks, block_src, block_pieces, block_line, n_lines,
         probs) = self._predict(html, encoding)
        if with_blocks:
            return blocks, probs
        if drop_title:
            probs = _drop_title(root, blocks, block_src, probs)
        if output == "text":
            return _plain_text(root, block_line, n_lines, probs)
        selected = set(np.flatnonzero(probs > THRESHOLD).tolist())
        if not selected:
            return ""
        base = effective_base(root, base_url)
        prune(root, block_pieces, selected)
        if base:
            make_absolute(root, base)
        if output == "html":
            return lxml_html.tostring(root, encoding="unicode")
        lines, _ = render_tree(root, links=True)
        return "\n".join(lines).strip("\n")

    def _predict(self, html, encoding=None):
        """html -> (root, blocks, block_src, block_pieces, block_line, n_lines, probs).
        blocks are the rstripped non-empty render lines; block_src[i] the DOM elements
        behind block i; block_pieces[i] the exact parts of them it holds; block_line[i] its
        index among the n_lines render lines. Empty page, or one that parses to no element
        -> (None, [], [], [], [], 0, empty array)."""
        empty = np.empty(0, dtype=np.float32)
        html = prepare_input(html, encoding)
        if not html or not html.strip():
            return None, [], [], [], [], 0, empty
        try:
            root = parse(html)
        except etree.ParserError:   # no elements: only comments, doctype, PIs, NULs
            return None, [], [], [], [], 0, empty
        lines, line_src, line_pieces = render_tree(root, pieces=True)
        blocks, block_src, block_pieces, block_line = [], [], [], []
        for i, (ln, src, pieces) in enumerate(zip(lines, line_src, line_pieces, strict=True)):
            if ln.strip():
                blocks.append(ln.rstrip())
                block_src.append(src)
                block_pieces.append(pieces)
                block_line.append(i)
        if not blocks:
            return root, [], [], [], [], len(lines), empty
        feats = self.model.features(lines, line_src)          # on RAW lines
        page = prep_page(self.model.tok, blocks, self.model.window,
                         cap=self.model.cap, feats=feats)
        return (root, blocks, block_src, block_pieces, block_line, len(lines),
                self.model.infer_fn(page))


_CACHE = {}


def extract(html, mode="mini", output="text", with_blocks=False, *, encoding=None,
            base_url=None, drop_title=False):
    """Module-level convenience: extract with a per-mode cached Extractor."""
    ex = _CACHE.get(mode)
    if ex is None:
        ex = _CACHE[mode] = Extractor(mode)
    return ex.extract(html, output=output, with_blocks=with_blocks, encoding=encoding,
                      base_url=base_url, drop_title=drop_title)
