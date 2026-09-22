"""The public extractor: HTML in, main content out. Holds a lazily-loaded Model and
runs one pipeline per call -- parse, render to lines, score each line, keep the lines
over threshold, serialize. Mirrors the research inference front-end (render_blocks +
prep_page + threshold); no sanitize step.
"""
import numpy as np

from .core.prep_page import prep_page
from .core.prune import prune, to_html
from .core.render import parse, render_tree
from .loader import load_model

THRESHOLD = 0.5


def _selected_srcs(block_src, probs):
    """DOM elements behind the blocks scoring over threshold (union, in order)."""
    return [el for src, p in zip(block_src, probs, strict=True) if p > THRESHOLD for el in src]


class Extractor:
    """Main-content extractor for one mode. The model loads on the first extract() call
    and is reused, so create one Extractor and call it many times."""

    def __init__(self, mode="mini", *, bundle_dir=None, threads=None):
        self.mode = mode
        self._bundle_dir = bundle_dir
        self._threads = threads
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = load_model(self.mode, bundle_dir=self._bundle_dir,
                                     threads=self._threads)
        return self._model

    def extract(self, html, output="text", with_blocks=False):
        """Extract main content from an HTML string.

        output="text" joins the selected lines; output="html" prunes the DOM to the
        selected blocks (structure, links, images kept) and serializes. with_blocks=True
        returns (blocks, probs) -- every block and its probability, unthresholded."""
        root, blocks, block_src, probs = self._predict(html)
        if with_blocks:
            return blocks, probs
        if output == "text":
            return "\n".join(b for b, p in zip(blocks, probs, strict=True) if p > THRESHOLD)
        if output == "html":
            selected = _selected_srcs(block_src, probs)
            return to_html(root, selected) if selected else ""
        if output == "markdown":
            selected = _selected_srcs(block_src, probs)
            if not selected:
                return ""
            prune(root, selected)
            lines, _ = render_tree(root, links=True)
            return "\n".join(lines).strip("\n")
        raise ValueError(f"unknown output {output!r}; expected 'text', 'html', or 'markdown'")

    def _predict(self, html):
        """html -> (root, blocks, block_src, probs). blocks are the rstripped non-empty
        render lines; block_src[i] the DOM elements behind block i. Empty page ->
        (None, [], [], empty array)."""
        if not html or not html.strip():
            return None, [], [], np.empty(0, dtype=np.float32)
        root = parse(html)
        lines, line_src = render_tree(root)
        blocks, block_src = [], []
        for ln, src in zip(lines, line_src, strict=True):
            if ln.strip():
                blocks.append(ln.rstrip())
                block_src.append(src)
        if not blocks:
            return root, [], [], np.empty(0, dtype=np.float32)
        feats = self.model.features(lines, line_src)          # on RAW lines
        page = prep_page(self.model.tok, blocks, self.model.window,
                         cap=self.model.cap, feats=feats)
        return root, blocks, block_src, self.model.infer_fn(page)


_CACHE = {}


def extract(html, mode="mini", output="text", with_blocks=False):
    """Module-level convenience: extract with a per-mode cached Extractor."""
    ex = _CACHE.get(mode)
    if ex is None:
        ex = _CACHE[mode] = Extractor(mode)
    return ex.extract(html, output=output, with_blocks=with_blocks)
