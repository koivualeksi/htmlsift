"""Leading page-title detection for drop_title: the first kept line, when it comes from a
heading or paragraph that matches the page's <title> or og:title."""
import difflib
import re
import unicodedata

TITLE_SEP = re.compile(r"\s[|\-–—:·•»/~]{1,2}\s|\s*[|·•»]\s*")
SIM = 0.8
_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6", "p"})


def norm(s):
    """NFKC, casefold, runs of non-alphanumerics to one space."""
    s = unicodedata.normalize("NFKC", s).casefold()
    return re.sub(r"[\W_]+", " ", s).strip()


def title_candidates(title, og):
    """Normalised whole titles and their segments split at TITLE_SEP."""
    out = set()
    for t in (title, og):
        if t:
            out.add(norm(t))
            out.update(norm(p) for p in TITLE_SEP.split(t))
    out.discard("")
    return out


def title_match(line, title, og):
    """Best SequenceMatcher ratio between the normalised line and any title candidate."""
    h = norm(line)
    return max((difflib.SequenceMatcher(None, h, c, autojunk=False).ratio()
                for c in title_candidates(title, og)), default=0.0)


def page_titles(root):
    """(first <title>, og:title), whitespace collapsed; None where absent."""
    t = root.xpath("//title")
    og = root.xpath("//meta[@property='og:title' or @name='og:title']/@content")
    title = " ".join(t[0].text_content().split()) if t else None
    return title, (" ".join(og[0].split()) if og else None)


def _enclosing(el):
    """Nearest ancestor-or-self that is h1-h6 or p, else None."""
    while el is not None:
        if isinstance(el.tag, str) and el.tag.lower() in _TAGS:
            return el
        el = el.getparent()
    return None


def find_title(root, k, line, block_src):
    """The h1-h6/p element behind block k (text: line) if it is the page title, else None.
    All of block k's elements must sit in that one element, the element's text must be
    this line alone, and line must match <title> or og:title."""
    encl = {_enclosing(e) for e in block_src[k]}
    if len(encl) != 1 or None in encl:
        return None
    el = encl.pop()
    if norm(el.text_content()) != norm(line):
        return None
    return el if title_match(line, *page_titles(root)) >= SIM else None
