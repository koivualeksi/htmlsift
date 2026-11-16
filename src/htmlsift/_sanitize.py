"""Strip active content from the pruned tree before html/markdown serialization.

A blocklist cleanup, not a guarantee: run a full sanitizer before embedding the output
in a web page."""
import re
from html import unescape

from lxml.html.defs import link_attrs

from .core.render import SKIP_TAGS

_DROP_TAGS = SKIP_TAGS | {"animate", "set"}
_URL_ATTRS = link_attrs | {"poster", "xlink:href"}
_SCHEMES = frozenset({"http", "https", "mailto", "tel", "ftp"})
_SCHEME = re.compile(r"([a-z][a-z0-9+.\-]*):", re.I)
_EDGE = "".join(map(chr, range(0x21)))


def _scheme(url):
    """The URL's scheme as a browser reads it, lowercased; None if relative."""
    url = re.sub(r"[\t\n\r]", "", unescape(url).strip(_EDGE))
    m = _SCHEME.match(url)
    return m.group(1).lower() if m else None


def _allowed(tag, name, value):
    scheme = _scheme(value)
    if scheme is None or scheme in _SCHEMES:
        return True
    return (scheme == "data" and tag == "img" and name == "src"
            and unescape(value).strip(_EDGE).lower().startswith("data:image/"))


def sanitize(root):
    """Drop SKIP_TAGS and SVG animation elements (tails kept), on* and srcdoc
    attributes, and link attributes whose scheme is not allowed. In place."""
    for el in [e for e in root.iter() if isinstance(e.tag, str)
               and e.tag.lower() in _DROP_TAGS and e is not root]:
        el.drop_tree()
    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        tag = el.tag.lower()
        for name, value in list(el.attrib.items()):
            low = name.lower()
            if (low.startswith("on") or low == "srcdoc"
                    or (low in _URL_ATTRS and not _allowed(tag, low, value))):
                del el.attrib[name]
