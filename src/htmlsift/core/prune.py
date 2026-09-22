"""Prune a parsed DOM to the elements behind the selected blocks, then serialize.

The renderer records, per rendered line, the DOM elements that produced it (line_src).
Given the elements behind the KEPT lines, keep exactly those, their ancestors (the
structural path to root) and their descendants (inline links/images inside a kept block),
and drop everything else. lxml only -- no html2text.
"""
from lxml import html as lxml_html


def keep_set(elements):
    """The nodes to keep: each element, its ancestors, and its descendants. Holding the
    proxies in the set keeps their lxml identity stable for the prune walk."""
    keep = set()
    for el in elements:
        if el in keep:
            continue
        keep.add(el)
        keep.update(el.iterancestors())
        keep.update(el.iterdescendants())
    return keep


def prune(root, elements):
    """Remove from root every element not in keep_set(elements); returns root (mutated)."""
    keep = keep_set(elements)

    def walk(el):
        for child in list(el):
            if child in keep:
                walk(child)
            else:
                el.remove(child)

    walk(root)
    return root


def to_html(root, elements):
    """Prune root to the selected elements and serialize to an HTML string."""
    prune(root, elements)
    return lxml_html.tostring(root, encoding="unicode")
