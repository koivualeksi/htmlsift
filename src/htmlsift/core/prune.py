"""Prune a parsed DOM to the text of the selected lines.

render_tree(pieces=True) records, per line, the parts of elements it holds: an element's
.text or .tail, a unit rendered whole (wrap tag, hr, data-table row) or one content line
of a pre. prune() keeps exactly the parts of the selected lines. lxml only.
"""
from .render import HEADING_TAGS, SIMPLE_BLOCK_TAGS

# Parents that start their own line: a child dropped there cannot glue its tail to text
# before the parent.
_LINE_PARENTS = SIMPLE_BLOCK_TAGS | set(HEADING_TAGS) | {
    "body", "li", "td", "th", "dd", "dt", "blockquote"}


def prune(root, block_pieces, selected):
    """Keep in root only the parts behind the blocks in selected (indices into
    block_pieces); returns root (mutated). An element holding no part follows its nearest
    ancestor that does. A pre stays whole when any of its lines is selected."""
    seen, kept = set(), set()
    for i, pieces in enumerate(block_pieces):
        for el, part, *_ in pieces:
            if part == "pre":
                part = "whole"
            seen.add((el, part))
            if i in selected:
                kept.add((el, part))
    _walk(root, False, seen, kept)
    return root


def _walk(el, inherited, seen, kept):
    """Prune el's subtree -> (el kept, el kept for more than its tail, subtree holds a
    part). inherited: whether the nearest ancestor holding a part is kept."""
    whole = (el, "whole") in kept
    if whole:
        keep = inh = True
    elif (el, "text") in seen:
        keep = inh = (el, "text") in kept
        if not keep:
            el.text = None
    elif (el, "whole") in seen:
        keep = inh = False
    elif (el, "tail") in seen:
        keep, inh = False, (el, "tail") in kept
    else:
        keep, inh = False, inherited
    owns = any((el, part) in seen for part in ("text", "tail", "whole"))
    kept_child = False
    if not whole:
        for child in list(el):
            child_kept, child_body, child_owns = _walk(child, inh, seen, kept)
            owns |= child_owns
            if child_kept:
                kept_child = True
                if (not child_body and child.tag != "br" and child.getprevious() is None
                        and not (el.text or "").strip() and el.tag in _LINE_PARENTS):
                    child.drop_tree()          # only its tail is kept and nothing precedes it
            else:
                el.remove(child)
    tail_kept = (el, "tail") in kept
    if (el, "tail") in seen and not tail_kept:
        el.tail = None
    body = keep or kept_child or (not owns and inherited)
    return body or tail_kept, body, owns
