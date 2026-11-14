"""Prune a parsed DOM to the text of the selected lines.

render_tree(pieces=True) records, per line, the parts of elements it holds: an element's
.text or .tail, a unit rendered whole (wrap tag, hr, data-table row) or one content line
of a pre. prune() keeps exactly the parts of the selected lines. lxml only.
"""
from .render import BLOCKISH_TAGS, HEADING_TAGS, LIST_TAGS, SIMPLE_BLOCK_TAGS

# Parents that start their own line: a child dropped there cannot glue its tail to text
# before the parent.
_LINE_PARENTS = SIMPLE_BLOCK_TAGS | set(HEADING_TAGS) | {
    "body", "li", "td", "th", "dd", "dt", "blockquote"}
# Elements the renderer starts a new line at.
_BREAK_TAGS = _LINE_PARENTS | LIST_TAGS | {"pre", "table", "tr", "option", "hr", "br"}


def prune(root, block_pieces, selected):
    """Keep in root only the parts behind the blocks in selected (indices into
    block_pieces); returns root (mutated). An element holding no part follows its nearest
    ancestor that does. A pre with no block elements inside (a code block) stays whole
    when any of its lines is selected; any other pre is cut to its selected lines."""
    seen, kept, pre_lines, line_of = set(), set(), {}, {}
    for i, pieces in enumerate(block_pieces):
        for el, part, *line in pieces:
            if part == "pre":
                part = "whole"
                if i in selected:
                    pre_lines.setdefault(el, set()).add(line[0])
            seen.add((el, part))
            if i in selected:
                kept.add((el, part))
                lo, hi = line_of.get((el, part), (i, i))
                line_of[(el, part)] = (min(lo, i), max(hi, i))
    for pre, lines in pre_lines.items():
        if not _is_code_block(pre):
            _cut_pre(pre, lines)
    _walk(root, False, seen, kept)
    _separate(root, line_of)
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
            elif child.tag != "br" or child_owns:   # a bare br stays in a kept parent
                el.remove(child)
    tail_kept = (el, "tail") in kept
    if (el, "tail") in seen and not tail_kept:
        el.tail = None
    body = keep or kept_child or (not owns and inherited)
    return body or tail_kept, body, owns


def _separate(root, line_of):
    """Insert a <br> where kept text from two different original lines now meets with
    nothing kept between them that starts a new line. line_of: kept piece ->
    (first, last) block index."""
    inserts = []
    prev = None          # last block index of the previous kept text
    broken = True        # a line break since then

    def text(span, where, el):
        nonlocal prev, broken
        if prev is not None and not broken and span[0] != prev:
            inserts.append((where, el))
        prev, broken = span[1], False

    def visit(el):
        nonlocal broken
        brk = isinstance(el.tag, str) and el.tag.lower() in _BREAK_TAGS
        broken |= brk
        if (el, "whole") in line_of:
            text(line_of[(el, "whole")], "before", el)
        else:
            if el.text and (el, "text") in line_of:
                text(line_of[(el, "text")], "before", el)
            for child in el:
                visit(child)
                if child.tail and (child, "tail") in line_of:
                    text(line_of[(child, "tail")], "after", child)
        broken |= brk

    visit(root)
    for where, el in inserts:
        br = el.makeelement("br")
        if where == "before":
            el.addprevious(br)
        else:
            br.tail, el.tail = el.tail, None
            el.addnext(br)


def _is_code_block(pre):
    return not any(isinstance(d.tag, str) and d.tag.lower() in BLOCKISH_TAGS
                   for d in pre.iterdescendants())


def _cut_pre(pre, lines):
    """Keep only the characters of pre on its content lines in lines; drop elements left
    empty."""
    n = 0
    had_text = set()

    def cut(s):
        nonlocal n
        if not s:
            return s
        out = []
        for ch in s:
            if n in lines:
                out.append(ch)
            if ch == "\n":
                n += 1
        return "".join(out) or None

    def rec(el):
        has = bool(el.text)
        el.text = cut(el.text)
        for child in el:
            has |= rec(child) | bool(child.tail)
            child.tail = cut(child.tail)
        if has:
            had_text.add(el)
        return has

    rec(pre)
    for d in reversed(list(pre.iterdescendants())):
        if d in had_text and not d.text and len(d) == 0:
            d.drop_tree()
