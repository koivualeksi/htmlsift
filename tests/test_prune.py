"""Pruner unit tests (no bundle): mark elements, assert the pruned/serialized tree keeps
exactly the marked subtrees + their ancestors + descendants and drops the rest."""
from lxml import html as lxml_html

from htmlsift.core.prune import keep_set, prune, to_html


def test_to_html_keeps_link_and_structure():
    root = lxml_html.fromstring(
        '<body><nav><a href="/">Home</a></nav>'
        '<main><p>Read the <a href="/x">report</a>.</p><aside>sidebar-junk</aside></main>'
        '<footer>copyright</footer></body>')
    p = root.find(".//main/p")
    out = to_html(root, [p])
    assert '<a href="/x">report</a>' in out          # marked block + its inline link kept
    assert "<main" in out                             # ancestor kept
    for gone in ("Home", "copyright", "sidebar-junk"):  # nav, footer, sibling aside dropped
        assert gone not in out


def test_inline_image_under_marked_block_kept():
    root = lxml_html.fromstring('<body><main><p>hi <img src="a.png" alt="A"></p></main></body>')
    out = to_html(root, [root.find(".//p")])
    assert 'src="a.png"' in out                  # descendant of a marked block survives


def test_two_marks_under_one_container_both_survive():
    root = lxml_html.fromstring(
        "<body><main><p>keep1</p><p>drop</p><p>keep2</p></main><nav>x</nav></body>")
    ps = root.findall(".//main/p")
    prune(root, [ps[0], ps[2]])
    out = lxml_html.tostring(root, encoding="unicode")
    assert "keep1" in out and "keep2" in out
    assert "drop" not in out and ">x<" not in out


def test_keep_set_empty():
    assert keep_set([]) == set()
