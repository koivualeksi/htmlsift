"""render.py links= extension (no bundle): links=False keeps the frozen model-input
behavior (flattened links, dropped images); links=True emits markdown."""
from htmlsift.core.render import render


def test_links_false_flattens_link():
    md = render('<p>see <a href="/x">the report</a> now</p>')
    assert "the report" in md
    assert "/x" not in md and "[" not in md          # href gone, no markdown link


def test_links_false_drops_image():
    md = render('<p>x <img src="a.png" alt="A"> y</p>')
    assert "a.png" not in md and "![" not in md


def test_links_true_renders_link():
    md = render('<p>see <a href="/x">the report</a> now</p>', links=True)
    assert "[the report](/x)" in md


def test_links_true_renders_image():
    md = render('<p>before <img src="a.png" alt="A"> after</p>', links=True)
    assert "![A](a.png)" in md


def test_link_without_href_is_plain_even_with_links():
    md = render('<p><a>anchor</a></p>', links=True)
    assert "anchor" in md and "[" not in md          # no href -> falls back to plain text
