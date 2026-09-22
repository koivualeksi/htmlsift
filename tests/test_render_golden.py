"""Golden test for the frozen renderer: render_tree(sample.html) must keep producing
the exact same lines. Any drift in render.py fails here.

The reference is captured with HTMLSIFT_BLESS=1 (eyeball it once, commit it); every run
after compares against it:

    HTMLSIFT_BLESS=1 pytest tests/test_render_golden.py
"""
import os
from pathlib import Path

from htmlsift.core.render import parse, render_tree

DATA = Path(__file__).parent / "data"
GOLDEN = DATA / "sample.rendered.txt"


def test_render_golden():
    html = (DATA / "sample.html").read_text(encoding="utf-8")
    got = "\n".join(render_tree(parse(html))[0]) + "\n"
    if os.environ.get("HTMLSIFT_BLESS"):
        GOLDEN.write_text(got, encoding="utf-8")
    assert GOLDEN.exists(), \
        "golden missing; run: HTMLSIFT_BLESS=1 pytest tests/test_render_golden.py"
    assert got == GOLDEN.read_text(encoding="utf-8")
