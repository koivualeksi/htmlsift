"""API-surface checks that run without a model bundle: empty HTML short-circuits before
the model loads, so these exercise the public contract with no network or weights.
"""
import pytest

from htmlsift import Extractor, extract


def test_empty_html_with_blocks():
    blocks, probs = Extractor().extract("", with_blocks=True)
    assert blocks == [] and len(probs) == 0


@pytest.mark.parametrize("out", ["text", "html", "markdown"])
def test_empty_all_outputs(out):
    assert Extractor().extract("", output=out) == ""


def test_unknown_output_raises():
    with pytest.raises(ValueError):
        Extractor().extract("", output="bogus")


def test_module_level_extract_empty():
    assert extract("") == ""
