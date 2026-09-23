"""htmlsift -- web main-content extraction (HTML in, clean text/HTML/markdown out).

    from htmlsift import Extractor, extract
"""
from .extractor import Extractor, extract

__version__ = "0.1.1"

__all__ = ["Extractor", "extract"]
