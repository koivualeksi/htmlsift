from types import SimpleNamespace

import numpy as np
import pytest

from htmlsift import Extractor


@pytest.fixture
def keep_all(monkeypatch):
    """An Extractor whose dummy model scores every line 1.0."""
    monkeypatch.setattr("htmlsift.extractor.prep_page", lambda tok, blocks, *a, **k: blocks)
    ex = Extractor()
    ex._model = SimpleNamespace(features=lambda lines, src: None, tok=None, window=0, cap=0,
                                infer_fn=lambda blocks: np.ones(len(blocks), dtype=np.float32))
    return ex
