"""Tokenizer loading from tokenizer.json or tokenizer.json.gz, and which of the two a
bundle resolves to (local dir and hub, with the plain-file fallback)."""
import gzip
import shutil
from pathlib import Path

import pytest
from huggingface_hub.utils import LocalEntryNotFoundError
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from htmlsift import _artifacts
from htmlsift._artifacts import TOKENIZER, TOKENIZER_GZ, resolve_bundle
from htmlsift.core.tokenizer import FastTokenizer

TEXT = "the cat sat on the mat , and a dog"


@pytest.fixture
def tok_files(tmp_path):
    """(plain, gz) paths of a tiny word-level tokenizer."""
    vocab = {w: i for i, w in enumerate(["[UNK]", "the", "cat", "sat", "on", "mat", ","])}
    tk = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = Whitespace()
    plain = tmp_path / TOKENIZER
    tk.save(str(plain))
    gz = tmp_path / TOKENIZER_GZ
    with open(plain, "rb") as src, gzip.open(gz, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return plain, gz


def test_gz_and_plain_encode_the_same(tok_files):
    plain, gz = tok_files
    a = FastTokenizer.from_file(plain)(TEXT)
    b = FastTokenizer.from_file(gz)(TEXT)
    assert a == b
    assert a["input_ids"][:3] == [1, 2, 3] and 0 in a["input_ids"]


@pytest.mark.parametrize("present, expected", [
    ([TOKENIZER_GZ], TOKENIZER_GZ),
    ([TOKENIZER], TOKENIZER),
    ([TOKENIZER, TOKENIZER_GZ], TOKENIZER_GZ),
])
def test_local_dir_picks_gz_when_present(tmp_path, present, expected):
    for fn in present:
        (tmp_path / fn).write_bytes(b"")
    for mode in ("mini", "base"):
        assert resolve_bundle(mode, local_dir=tmp_path)["tokenizer"] == tmp_path / expected


def fake_hub(monkeypatch, has_gz):
    """Replace hf_hub_download; returns the list of requested filenames."""
    calls = []

    def download(repo, filename, revision=None):
        calls.append(filename)
        if filename.endswith(TOKENIZER_GZ) and not has_gz:
            raise LocalEntryNotFoundError(f"{filename} not found")
        return f"/cache/{filename}"

    monkeypatch.setattr(_artifacts, "hf_hub_download", download)
    return calls


def test_hub_falls_back_to_plain_without_gz(monkeypatch):
    calls = fake_hub(monkeypatch, has_gz=False)
    paths = resolve_bundle("mini")
    assert paths["tokenizer"] == Path(f"/cache/mini/{TOKENIZER}")
    assert calls.index(f"mini/{TOKENIZER_GZ}") < calls.index(f"mini/{TOKENIZER}")
    assert len(calls) == len(set(calls)) == 5
    assert set(paths) == {"table", "head", "tokenizer", "manifest"}


def test_hub_uses_gz_when_present(monkeypatch):
    calls = fake_hub(monkeypatch, has_gz=True)
    paths = resolve_bundle("base")
    assert paths["tokenizer"] == Path(f"/cache/base/{TOKENIZER_GZ}")
    assert f"base/{TOKENIZER}" not in calls
    assert len(calls) == len(set(calls)) == 4
