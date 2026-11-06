"""`htmlsift extract` with a fake Extractor: input resolution, raw vs JSONL output, errors
and exit codes. No model bundle needed.
"""
import io
import json
from types import SimpleNamespace

import pytest

from htmlsift.cli import main


class FakeExtractor:
    """Returns the page decoded as-is; raises for pages containing FAIL."""
    last = None

    def __init__(self, mode="mini", *, device=None):
        self.mode, self.device, self.calls = mode, device, []
        FakeExtractor.last = self

    @property
    def model(self):
        return object()

    def extract(self, html, output="text", *, encoding=None, base_url=None):
        self.calls.append({"output": output, "encoding": encoding, "base_url": base_url})
        text = html.decode(encoding or "utf-8")
        if "FAIL" in text:
            raise ValueError("bad page")
        return text


class BrokenModelExtractor(FakeExtractor):
    @property
    def model(self):
        raise OSError("no bundle")


@pytest.fixture
def run(monkeypatch, tmp_path, capfdbinary):
    """Run main() in tmp_path with the fake Extractor -> (exit code, stdout, stderr)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("htmlsift.cli.Extractor", FakeExtractor)

    def _run(*argv):
        code = main(["extract", *argv])
        out, err = capfdbinary.readouterr()
        return code, out, err.decode("utf-8")
    return _run


def _files(tmp_path, **pages):
    for name, text in pages.items():
        p = tmp_path / name.replace("__", "/")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(text.encode("utf-8"))


def _records(out):
    return [json.loads(line) for line in out.decode("utf-8").splitlines()]


def test_single_file_raw(run, tmp_path):
    _files(tmp_path, **{"a.html": "hello\n\n"})
    assert run("a.html") == (0, b"hello\n", "")


def test_empty_content_raw(run, tmp_path):
    _files(tmp_path, **{"a.html": ""})
    assert run("a.html") == (0, b"\n", "")


def test_single_file_jsonl(run, tmp_path):
    _files(tmp_path, **{"a.html": "hello"})
    code, out, _ = run("a.html", "--jsonl")
    assert code == 0
    assert _records(out) == [{"path": "a.html", "content": "hello"}]


def test_several_files_in_argument_order(run, tmp_path):
    _files(tmp_path, **{"a.html": "A", "b.html": "B"})
    code, out, _ = run("b.html", "a.html")
    assert code == 0
    assert _records(out) == [{"path": "b.html", "content": "B"},
                             {"path": "a.html", "content": "A"}]


def test_glob_with_one_match_is_jsonl(run, tmp_path):
    _files(tmp_path, **{"a.html": "A"})
    _, out, _ = run("*.html")
    assert _records(out) == [{"path": "a.html", "content": "A"}]


def test_glob_matches_sorted_posix_dirs_skipped(run, tmp_path):
    _files(tmp_path, **{"d__b.html": "B", "d__a.html": "A", "d__sub__c.html": "C"})
    _, out, _ = run("d/*")
    assert [r["path"] for r in _records(out)] == ["d/a.html", "d/b.html"]


def test_recursive_glob(run, tmp_path):
    _files(tmp_path, **{"d__a.html": "A", "d__sub__c.html": "C"})
    _, out, _ = run("d/**/*.html")
    assert [r["path"] for r in _records(out)] == ["d/a.html", "d/sub/c.html"]


def test_existing_path_with_glob_chars_is_literal(run, tmp_path):
    _files(tmp_path, **{"page[1].html": "P"})
    assert run("page[1].html") == (0, b"P\n", "")


def test_stdin(run, monkeypatch):
    monkeypatch.setattr("sys.stdin", SimpleNamespace(buffer=io.BytesIO(b"from stdin")))
    assert run("-") == (0, b"from stdin\n", "")


def test_input_errors_are_records(run, tmp_path):
    _files(tmp_path, **{"a.html": "A", "d__x.html": "X", "bad.html": "FAIL"})
    code, out, _ = run("a.html", "missing.html", "d", "*.zz", "bad.html")
    recs = _records(out)
    assert code == 1
    assert recs[0] == {"path": "a.html", "content": "A"}
    assert recs[1]["path"] == "missing.html" and "No such file" in recs[1]["error"]
    assert recs[2] == {"path": "d", "error": 'is a directory; use "d/*.html"'}
    assert recs[3] == {"path": "*.zz", "error": "no files match"}
    assert recs[4] == {"path": "bad.html", "error": "bad page"}


def test_single_failing_input_goes_to_stderr(run, tmp_path):
    _files(tmp_path, **{"bad.html": "FAIL"})
    code, out, err = run("bad.html")
    assert (code, out) == (1, b"")
    assert err == "htmlsift: bad.html: bad page\n"


def test_model_load_failure(run, monkeypatch, tmp_path):
    _files(tmp_path, **{"a.html": "A"})
    monkeypatch.setattr("htmlsift.cli.Extractor", BrokenModelExtractor)
    code, out, err = run("a.html", "b.html")
    assert (code, out) == (1, b"")
    assert err == "htmlsift: cannot load model: no bundle\n"


def test_base_without_deps_hint(monkeypatch, tmp_path, capfd):
    """Real Extractor: the missing-deps hint reaches stderr and no bundle is fetched."""
    (tmp_path / "a.html").write_text("<p>A</p>")
    monkeypatch.setattr("htmlsift.loader.find_spec", lambda name: None)

    def no_download(*a, **k):
        raise AssertionError("bundle fetched")
    monkeypatch.setattr("htmlsift.loader.resolve_bundle", no_download)
    code = main(["extract", str(tmp_path / "a.html"), "--mode", "base"])
    out, err = capfd.readouterr()
    assert (code, out) == (1, "")
    assert err == ("htmlsift: cannot load model: mode='base' needs torch, transformers: "
                   'pip install "htmlsift[full]"\n')


@pytest.mark.parametrize("argv", [["-", "-"], ["a.html", "--encoding", "bogus"]])
def test_usage_errors(run, argv):
    with pytest.raises(SystemExit) as exc:
        run(*argv)
    assert exc.value.code == 2


def test_options_reach_extractor(run, tmp_path):
    (tmp_path / "a.html").write_bytes("caf\xe9".encode("cp1252"))
    code, out, _ = run("a.html", "--output", "markdown", "--encoding", "cp1252",
                       "--base-url", "https://example.com/", "--mode", "base",
                       "--device", "cpu")
    ex = FakeExtractor.last
    assert (code, out) == (0, "café\n".encode())
    assert (ex.mode, ex.device) == ("base", "cpu")
    assert ex.calls == [{"output": "markdown", "encoding": "cp1252",
                         "base_url": "https://example.com/"}]


def test_utf8_output_not_escaped(run, tmp_path):
    _files(tmp_path, **{"ja.html": "日本語", "ru.html": "Привет"})
    code, out, _ = run("ja.html", "ru.html")
    assert code == 0
    assert "日本語".encode() in out and "Привет".encode() in out
    assert b"\\u" not in out
