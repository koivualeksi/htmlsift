# Changelog

All notable changes to htmlsift are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `htmlsift --version` prints the htmlsift and Python versions and the installed
  backends: onnxruntime for `mini`, torch and transformers for `base`.

### Changed
- With w3lib 2.6 or newer, the five bytes windows-1252 leaves undefined (`0x81`, `0x8D`,
  `0x8F`, `0x90`, `0x9D`) decode to C1 control characters, as in browsers, instead of
  `U+FFFD`.
- `output="text"` (the default) returns plain text, as documented. Heading `#`, emphasis,
  inline-code and strikethrough markers, `>` quote prefixes, `* * *` rules and code-fence
  lines are no longer in it; list bullets, ordered-list numbers and table rows stay.
  Literal `#`, `*`, `_` or `|` in the page text are kept. The selected content is the
  same as in 0.2.0, and the model input and predictions are unchanged.
- `text` output renders the page a second time, without markers: about 3 ms per page
  more on one CPU core (mini, measured on CommonCrawl pages).
- The lines as the model scored them, markers included, are still available through
  `with_blocks=True`.

### Fixed
- `mode="base"` without torch or transformers installed raised a bare
  `No module named 'torch'`, and only after downloading the ~1 GB base bundle. It now
  raises `ImportError` before any download, naming the missing packages and
  `pip install "htmlsift[full]"`. The CLI prints the same message.

## [0.2.0] - 2026-11-02

### Added
- `extract()` accepts `bytes` and decodes them: byte order mark, then `encoding=`, then the
  page's `<meta>` declaration, then utf-8, falling back to cp1252. Legacy charset labels map
  the way browsers read them (`latin1` → cp1252, `shift_jis` → cp932, `gb2312` → gb18030).
- Keyword-only `encoding=` on `Extractor.extract` and `extract`, for a charset known from
  an HTTP or WARC header.
- Dependency: `w3lib` (BSD-3-Clause) for charset sniffing.
- Keyword-only `base_url=` on `Extractor.extract` and `extract`: links and image URLs in
  `html` and `markdown` output are made absolute against it. The page's first
  `<base href>` is resolved against `base_url`; malformed links are left as they are.
  `text` output and `with_blocks` are unaffected.
- `htmlsift extract` command: files, quoted globs or stdin in; the content of a single
  input, or one JSON line per file (`--jsonl` to force), out. Flags `--output`,
  `--mode`, `--device`, `--base-url`, `--encoding`. Failed files are reported as
  records and the run exits with 1.

### Changed
- Input other than `str` or `bytes` raises `TypeError`; `extract(None)` previously
  returned `""`.
- `encoding=` with `str` input raises `TypeError`, since a `str` is already decoded.
- A page with an absolute `<base href>` gets absolute links in `html` and `markdown` output
  even without `base_url`. Previously `<base>` was dropped with `<head>`, leaving its links
  relative.

### Fixed
- Non-utf-8 `bytes` input crashed with `UnicodeDecodeError`.
- `html` and `markdown` output could drop images and other inline content inside a kept
  block, varying from run to run on the same page. `text` output was unaffected.

## [0.1.1] - 2026-09-23

Documentation and packaging metadata only — no code or model changes since 0.1.0.

### Added
- Project URLs in package metadata (research repo, Hugging Face weights, issue tracker).
- PyPI Trusted Publishing workflow (`.github/workflows/publish.yml`).
- Cross-benchmark accuracy table (`base`/`mini` vs trafilatura, readability, resiliparse on
  WMB, WCXB, DAnIEL) and an "Annotation is a policy" section in the README.
- GPU-on-Windows install note (CUDA torch from the PyTorch index).

### Changed
- Stated the model-weight license (Apache-2.0) and added WebMainBench attribution.
- Cited the WebMainBench source paper (Liu et al., *Dripper*, arXiv:2511.23119).
- Clarified model shorthand and tightened copy in the README and the PyPI description.

## [0.1.0] - 2026-09-22

### Added
- Initial release.
- `mini` model (int8 ONNX, runs on CPU) and `base` model (311M encoder, GPU-capable),
  installed via `pip install htmlsift` and `pip install htmlsift[full]` respectively.
- `Extractor(mode, *, bundle_dir, threads, device)` with
  `.extract(html, output=..., with_blocks=...)`, plus the module-level `extract(...)`.
- Output formats: plain text, HTML, and markdown.
- Device selection for `base`: `device="cuda"` / `"cpu"`, auto-detecting a GPU and
  warning on CPU fallback.
- Model bundles fetched from the Hugging Face Hub, pinned to a fixed commit revision.

[Unreleased]: https://github.com/koivualeksi/htmlsift/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/koivualeksi/htmlsift/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/koivualeksi/htmlsift/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/koivualeksi/htmlsift/releases/tag/v0.1.0
