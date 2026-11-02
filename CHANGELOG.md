# Changelog

All notable changes to htmlsift are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/koivualeksi/htmlsift/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/koivualeksi/htmlsift/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/koivualeksi/htmlsift/releases/tag/v0.1.0
