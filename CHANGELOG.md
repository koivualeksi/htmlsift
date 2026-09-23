# Changelog

All notable changes to htmlsift are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/koivualeksi/htmlsift/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/koivualeksi/htmlsift/releases/tag/v0.1.0
