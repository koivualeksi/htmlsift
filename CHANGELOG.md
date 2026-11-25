# Changelog

All notable changes to htmlsift are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `dedup=True` (CLI `--dedup`): leaves out each selected line of 100 or more characters
  that repeats an earlier selected line, keeping the first copy, in `text`, `html` and
  `markdown`. Lines are compared ignoring case, punctuation and markers. Lines in a
  `<pre>` are never deduped. Off by default; `with_blocks=True`, the model input and the
  predictions are unchanged.
- `threshold=` (CLI `--threshold`): a line is kept when its probability is above it,
  0 to 1, default 0.5. Lower keeps more, higher keeps less. Only 0.5 is benchmarked.
  `with_blocks=True` still returns every probability.

### Changed
- New mini model: the embedding table is int8 rank-128 (34.6 MB, was 768 wide and
  202 MB). The release pin moves to Hub commit `a86f08a`; `base/` is unchanged apart from
  its added `tokenizer.json.gz`. The first mini download is ~43 MB, and base fetches the
  5.4 MB gzipped tokenizer instead of the 33 MB plain one.
- Dependencies now have tested bounds: `lxml>=6.0.4,<7`, `tokenizers>=0.21,<0.24`,
  `onnxruntime>=1.20.1`, `numpy>=1.26.4`, `huggingface-hub>=0.26.5`, `w3lib>=2.3.1,<3`
  (was `>=2.5,<3`), and for `[full]` `torch>=2.7.1`, `transformers>=5.1,<6`. Tested on
  Linux: across these ranges the mini model gives the same `text`, `html` and `markdown`
  on 2,040 test pages (onnxruntime before 1.30 shifts probabilities by at most 2.4e-6).
  The base model gives the same line selection on a 13-page sample, except that torch
  2.7-2.12 shifts probabilities by up to 0.014 and can flip a line that sits right at the
  0.5 threshold. lxml 5.x is excluded: its older libxml2 renders some pages differently from
  the libxml2 2.14 the models were trained on. CI now also runs the suite with the lowest
  allowed versions.
- The mini model now mean-pools each block's token embeddings with one vectorized
  segment sum, in batches of 32,768 tokens, instead of a Python loop over blocks.
  Probabilities can differ from before by about 1e-6; on 2,040 test pages no block
  changed side of the 0.5 threshold.
- The mini model's per-page memory estimate is sized for the rank-128 table: 1 KB per
  distinct token id and 12 KB per block (was 12 and 21, sized at 768 wide). Pages
  estimated above 100 MB still give onnxruntime's arena back after they run; with the
  new sizes that is pages with about 8,500 or more blocks. Output is unchanged.
- The tokenizer is read from `tokenizer.json.gz` when the bundle has one, and from
  `tokenizer.json` otherwise, both from the Hub and from a local bundle folder. Pinned
  bundles without the gzipped file load as before.

### Fixed
- A document that parses to no element (only comments, a doctype, an `<?xml?>`
  declaration or other processing instructions; with libxml2 2.11, as in lxml's Windows
  wheels, also a lone BOM or leading NUL bytes) raised
  `lxml.etree.ParserError: Document is empty`. It now returns `""` in every output, and
  `([], [])` with `with_blocks=True`, as an empty string does.
- An unknown `output` (e.g. `output="md"`) raised `ValueError` only after the model had
  run on the page, and not at all with `with_blocks=True`. It now raises before any
  inference, in every case.
- `output="html"` and `output="markdown"` no longer contain text from lines the model
  rejected. They kept whole elements, so the text of a parent element, the text right
  after a kept element and the children of a kept element came back with it. They now
  keep exactly the text of the selected lines, as `output="text"` does. A code block (a
  `<pre>` with no block elements inside) stays whole when any of its lines is selected.
  `text` output, the model input and the predictions are unchanged.
- `output="html"` and `output="markdown"` no longer glue separate lines together
  (`Personal incomePopulation`) when the `<br>` or block element between them was not
  kept; a `<br>` takes its place.
- With `base_url=` (or an absolute `<base href>`), a relative SVG `xlink:href` stayed
  relative in `output="html"` on lxml < 6.1. It is now made absolute on every lxml
  version, as on 6.1+.
- With libxml2 2.14 (lxml's Linux wheels), `output="html"` and `output="markdown"`
  raised `ValueError: All strings must be XML compatible` on pages with C0 control
  characters (such as `\x03` or `&#3;`) in text or attributes. The parser now removes
  them, as libxml2 2.11 (lxml's Windows wheels) already did. Pages without them, and every
  page on Windows, give the same output as before; on Linux, the `text` output and the
  model input of affected pages lose the characters too. None of the 7,825 WebMainBench
  training and test pages contain them.

### Security
- `output="html"` and `output="markdown"` are now cleaned of active content. Kept
  elements could carry `<script>`, `<style>`, event handlers (`onclick=`), `srcdoc` and
  `javascript:` links, and markdown wrote `[text](javascript:...)`. Now the elements the
  model never reads (`script`, `style`, `title`, `meta`, `link`, `base`) and SVG
  `<animate>` / `<set>` are dropped, `on*` and `srcdoc` attributes are removed, and a link
  or source whose scheme is not `http`, `https`, `mailto`, `tel`, `ftp` or relative loses
  its target (its text stays); `data:image/` in `<img src>` is kept. SVG `xlink:href` is
  checked the same way on every lxml version (lxml < 6.1 does not list it as a link
  attribute). A basic cleanup, not a guarantee. `text` output, the model input and the
  predictions are unchanged.

## [0.2.1] - 2026-11-08

### Added
- `htmlsift --version` prints the htmlsift and Python versions and the installed
  backends: onnxruntime for `mini`, torch and transformers for `base`.
- `drop_title=True` in `extract()` and `--drop-title` in the CLI leave out the page title
  that starts the content: the first selected line, when it comes from an `h1`–`h6` or
  `p` element holding only that line and matches the page's `<title>` or `og:title`.
  Applies to `text`, `html` and `markdown`. Off by default.

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
- `output="text"` no longer contains lines made only of invisible characters (zero-width
  spaces and joiners, byte order marks, soft hyphens, control characters), which pages
  use as spacers.
- `mini` memory on very long pages. The embedding step no longer grows with page length
  (a 3M-token page peaked at 39 GB, now 1.2 GB), and pages estimated above 100 MB hand
  their memory back to the system when they finish instead of keeping it for the life
  of the process. Output is unchanged; model time is about 15 % lower on CommonCrawl
  pages.
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

[Unreleased]: https://github.com/koivualeksi/htmlsift/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/koivualeksi/htmlsift/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/koivualeksi/htmlsift/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/koivualeksi/htmlsift/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/koivualeksi/htmlsift/releases/tag/v0.1.0
