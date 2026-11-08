# htmlsift

Web main-content extraction: HTML in, clean text / HTML / markdown out. Trained on
[WebMainBench](https://github.com/opendatalab/WebMainBench), scores each rendered line of a
page as main content or boilerplate (nav, ads, cookie banners, and footers) and returns just
the content.

Apache-2.0. Full documentation, diagrams, and benchmarks:
https://github.com/koivualeksi/htmlsift

## Results

| model | desc | WMB test F1 | seeds | CPU (pages/s) | GPU (pages/s) | install |
|---|---|---|---|---|---|---|
| `base` | 311M, 10 layers | **0.9311** | 3 | 0.4 | 24.6 | `htmlsift[full]` |
| unpublished | 97M, 6 layers, int8 QAT | 0.9256 | 3 | 0.9 | 30.2 | — |
| `mini` | int8 embedding table | 0.9010 | 3 | **27.7** | — | `htmlsift` |

GPU = RTX 4090, CPU = Ryzen 5 3600 single thread; WMB test = 544 pages (one of the 545 has
no gold reference). On WMB (in-domain), `mini` scores ~15 F1 points above trafilatura at
comparable CPU speed.

### Cross-benchmark accuracy (zero-shot)

Trained on WMB, scored on benchmarks the model never saw. Each column uses that benchmark's own metric.

| model | WMB (ROUGE-5) | WCXB (word-F1) | DAnIEL (ROUGE-L) |
|---|---|---|---|
| `base` | 0.9311 | 0.8633 | 0.9175 |
| `mini` | 0.9010 | 0.8474 | 0.8797 |
| trafilatura | 0.7525 | 0.8584 | 0.8265 |
| readability | 0.8016 | 0.7653 | 0.8925 |
| resiliparse | 0.7135 | 0.7909 | 0.7094 |

Model rows are 3-seed test means; heuristics are single deterministic runs. "Main content" is a labeling policy: htmlsift leads in-domain and stays competitive off-policy — retrain on your own labels if your definition differs. Details in the [research repo](https://github.com/koivualeksi/htmlsift-research).

## Install

```bash
pip install htmlsift          # default: the mini model (CPU, ONNX)
pip install htmlsift[full]    # adds the base 311M model (torch; GPU-capable)
```

- `htmlsift` — the `mini` model only; few dependencies (onnxruntime + tokenizers + lxml +
  numpy + huggingface-hub + w3lib). First use downloads ~230 MB of model files.
- `htmlsift[full]` — adds `torch` + `transformers` to run `base`. First use of `base`
  downloads ~1 GB.

Requires Python 3.10+. Model files download from the Hugging Face Hub on first use and are
cached; run `htmlsift download <mode>` to prefetch, or set `HF_HUB_OFFLINE=1` to run from
cache only.

### Memory (`mini`)

~340 MB once loaded; 99 % of CommonCrawl pages need under 50 MB more. Memory grows with
page length (~21 KB per rendered line), so a very large page can exhaust RAM on a 1–2 GB
machine. Cap the input size before calling `extract()`.

## Usage

```python
from htmlsift import Extractor

ex = Extractor()                     # defaults to "mini"
ex.extract(html)                     # -> str (plain text)
ex.extract(html, output="html")      # structure, links, images kept
ex.extract(html, output="markdown")  # clean markdown: [text](url), ![alt](src)
ex.extract(html, with_blocks=True)   # -> (blocks, probs)

import htmlsift
htmlsift.extract(html)               # module-level convenience
```

Raw bytes are decoded for you (byte order mark, then `encoding=`, then `<meta charset>`,
then utf-8 / cp1252):

```python
ex.extract(response.content)                       # charset from the page itself
ex.extract(payload, encoding="windows-1251")       # charset from the HTTP / WARC header
```

Make links and image URLs absolute in `html` / `markdown` output (the page's `<base href>`
is honoured):

```python
ex.extract(html, output="markdown", base_url="https://example.com/blog/post")
```

Leave out the page title that starts the content, as pretraining text pipelines usually do
(the first line, when it is an `h1`–`h6` or `p` matching the page's `<title>` or
`og:title`; kept by default):

```python
ex.extract(html, drop_title=True)
```

The larger model, from `htmlsift[full]`:

```python
ex = Extractor("base")               # 311M encoder; GPU when present, else CPU (with a warning)
ex = Extractor("base", device="cuda")  # require the GPU
```

`mini` is CPU-only; only `base` runs on the GPU.

## Output modes

| `output` | what you get |
|---|---|
| `text` (default) | selected content as plain text: no heading, emphasis, code, quote, rule or fence markers; list bullets, ordered-list numbers and table rows (`\| a \| b \|`) stay |
| `html` | selected DOM, with structure / links / images kept |
| `markdown` | clean markdown, links and images preserved |

The lines exactly as the model scored them, markdown markers included, come from
`with_blocks=True`:

```python
blocks, probs = ex.extract(html, with_blocks=True)
scored = "\n".join(b for b, p in zip(blocks, probs) if p > 0.5)
```

## Command line

```bash
htmlsift extract page.html                          # main content of one page
curl -s https://example.com/a | htmlsift extract - --base-url https://example.com/a
htmlsift extract "crawl/*.html" --output markdown > out.jsonl   # one JSON line per file
htmlsift --version                                  # htmlsift, Python and backend versions
```

Flags: `--output`, `--mode`, `--device`, `--base-url`, `--encoding`, `--drop-title`,
`--jsonl` (records
even for a single file). Failed files become `{"path", "error"}` records and the exit code
is 1. Output is utf-8.

## License

Apache-2.0. Model weights are distributed from the Hugging Face Hub, also under Apache-2.0, and
are trained on [WebMainBench](https://github.com/opendatalab/WebMainBench) (Apache-2.0); please
attribute WebMainBench when you use them.
