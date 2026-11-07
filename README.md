# htmlsift

Main-content extraction for web pages: hand it a raw HTML string and get back the main content
 - nav, ads, cookie banners, and footers stripped
 - outputs plain text, **HTML**, or **markdown**.

Two models shipped, both trained on [WebMainBench][wmb] (WMB) — 7,825 labelled pages, custom split
6,548 train / 732 val / 545 test:

- **`mini`** (default) - a fast int8-ONNX model that runs on CPU. `pip install htmlsift`
- **`base`** - a 311M encoder, most accurate and GPU-capable. `pip install htmlsift[full]`

Each model is fully re-trainable with your own annotated data and the whole research suite is available at [htmlsift-research](https://github.com/koivualeksi/htmlsift-research)

[wmb]: https://github.com/opendatalab/WebMainBench

## Results

| model | desc | WMB test F1 | seeds | CPU (pages/s) | GPU (pages/s) | install |
|---|---|---|---|---|---|---|
| `base` | 311M, 10 layers | **0.9311** | 3 | 0.4 | 24.6 | `htmlsift[full]` |
| unpublished | 97M, 6 layers, int8 QAT | 0.9256 | 3 | 0.9 | 30.2 | — |
| `mini` | int8 embedding table | 0.9010 | 3 | **27.7** | — | `htmlsift` |

* GPU = RTX 4090
* CPU = Ryzen 5 3600 single thread
* WMB test = 544 pages: one of the 545 test pages has no gold reference and is skipped in scoring.

### On CPU

![htmlsift mini vs trafilatura, readability, resiliparse: F1 and ms/page on one CPU core](docs/assets/cpu_speed_accuracy.svg)

One CPU core, WMB test (544 pages), extraction only. `mini` gets close to trafilatura's
per-page time (36 vs 31 ms median) and scores ~15 F1 points higher. readability and resiliparse are faster but markedly less accurate. 

### On GPU

![htmlsift base sustains 21x the throughput of MinerU-HTML at matched F1 on an RTX 4090](docs/assets/gpu_speed_accuracy_lite.svg)

RTX 4090, WMB test (544 pages). `base` sustains 24.6 pages/s against
MinerU-HTML v1.1's 1.16. A 21x gap at matched accuracy (F1 0.9311 vs 0.9306).

### Cross benchmark accuracy

The shipped models are trained on WMB, so WMB is in-domain. The other two columns are
**zero-shot**, the model never saw WCXB or the multilingual DAnIEL in training. Each column is
scored with that benchmark's own metric. `base` leads every heuristic here; `mini` lands just
behind the best on each board — see **Annotation is a policy** under Concept.

| model | WMB (ROUGE-5) | WCXB (word-F1, zero-shot) | DAnIEL (ROUGE-L, zero-shot) |
|---|---|---|---|
| `base` | 0.9311 | 0.8633 | 0.9175 |
| `mini` | 0.9010 | 0.8474 | 0.8797 |
| trafilatura | 0.7525 | 0.8584 | 0.8265 |
| readability | 0.8016 | 0.7653 | 0.8925 |
| resiliparse | 0.7135 | 0.7909 | 0.7094 |

Model rows are 3-seed test means; heuristics are single deterministic runs.

## Installation

```bash
pip install htmlsift          # default: the mini model (CPU, ONNX)
pip install htmlsift[full]    # adds the base 311M model (torch; GPU-capable)
```

- **`htmlsift`** — the `mini` model only; few dependencies (onnxruntime + tokenizers + lxml +
  numpy + huggingface-hub + w3lib). First use downloads ~230 MB of model files.
- **`htmlsift[full]`** — adds `torch` + `transformers` to run `base`. First use of `base`
  downloads ~1 GB.

Requires Python 3.10+. Model files download from the Hugging Face Hub on first use and are
cached; prefetch with `htmlsift download mini` (or `base`), or set `HF_HUB_OFFLINE=1` to run
from cache only.

### Memory (`mini`)

- **Fixed:** ~340 MB once the model is loaded.
- **Per page:** 99 % of CommonCrawl pages need under 50 MB on top of that.
- **Long pages:** memory grows with the page, about 21 KB per rendered line plus ~0.2 KB
  per token. The largest page in an 8,820-page CommonCrawl sample needed ~0.8 GB.
- **Release:** a page estimated above 100 MB hands its memory back to the system when it
  finishes, so one large page doesn't leave a long-running worker big.
- **Small machines:** page size has no built-in limit, so a very large page can exhaust
  RAM on a 1–2 GB machine. Cap the input size before calling `extract()`.

## Usage

```python
from htmlsift import Extractor

ex = Extractor()                     # defaults to "mini"
ex.extract(html)                     # -> str (plain text)
ex.extract(html, output="html")      # keep structure, links, images
ex.extract(html, output="markdown")  # clean markdown: [text](url), ![alt](src)
ex.extract(html, with_blocks=True)   # -> (blocks, probs) for callers who want each line and its prediction individually
```

Module-level convenience (cached singleton):

```python
import htmlsift
htmlsift.extract(html)               # default mini
```

### Bytes and encodings

Pass raw bytes from a file, an HTTP response or a WARC record, and htmlsift decodes them:

```python
ex.extract(response.content)                       # charset from the page itself
ex.extract(payload, encoding="windows-1251")       # charset from the HTTP / WARC header
```

The charset is taken from, in order: a byte order mark, `encoding=`, the page's
`<meta charset>`; an undeclared page is read as utf-8, falling back to cp1252. Unknown
labels are skipped and undecodable bytes become `�`. A `str` counts as already decoded, so
`encoding=` with a `str` raises `TypeError`. Prefer passing bytes over a `str` decoded
upstream: `requests`' `response.text`, for one, guesses ISO-8859-1 for pages without a
charset header.

### Absolute links

Pass the page's URL to make links and image URLs absolute in `html` and `markdown` output:

```python
ex.extract(html, output="markdown", base_url="https://example.com/blog/post")
# [the report](report)  ->  [the report](https://example.com/blog/report)
```

The page's first `<base href>` is resolved against `base_url`, as a browser does. Without
`base_url`, an absolute `<base href>` is used on its own; with neither, links stay as they
are. Malformed links are left unchanged. `text` output has no links, so it ignores
`base_url`.

The larger model, from the `[full]` install:

```python
ex = Extractor("base")               # 311M encoder; uses the GPU when one is present,
                                     # else CPU (with a warning)
ex = Extractor("base", device="cuda")  # require the GPU (errors if none is available)
ex = Extractor("base", device="cpu", threads=4)  # force CPU; `threads` sets parallelism
```

`mini` is CPU-only (int8 ONNX); asking it for a GPU raises. Only `base` runs on the GPU.

### Example

Nearly 1 MB of Wikipedia HTML ([Photosynthesis](https://en.wikipedia.org/wiki/Photosynthesis))
in — the boilerplate (nav, sidebar, citations, footer) dropped, the article kept. The verbatim
head of the markdown (`mini` and `base` return the same here):

```python
md = Extractor().extract(html, output="markdown")
```

```markdown
# Photosynthesis

From Wikipedia, the free encyclopedia

For other uses, see [Photosynthesis (disambiguation)](https://en.wikipedia.org/wiki/Photosynthesis_(disambiguation)).

Schematic of photosynthesis in plants. The [carbohydrates](https://en.wikipedia.org/wiki/Carbohydrate) produced are stored in or used by the plant.

Composite image showing the global distribution of photosynthesis, including both oceanic [phytoplankton](https://en.wikipedia.org/wiki/Phytoplankton) and terrestrial [vegetation](https://en.wikipedia.org/wiki/Vegetation). Dark red and blue-green indicate regions of high photosynthetic activity in the ocean and on land, respectively.

**Photosynthesis** is a [system](https://en.wikipedia.org/wiki/Biological_system) of [biological processes](https://en.wikipedia.org/wiki/Biological_process) by which [photopigment](https://en.wikipedia.org/wiki/Photopigment)-bearing [autotrophic](https://en.wikipedia.org/wiki/Autotroph) [organisms](https://en.wikipedia.org/wiki/Organism), such as most [plants](https://en.wikipedia.org/wiki/Plant), [algae](https://en.wikipedia.org/wiki/Algae) and [cyanobacteria](https://en.wikipedia.org/wiki/Cyanobacteria), convert light energy—typically from sunlight—into the chemical energy necessary to fuel their metabolism…
```

### Output modes

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
htmlsift extract page.html --output markdown --base-url https://example.com/page
curl -s https://example.com/a | htmlsift extract - --base-url https://example.com/a
htmlsift extract "crawl/*.html" > out.jsonl         # one JSON record per file
htmlsift extract "crawl/**/*.html" --mode base --device cuda > out.jsonl
htmlsift --version                                  # htmlsift, Python and backend versions
```

One file or `-` (stdin) prints the content itself. Several files or a glob print one JSON
line per file, `{"path": ..., "content": ...}`, or `{"path": ..., "error": ...}` for a file
that failed; the run continues and exits with 1 at the end. Pass `--jsonl` to get records
for a single file too, which scripts should do: bash expands an unquoted `*.html` before
htmlsift sees it. Quoted globs are expanded by htmlsift itself (`**` recurses), so they
work the same in every shell; directories are not walked.

| flag | |
|---|---|
| `--output text\|html\|markdown` | as `output=` above (default `text`) |
| `--mode mini\|base`, `--device` | model and torch device, as for `Extractor` |
| `--base-url URL` | make links absolute; applies to every input |
| `--encoding ENC` | charset of the input bytes; applies to every input |
| `--jsonl` | always print JSON records |

Files are read as bytes, so the charset is detected as described above. Output is utf-8.
Exit codes: 0 ok, 1 a file or the model failed, 2 a usage error. In Windows PowerShell 5.1,
`>` re-encodes a program's output; redirect from bash, cmd or PowerShell 7 instead.

## Concept

<details>
<summary><b>How it works</b></summary>

### Line level labeling
![htmlsift labels rendered lines, not DOM elements](docs/assets/concept.svg)

Most extractors decide keep-or-drop on HTML *elements*. htmlsift first renders the page to
markdown *lines*, then decides keep-or-drop on each line. The trade-off is in the figure above, a line-based ceiling sits below the element level gold ceiling but there are far fewer units to classify per page.

### HTML 'signals' utilized

- **Where it sits**: in the DOM - tag, nesting depth, whether it's inside a link.
- **What it says**: a multilingual embedding of the line's tokens.
- **The whole-page context**: repeated boilerplate only stands out in contrast to the rest
  of the page, so every line is judged against all the others by a BiGRU.

![The three signals, and which model tier carries each](docs/assets/signals.svg)

Heuristic extractors carry the first one and a half of these; an LLM carries all three but
pays to generate the page token by token. htmlsift carries all three in one pass, and outputs predictions without needing a decoder.

![One pass over the page: base vs mini](docs/assets/architecture.svg)

Both tiers share the render, the per-line pooling, and the BiGRU head. They differ only in the text processing section
- mini converts text to tokens and mean-pools them, has no internal attention allowing fast CPU processing
- base uses granite r2 311m embedder using only first 10 layers requiring GPU for efficient usage 


</details>

<details>
<summary><b>Annotation is a policy</b></summary>

![The same page kept differently under different benchmark annotation policies](docs/assets/annotation-example.svg)

htmlsift learns what counts as main content from a benchmark's labels, so it's most accurate on pages that follow that policy. On WebMainBench (the benchmark the shipped models train on) it beats the common heuristic extractors by a wide margin. Zero-shot on benchmarks with different policies, the gap narrows: the 311M `base` still edges every heuristic tested (trafilatura, readability, resiliparse) on both WCXB and DAnIEL, and `mini` lands just behind the strongest heuristic on each board, ahead of the rest.

Train htmlsift on a different policy and it turns around. Fit it to WCXB instead of WMB and it tops WCXB but loses points on other benchmarks. For example when training `base` model on WCXB data, WMB falls to 0.8479, DAnIEL to 0.8826 while WCXB climbs up to 0.9209. Whichever policy you train on, htmlsift leads there and pays for it elsewhere.

For more information, review the [htmlsift-research](https://github.com/koivualeksi/htmlsift-research) readme.

</details>

<details>
<summary><b>Training and retraining</b></summary>

Most of the accuracy arrives in the first few hundred labelled pages — the model learns a
benchmark's labelling policy quickly, so retraining on your own labels is cheap.

![Accuracy vs training-set size, three model arms](docs/assets/data-efficiency.svg)

These models were trained in the **htmlsift-research** repository. To reproduce the boards or
retrain on your own data, see [htmlsift-research](https://github.com/koivualeksi/htmlsift-research)
and its `docs/REPRODUCE.md`.

</details>

<details>
<summary><b>Known limits</b></summary>

A standalone hero image that is its own block renders to no text line, so it can't be selected
today. Inline images inside a kept block are preserved in `html` and `markdown`.

`base_url` does not rewrite `srcset`, so responsive images in `html` output keep relative
candidates (`src` is rewritten; `markdown` uses only `src`).

</details>

## License

Apache-2.0. See [LICENSE](LICENSE).

Model weights are distributed from the Hugging Face Hub, also under Apache-2.0, and are
trained on [WebMainBench](https://github.com/opendatalab/WebMainBench) (Apache-2.0); please
attribute WebMainBench when you use them.

## References

- **WMB — WebMainBench.** The training benchmark; ROUGE-5 main-content metric.
  Liu et al., *Dripper: Token-Efficient Main HTML Extraction with a Lightweight LM*.
  [arXiv:2511.23119](https://arxiv.org/abs/2511.23119) ·
  [repo](https://github.com/opendatalab/WebMainBench)
- **WCXB — Web Content Extraction Benchmark.** Foley, *WCXB: A Multi-Type Web Content Extraction
  Benchmark*. [arXiv:2605.21097](https://arxiv.org/abs/2605.21097) ·
  [repo](https://github.com/Murrough-Foley/web-content-extraction-benchmark)
- **DAnIEL.** Multilingual news corpus (Mutuvi et al.,
  [LREC 2020](https://aclanthology.org/2020.lrec-1.509/)); main-content scoring follows
  *Multilingual Evaluation of Main Content Extractors for Web Pages*, SIGIR 2025
  ([ACM](https://dl.acm.org/doi/10.1145/3726302.3730234)).

