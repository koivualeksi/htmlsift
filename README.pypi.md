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
  numpy + huggingface-hub). First use downloads ~230 MB of model files.
- `htmlsift[full]` — adds `torch` + `transformers` to run `base`. First use of `base`
  downloads ~1 GB.

Requires Python 3.10+. Model files download from the Hugging Face Hub on first use and are
cached; run `htmlsift download <mode>` to prefetch, or set `HF_HUB_OFFLINE=1` to run from
cache only.

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

The larger model, from `htmlsift[full]`:

```python
ex = Extractor("base")               # 311M encoder; GPU when present, else CPU (with a warning)
ex = Extractor("base", device="cuda")  # require the GPU
```

`mini` is CPU-only; only `base` runs on the GPU.

## Output modes

| `output` | what you get |
|---|---|
| `text` (default) | selected content as plain text |
| `html` | selected DOM, with structure / links / images kept |
| `markdown` | clean markdown, links and images preserved |

## License

Apache-2.0. Model weights are distributed from the Hugging Face Hub, also under Apache-2.0, and
are trained on [WebMainBench](https://github.com/opendatalab/WebMainBench) (Apache-2.0); please
attribute WebMainBench when you use them.
