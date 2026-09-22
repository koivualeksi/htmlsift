# htmlsift

Web main-content extraction: HTML in, clean text / HTML / markdown out. A small model,
trained on [WebMainBench](https://github.com/opendatalab/WebMainBench), scores each rendered
line of a page as main content or boilerplate (nav, ads, cookie banners, and footers) and
returns just the content.

Apache-2.0. Full documentation, diagrams, and benchmarks:
https://github.com/koivualeksi/htmlsift

## Install

```bash
pip install htmlsift          # default: the mini model (CPU, ONNX)
pip install htmlsift[full]    # adds the base 311M model (torch; GPU-capable)
```

- `htmlsift` — the `mini` model only; a small install (onnxruntime + tokenizers + lxml +
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
ex = Extractor("base", threads=4)    # 311M encoder
```

## Output modes

| `output` | what you get |
|---|---|
| `text` (default) | selected content as plain text |
| `html` | selected DOM, with structure / links / images kept |
| `markdown` | clean markdown, links and images preserved |

## License

Apache-2.0. Model weights are distributed separately, from the Hugging Face Hub.
