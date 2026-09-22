# AGENTS.md — htmlsift

## What this is

A pip-installable web main-content extractor. HTML string in, clean
text / HTML / markdown out, using the trained htmlsift models. Apache-2.0.
This is the production inference package; model training and reproduction
live in the separate research repo (see Resources).

Two modes:
- `mini` (default) — int8 ONNX, CPU, torch-free. `pip install htmlsift`
- `base` — 311M torch encoder, GPU-capable. `pip install htmlsift[full]`

Model weights are not in the repo. They are fetched from the Hugging Face
Hub on first use, pinned by commit SHA per release (see
`src/htmlsift/_artifacts.py`).

## Licence and dependencies

Apache-2.0. Rendering and serialization are lxml-only, and the package
keeps its dependency tree free of GPL-licensed code — that is what lets it
stay Apache-2.0. Changes that pull in a GPL dependency (for example
`html2text`) won't be accepted into the project.

## Layout

- `src/htmlsift/` — the package. Public API in `extractor.py`
  (`Extractor`, `extract`); model assembly in `loader.py`; bundle fetch
  in `_artifacts.py`; CLI in `cli.py`.
- `src/htmlsift/core/` — the pipeline: parse/render, prep, features,
  inference (`onnx_infer.py` for mini, `torch_infer.py` for base).
- `tests/` — golden and unit tests.

## Working with the code

- Backends are imported lazily, so a default install never loads torch;
  the torch/transformers imports live in `core/torch_infer.py`.
- Lint: `ruff check src tests`
- Test: `pytest -q`. Model golden tests skip unless `HTMLSIFT_BUNDLE`
  points at a local bundle (e.g. `localbundle/mini`).

## Resources

- `htmlsift-research` — training, evaluation, and reproduction:
  https://github.com/koivualeksi/htmlsift-research
