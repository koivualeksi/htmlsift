"""Where the model bundles live and how the package fetches them.

Bundles are published to a public HuggingFace repo and pinned by commit SHA per
release, so a given htmlsift version always fetches byte-identical files. Fetching
uses hf_hub_download, which caches to the standard HF cache; a local directory can
override the fetch during development.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "koivualeksi/htmlsift-release"
REVISION = "73ac2b978aef356e9e5772640a0c9444f2b67705"   # release commit with both mini/
                         # and base/ bundles. Pin per release so a version fetches byte-
                         # identical files; None (dev only) would track the default branch.

# Bundle filenames. TABLE/HEAD are the mini ONNX graphs and must match what
# tools/release.py publishes (research-side).
TABLE = "mini-table-int8.onnx"
HEAD = "mini-head-fABC.onnx"
TOKENIZER = "tokenizer.json"
MANIFEST = "manifest.json"

# Each mode's bundle as {logical name -> filename}. Remote layout is <mode>/<filename>.
BUNDLE_FILES = {
    "mini": {"table": TABLE, "head": HEAD, "tokenizer": TOKENIZER, "manifest": MANIFEST},
    "base": {"config": "config.json", "weights": "weights.pt",
             "tokenizer": TOKENIZER, "manifest": MANIFEST},
}


@dataclass
class Manifest:
    """The per-model config that travels in the bundle (feats spec + z-score stats +
    sizes), read from manifest.json so the package needs no torch to load it."""
    mode: str
    feats: str | None = None
    hidden: int | None = None
    window: int = 8192
    cap: int = 0
    zscore_mean: list = field(default_factory=list)
    zscore_std: list = field(default_factory=list)

    @classmethod
    def from_file(cls, path):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        z = d.get("zscore") or {}
        return cls(mode=d["mode"], feats=d.get("feats"), hidden=d.get("hidden"),
                   window=d.get("window", 8192), cap=d.get("cap", 0),
                   zscore_mean=z.get("mean", []), zscore_std=z.get("std", []))


def resolve_bundle(mode, local_dir=None):
    """{logical name -> local file path} for `mode`'s bundle. A local_dir reads the
    files straight off disk (development); otherwise each is pulled from the release
    repo at the pinned revision and cached by huggingface_hub."""
    files = BUNDLE_FILES[mode]
    if local_dir:
        base = Path(local_dir)
        return {name: base / fn for name, fn in files.items()}
    return {name: Path(hf_hub_download(REPO, f"{mode}/{fn}", revision=REVISION))
            for name, fn in files.items()}
