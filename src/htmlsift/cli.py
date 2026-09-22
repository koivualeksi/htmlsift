"""`htmlsift` command line: prefetch model bundles into the Hugging Face cache.

    htmlsift download mini        # fetch + cache the mini bundle
    htmlsift download base        # fetch + cache the base bundle
"""
import argparse

from ._artifacts import BUNDLE_FILES, resolve_bundle


def _download(mode):
    paths = resolve_bundle(mode)          # hf_hub_download each file -> cached
    print(f"{mode}: cached {len(paths)} files")
    for name, p in paths.items():
        print(f"  {name:10} {p}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="htmlsift")
    sub = ap.add_subparsers(dest="cmd", required=True)
    dl = sub.add_parser("download", help="prefetch a model bundle into the HF cache")
    dl.add_argument("mode", choices=list(BUNDLE_FILES))
    args = ap.parse_args(argv)
    if args.cmd == "download":
        _download(args.mode)


if __name__ == "__main__":
    main()
