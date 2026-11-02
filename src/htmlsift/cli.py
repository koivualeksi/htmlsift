"""`htmlsift` command line.

    htmlsift extract page.html                      # main content of one page
    htmlsift extract "crawl/*.html" > out.jsonl     # one JSON record per file
    curl -s URL | htmlsift extract - --base-url URL
    htmlsift download mini                          # prefetch a model bundle
"""
import argparse
import errno
import glob
import json
import os
import sys
from pathlib import Path

from w3lib.encoding import resolve_encoding

from ._artifacts import BUNDLE_FILES, resolve_bundle
from .extractor import Extractor


def _download(mode):
    paths = resolve_bundle(mode)          # hf_hub_download each file -> cached
    print(f"{mode}: cached {len(paths)} files")
    for name, p in paths.items():
        print(f"  {name:10} {p}")
    return 0


def _is_glob(arg):
    """A pattern to expand: not stdin, not an existing path, and has glob characters."""
    return arg != "-" and not os.path.exists(arg) and any(c in arg for c in "*?[")


def _resolve(arg):
    """One argument -> [(path, error)]; error is None for an input to read."""
    if os.path.isdir(arg):
        return [(arg, f'is a directory; use "{arg}/*.html"')]
    if _is_glob(arg):
        matches = sorted(p for p in glob.glob(arg, recursive=True) if not os.path.isdir(p))
        return [(Path(p).as_posix(), None) for p in matches] or [(arg, "no files match")]
    return [(arg, None)]


def _read(path):
    return sys.stdin.buffer.read() if path == "-" else Path(path).read_bytes()


class _PipeClosed(Exception):
    pass


def _write(text):
    try:
        sys.stdout.buffer.write(text.encode("utf-8"))
        sys.stdout.buffer.flush()
    except OSError as e:
        # Reader closed the pipe (e.g. `| head`); Windows reports it as EINVAL.
        if isinstance(e, BrokenPipeError) or e.errno == errno.EINVAL:
            raise _PipeClosed from e
        raise


def _emit(path, content, error, raw):
    if raw and error is not None:
        print(f"htmlsift: {path}: {error}", file=sys.stderr)
    elif raw:
        _write(content.rstrip("\n") + "\n")
    else:
        rec = {"path": path, "error": error} if error is not None else {"path": path,
                                                                        "content": content}
        _write(json.dumps(rec, ensure_ascii=False) + "\n")


def _extract(ap, args):
    if args.inputs.count("-") > 1:
        ap.error("'-' (stdin) can be given only once")
    if args.encoding is not None and resolve_encoding(args.encoding.strip()) is None:
        ap.error(f"unknown encoding: {args.encoding}")
    ex = Extractor(args.mode, device=args.device)
    try:
        _ = ex.model
    except Exception as e:
        print(f"htmlsift: cannot load model: {e}", file=sys.stderr)
        return 1
    raw = len(args.inputs) == 1 and not _is_glob(args.inputs[0]) and not args.jsonl
    failed = False
    try:
        for arg in args.inputs:
            for path, error in _resolve(arg):
                content = None
                if error is None:
                    try:
                        content = ex.extract(_read(path), output=args.output,
                                             encoding=args.encoding, base_url=args.base_url)
                    except Exception as e:
                        error = str(e) or type(e).__name__
                failed = failed or error is not None
                _emit(path, content, error, raw)
    except _PipeClosed:
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 1
    return 1 if failed else 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="htmlsift")
    sub = ap.add_subparsers(dest="cmd", required=True)

    dl = sub.add_parser("download", help="prefetch a model bundle into the HF cache")
    dl.add_argument("mode", choices=list(BUNDLE_FILES))

    ex = sub.add_parser(
        "extract", help="extract main content from HTML files",
        description="Extract main content. One file or '-' (stdin) prints the content; "
                    "several files or a glob print one JSON record per file.")
    ex.add_argument("inputs", nargs="+", metavar="INPUT",
                    help="HTML file, glob pattern (quote it; ** recurses), or - for stdin")
    ex.add_argument("--output", choices=["text", "html", "markdown"], default="text")
    ex.add_argument("--mode", choices=list(BUNDLE_FILES), default="mini")
    ex.add_argument("--device", help="torch device for --mode base, e.g. cuda or cpu")
    ex.add_argument("--base-url", help="make links absolute (html/markdown); applies to "
                                       "every input")
    ex.add_argument("--encoding", help="charset of the input bytes; applies to every input")
    ex.add_argument("--jsonl", action="store_true",
                    help="always print JSON records, even for a single input")

    args = ap.parse_args(argv)
    if args.cmd == "download":
        return _download(args.mode)
    return _extract(ap, args)


if __name__ == "__main__":
    sys.exit(main())
