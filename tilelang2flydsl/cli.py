"""CLI: `python -m tilelang2flydsl <input.py> -o <output.py>`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .diagnostics import TranslationError
from .translate import translate_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tilelang2flydsl",
        description="Translate a TileLang kernel into a FlyDSL (gfx942) kernel.",
    )
    parser.add_argument("input", help="path to the TileLang Python source file")
    parser.add_argument("-o", "--output", help="output path (defaults to stdout)")
    parser.add_argument(
        "--no-format",
        action="store_true",
        help="skip black formatting of the emitted source",
    )
    args = parser.parse_args(argv)

    try:
        src = translate_file(args.input)
    except TranslationError as e:
        print(f"tilelang2flydsl: error: {e}", file=sys.stderr)
        return 2

    if args.output:
        Path(args.output).write_text(src)
    else:
        sys.stdout.write(src)
    return 0
