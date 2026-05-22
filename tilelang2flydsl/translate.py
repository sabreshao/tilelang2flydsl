"""Public translator entry points."""

from __future__ import annotations

from pathlib import Path

from .emit import emit_source
from .lower import Lowerer
from .parse import parse_file, parse_source


def translate_source(source: str, path: str = "<string>") -> str:
    ir = parse_source(source, path=path)
    mod = Lowerer(ir).build_module()
    return emit_source(mod)


def translate_file(path: str) -> str:
    ir = parse_file(path)
    mod = Lowerer(ir).build_module()
    return emit_source(mod)


def translate_file_to(path: str, output: str) -> None:
    src = translate_file(path)
    Path(output).write_text(src)
