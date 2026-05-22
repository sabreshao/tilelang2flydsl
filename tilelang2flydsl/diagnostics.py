from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SourceLocation:
    path: str
    line: int
    col: int

    def __str__(self) -> str:
        return f"{self.path}:{self.line}:{self.col}"

    @classmethod
    def from_node(cls, path: str, node: ast.AST) -> "SourceLocation":
        line = getattr(node, "lineno", 0) or 0
        col = getattr(node, "col_offset", 0) or 0
        return cls(path=path, line=line, col=col)


class TranslationError(Exception):
    """Base class for any translator-level failure."""

    def __init__(self, message: str, location: Optional[SourceLocation] = None):
        super().__init__(message)
        self.message = message
        self.location = location

    def __str__(self) -> str:
        if self.location is None:
            return self.message
        return f"{self.location}: {self.message}"


class UnsupportedConstruct(TranslationError):
    """Raised when the translator hits a TileLang construct it cannot lower."""

    def __init__(self, symbol: str, location: SourceLocation, hint: str = ""):
        msg = f"unsupported TileLang construct '{symbol}'"
        if hint:
            msg += f"; {hint}"
        super().__init__(msg, location)
        self.symbol = symbol
        self.hint = hint


class MalformedKernel(TranslationError):
    """Raised when the translator cannot recognise the kernel shape at all."""
