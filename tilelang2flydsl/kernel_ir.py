"""Typed IR produced by parse.py and consumed by lower/.

We keep raw `ast.expr` nodes for arithmetic — the lowerer copies them straight
into the output AST. This avoids reinventing a parallel arithmetic IR for what
is just Python's existing one.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, Union

from .diagnostics import SourceLocation


# ---------------------------------------------------------------------------
# Types appearing on kernel parameters
# ---------------------------------------------------------------------------


@dataclass
class TensorParam:
    """A tensor parameter: `T.Tensor[(M, K), dtype]` or `T.Tensor((M, K), dtype)`."""

    shape: list[ast.expr]
    dtype: ast.expr
    strided: bool = False
    strides: Optional[list[ast.expr]] = None


@dataclass
class ScalarParam:
    """A scalar / symbolic parameter (e.g. `block_M: int = 128`)."""

    annotation: Optional[ast.expr]
    default: Optional[ast.expr]


@dataclass
class Param:
    name: str
    kind: Literal["tensor", "scalar"]
    tensor: Optional[TensorParam] = None
    scalar: Optional[ScalarParam] = None


# ---------------------------------------------------------------------------
# Closure / outer-function bindings
# ---------------------------------------------------------------------------


@dataclass
class DynamicSym:
    """`x = T.dynamic("name")` — captured at translate-time."""

    name: str


@dataclass
class LoopLayout:
    """`T.Fragment((by, bx), forward_fn=fn)` captured at translate-time.

    `forward_fn` is the actual Python callable from the input file's namespace,
    used to enumerate the (thread, local) mapping when lowering T.Parallel.
    """

    extents: tuple[int, int]
    forward_fn: Optional[Callable[..., tuple[int, int]]] = None


@dataclass
class ClosureBindings:
    """Statically-evaluated assignments in the outer factory function."""

    # name -> python value (ints, strings, DynamicSym, LoopLayout, etc.)
    values: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Statements inside the kernel body
# ---------------------------------------------------------------------------


@dataclass
class Alloc:
    name: str
    kind: Literal["shared", "fragment", "local", "var"]
    shape: list[ast.expr]
    dtype: ast.expr
    src: SourceLocation
    init: Optional[ast.expr] = None  # for alloc_var


@dataclass
class Clear:
    target: ast.expr  # usually a Name
    src: SourceLocation


@dataclass
class Fill:
    target: ast.expr
    value: ast.expr
    src: SourceLocation


@dataclass
class Copy:
    src_expr: ast.expr  # source operand (may be subscript)
    dst_expr: ast.expr  # destination operand (may be subscript)
    src: SourceLocation


@dataclass
class Gemm:
    a: ast.expr
    b: ast.expr
    c: ast.expr
    transpose_a: bool
    transpose_b: bool
    src: SourceLocation


@dataclass
class Pipelined:
    var: str
    trip: ast.expr
    num_stages: int
    body: list["Stmt"]
    src: SourceLocation


@dataclass
class Parallel:
    vars: list[str]
    extents: list[ast.expr]
    loop_layout: Optional[LoopLayout]
    body: list["Stmt"]
    src: SourceLocation


@dataclass
class Unroll:
    var: str
    start: ast.expr
    stop: ast.expr
    body: list["Stmt"]
    src: SourceLocation


@dataclass
class Vectorized:
    var: str
    start: ast.expr
    stop: ast.expr
    body: list["Stmt"]
    src: SourceLocation


@dataclass
class Serial:
    var: str
    start: ast.expr
    stop: ast.expr
    step: Optional[ast.expr]
    body: list["Stmt"]
    src: SourceLocation


@dataclass
class SyncThreads:
    src: SourceLocation


@dataclass
class Assume:
    pred: ast.expr
    src: SourceLocation


@dataclass
class Assign:
    """A plain Python assignment inside the kernel body (per-thread scalar code)."""

    target: ast.expr  # ast.Name or ast.Subscript or ast.Tuple
    value: ast.expr
    src: SourceLocation


@dataclass
class IfStmt:
    test: ast.expr
    body: list["Stmt"]
    orelse: list["Stmt"]
    src: SourceLocation


@dataclass
class Raw:
    """Pass-through statement we don't structurally interpret.

    Used for `print`, `T.print`, and other diagnostic calls we want to drop
    or pass through as comments. Lowerer decides what to do with it.
    """

    node: ast.stmt
    note: str
    src: SourceLocation


Stmt = Union[
    Alloc,
    Clear,
    Fill,
    Copy,
    Gemm,
    Pipelined,
    Parallel,
    Unroll,
    Vectorized,
    Serial,
    SyncThreads,
    Assume,
    Assign,
    IfStmt,
    Raw,
]


# ---------------------------------------------------------------------------
# Top-level kernel IR
# ---------------------------------------------------------------------------


@dataclass
class KernelIR:
    name: str
    params: list[Param]
    decorators_meta: dict[str, Any]
    grid: list[ast.expr]
    block_threads: ast.expr
    block_axes: list[str]
    body: list[Stmt]
    returns: Optional[str]
    const_names: list[str]  # names introduced by T.const("M, N, K")
    closure: ClosureBindings
    src_location: SourceLocation
    # Constexpr parameter names (those with defaults that are plain ints, the
    # block_M/block_N/block_K-style knobs). Emitted as `fx.Constexpr[int]`.
    constexpr_params: list[str] = field(default_factory=list)
    # If the kernel returns a tensor created via `T.empty(shape, dtype)` at
    # function level, record the shape/dtype expressions here so the host
    # wrapper can allocate it.
    empty_shape: Optional[list[ast.expr]] = None
    empty_dtype: Optional[ast.expr] = None
