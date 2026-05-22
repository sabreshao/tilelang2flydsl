"""Stage 2: KernelIR → ast.Module containing a @flyc.kernel + @flyc.jit pair."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Iterable, Optional

from .. import arch
from .. import tilelang_api as api
from ..diagnostics import SourceLocation, TranslationError, UnsupportedConstruct
from ..kernel_ir import (
    Alloc,
    Assign,
    Assume,
    Clear,
    Copy,
    DynamicSym,
    Fill,
    Gemm,
    IfStmt,
    KernelIR,
    LoopLayout,
    Parallel,
    Pipelined,
    Raw,
    Serial,
    Stmt,
    SyncThreads,
    Unroll,
    Vectorized,
)


# ---------------------------------------------------------------------------
# Helpers to build ast nodes succinctly
# ---------------------------------------------------------------------------


def _name(n: str, ctx=None) -> ast.Name:
    return ast.Name(id=n, ctx=ctx or ast.Load())


def _attr(value: ast.expr, attr: str, ctx=None) -> ast.Attribute:
    return ast.Attribute(value=value, attr=attr, ctx=ctx or ast.Load())


def _call(func: ast.expr, args: list[ast.expr] | None = None, kwargs: list[ast.keyword] | None = None) -> ast.Call:
    return ast.Call(func=func, args=args or [], keywords=kwargs or [])


def _const(v) -> ast.Constant:
    return ast.Constant(value=v)


def _tuple(elts: Iterable[ast.expr]) -> ast.Tuple:
    return ast.Tuple(elts=list(elts), ctx=ast.Load())


def _comment(text: str) -> ast.stmt:
    """Emit a comment as a string expression. ast.unparse will render it as a
    bare string literal which any Python linter will tolerate; black formats it
    cleanly. Not a real comment — Python's AST can't express those — but the
    closest thing without dropping back to text emission."""
    return ast.Expr(value=_const(f"# {text}"))


def _binop(op: ast.operator, *operands: ast.expr) -> ast.expr:
    it = iter(operands)
    acc = next(it)
    for o in it:
        acc = ast.BinOp(left=acc, op=op, right=o)
    return acc


def _floordiv(a: ast.expr, b: ast.expr) -> ast.expr:
    return ast.BinOp(left=a, op=ast.FloorDiv(), right=b)


def _add(a: ast.expr, b: ast.expr) -> ast.expr:
    return ast.BinOp(left=a, op=ast.Add(), right=b)


def _sub(a: ast.expr, b: ast.expr) -> ast.expr:
    return ast.BinOp(left=a, op=ast.Sub(), right=b)


def _ceildiv(a: ast.expr, b: ast.expr) -> ast.expr:
    return _floordiv(_add(_sub(a, _const(1)), b), b) if False else _floordiv(_add(a, _sub(b, _const(1))), b)


# Rewrite `T.ceildiv(a, b)` calls in any expression node into `(a + b - 1) // b`.
class _CeildivRewriter(ast.NodeTransformer):
    def visit_Call(self, node: ast.Call) -> ast.expr:
        self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "T" and node.func.attr == "ceildiv":
            if len(node.args) == 2:
                return _ceildiv(node.args[0], node.args[1])
        return node


# Rewrite `T.get_thread_binding()` into `tid`.
class _ThreadBindingRewriter(ast.NodeTransformer):
    def visit_Call(self, node: ast.Call) -> ast.expr:
        self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "T" and node.func.attr == "get_thread_binding":
            return _name("tid")
        return node


# Rewrite `T.float16` → `fx.Float16` everywhere.
class _DtypeRewriter(ast.NodeTransformer):
    def visit_Attribute(self, node: ast.Attribute) -> ast.expr:
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == "T" and node.attr in api.DTYPES:
            return _attr(_name("fx"), api.DTYPES[node.attr])
        return node


# Rewrite `A[...]` → `A_g[...]` for tensor params, when A appears in scalar
# kernel-body code. Skip if A already ends in `_g`.
class _TensorRefRewriter(ast.NodeTransformer):
    def __init__(self, tensor_names: set[str]):
        self.tensor_names = tensor_names

    def visit_Subscript(self, node: ast.Subscript) -> ast.expr:
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id in self.tensor_names:
            node.value = ast.Name(id=f"{node.value.id}_g", ctx=ast.Load())
        return node


# Rewrite `dtype.bytes` (used e.g. in transpose swizzle) when `dtype` is a
# T.dtype attribute — leave as-is; FlyDSL has no equivalent symbol. We instead
# just keep the input expression intact since FlyDSL accepts arbitrary Python
# ints at trace time, but emit a comment near the usage. This is a passthrough
# for v1.
def _rewrite_expr(e: ast.expr, tensor_names: Optional[set[str]] = None) -> ast.expr:
    e = _CeildivRewriter().visit(e)
    e = _ThreadBindingRewriter().visit(e)
    e = _DtypeRewriter().visit(e)
    if tensor_names:
        e = _TensorRefRewriter(tensor_names).visit(e)
    ast.fix_missing_locations(e)
    return e


def _dtype_to_fly(node: ast.expr) -> ast.expr:
    """Map `T.float16` → `fx.Float16`, `T.dtype(...)` → `fx.dtype(...)`-passthrough.
    For a plain Name (e.g. a parameter `dtype`), pass it through."""
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "T":
        if node.attr in api.DTYPES:
            return _attr(_name("fx"), api.DTYPES[node.attr])
    return node


# ---------------------------------------------------------------------------
# Lowerer state
# ---------------------------------------------------------------------------


@dataclass
class _AllocInfo:
    name: str
    kind: str           # 'shared' | 'fragment' | 'local'
    shape: list[ast.expr]
    dtype: ast.expr     # already mapped to fx.<DType>


@dataclass
class _State:
    """Per-kernel codegen state."""

    smem_emitted: bool = False
    has_gemm: bool = False
    mma_in_dtype_name: Optional[str] = None       # 'Float16' etc.
    mma_accum_dtype_name: Optional[str] = None
    mma_m: int = 16
    mma_n: int = 16
    mma_k: int = 16
    # Copy atoms — keyed by dtype attr name; value is the python var name we
    # used in the emitted source for that atom.
    copy_atom_names: dict[str, str] = field(default_factory=dict)
    allocs: dict[str, _AllocInfo] = field(default_factory=dict)
    threads_const: Optional[int] = None
    notes: list[str] = field(default_factory=list)


class Lowerer:
    def __init__(self, ir: KernelIR):
        self.ir = ir
        self.state = _State()
        # numeric threads_const if literal
        if isinstance(ir.block_threads, ast.Constant) and isinstance(ir.block_threads.value, int):
            self.state.threads_const = ir.block_threads.value

    # public ----------------------------------------------------------------
    def build_module(self) -> ast.Module:
        # Fast path: standard TileLang GEMM pattern → emit the validated
        # LDS-staged FlyDSL form from a template. Bypasses the per-statement
        # lowerer entirely.
        from . import gemm_template

        info = gemm_template.match(self.ir)
        if info is not None:
            mod = gemm_template.emit(info)
            ast.fix_missing_locations(mod)
            return mod

        body: list[ast.stmt] = []
        body.extend(self._build_prelude())
        body.append(self._build_kernel_fn())
        body.append(self._build_host_wrapper())
        mod = ast.Module(body=body, type_ignores=[])
        ast.fix_missing_locations(mod)
        return mod

    # ----------------------------------------------------------------------
    def _build_prelude(self) -> list[ast.stmt]:
        # `import flydsl as fx`
        imports: list[ast.stmt] = [
            ast.ImportFrom(module=None, names=[], level=0) if False else  # placeholder branch never taken
            ast.Import(names=[ast.alias(name="flydsl", asname="fx")]),
            ast.ImportFrom(module="flydsl", names=[ast.alias(name="compiler", asname="flyc")], level=0),
            ast.ImportFrom(
                module="flydsl.expr.rocdl",
                names=[
                    ast.alias(name="MFMA", asname=None),
                    ast.alias(name="BufferCopy128b", asname=None),
                    ast.alias(name="BufferCopy64b", asname=None),
                    ast.alias(name="BufferCopy32b", asname=None),
                    ast.alias(name="make_buffer_tensor", asname=None),
                ],
                level=0,
            ),
        ]
        # Header comment line as a docstring-style string
        if self.ir.decorators_meta.get("jit_kwargs"):
            note = ast.Expr(value=_const(
                f"# Generated by tilelang2flydsl from kernel '{self.ir.name}' "
                f"(original @tilelang.jit kwargs: {self.ir.decorators_meta['jit_kwargs']})"
            ))
        else:
            note = ast.Expr(value=_const(
                f"# Generated by tilelang2flydsl from kernel '{self.ir.name}'."
            ))
        return [note, *imports]

    # ----------------------------------------------------------------------
    def _build_kernel_fn(self) -> ast.FunctionDef:
        # Pre-pass: scan body to determine what setup the kernel top needs.
        self._scan_body(self.ir.body)
        # Free-name analysis: any Name referenced in the body that isn't a
        # kernel arg, an alloc, a loop variable, or a known intrinsic, must be
        # promoted to a kernel arg (e.g. closure-captured block sizes).
        self._free_names = self._collect_free_names()

        args = self._build_kernel_args()
        body: list[ast.stmt] = []

        # Bind block / thread indices.
        for axis_idx, axis_name in enumerate(self.ir.block_axes):
            xyz = "xyz"[axis_idx] if axis_idx < 3 else f"d{axis_idx}"
            body.append(
                ast.Assign(
                    targets=[_name(axis_name, ast.Store())],
                    value=_attr(_attr(_name("fx"), "block_idx"), xyz),
                )
            )
        body.append(
            ast.Assign(
                targets=[_name("tid", ast.Store())],
                value=_attr(_attr(_name("fx"), "thread_idx"), "x"),
            )
        )

        # Buffer-tensor views for all tensor params (and output).
        for p in self.ir.params:
            if p.kind == "tensor":
                body.append(
                    ast.Assign(
                        targets=[_name(f"{p.name}_g", ast.Store())],
                        value=_call(_name("make_buffer_tensor"), [_name(p.name)]),
                    )
                )
        if self.ir.returns and self.ir.returns not in [p.name for p in self.ir.params]:
            body.append(
                ast.Assign(
                    targets=[_name(f"{self.ir.returns}_g", ast.Store())],
                    value=_call(_name("make_buffer_tensor"), [_name(self.ir.returns)]),
                )
            )

        # MMA setup at kernel top (only if there's any T.gemm).
        if self.state.has_gemm:
            body.extend(self._emit_mma_setup())

        # Lower body statements first (populates state.copy_atom_names and
        # state.allocs), then splice the copy-atom declarations between the
        # MMA setup and the body. This keeps the kernel top tidy.
        lowered_body = self._lower_stmts(self.ir.body)
        body.extend(self._emit_copy_atom_setup())
        body.extend(lowered_body)

        return ast.FunctionDef(
            name=self.ir.name,
            args=args,
            body=body or [ast.Pass()],
            decorator_list=[_attr(_name("flyc"), "kernel")],
            returns=None,
        )

    # ----------------------------------------------------------------------
    def _scan_body(self, stmts: list[Stmt]) -> None:
        """Walk the IR before lowering to set kernel-wide flags (has_gemm, …)."""
        for s in stmts:
            if isinstance(s, Gemm):
                self.state.has_gemm = True
                # Look up the A/B/C dtypes from any matching allocs we've seen so far
            elif isinstance(s, (Pipelined, Parallel, Unroll, Vectorized, Serial)):
                self._scan_body(s.body)
            elif isinstance(s, IfStmt):
                self._scan_body(s.body)
                self._scan_body(s.orelse)

    def _emit_mma_setup(self) -> list[ast.stmt]:
        """Emit `mma_atom`, `tiled_mma`, `thr_mma` once at kernel top."""
        # Find first Gemm and infer dtypes from the allocs it references.
        gemm = self._find_first(self.ir.body, Gemm)
        a_dtype = b_dtype = "Float16"
        c_dtype = "Float32"
        if gemm is not None:
            a_name = _root_name(gemm.a)
            b_name = _root_name(gemm.b)
            c_name = _root_name(gemm.c)
            # The allocs map isn't populated yet (we lower in order), but we can
            # walk the IR for the alloc statements to find dtypes.
            for alloc in self._all_allocs(self.ir.body):
                if alloc.name == a_name:
                    a_dtype = _dtype_repr(_dtype_to_fly(_clone(alloc.dtype)))
                if alloc.name == b_name:
                    b_dtype = _dtype_repr(_dtype_to_fly(_clone(alloc.dtype)))
                if alloc.name == c_name:
                    c_dtype = _dtype_repr(_dtype_to_fly(_clone(alloc.dtype)))
        m, n, k, in_dtype_name = arch.pick_mfma(a_dtype, b_dtype, c_dtype)
        self.state.mma_in_dtype_name = in_dtype_name
        self.state.mma_accum_dtype_name = c_dtype
        self.state.mma_m, self.state.mma_n, self.state.mma_k = m, n, k
        return [
            ast.Assign(
                targets=[_name("mma_atom", ast.Store())],
                value=_call(
                    _attr(_name("fx"), "make_mma_atom"),
                    [_call(_name("MFMA"), [_const(m), _const(n), _const(k), _attr(_name("fx"), in_dtype_name)])],
                ),
            ),
            ast.Assign(
                targets=[_name("tiled_mma", ast.Store())],
                value=_call(
                    _attr(_name("fx"), "make_tiled_mma"),
                    [_name("mma_atom"), _call(_attr(_name("fx"), "make_layout"), [_tuple([_const(2), _const(2), _const(1)])])],
                ),
            ),
            ast.Assign(
                targets=[_name("thr_mma", ast.Store())],
                value=_call(_attr(_name("tiled_mma"), "thr_slice"), [_name("tid")]),
            ),
        ]

    def _find_first(self, stmts: list[Stmt], cls):
        for s in stmts:
            if isinstance(s, cls):
                return s
            if isinstance(s, (Pipelined, Parallel, Unroll, Vectorized, Serial)):
                r = self._find_first(s.body, cls)
                if r is not None:
                    return r
            if isinstance(s, IfStmt):
                r = self._find_first(s.body, cls) or self._find_first(s.orelse, cls)
                if r is not None:
                    return r
        return None

    def _collect_free_names(self) -> list[str]:
        """Return ordered list of free-name references that need to become kernel args."""
        bound: set[str] = {p.name for p in self.ir.params}
        bound.update(self.ir.const_names)
        for v in self.ir.closure.values.values():
            if isinstance(v, DynamicSym):
                bound.add(v.name)
        bound.update(self.ir.block_axes)
        bound.add("tid")
        if self.ir.returns:
            bound.add(self.ir.returns)
        # Locally-bound by allocations and assignments anywhere in the body
        for s in self._all_iter(self.ir.body):
            if isinstance(s, Alloc):
                bound.add(s.name)
            elif isinstance(s, Assign) and isinstance(s.target, ast.Name):
                bound.add(s.target.id)
            elif isinstance(s, Assign) and isinstance(s.target, ast.Tuple):
                for e in s.target.elts:
                    if isinstance(e, ast.Name):
                        bound.add(e.id)
            elif isinstance(s, (Pipelined, Unroll, Vectorized, Serial)):
                bound.add(s.var)
            elif isinstance(s, Parallel):
                bound.update(s.vars)
        # Names referenced anywhere in the body but not bound
        free: list[str] = []
        seen: set[str] = set()
        # Modules / well-known identifiers we never want as args
        builtins = {"T", "fx", "flyc", "True", "False", "None", "int", "float", "str", "bool", "object", "range", "len", "abs", "min", "max"}
        for node in self._iter_expr_nodes(self.ir.body):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                n = node.id
                if n in bound or n in builtins or n in seen:
                    continue
                seen.add(n)
                free.append(n)
        return free

    def _all_iter(self, stmts: list[Stmt]):
        for s in stmts:
            yield s
            if isinstance(s, (Pipelined, Parallel, Unroll, Vectorized, Serial)):
                yield from self._all_iter(s.body)
            elif isinstance(s, IfStmt):
                yield from self._all_iter(s.body)
                yield from self._all_iter(s.orelse)

    def _iter_expr_nodes(self, stmts: list[Stmt]):
        """Yield every ast.expr node appearing in the IR (for free-name scan)."""
        for s in self._all_iter(stmts):
            for attr in ("trip", "start", "stop", "step", "pred", "value", "test", "target", "a", "b", "c", "src_expr", "dst_expr", "init"):
                val = getattr(s, attr, None)
                if isinstance(val, ast.AST):
                    yield from ast.walk(val)
            for attr in ("shape", "extents"):
                vals = getattr(s, attr, None)
                if isinstance(vals, list):
                    for v in vals:
                        if isinstance(v, ast.AST):
                            yield from ast.walk(v)

    def _all_allocs(self, stmts: list[Stmt]):
        for s in stmts:
            if isinstance(s, Alloc):
                yield s
            elif isinstance(s, (Pipelined, Parallel, Unroll, Vectorized, Serial)):
                yield from self._all_allocs(s.body)
            elif isinstance(s, IfStmt):
                yield from self._all_allocs(s.body)
                yield from self._all_allocs(s.orelse)

    def _build_kernel_args(self) -> ast.arguments:
        # Build a list of (arg, default-or-None) pairs in the natural declaration
        # order, then split into args[] + defaults[] with all defaultless args
        # coming first. This guarantees a valid Python function signature even
        # when categories interleave.
        pairs: list[tuple[ast.arg, Optional[ast.expr]]] = []

        # Tensor params
        for p in self.ir.params:
            if p.kind == "tensor":
                pairs.append((ast.arg(arg=p.name, annotation=_attr(_name("fx"), "Tensor")), None))

        # Output tensor (if returned via T.empty inside body)
        if self.ir.returns and self.ir.returns not in [p.name for p in self.ir.params]:
            pairs.append((ast.arg(arg=self.ir.returns, annotation=_attr(_name("fx"), "Tensor")), None))

        # const() symbols
        for cn in self.ir.const_names:
            pairs.append(
                (
                    ast.arg(
                        arg=cn,
                        annotation=ast.Subscript(value=_attr(_name("fx"), "Constexpr"), slice=_name("int"), ctx=ast.Load()),
                    ),
                    None,
                )
            )

        # dynamic() symbols captured in closure
        for v in self.ir.closure.values.values():
            if isinstance(v, DynamicSym):
                pairs.append((ast.arg(arg=v.name, annotation=_attr(_name("fx"), "Int32")), None))

        # Scalar constexpr params (block_M, block_N, …, dtype=fx.Float16, …)
        for p in self.ir.params:
            if p.kind == "scalar" and p.name in self.ir.constexpr_params:
                ann = _constexpr_annotation_for(p)
                default = _rewrite_expr(_clone(p.scalar.default)) if (p.scalar and p.scalar.default is not None) else None
                pairs.append((ast.arg(arg=p.name, annotation=ann), default))

        # Free-name args
        for n in self._free_names:
            v = self.ir.closure.values.get(n)
            default = _const(v) if isinstance(v, (int, str)) else None
            pairs.append(
                (
                    ast.arg(
                        arg=n,
                        annotation=ast.Subscript(value=_attr(_name("fx"), "Constexpr"), slice=_name("int"), ctx=ast.Load()),
                    ),
                    default,
                )
            )

        return _split_args_defaults(pairs)

    # ----------------------------------------------------------------------
    def _build_host_wrapper(self) -> ast.FunctionDef:
        pairs: list[tuple[ast.arg, Optional[ast.expr]]] = []

        for p in self.ir.params:
            if p.kind == "tensor":
                pairs.append((ast.arg(arg=p.name, annotation=_attr(_name("fx"), "Tensor")), None))
        for cn in self.ir.const_names:
            pairs.append((ast.arg(arg=cn, annotation=_attr(_name("fx"), "Int32")), None))
        for v in self.ir.closure.values.values():
            if isinstance(v, DynamicSym):
                pairs.append((ast.arg(arg=v.name, annotation=_attr(_name("fx"), "Int32")), None))
        for p in self.ir.params:
            if p.kind == "scalar" and p.name in self.ir.constexpr_params:
                ann = _host_annotation_for(p)
                default = _rewrite_expr(_clone(p.scalar.default)) if (p.scalar and p.scalar.default is not None) else None
                pairs.append((ast.arg(arg=p.name, annotation=ann), default))
        for n in self._free_names:
            v = self.ir.closure.values.get(n)
            default = _const(v) if isinstance(v, (int, str)) else None
            pairs.append((ast.arg(arg=n, annotation=_name("int")), default))
        # stream kwarg (always defaulted, always last)
        pairs.append(
            (
                ast.arg(arg="stream", annotation=_attr(_name("fx"), "Stream")),
                _call(_attr(_name("fx"), "Stream"), [_const(None)]),
            )
        )

        host_args = _split_args_defaults(pairs)
        params_ast = host_args.args
        defaults = host_args.defaults

        body: list[ast.stmt] = []

        # If kernel returns a T.empty output, allocate it
        output_name = self.ir.returns
        output_alloc_inside_body = output_name and output_name not in [p.name for p in self.ir.params]
        if output_alloc_inside_body:
            # Find the T.empty inside body
            shape, dtype = self._find_empty_alloc(output_name)
            if shape is None:
                shape = [_const(0)]
            if dtype is None:
                dtype = _attr(_name("fx"), "Float16")
            body.append(
                ast.Assign(
                    targets=[_name(output_name, ast.Store())],
                    value=_call(
                        _attr(_name("fx"), "empty"),
                        [_tuple(shape), dtype],
                        [ast.keyword(arg="device", value=_attr(_name(self.ir.params[0].name), "device"))],
                    ),
                )
            )

        # Compute grid expression
        grid_elts: list[ast.expr] = [_rewrite_expr(_clone(g)) for g in self.ir.grid]
        while len(grid_elts) < 3:
            grid_elts.append(_const(1))
        grid_tuple = _tuple(grid_elts)
        body.append(ast.Assign(targets=[_name("grid", ast.Store())], value=grid_tuple))

        # Build kernel call
        call_args: list[ast.expr] = []
        for p in self.ir.params:
            if p.kind == "tensor":
                call_args.append(_name(p.name))
        if output_alloc_inside_body:
            call_args.append(_name(output_name))
        for cn in self.ir.const_names:
            call_args.append(_name(cn))
        for k, v in self.ir.closure.values.items():
            if isinstance(v, DynamicSym):
                call_args.append(_name(v.name))
        for p in self.ir.params:
            if p.kind == "scalar":
                call_args.append(_name(p.name))
        for n in self._free_names:
            call_args.append(_name(n))

        threads_expr = _rewrite_expr(_clone(self.ir.block_threads))
        kernel_call = _call(_name(self.ir.name), call_args)
        launch_call = _call(
            _attr(kernel_call, "launch"),
            [],
            [
                ast.keyword(arg="grid", value=_name("grid")),
                ast.keyword(arg="block", value=_tuple([threads_expr, _const(1), _const(1)])),
                ast.keyword(arg="stream", value=_name("stream")),
            ],
        )
        body.append(ast.Expr(value=launch_call))

        if output_alloc_inside_body:
            body.append(ast.Return(value=_name(output_name)))

        return ast.FunctionDef(
            name=f"{self.ir.name}_launch",
            args=host_args,
            body=body,
            decorator_list=[_attr(_name("flyc"), "jit")],
            returns=None,
        )

    def _find_empty_alloc(self, name: str) -> tuple[Optional[list[ast.expr]], Optional[ast.expr]]:
        # Prefer the IR-level capture from parse.py (function-level T.empty()).
        if self.ir.empty_shape is not None:
            shape = [_clone(e) for e in self.ir.empty_shape]
            dtype = _dtype_to_fly(_clone(self.ir.empty_dtype)) if self.ir.empty_dtype is not None else _attr(_name("fx"), "Float16")
            return shape, dtype
        # Fallback: inline T.empty() inside the with-block body.
        for s in self.ir.body:
            if isinstance(s, Assign) and isinstance(s.target, ast.Name) and s.target.id == name:
                v = s.value
                if isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and v.func.attr == "empty" and isinstance(v.func.value, ast.Name) and v.func.value.id == "T":
                    args = v.args
                    if args:
                        shape = args[0].elts if isinstance(args[0], ast.Tuple) else [args[0]]
                        dtype = args[1] if len(args) >= 2 else _attr(_name("fx"), "Float16")
                        return shape, _dtype_to_fly(dtype)
        return None, None

    # ----------------------------------------------------------------------
    # Statement lowering
    # ----------------------------------------------------------------------
    def _lower_stmts(self, stmts: list[Stmt]) -> list[ast.stmt]:
        out: list[ast.stmt] = []
        for s in stmts:
            r = self._lower_stmt(s)
            if r is not None:
                out.extend(r)
        return out

    def _lower_stmt(self, s: Stmt) -> list[ast.stmt]:
        if isinstance(s, Alloc):
            return self._lower_alloc(s)
        if isinstance(s, Clear):
            return [ast.Expr(value=_call(_attr(_name("fx"), "fill"), [_rewrite_expr(_clone(s.target)), _const(0)]))]
        if isinstance(s, Fill):
            return [ast.Expr(value=_call(_attr(_name("fx"), "fill"), [_rewrite_expr(_clone(s.target)), _rewrite_expr(_clone(s.value))]))]
        if isinstance(s, Copy):
            return self._lower_copy(s)
        if isinstance(s, Gemm):
            return self._lower_gemm(s)
        if isinstance(s, Pipelined):
            return self._lower_pipelined(s)
        if isinstance(s, Parallel):
            return self._lower_parallel(s)
        if isinstance(s, Unroll):
            return self._lower_unroll(s)
        if isinstance(s, Vectorized):
            # Vectorize → constexpr loop + comment
            stmts = self._lower_unroll(Unroll(var=s.var, start=s.start, stop=s.stop, body=s.body, src=s.src))
            return [_comment("vectorized hint (FlyDSL ROCDL codegen will vectorize)"), *stmts]
        if isinstance(s, Serial):
            return self._lower_serial(s)
        if isinstance(s, SyncThreads):
            return [ast.Expr(value=_call(_attr(_attr(_name("fx"), "gpu"), "barrier"), []))]
        if isinstance(s, Assume):
            return [_comment(f"T.assume({ast.unparse(s.pred)})")]
        if isinstance(s, Assign):
            tensor_names = {p.name for p in self.ir.params if p.kind == "tensor"}
            target = _rewrite_expr(_clone(s.target), tensor_names=tensor_names)
            value = _rewrite_expr(_clone(s.value), tensor_names=tensor_names)
            if isinstance(target, ast.Name):
                # Skip output T.empty allocations and T.const tuple unpackings
                if isinstance(s.value, ast.Call):
                    name = _t_attr_name(s.value)
                    if name in ("empty", "const"):
                        return []
                # Skip `tid = T.get_thread_binding()` — we already emit it.
                if isinstance(s.value, ast.Call) and _t_attr_name(s.value) == "get_thread_binding":
                    return []
                return [ast.Assign(targets=[ast.Name(id=target.id, ctx=ast.Store())], value=value)]
            target.ctx = ast.Store()
            return [ast.Assign(targets=[target], value=value)]
        if isinstance(s, IfStmt):
            return [
                ast.If(
                    test=_rewrite_expr(_clone(s.test)),
                    body=self._lower_stmts(s.body) or [ast.Pass()],
                    orelse=self._lower_stmts(s.orelse),
                )
            ]
        if isinstance(s, Raw):
            note_stmt = _comment(s.note)
            if isinstance(s.node, ast.Expr) and isinstance(s.node.value, ast.Call) and _t_attr_name(s.node.value) in api.DROPPABLE_ANNOTATIONS:
                return [note_stmt]
            # Otherwise emit a comment + the original node verbatim
            return [note_stmt]
        raise TranslationError(f"don't know how to lower {type(s).__name__}", getattr(s, "src", None))

    # ----- alloc ---------------------------------------------------------
    def _lower_alloc(self, s: Alloc) -> list[ast.stmt]:
        dtype_fly = _dtype_to_fly(_clone(s.dtype))
        info = _AllocInfo(name=s.name, kind=s.kind, shape=[_rewrite_expr(_clone(e)) for e in s.shape], dtype=dtype_fly)
        self.state.allocs[s.name] = info
        if s.kind == "shared":
            stmts: list[ast.stmt] = []
            if not self.state.smem_emitted:
                stmts.append(
                    ast.Assign(
                        targets=[_name("smem", ast.Store())],
                        value=_call(
                            _attr(_attr(_name("fx"), "gpu"), "SharedAllocator"),
                            [],
                            [ast.keyword(arg="static", value=_const(True))],
                        ),
                    )
                )
                self.state.smem_emitted = True
            stmts.append(
                ast.Assign(
                    targets=[_name(s.name, ast.Store())],
                    value=_call(_attr(_name("smem"), "alloc"), [_tuple(info.shape), dtype_fly]),
                )
            )
            return stmts
        if s.kind == "fragment":
            # Bind to thr_mma.make_fragment_C() — thr_mma was emitted at kernel top.
            return [
                ast.Assign(
                    targets=[_name(s.name, ast.Store())],
                    value=_call(_attr(_name("thr_mma"), "make_fragment_C"), [_tuple(info.shape)]),
                )
            ]
        if s.kind == "local":
            # `fx.make_rmem_tensor(fx.make_layout(*shape), dtype)`
            layout = _call(_attr(_name("fx"), "make_layout"), list(info.shape))
            return [
                ast.Assign(
                    targets=[_name(s.name, ast.Store())],
                    value=_call(_attr(_name("fx"), "make_rmem_tensor"), [layout, dtype_fly]),
                )
            ]
        if s.kind == "var":
            init = _rewrite_expr(_clone(s.init)) if s.init is not None else _const(0)
            return [ast.Assign(targets=[_name(s.name, ast.Store())], value=init)]
        raise TranslationError(f"unknown alloc kind {s.kind!r}", s.src)

    # ----- copy ----------------------------------------------------------
    def _lower_copy(self, s: Copy) -> list[ast.stmt]:
        src = s.src_expr
        dst = s.dst_expr
        src_name = _root_name(src)
        dst_name = _root_name(dst)
        src_alloc = self.state.allocs.get(src_name or "")
        dst_alloc = self.state.allocs.get(dst_name or "")

        # Fragment→global epilogue store
        if src_alloc is not None and src_alloc.kind == "fragment":
            return self._lower_copy_fragment_to_global(s, src_alloc)
        # global→shared bulk copy (the most common load path)
        if dst_alloc is not None and dst_alloc.kind == "shared":
            return self._lower_copy_global_to_shared(s, dst_alloc)
        # shared→global (uncommon — direct store)
        if src_alloc is not None and src_alloc.kind == "shared":
            return self._lower_copy_shared_to_global(s, src_alloc)
        # local→shared, shared→local, etc. — emit a TODO + verbatim assignment.
        # FlyDSL supports `fx.copy(copy_atom, src, dst)` for arbitrary partitions
        # but choosing the atom needs more context than we have here.
        return [
            _comment(f"copy {src_name} -> {dst_name}: kinds unrecognised; emit raw fx.copy with default atom"),
            ast.Expr(value=_call(
                _attr(_name("fx"), "copy"),
                [_name(self._copy_atom_for(dst_alloc.dtype if dst_alloc else (src_alloc.dtype if src_alloc else _attr(_name("fx"), "Float16")))), _rewrite_expr(_clone(src)), _rewrite_expr(_clone(dst))],
            )),
        ]

    def _copy_atom_for(self, dtype_node: ast.expr) -> str:
        """Return the python var name for a copy atom of this dtype.

        Registers the dtype so the kernel-top pre-pass can emit the declaration.
        """
        name = _dtype_repr(dtype_node)
        var = f"copy_atom_{name.lower()}"
        self.state.copy_atom_names[name] = var
        return var

    def _emit_copy_atom_setup(self) -> list[ast.stmt]:
        out: list[ast.stmt] = []
        for dtype_name, var in self.state.copy_atom_names.items():
            out.append(
                ast.Assign(
                    targets=[_name(var, ast.Store())],
                    value=_call(
                        _attr(_name("fx"), "make_copy_atom"),
                        [_call(_name("BufferCopy128b"), []), _attr(_name("fx"), dtype_name)],
                    ),
                )
            )
        return out

    def _lower_copy_global_to_shared(self, s: Copy, dst_alloc: _AllocInfo) -> list[ast.stmt]:
        dst = _rewrite_expr(_clone(s.dst_expr))
        atom_var = self._copy_atom_for(_dtype_to_fly(_clone(dst_alloc.dtype)))
        rows = _maybe_literal(dst_alloc.shape[0]) if dst_alloc.shape else None
        cols = _maybe_literal(dst_alloc.shape[1]) if len(dst_alloc.shape) >= 2 else None
        tv_layout = self._tv_layout_call(self.state.threads_const, rows, cols)
        src_tile = _rewrite_src_to_g(_clone(s.src_expr))
        return [
            ast.Assign(
                targets=[_name("tiled_copy", ast.Store())],
                value=_call(_attr(_name("fx"), "make_tiled_copy"), [_name(atom_var), tv_layout]),
            ),
            ast.Assign(
                targets=[_name("thr_copy", ast.Store())],
                value=_call(_attr(_name("tiled_copy"), "get_slice"), [_name("tid")]),
            ),
            ast.Expr(value=_call(
                _attr(_name("fx"), "copy"),
                [
                    _name(atom_var),
                    _call(_attr(_name("thr_copy"), "partition_S"), [src_tile]),
                    _call(_attr(_name("thr_copy"), "partition_D"), [dst]),
                ],
            )),
        ]

    def _lower_copy_shared_to_global(self, s: Copy, src_alloc: _AllocInfo) -> list[ast.stmt]:
        dst_tile = _rewrite_src_to_g(_clone(s.dst_expr))
        src = _rewrite_expr(_clone(s.src_expr))
        atom_var = self._copy_atom_for(_dtype_to_fly(_clone(src_alloc.dtype)))
        rows = _maybe_literal(src_alloc.shape[0]) if src_alloc.shape else None
        cols = _maybe_literal(src_alloc.shape[1]) if len(src_alloc.shape) >= 2 else None
        tv_layout = self._tv_layout_call(self.state.threads_const, rows, cols)
        return [
            ast.Assign(
                targets=[_name("tiled_copy", ast.Store())],
                value=_call(_attr(_name("fx"), "make_tiled_copy"), [_name(atom_var), tv_layout]),
            ),
            ast.Assign(
                targets=[_name("thr_copy", ast.Store())],
                value=_call(_attr(_name("tiled_copy"), "get_slice"), [_name("tid")]),
            ),
            ast.Expr(value=_call(
                _attr(_name("fx"), "copy"),
                [
                    _name(atom_var),
                    _call(_attr(_name("thr_copy"), "partition_S"), [src]),
                    _call(_attr(_name("thr_copy"), "partition_D"), [dst_tile]),
                ],
            )),
        ]

    def _lower_copy_fragment_to_global(self, s: Copy, src_alloc: _AllocInfo) -> list[ast.stmt]:
        dst_tile = _rewrite_src_to_g(_clone(s.dst_expr))
        atom_var = self._copy_atom_for(_dtype_to_fly(_clone(src_alloc.dtype)))
        return [
            _comment(f"epilogue store: {src_alloc.name} -> {ast.unparse(s.dst_expr)}"),
            ast.Assign(
                targets=[_name("c_partition", ast.Store())],
                value=_call(_attr(_name("thr_mma"), "partition_C"), [dst_tile]),
            ),
            ast.Expr(value=_call(
                _attr(_name("fx"), "copy"),
                [
                    _name(atom_var),
                    _name(src_alloc.name),
                    _name("c_partition"),
                ],
            )),
        ]

    def _tv_layout_call(self, threads: Optional[int], rows: Optional[int], cols: Optional[int]) -> ast.expr:
        # If we have static numbers, plug in a coalesced TV layout. Otherwise
        # emit `fx.make_layout(threads)` and a comment to fix up.
        if threads is not None and rows is not None and cols is not None:
            (tpr, tpc), (vpr, vpc) = arch.default_copy_tv_layout(threads, rows, cols)
            return _call(
                _attr(_name("fx"), "make_layout"),
                [_tuple([_const(tpr), _const(tpc)]), _tuple([_const(vpr), _const(vpc)])],
            )
        # Dynamic: emit a 1D placeholder layout — caller may need to retune.
        return _call(
            _attr(_name("fx"), "make_layout"),
            [_const(threads if threads is not None else 1)],
        )

    # ----- gemm ----------------------------------------------------------
    def _lower_gemm(self, s: Gemm) -> list[ast.stmt]:
        a_name = _root_name(s.a)
        b_name = _root_name(s.b)
        c_name = _root_name(s.c)
        # mma_atom / tiled_mma / thr_mma were emitted at kernel top by
        # _emit_mma_setup(); nothing to set up here.
        frag_a_name = f"frag_{a_name}"
        frag_b_name = f"frag_{b_name}"
        return [
            ast.Assign(
                targets=[_name(frag_a_name, ast.Store())],
                value=_call(_attr(_name("thr_mma"), "make_fragment_A"), [_name(a_name or "A_shared")]),
            ),
            ast.Assign(
                targets=[_name(frag_b_name, ast.Store())],
                value=_call(_attr(_name("thr_mma"), "make_fragment_B"), [_name(b_name or "B_shared")]),
            ),
            ast.Expr(value=_call(
                _attr(_name("fx"), "gemm"),
                [
                    _name("mma_atom"),
                    _name(c_name or "C_local"),
                    _name(frag_a_name),
                    _name(frag_b_name),
                    _name(c_name or "C_local"),
                ],
            )),
        ]

    # ----- loops --------------------------------------------------------
    def _lower_pipelined(self, s: Pipelined) -> list[ast.stmt]:
        trip = _rewrite_expr(_clone(s.trip))
        body = self._lower_stmts(s.body)
        # Bracket the body with barriers — minimal correct synchronisation for v1.
        body_with_barriers = [
            ast.Expr(value=_call(_attr(_attr(_name("fx"), "gpu"), "barrier"), [])),
            *body,
            ast.Expr(value=_call(_attr(_attr(_name("fx"), "gpu"), "barrier"), [])),
        ]
        note = _comment(f"TODO: software-pipeline this loop ({s.num_stages} stages) via range(..., init=[...])")
        return [
            note,
            ast.For(
                target=_name(s.var, ast.Store()),
                iter=_call(_attr(_name("fx"), "range"), [_const(0), trip, _const(1)]),
                body=body_with_barriers,
                orelse=[],
            ),
        ]

    def _lower_unroll(self, s: Unroll) -> list[ast.stmt]:
        start = _rewrite_expr(_clone(s.start))
        stop = _rewrite_expr(_clone(s.stop))
        body = self._lower_stmts(s.body)
        iter_call = _call(_attr(_name("fx"), "range_constexpr"), [start, stop] if not _is_zero(start) else [stop])
        return [
            ast.For(target=_name(s.var, ast.Store()), iter=iter_call, body=body or [ast.Pass()], orelse=[]),
        ]

    def _lower_serial(self, s: Serial) -> list[ast.stmt]:
        start = _rewrite_expr(_clone(s.start))
        stop = _rewrite_expr(_clone(s.stop))
        args = [start, stop]
        if s.step is not None:
            args.append(_rewrite_expr(_clone(s.step)))
        body = self._lower_stmts(s.body)
        return [
            ast.For(target=_name(s.var, ast.Store()), iter=_call(_attr(_name("fx"), "range"), args), body=body or [ast.Pass()], orelse=[]),
        ]

    def _lower_parallel(self, s: Parallel) -> list[ast.stmt]:
        # T.Parallel(by, bx) is a "thread-parallel multi-dim loop". We map to
        # a tiled-copy partition when used for a write-back like the transpose
        # epilogue, or fall back to unrolled fx.range_constexpr loops over the
        # extents (since FlyDSL doesn't have a direct `Parallel` analog and the
        # body usually does scalar writes that the threads share via tid).
        extents = [_rewrite_expr(_clone(e)) for e in s.extents]
        body = self._lower_stmts(s.body)
        notes: list[ast.stmt] = []
        if s.loop_layout is not None:
            notes.append(
                _comment(
                    "T.Parallel with custom loop_layout: enumerated thread-value layout below "
                    f"(extents={s.loop_layout.extents}). FlyDSL author should validate the TV-layout."
                )
            )
            try:
                tv = _enumerate_forward_fn_layout(s.loop_layout)
                notes.append(
                    ast.Assign(
                        targets=[_name("tv_layout", ast.Store())],
                        value=tv,
                    )
                )
            except Exception as exc:
                notes.append(_comment(f"could not enumerate loop_layout forward_fn: {exc!r}"))
        # Emit nested unrolled loops over the extents. The body is preserved.
        result: list[ast.stmt] = list(notes)
        inner_body: list[ast.stmt] = body or [ast.Pass()]
        for var, ext in zip(reversed(s.vars), reversed(extents)):
            inner_body = [
                ast.For(
                    target=_name(var, ast.Store()),
                    iter=_call(_attr(_name("fx"), "range_constexpr"), [ext]),
                    body=inner_body,
                    orelse=[],
                )
            ]
        result.extend(inner_body)
        return result


def _enumerate_forward_fn_layout(ll: LoopLayout) -> ast.expr:
    """Run `ll.forward_fn(i, j)` over the extents and build an fx.make_layout(...)
    call with the enumerated (thread, local) coordinate mapping recorded as a
    Python list literal."""
    if ll.forward_fn is None:
        raise ValueError("loop_layout has no forward_fn")
    by, bx = ll.extents
    table: list[tuple[int, int, int, int]] = []
    for i in range(by):
        for j in range(bx):
            res = ll.forward_fn(i, j)
            if not (isinstance(res, tuple) and len(res) == 2):
                raise ValueError(f"forward_fn({i},{j}) did not return a 2-tuple")
            t, lv = res
            table.append((i, j, int(t), int(lv)))
    # Emit as a static tuple-of-tuples; FlyDSL author then converts it.
    literal = _tuple(
        [
            _tuple([_const(i), _const(j), _const(t), _const(lv)])
            for (i, j, t, lv) in table
        ]
    )
    return _call(
        _attr(_name("fx"), "make_layout"),
        [literal],
    )


# ---------------------------------------------------------------------------
# Misc helpers (free functions used by Lowerer)
# ---------------------------------------------------------------------------


def _clone(node: ast.expr) -> ast.expr:
    """Deep-copy an ast node so we can mutate it freely."""
    import copy

    return copy.deepcopy(node)


def _root_name(e: ast.expr) -> Optional[str]:
    if isinstance(e, ast.Name):
        return e.id
    if isinstance(e, ast.Subscript):
        return _root_name(e.value)
    if isinstance(e, ast.Attribute):
        return _root_name(e.value)
    return None


def _rewrite_src_to_g(e: ast.expr) -> ast.expr:
    """Rewrite `A[...]` → `A_g[...]` so global accesses use the buffer-tensor view."""
    if isinstance(e, ast.Subscript) and isinstance(e.value, ast.Name):
        e.value = ast.Name(id=f"{e.value.id}_g", ctx=ast.Load())
        return e
    if isinstance(e, ast.Name):
        return ast.Name(id=f"{e.id}_g", ctx=ast.Load())
    return e


def _maybe_literal(e: ast.expr) -> Optional[int]:
    if isinstance(e, ast.Constant) and isinstance(e.value, int):
        return e.value
    return None


def _is_zero(e: ast.expr) -> bool:
    return isinstance(e, ast.Constant) and e.value == 0


def _dtype_repr(node: ast.expr) -> str:
    """Return the FlyDSL dtype attribute name (e.g. 'Float16') for an fx.* dtype
    expression. Returns 'Float16' as fallback."""
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "fx":
        return node.attr
    return "Float16"


def _t_attr_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "T":
        return node.attr
    if isinstance(node, ast.Call):
        return _t_attr_name(node.func)
    return None


def _split_args_defaults(pairs: list[tuple[ast.arg, Optional[ast.expr]]]) -> ast.arguments:
    """Given (arg, default-or-None) pairs in declaration order, produce a valid
    ast.arguments with all defaultless args first and the defaults aligned."""
    no_default = [a for (a, d) in pairs if d is None]
    with_default = [(a, d) for (a, d) in pairs if d is not None]
    args = [*no_default, *(a for (a, _) in with_default)]
    defaults = [d for (_, d) in with_default]
    return ast.arguments(
        posonlyargs=[],
        args=args,
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=defaults,
    )


def _is_dtype_default(default: Optional[ast.expr]) -> bool:
    """True if `default` looks like a dtype expression — either `T.<dtype>` or
    `fx.<DType>`."""
    if default is None:
        return False
    if isinstance(default, ast.Attribute) and isinstance(default.value, ast.Name):
        if default.value.id == "T" and default.attr in api.DTYPES:
            return True
        if default.value.id == "fx" and default.attr in api.DTYPES.values():
            return True
    return False


def _constexpr_annotation_for(p) -> ast.expr:
    """Choose a Constexpr annotation for a scalar param on the @flyc.kernel side.

    - default is a literal int  → `fx.Constexpr[int]`
    - default is a dtype        → `fx.Constexpr[fx.dtype]`
    - no default                → `fx.Constexpr[int]` (best guess; block_*).
    """
    default = p.scalar.default if p.scalar else None
    if default is None or _maybe_literal(default) is not None:
        return ast.Subscript(value=_attr(_name("fx"), "Constexpr"), slice=_name("int"), ctx=ast.Load())
    if _is_dtype_default(default):
        return ast.Subscript(value=_attr(_name("fx"), "Constexpr"), slice=_attr(_name("fx"), "dtype"), ctx=ast.Load())
    return _attr(_name("fx"), "Constexpr")


def _host_annotation_for(p) -> ast.expr:
    default = p.scalar.default if p.scalar else None
    if default is None or _maybe_literal(default) is not None:
        return _name("int")
    if _is_dtype_default(default):
        return _attr(_name("fx"), "dtype")
    return _name("object")
