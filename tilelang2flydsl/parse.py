"""Stage 1: TileLang Python AST → KernelIR.

We recognise two kernel shapes:

  Direct:
      @tilelang.jit
      def matmul(A, B, ...):
          ...
          with T.Kernel(...) as (bx, by):
              ...
          return C

  Factory:
      @tilelang.jit(pass_configs={...})
      def get_batched_transpose_kernel(...):
          # outer locals captured as closure constants
          num_threads = 256
          shape_x = T.dynamic("shape_x")
          loop_layout = T.Fragment(..., forward_fn=create_loop_layout_fn(...))

          @T.prim_func
          def batched_transpose_kernel(x: T.StridedTensor[...], out: T.Tensor[...]):
              with T.Kernel(...) as (pid_y, pid_x, pid_batch):
                  ...
          return batched_transpose_kernel

In both shapes the actual kernel is the function containing a `with T.Kernel(...)`.
The parser locates that function, captures outer-scope constants (in factory shape),
and walks the kernel body producing typed KernelIR statements.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any, Optional

from . import tilelang_api as api
from .diagnostics import (
    MalformedKernel,
    SourceLocation,
    TranslationError,
    UnsupportedConstruct,
)
from .kernel_ir import (
    Alloc,
    Assign,
    Assume,
    Clear,
    ClosureBindings,
    Copy,
    DynamicSym,
    Fill,
    Gemm,
    IfStmt,
    KernelIR,
    LoopLayout,
    Param,
    Parallel,
    Pipelined,
    Raw,
    ScalarParam,
    Serial,
    Stmt,
    SyncThreads,
    TensorParam,
    Unroll,
    Vectorized,
)


# ---------------------------------------------------------------------------
# Small AST helpers
# ---------------------------------------------------------------------------


def _is_T_attr(node: ast.AST, name: Optional[str] = None) -> bool:
    """Is `node` of the form `T.<name>`? If `name` is None, any attribute on T."""
    if not isinstance(node, ast.Attribute):
        return False
    if not isinstance(node.value, ast.Name) or node.value.id != "T":
        return False
    return name is None or node.attr == name


def _is_T_call(node: ast.AST, name: Optional[str] = None) -> bool:
    """Is `node` a call of the form `T.<name>(...)`?"""
    return isinstance(node, ast.Call) and _is_T_attr(node.func, name)


def _t_attr_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "T":
        return node.attr
    if isinstance(node, ast.Call):
        return _t_attr_name(node.func)
    if isinstance(node, ast.Subscript):
        return _t_attr_name(node.value)
    return None


def _decorator_name(dec: ast.expr) -> Optional[str]:
    """Return e.g. 'tilelang.jit' or 'T.prim_func' for a decorator expression."""
    if isinstance(dec, ast.Call):
        return _decorator_name(dec.func)
    if isinstance(dec, ast.Attribute):
        left = _decorator_name(dec.value)
        return f"{left}.{dec.attr}" if left else dec.attr
    if isinstance(dec, ast.Name):
        return dec.id
    return None


def _literal_int(node: ast.expr) -> Optional[int]:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    return None


def _name_id(node: ast.expr) -> Optional[str]:
    return node.id if isinstance(node, ast.Name) else None


# ---------------------------------------------------------------------------
# Locating the kernel function inside a file
# ---------------------------------------------------------------------------


def _has_decorator(func: ast.FunctionDef, *names: str) -> bool:
    for dec in func.decorator_list:
        n = _decorator_name(dec)
        if n in names:
            return True
    return False


def _find_kernels(tree: ast.Module) -> list[tuple[ast.FunctionDef, Optional[ast.FunctionDef]]]:
    """Return [(kernel_function, outer_factory_or_None), ...].

    Looks for top-level functions decorated with `@tilelang.jit` (direct) or
    outer functions whose body contains an inner `@T.prim_func` function (factory).
    """
    found: list[tuple[ast.FunctionDef, Optional[ast.FunctionDef]]] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if _has_decorator(node, "tilelang.jit"):
            # Either the body is the kernel, or the body builds an inner kernel.
            inner = _inner_prim_func(node)
            if inner is not None:
                found.append((inner, node))
            elif _has_T_kernel_block(node):
                found.append((node, None))
        elif _has_T_kernel_block(node):
            # Standalone @T.prim_func-style; still translate.
            inner = _inner_prim_func(node)
            if inner is not None:
                found.append((inner, node))
    return found


def _inner_prim_func(outer: ast.FunctionDef) -> Optional[ast.FunctionDef]:
    for stmt in outer.body:
        if isinstance(stmt, ast.FunctionDef) and _has_decorator(stmt, "T.prim_func"):
            return stmt
    return None


def _has_T_kernel_block(func: ast.FunctionDef) -> bool:
    class V(ast.NodeVisitor):
        found = False

        def visit_With(self, node: ast.With) -> None:
            for item in node.items:
                if _is_T_call(item.context_expr, "Kernel"):
                    self.found = True
            self.generic_visit(node)

    v = V()
    v.visit(func)
    return v.found


def _find_T_kernel(func: ast.FunctionDef) -> Optional[ast.With]:
    for node in ast.walk(func):
        if isinstance(node, ast.With):
            for item in node.items:
                if _is_T_call(item.context_expr, "Kernel"):
                    return node
    return None


# ---------------------------------------------------------------------------
# Parameter and annotation parsing
# ---------------------------------------------------------------------------


def _parse_tensor_annotation(ann: ast.expr) -> Optional[TensorParam]:
    """Recognise `T.Tensor[(shape), dtype]`, `T.Tensor((shape), dtype)`,
    and `T.StridedTensor[(shape), (strides), dtype]`."""
    name = _t_attr_name(ann)
    if name not in ("Tensor", "StridedTensor"):
        return None

    if isinstance(ann, ast.Subscript):
        slice_expr = ann.slice
        if isinstance(slice_expr, ast.Tuple):
            elts = list(slice_expr.elts)
        else:
            elts = [slice_expr]
    elif isinstance(ann, ast.Call):
        elts = list(ann.args)
    else:
        return None

    if name == "Tensor":
        if len(elts) < 2:
            return None
        shape = _unpack_tuple(elts[0])
        dtype = elts[1]
        return TensorParam(shape=shape, dtype=dtype, strided=False)
    # StridedTensor
    if len(elts) < 3:
        return None
    shape = _unpack_tuple(elts[0])
    strides = _unpack_tuple(elts[1])
    dtype = elts[2]
    return TensorParam(shape=shape, dtype=dtype, strided=True, strides=strides)


def _unpack_tuple(node: ast.expr) -> list[ast.expr]:
    if isinstance(node, ast.Tuple):
        return list(node.elts)
    if isinstance(node, ast.List):
        return list(node.elts)
    return [node]


def _parse_params(func: ast.FunctionDef) -> tuple[list[Param], list[str]]:
    """Return (params, constexpr_param_names).

    Tensor parameters can be annotated either on the signature
    (`def k(A: T.Tensor[…])` — used by `@T.prim_func`-style factory kernels)
    or **inside the body** as standalone AnnAssigns (`A: T.Tensor((M,K), dtype)`
    — the idiom used by `@tilelang.jit` body kernels). We honour both forms.
    """
    params: list[Param] = []
    constexpr_names: list[str] = []
    args = func.args
    defaults: dict[str, Optional[ast.expr]] = {}
    n_defaults = len(args.defaults)
    n_positional = len(args.args)
    for i, a in enumerate(args.args):
        d_idx = i - (n_positional - n_defaults)
        defaults[a.arg] = args.defaults[d_idx] if d_idx >= 0 else None
    for kw in args.kwonlyargs:
        defaults[kw.arg] = None
    for i, kw in enumerate(args.kwonlyargs):
        if i < len(args.kw_defaults) and args.kw_defaults[i] is not None:
            defaults[kw.arg] = args.kw_defaults[i]

    # Body-level tensor annotations: collect `A: T.Tensor((M, K), dtype)`
    # statements that re-type a parameter name.
    body_tensor_anns: dict[str, TensorParam] = {}
    for stmt in func.body:
        if isinstance(stmt, ast.AnnAssign) and stmt.value is None and isinstance(stmt.target, ast.Name):
            tp = _parse_tensor_annotation(stmt.annotation)
            if tp is not None:
                body_tensor_anns[stmt.target.id] = tp

    for a in list(args.args) + list(args.kwonlyargs):
        # Prefer body annotation, then signature annotation.
        tp = body_tensor_anns.get(a.arg)
        if tp is None and a.annotation is not None:
            tp = _parse_tensor_annotation(a.annotation)
        if tp is not None:
            params.append(Param(name=a.arg, kind="tensor", tensor=tp))
            continue
        default = defaults.get(a.arg)
        params.append(
            Param(
                name=a.arg,
                kind="scalar",
                scalar=ScalarParam(annotation=a.annotation, default=default),
            )
        )
        # All scalar params to a @tilelang.jit kernel are compile-time bound.
        constexpr_names.append(a.arg)
    return params, constexpr_names


# ---------------------------------------------------------------------------
# Outer-factory closure extraction
# ---------------------------------------------------------------------------


def _load_module_for_closure(path: str) -> Optional[types.ModuleType]:
    """Import the input file as a module so we can evaluate factory closures.

    We need this because `T.Fragment(forward_fn=create_loop_layout_fn(...))` has a
    Python callable that has to be evaluated to enumerate the TV-layout. Pure-AST
    analysis can't do that — we actually run the outer factory.
    """
    p = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(f"_tl2fly_{p.stem}", p)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    try:
        # Stub `tilelang` and `T` so import doesn't fail in env without tilelang.
        _install_tilelang_stub(mod)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
    except Exception:
        # We tolerate import failure — closure-dependent kernels will fail later
        # with a clearer error at the use site.
        return None
    return mod


def _install_tilelang_stub(mod: types.ModuleType) -> None:
    """If `tilelang` isn't importable, install a minimal stub good enough for
    the outer factory's literal/closure assignments to evaluate."""
    if "tilelang" in sys.modules:
        return
    stub = types.ModuleType("tilelang")

    def _jit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]

        def deco(fn):
            return fn

        return deco

    stub.jit = _jit
    stub.PassConfigKey = types.SimpleNamespace(TL_DISABLE_WARP_SPECIALIZED="TL_DISABLE_WARP_SPECIALIZED")
    lang = types.ModuleType("tilelang.language")

    class _StubT:
        # dtypes
        float16 = "float16"
        float32 = "float32"
        bfloat16 = "bfloat16"
        int8 = "int8"
        uint8 = "uint8"
        int32 = "int32"
        uint32 = "uint32"

        def __getattr__(self, name):
            return lambda *a, **k: None

    class _Fragment:
        def __init__(self, shape, forward_fn=None):
            self.shape = shape
            self.forward_fn = forward_fn

    class _Dtype(str):
        def __new__(cls, value):
            return str.__new__(cls, str(value))

        @property
        def bytes(self):
            mapping = {
                "float16": 2,
                "bfloat16": 2,
                "float32": 4,
                "float64": 8,
                "int8": 1,
                "uint8": 1,
                "int16": 2,
                "uint16": 2,
                "int32": 4,
                "uint32": 4,
                "int64": 8,
                "uint64": 8,
                "bool": 1,
            }
            return mapping.get(str(self), 1)

    def _dynamic(name):
        return DynamicSym(name=name)

    def _const(names):
        return tuple(n.strip() for n in names.split(","))

    def _Tensor(*a, **k):
        return None

    def _StridedTensor(*a, **k):
        return None

    def _prim_func(fn):
        return fn

    T_ns = types.SimpleNamespace(
        dynamic=_dynamic,
        const=_const,
        Fragment=_Fragment,
        Tensor=_Tensor,
        StridedTensor=_StridedTensor,
        prim_func=_prim_func,
        dtype=_Dtype,
        float16="float16",
        float32="float32",
        bfloat16="bfloat16",
        int8="int8",
        uint8="uint8",
        int32="int32",
    )
    lang.T = T_ns
    for attr in [
        "dynamic",
        "const",
        "Fragment",
        "Tensor",
        "StridedTensor",
        "prim_func",
        "dtype",
        "float16",
        "float32",
        "bfloat16",
        "int8",
        "uint8",
        "int32",
    ]:
        setattr(lang, attr, getattr(T_ns, attr))
    stub.language = lang
    sys.modules["tilelang"] = stub
    sys.modules["tilelang.language"] = lang


def _extract_closure(outer: ast.FunctionDef, mod: Optional[types.ModuleType]) -> ClosureBindings:
    """Capture literal-assignment closure bindings from the outer factory.

    Tries two passes:
      1. If `mod` imported successfully, eval each assignment against module
         globals to get real Python values (catches `T.Fragment(..., forward_fn=fn)`).
      2. Static AST pass: recognise `T.dynamic("name")` and literal-int/string
         assignments and a few constant expressions, in case (1) failed.
    """
    bindings = ClosureBindings()
    if mod is not None:
        globs = vars(mod)
        locs: dict[str, Any] = {}
        for stmt in outer.body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                name = stmt.targets[0].id
                try:
                    code = compile(ast.Expression(stmt.value), "<closure>", "eval")
                    value = eval(code, globs, locs)
                except Exception:
                    continue
                locs[name] = value
                if hasattr(value, "forward_fn"):
                    extents = getattr(value, "shape", None)
                    if isinstance(extents, tuple) and len(extents) == 2 and all(isinstance(e, int) for e in extents):
                        bindings.values[name] = LoopLayout(extents=extents, forward_fn=value.forward_fn)
                        continue
                bindings.values[name] = value
    # Static fallback: catch the things we always want even when import failed.
    for stmt in outer.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            name = stmt.targets[0].id
            if name in bindings.values:
                continue
            val = stmt.value
            if _is_T_call(val, "dynamic") and val.args and isinstance(val.args[0], ast.Constant) and isinstance(val.args[0].value, str):
                bindings.values[name] = DynamicSym(name=val.args[0].value)
            elif isinstance(val, ast.Constant) and isinstance(val.value, (int, str, float, bool)):
                bindings.values[name] = val.value
    return bindings


# ---------------------------------------------------------------------------
# T.Kernel block parsing
# ---------------------------------------------------------------------------


def _parse_kernel_with(node: ast.With) -> tuple[list[ast.expr], ast.expr, list[str]]:
    """Return (grid_exprs, threads_expr, block_axis_names)."""
    item = node.items[0]
    call = item.context_expr
    assert isinstance(call, ast.Call)
    # Grid args: all positional args before kwargs
    grid = list(call.args)
    threads_expr: ast.expr = ast.Constant(value=128)
    for kw in call.keywords:
        if kw.arg == "threads":
            threads_expr = kw.value
    # Optional ctx var `as (bx, by, ...)`
    axes: list[str] = []
    if item.optional_vars is not None:
        ov = item.optional_vars
        if isinstance(ov, ast.Tuple):
            for e in ov.elts:
                if isinstance(e, ast.Name):
                    axes.append(e.id)
                else:
                    axes.append("<unnamed>")
        elif isinstance(ov, ast.Name):
            axes = [ov.id]
    return grid, threads_expr, axes


# ---------------------------------------------------------------------------
# Body parsing
# ---------------------------------------------------------------------------


class _BodyParser:
    def __init__(self, path: str, factory_locals: set[str], constexpr_param_names: set[str]):
        self.path = path
        self.factory_locals = factory_locals
        self.constexpr_param_names = constexpr_param_names

    def loc(self, node: ast.AST) -> SourceLocation:
        return SourceLocation.from_node(self.path, node)

    # public ---------------------------------------------------------------
    def parse_block(self, stmts: list[ast.stmt]) -> list[Stmt]:
        out: list[Stmt] = []
        for s in stmts:
            r = self.parse_stmt(s)
            if r is not None:
                out.extend(r if isinstance(r, list) else [r])
        return out

    def parse_stmt(self, node: ast.stmt) -> list[Stmt] | Stmt | None:
        loc = self.loc(node)
        # Assignments — either an Alloc, or a plain per-thread assignment
        if isinstance(node, ast.Assign):
            return self._parse_assign(node, loc)
        if isinstance(node, ast.AugAssign):
            return Assign(target=node.target, value=ast.BinOp(left=node.target, op=node.op, right=node.value), src=loc)
        if isinstance(node, ast.AnnAssign):
            return self._parse_ann_assign(node, loc)
        if isinstance(node, ast.Expr):
            return self._parse_expr_stmt(node, loc)
        if isinstance(node, ast.For):
            return self._parse_for(node, loc)
        if isinstance(node, ast.If):
            body = self.parse_block(node.body)
            orelse = self.parse_block(node.orelse)
            return IfStmt(test=node.test, body=body, orelse=orelse, src=loc)
        if isinstance(node, ast.With):
            # Nested T.Kernel not supported; pass through inner block.
            return self.parse_block(node.body)
        if isinstance(node, ast.Return):
            return None
        if isinstance(node, ast.Pass):
            return None
        # Unknown stmt — keep as raw passthrough
        return Raw(node=node, note="unhandled statement", src=loc)

    # ---------------------------------------------------------------------
    def _parse_assign(self, node: ast.Assign, loc: SourceLocation) -> Stmt | list[Stmt]:
        if len(node.targets) != 1:
            return Assign(target=ast.Tuple(elts=node.targets, ctx=ast.Store()), value=node.value, src=loc)
        target = node.targets[0]
        # Allocation forms
        if isinstance(node.value, ast.Call):
            name = _t_attr_name(node.value)
            if name in ("alloc_shared", "alloc_fragment", "alloc_local", "alloc_var"):
                kind = {"alloc_shared": "shared", "alloc_fragment": "fragment", "alloc_local": "local", "alloc_var": "var"}[name]
                args = node.value.args
                if kind == "var":
                    dtype = args[0]
                    shape: list[ast.expr] = []
                    init = None
                    for kw in node.value.keywords:
                        if kw.arg == "init":
                            init = kw.value
                else:
                    if not args:
                        self._fail(node.value, "alloc_* requires a shape argument")
                    shape = _unpack_tuple(args[0])
                    dtype = args[1] if len(args) >= 2 else self._fail(node.value, "alloc_* requires a dtype argument")
                    init = None
                if not isinstance(target, ast.Name):
                    self._fail(node, "allocation must bind to a simple name")
                return Alloc(name=target.id, kind=kind, shape=shape, dtype=dtype, init=init, src=loc)
            # Multiple-name T.const("M, N, K") — handled at outer level too
            if name == "const":
                return None
            # `tid = T.get_thread_binding()` — preserve as plain Assign
        # Plain assignment (per-thread scalar code, register tile writes, etc.)
        return Assign(target=target, value=node.value, src=loc)

    def _parse_ann_assign(self, node: ast.AnnAssign, loc: SourceLocation) -> Optional[Stmt]:
        # Most ann-assigns inside the body are type-decl-only (e.g. `A: T.Tensor[...]`)
        # which the parser already handled at the param level. Drop them.
        if node.value is None:
            return None
        if not isinstance(node.target, ast.Name):
            return Assign(target=node.target, value=node.value, src=loc)
        return Assign(target=node.target, value=node.value, src=loc)

    def _parse_expr_stmt(self, node: ast.Expr, loc: SourceLocation) -> Optional[Stmt]:
        call = node.value
        if not isinstance(call, ast.Call):
            return None
        name = _t_attr_name(call)
        if name is None:
            return Raw(node=node, note="non-T expression statement", src=loc)
        cls = api.classify_symbol(name)
        if cls == "nvidia_only":
            raise UnsupportedConstruct(f"T.{name}", loc, hint="no AMD equivalent in v1")
        if cls == "unsupported":
            raise UnsupportedConstruct(f"T.{name}", loc, hint="not handled in v1")
        if cls == "droppable":
            return Raw(node=node, note=f"dropped annotation T.{name}", src=loc)
        if cls == "unknown":
            raise UnsupportedConstruct(f"T.{name}", loc, hint="symbol not in v1 catalog")

        if name == "copy":
            if len(call.args) < 2:
                self._fail(call, "T.copy requires (src, dst)")
            return Copy(src_expr=call.args[0], dst_expr=call.args[1], src=loc)
        if name == "clear":
            return Clear(target=call.args[0], src=loc)
        if name == "fill":
            return Fill(target=call.args[0], value=call.args[1], src=loc)
        if name == "gemm":
            transpose_a = False
            transpose_b = False
            for kw in call.keywords:
                if kw.arg == "transpose_A":
                    transpose_a = bool(_literal_int(kw.value))
                elif kw.arg == "transpose_B":
                    transpose_b = bool(_literal_int(kw.value))
            a, b, c = call.args[0], call.args[1], call.args[2]
            return Gemm(a=a, b=b, c=c, transpose_a=transpose_a, transpose_b=transpose_b, src=loc)
        if name == "sync_threads":
            return SyncThreads(src=loc)
        if name == "assume":
            return Assume(pred=call.args[0], src=loc)
        # Anything else is treated as a raw expression with a comment.
        return Raw(node=node, note=f"T.{name} not yet handled", src=loc)

    def _parse_for(self, node: ast.For, loc: SourceLocation) -> Stmt:
        iter_call = node.iter
        target = node.target
        body = self.parse_block(node.body)
        # T.Pipelined / T.Parallel / T.unroll / T.vectorized / T.serial
        if isinstance(iter_call, ast.Call):
            name = _t_attr_name(iter_call)
            if name == "Pipelined":
                trip = iter_call.args[0]
                num_stages = 1
                for kw in iter_call.keywords:
                    if kw.arg == "num_stages":
                        v = _literal_int(kw.value)
                        if v is not None:
                            num_stages = v
                if not isinstance(target, ast.Name):
                    self._fail(node, "T.Pipelined iteration target must be a simple name")
                return Pipelined(var=target.id, trip=trip, num_stages=num_stages, body=body, src=loc)
            if name == "Parallel":
                extents = list(iter_call.args)
                ll = None
                for kw in iter_call.keywords:
                    if kw.arg == "loop_layout":
                        ll = self._resolve_loop_layout(kw.value, loc)
                vars_ = self._unpack_for_target(target, loc)
                return Parallel(vars=vars_, extents=extents, loop_layout=ll, body=body, src=loc)
            if name in ("unroll", "Unroll"):
                start, stop = self._parse_loop_bounds(iter_call, loc)
                if not isinstance(target, ast.Name):
                    self._fail(node, "T.unroll iteration target must be a simple name")
                return Unroll(var=target.id, start=start, stop=stop, body=body, src=loc)
            if name in ("vectorized", "Vectorized"):
                start, stop = self._parse_loop_bounds(iter_call, loc)
                if not isinstance(target, ast.Name):
                    self._fail(node, "T.vectorized iteration target must be a simple name")
                return Vectorized(var=target.id, start=start, stop=stop, body=body, src=loc)
            if name in ("serial", "Serial"):
                start, stop = self._parse_loop_bounds(iter_call, loc)
                step = None
                if len(iter_call.args) >= 3:
                    step = iter_call.args[2]
                if not isinstance(target, ast.Name):
                    self._fail(node, "T.serial iteration target must be a simple name")
                return Serial(var=target.id, start=start, stop=stop, step=step, body=body, src=loc)
        # Plain Python `for` (treated as serial)
        if isinstance(iter_call, ast.Call) and isinstance(iter_call.func, ast.Name) and iter_call.func.id == "range":
            start, stop = self._parse_loop_bounds(iter_call, loc)
            step = iter_call.args[2] if len(iter_call.args) >= 3 else None
            if not isinstance(target, ast.Name):
                self._fail(node, "for-range target must be a simple name")
            return Serial(var=target.id, start=start, stop=stop, step=step, body=body, src=loc)
        # Anything else is unsupported as a loop iterator
        raise UnsupportedConstruct("for-loop iterator", loc, hint=f"only T.Pipelined/Parallel/unroll/vectorized/serial/range are supported; got {ast.unparse(iter_call) if isinstance(iter_call, ast.AST) else type(iter_call).__name__}")

    # ---------------------------------------------------------------------
    def _unpack_for_target(self, target: ast.expr, loc: SourceLocation) -> list[str]:
        if isinstance(target, ast.Name):
            return [target.id]
        if isinstance(target, ast.Tuple):
            out = []
            for e in target.elts:
                if not isinstance(e, ast.Name):
                    self._fail(target, "tuple-unpack for target must contain only names")
                out.append(e.id)
            return out
        self._fail(target, "unrecognised for-loop target")
        return []  # unreachable

    def _parse_loop_bounds(self, call: ast.Call, loc: SourceLocation) -> tuple[ast.expr, ast.expr]:
        args = call.args
        if len(args) == 0:
            self._fail(call, "loop iterator needs at least one argument")
        if len(args) == 1:
            return ast.Constant(value=0), args[0]
        return args[0], args[1]

    def _resolve_loop_layout(self, value: ast.expr, loc: SourceLocation) -> Optional[LoopLayout]:
        # Two cases: a direct T.Fragment(...) call, or a Name that references a
        # closure-captured LoopLayout.
        if isinstance(value, ast.Name):
            return self.factory_layouts.get(value.id) if hasattr(self, "factory_layouts") else None
        if _is_T_call(value, "Fragment"):
            # Not statically resolvable without the closure
            raise UnsupportedConstruct(
                "T.Fragment(...) inline",
                loc,
                hint="bind T.Fragment to a name in the outer factory; in-place forward_fn closures aren't statically resolved",
            )
        return None

    # ---------------------------------------------------------------------
    def _fail(self, node: ast.AST, msg: str):
        raise TranslationError(msg, SourceLocation.from_node(self.path, node))


# ---------------------------------------------------------------------------
# Top-level entry points
# ---------------------------------------------------------------------------


def parse_source(source: str, path: str = "<string>") -> KernelIR:
    tree = ast.parse(source, filename=path)
    return _parse_tree(tree, path)


def parse_file(path: str) -> KernelIR:
    src = Path(path).read_text()
    tree = ast.parse(src, filename=path)
    return _parse_tree(tree, path)


def _parse_tree(tree: ast.Module, path: str) -> KernelIR:
    found = _find_kernels(tree)
    if not found:
        raise MalformedKernel(
            "no @tilelang.jit kernel or T.Kernel block found in this file",
            SourceLocation(path=path, line=1, col=0),
        )
    if len(found) > 1:
        raise MalformedKernel(
            f"found {len(found)} kernels; v1 supports exactly one per file",
            SourceLocation.from_node(path, found[1][0]),
        )
    kernel_fn, outer_fn = found[0]
    loc = SourceLocation.from_node(path, kernel_fn)
    params, constexpr_names = _parse_params(kernel_fn)
    # Outer factory closure (LoopLayout, DynamicSym, ints)
    if outer_fn is not None:
        mod = _load_module_for_closure(path)
        closure = _extract_closure(outer_fn, mod)
    else:
        closure = ClosureBindings()
    # Decorator metadata
    decorators_meta = _decorator_meta(outer_fn or kernel_fn)
    # Find the T.Kernel with-block
    with_node = _find_T_kernel(kernel_fn)
    if with_node is None:
        raise MalformedKernel("kernel body does not contain a `with T.Kernel(...)` block", loc)
    grid, threads_expr, axes = _parse_kernel_with(with_node)
    # Pull out top-of-body T.const declarations to record symbol names.
    const_names = _collect_const_names(kernel_fn)
    # Body parser knows about the factory layouts (loop_layout=name).
    parser = _BodyParser(path, factory_locals=set(closure.values.keys()), constexpr_param_names=set(constexpr_names))
    parser.factory_layouts = {k: v for k, v in closure.values.items() if isinstance(v, LoopLayout)}  # type: ignore[attr-defined]
    body = parser.parse_block(with_node.body)
    # Determine returned tensor name (T.empty pattern) and capture its shape/dtype.
    returns = _find_returned_name(kernel_fn)
    empty_shape, empty_dtype = _find_empty_shape_dtype(kernel_fn, returns) if returns else (None, None)
    return KernelIR(
        name=kernel_fn.name,
        params=params,
        decorators_meta=decorators_meta,
        grid=grid,
        block_threads=threads_expr,
        block_axes=axes,
        body=body,
        returns=returns,
        const_names=const_names,
        closure=closure,
        src_location=loc,
        constexpr_params=constexpr_names,
        empty_shape=empty_shape,
        empty_dtype=empty_dtype,
    )


def _find_empty_shape_dtype(func: ast.FunctionDef, name: str) -> tuple[Optional[list[ast.expr]], Optional[ast.expr]]:
    for stmt in ast.walk(func):
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == name
            and isinstance(stmt.value, ast.Call)
            and _is_T_call(stmt.value, "empty")
        ):
            args = stmt.value.args
            if args:
                shape = list(args[0].elts) if isinstance(args[0], ast.Tuple) else [args[0]]
                dtype = args[1] if len(args) >= 2 else None
                return shape, dtype
    return None, None


def _decorator_meta(func: ast.FunctionDef) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    for dec in func.decorator_list:
        if isinstance(dec, ast.Call):
            name = _decorator_name(dec.func)
            if name == "tilelang.jit":
                kw = {}
                for k in dec.keywords:
                    try:
                        kw[k.arg] = ast.unparse(k.value)
                    except Exception:
                        kw[k.arg] = "<unrepr>"
                meta["jit_kwargs"] = kw
    return meta


def _collect_const_names(func: ast.FunctionDef) -> list[str]:
    names: list[str] = []
    for stmt in ast.walk(func):
        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call) and _is_T_call(stmt.value, "const"):
            if stmt.value.args and isinstance(stmt.value.args[0], ast.Constant) and isinstance(stmt.value.args[0].value, str):
                for n in stmt.value.args[0].value.split(","):
                    names.append(n.strip())
    return names


def _find_returned_name(func: ast.FunctionDef) -> Optional[str]:
    for stmt in ast.walk(func):
        if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Name):
            return stmt.value.id
    # Or look for `<name> = T.empty(...)` and assume that's the output
    for stmt in ast.walk(func):
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and isinstance(stmt.value, ast.Call)
            and _is_T_call(stmt.value, "empty")
        ):
            return stmt.targets[0].id
    return None
