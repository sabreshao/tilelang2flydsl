"""Test setup: install a stub `flydsl` package so the emitted source `import`s
without needing a real FlyDSL install. The stub exposes everything the emitter
references as no-op callables/objects."""

from __future__ import annotations

import sys
import types
from pathlib import Path


def _install_flydsl_stub() -> None:
    if "flydsl" in sys.modules:
        return

    flydsl = types.ModuleType("flydsl")

    class _Anything:
        """Permissive stand-in: any attribute access yields another _Anything,
        any call returns _Anything. Lets the emitted module load without errors."""

        def __init__(self, *a, **k):
            pass

        def __getattr__(self, name):
            return _Anything()

        def __call__(self, *a, **k):
            return _Anything()

        def __getitem__(self, key):
            return _Anything()

    anything = _Anything()

    # dtypes
    for n in [
        "Float16",
        "Float32",
        "BFloat16",
        "Float64",
        "Int8",
        "UInt8",
        "Int16",
        "UInt16",
        "Int32",
        "UInt32",
        "Int64",
        "UInt64",
        "Bool",
    ]:
        setattr(flydsl, n, anything)
    flydsl.Tensor = _Anything
    flydsl.Stream = _Anything

    class _Constexpr:
        def __getitem__(self, item):
            return _Anything()

    flydsl.Constexpr = _Constexpr()

    # API surface used by emitted code
    for n in [
        "make_layout",
        "make_copy_atom",
        "make_tiled_copy",
        "make_mma_atom",
        "make_tiled_mma",
        "make_rmem_tensor",
        "empty",
        "fill",
        "copy",
        "gemm",
        "range",
        "range_constexpr",
        "zipped_divide",
        "logical_divide",
        "flat_divide",
        "slice",
    ]:
        setattr(flydsl, n, _Anything())

    # fx.thread_idx.x / fx.block_idx.x
    flydsl.thread_idx = _Anything()
    flydsl.block_idx = _Anything()

    # fx.gpu.barrier() / fx.gpu.SharedAllocator(...)
    gpu = types.ModuleType("flydsl.gpu")
    gpu.barrier = _Anything()
    gpu.SharedAllocator = _Anything
    flydsl.gpu = gpu
    sys.modules["flydsl.gpu"] = gpu

    # fx.rocdl.make_buffer_tensor / MFMA / BufferCopy*
    rocdl = types.ModuleType("flydsl.rocdl")
    rocdl.make_buffer_tensor = _Anything()
    rocdl.MFMA = _Anything
    rocdl.BufferCopy128b = _Anything
    rocdl.BufferCopy64b = _Anything
    rocdl.BufferCopy32b = _Anything
    flydsl.rocdl = rocdl
    sys.modules["flydsl.rocdl"] = rocdl

    expr_rocdl = types.ModuleType("flydsl.expr.rocdl")
    expr_rocdl.MFMA = _Anything
    expr_rocdl.BufferCopy128b = _Anything
    expr_rocdl.BufferCopy64b = _Anything
    expr_rocdl.BufferCopy32b = _Anything
    expr_rocdl.make_buffer_tensor = _Anything()
    sys.modules["flydsl.expr"] = types.ModuleType("flydsl.expr")
    sys.modules["flydsl.expr.rocdl"] = expr_rocdl

    compiler = types.ModuleType("flydsl.compiler")

    def _kernel(fn):
        return fn

    def _jit(fn):
        return fn

    compiler.kernel = _kernel
    compiler.jit = _jit
    flydsl.compiler = compiler
    sys.modules["flydsl.compiler"] = compiler

    sys.modules["flydsl"] = flydsl

    # The retargeted emitter does `import flydsl.expr as fx`, so install the
    # same symbol surface on the `expr` submodule as well.
    expr = types.ModuleType("flydsl.expr")
    for n in dir(flydsl):
        if not n.startswith("_"):
            setattr(expr, n, getattr(flydsl, n))
    # Additional names referenced by the GEMM template that aren't on the
    # top-level stub.
    for n in [
        "TiledMma", "TiledCopy", "UniversalCopy128b",
        "SwizzleType", "make_composed_layout", "make_ordered_layout",
        "make_fragment_like", "make_tiled_copy_A", "make_tiled_copy_B",
        "make_tiled_copy_C", "make_tile", "static", "get_dyn_shared",
        "make_view", "add_offset", "arith", "T", "GemmTraversalOrder",
    ]:
        if not hasattr(expr, n):
            setattr(expr, n, _Anything() if n[0].islower() else _Anything)
    flydsl.expr = expr
    sys.modules["flydsl.expr"] = expr
    # Re-link the rocdl / gpu submodules so `flydsl.expr.rocdl` resolves.
    expr.rocdl = flydsl.rocdl
    expr.gpu = flydsl.gpu
    sys.modules["flydsl.expr.rocdl"] = flydsl.rocdl


_install_flydsl_stub()


# Ensure the tilelang2flydsl source tree is importable when tests run from
# the repo without `pip install -e .`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
