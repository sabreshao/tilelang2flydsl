"""Stdlib-only smoke runner used in environments without pytest.

Runs the same logic as the test_*.py files. Usage:
    python3 tests/run_tests.py
"""

from __future__ import annotations

import py_compile
import sys
import tempfile
import textwrap
import traceback
from pathlib import Path

# Make `tilelang2flydsl` importable and install the flydsl stub.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

# Re-use the same stub as the conftest module.
import conftest  # noqa: F401  (side-effect: installs flydsl stub)

from tilelang2flydsl.translate import translate_file, translate_source  # noqa: E402
from tilelang2flydsl.diagnostics import UnsupportedConstruct  # noqa: E402
from tilelang2flydsl.parse import parse_file  # noqa: E402
from tilelang2flydsl.kernel_ir import Clear, Copy, Gemm, Parallel, Pipelined  # noqa: E402


FIXTURES = HERE / "fixtures"


_results: list[tuple[str, str, str]] = []  # (name, status, detail)


def _run(name: str, fn) -> None:
    try:
        fn()
    except Exception as e:
        tb = traceback.format_exc()
        _results.append((name, "FAIL", f"{e!r}\n{tb}"))
        return
    _results.append((name, "OK", ""))


def test_parse_gemm():
    ir = parse_file(str(FIXTURES / "example_gemm.py"))
    assert ir.name == "matmul", ir.name
    names = [p.name for p in ir.params]
    assert names == ["A", "B", "block_M", "block_N", "block_K", "dtype", "accum_dtype"], names
    assert ir.const_names == ["M", "N", "K"], ir.const_names
    assert "block_M" in ir.constexpr_params, ir.constexpr_params
    assert len(ir.grid) == 2
    assert ir.block_axes == ["bx", "by"]
    kinds = [type(s).__name__ for s in ir.body]
    assert "Clear" in kinds and "Pipelined" in kinds and "Copy" in kinds, kinds
    pipe = next(s for s in ir.body if isinstance(s, Pipelined))
    inner = [type(s).__name__ for s in pipe.body]
    assert inner.count("Copy") == 2 and inner.count("Gemm") == 1, inner
    assert ir.returns == "C", ir.returns


def test_parse_transpose():
    ir = parse_file(str(FIXTURES / "batched_transpose_kernel.py"))
    assert ir.name == "batched_transpose_kernel", ir.name
    tparams = [p.name for p in ir.params if p.kind == "tensor"]
    assert tparams == ["x", "out"], tparams
    assert len(ir.grid) == 3
    assert ir.block_axes == ["pid_y", "pid_x", "pid_batch"]
    kinds = [type(s).__name__ for s in ir.body]
    assert "Unroll" in kinds, kinds
    assert "SyncThreads" in kinds, kinds
    assert "Parallel" in kinds, kinds


def test_translate_gemm_smoke():
    src = translate_file(str(FIXTURES / "example_gemm.py"))
    # LDS-staged template; mirrors tests/runnable/example_gemm.py.
    for needle in [
        "import flydsl.expr as fx",
        "@flyc.kernel",
        "@flyc.jit",
        "fx.rocdl.make_buffer_tensor",
        "fx.get_dyn_shared",
        "fx.make_composed_layout",
        "make_tiled_copy_A",
        "make_tiled_copy_B",
        "make_fragment_A",
        "make_fragment_B",
        "make_fragment_C",
        "fx.gemm(",
        "fx.gpu.barrier(",
        "fx.block_idx",
        "fx.thread_idx",
        "GemmTraversalOrder",
        "range_constexpr",
    ]:
        assert needle in src, f"missing {needle} in:\n{src}"
    # py_compile the generated source
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
        path = f.name
    py_compile.compile(path, doraise=True)


def test_translate_transpose_smoke():
    src = translate_file(str(FIXTURES / "batched_transpose_kernel.py"))
    for needle in [
        # Transpose template output — direct gmem→gmem element-wise transpose
        # (LDS swizzle from the TileLang fixture is intentionally dropped).
        "import flydsl.expr as fx",
        "@flyc.kernel",
        "@flyc.jit",
        "fx.rocdl.make_buffer_tensor",
        "BLOCK_X",
        "BLOCK_Y",
        "NUM_THREADS",
        "range_constexpr",
        "fx.block_idx.z",
    ]:
        assert needle in src, f"missing {needle} in:\n{src}"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
        path = f.name
    py_compile.compile(path, doraise=True)


def test_generated_source_imports_under_stub():
    """The emitted module should import cleanly with the flydsl stub installed
    by conftest.py. This catches name typos and signature errors that
    py_compile would miss."""
    import importlib.util

    for fixture in ("example_gemm.py", "batched_transpose_kernel.py"):
        src = translate_file(str(FIXTURES / fixture))
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(src)
            path = f.name
        spec = importlib.util.spec_from_file_location(f"_t_{Path(path).stem}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)


def test_matches_golden():
    """Generated source must match the checked-in golden file byte-for-byte.

    Regenerate goldens with `python -m tilelang2flydsl tests/fixtures/<f>.py -o tests/expected/<f>.flydsl.py`.
    """
    for src_name, gold_name in (
        ("example_gemm.py", "example_gemm.flydsl.py"),
        ("batched_transpose_kernel.py", "batched_transpose_kernel.flydsl.py"),
    ):
        actual = translate_file(str(FIXTURES / src_name))
        expected_path = HERE / "expected" / gold_name
        expected = expected_path.read_text()
        if actual != expected:
            # Surface a useful diff snippet
            import difflib

            diff = "".join(
                difflib.unified_diff(
                    expected.splitlines(keepends=True),
                    actual.splitlines(keepends=True),
                    fromfile=str(expected_path),
                    tofile=f"<actual {src_name}>",
                    n=3,
                )
            )
            raise AssertionError(f"golden mismatch for {src_name}:\n{diff}")


def test_unsupported_construct_raises():
    src = textwrap.dedent(
        """
        import tilelang
        import tilelang.language as T

        @tilelang.jit
        def k(A, B):
            M, N = T.const("M, N")
            A: T.Tensor((M, N), T.float16)
            B: T.Tensor((M, N), T.float16)
            with T.Kernel(M, N, threads=128) as (bx, by):
                A_shared = T.alloc_shared((128, 32), T.float16)
                T.tma_copy(A[bx*128, by*32], A_shared)
            return B
        """
    )
    try:
        translate_source(src, path="<inline>")
    except UnsupportedConstruct as e:
        msg = str(e)
        assert "T.tma_copy" in msg, msg
        # file:line:col header
        assert "<inline>:" in msg, msg
        return
    raise AssertionError("expected UnsupportedConstruct to be raised")


def main():
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            _run(name, fn)
    fails = [(n, d) for n, s, d in _results if s == "FAIL"]
    for n, s, _ in _results:
        print(f"{s:4} {n}")
    if fails:
        print("\n----- failure details -----")
        for n, d in fails:
            print(f"\n{n}:\n{d}")
        sys.exit(1)
    print(f"\nall {len(_results)} tests passed.")


if __name__ == "__main__":
    main()
