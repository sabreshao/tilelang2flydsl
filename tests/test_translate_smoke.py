from __future__ import annotations

import py_compile
import textwrap
from pathlib import Path

import pytest

from tilelang2flydsl.translate import translate_file

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("fixture", ["example_gemm.py", "batched_transpose_kernel.py"])
def test_translate_produces_compilable_source(tmp_path, fixture):
    src = translate_file(str(FIXTURES / fixture))
    assert "import flydsl" in src
    assert "@flyc.kernel" in src
    assert "@flyc.jit" in src
    out = tmp_path / fixture.replace(".py", ".flydsl.py")
    out.write_text(src)
    # py_compile catches any syntactic problem in the emitted file.
    py_compile.compile(str(out), doraise=True)


def test_translate_gemm_contains_expected_idioms():
    src = translate_file(str(FIXTURES / "example_gemm.py"))
    # LDS-staged template output (runnable on MI300X — see tests/runnable/example_gemm.py).
    for needle in [
        "import flydsl.expr as fx",
        "fx.rocdl.make_buffer_tensor",
        "fx.get_dyn_shared",
        "fx.make_composed_layout",
        "fx.make_ordered_layout",
        "fx.add_offset",
        "fx.flat_divide",
        "make_tiled_copy_A",
        "make_tiled_copy_B",
        "make_tiled_copy_C",
        "make_fragment_A",
        "make_fragment_B",
        "make_fragment_C",
        "fx.gemm(",
        "fx.gpu.barrier(",
        "GemmTraversalOrder",
        "range_constexpr",
    ]:
        assert needle in src, f"missing {needle} in emitted source:\n{src}"


def test_translate_transpose_contains_expected_idioms():
    src = translate_file(str(FIXTURES / "batched_transpose_kernel.py"))
    # Transpose template: direct gmem→gmem element-wise transpose. LDS staging
    # + swizzle from the TileLang fixture is intentionally dropped.
    for needle in [
        "import flydsl.expr as fx",
        "fx.rocdl.make_buffer_tensor",
        "BLOCK_X",
        "BLOCK_Y",
        "NUM_THREADS",
        "range_constexpr",
        "fx.block_idx.z",
    ]:
        assert needle in src, f"missing {needle} in emitted source:\n{src}"


def test_unsupported_construct_raises():
    from tilelang2flydsl.diagnostics import UnsupportedConstruct
    from tilelang2flydsl.translate import translate_source

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
    with pytest.raises(UnsupportedConstruct) as ei:
        translate_source(src, path="<inline>")
    msg = str(ei.value)
    assert "T.tma_copy" in msg
    assert ":" in msg  # contains file:line:col
