from __future__ import annotations

from pathlib import Path

from tilelang2flydsl.parse import parse_file
from tilelang2flydsl.kernel_ir import Copy, Gemm, Pipelined, Clear, Parallel, Unroll, Vectorized

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_gemm_basic_shape():
    ir = parse_file(str(FIXTURES / "example_gemm.py"))
    assert ir.name == "matmul"
    assert [p.name for p in ir.params] == ["A", "B", "block_M", "block_N", "block_K", "dtype", "accum_dtype"]
    assert ir.const_names == ["M", "N", "K"]
    assert "block_M" in ir.constexpr_params
    # 2D grid
    assert len(ir.grid) == 2
    assert ir.block_axes == ["bx", "by"]
    # body contains: clear, pipelined(copy, copy, gemm), copy
    kinds = [type(s).__name__ for s in ir.body]
    assert "Clear" in kinds
    assert "Pipelined" in kinds
    assert "Copy" in kinds
    pipe = next(s for s in ir.body if isinstance(s, Pipelined))
    inner_kinds = [type(s).__name__ for s in pipe.body]
    assert inner_kinds.count("Copy") == 2
    assert inner_kinds.count("Gemm") == 1
    assert ir.returns == "C"


def test_parse_transpose_shape():
    ir = parse_file(str(FIXTURES / "batched_transpose_kernel.py"))
    assert ir.name == "batched_transpose_kernel"
    # tensor params
    tensor_params = [p.name for p in ir.params if p.kind == "tensor"]
    assert tensor_params == ["x", "out"]
    # 3D grid, threads=num_threads (a closure constant)
    assert len(ir.grid) == 3
    assert ir.block_axes == ["pid_y", "pid_x", "pid_batch"]
    # Body contains nested unroll loops, sync_threads, parallel
    kinds = [type(s).__name__ for s in ir.body]
    assert "Unroll" in kinds
    assert "SyncThreads" in kinds
    assert "Parallel" in kinds
    par = next(s for s in ir.body if isinstance(s, Parallel))
    # The loop_layout closure should have been captured
    assert par.loop_layout is not None
    assert par.loop_layout.extents == (128, 128) or par.loop_layout.extents == (64, 128) or par.loop_layout.extents == (128, 64) or par.loop_layout.extents == (64, 64)
