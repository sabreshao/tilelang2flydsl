# tilelang2flydsl — Build Summary

Built `tilelang2flydsl/` — a stdlib-only Python-AST → Python-AST translator that turns a `@tilelang.jit` kernel into a `@flyc.kernel` + `@flyc.jit` pair targeting gfx942 (CDNA3 MFMA).

**Pipeline:** `parse.py` → `lower/__init__.py` → `emit.py`, all driven by `translate.py` and the `python -m tilelang2flydsl <input>` CLI.

## End-to-end run on the two reference inputs

| Input | Result |
|---|---|
| `tilelang/examples/gemm/example_gemm.py` | Clean GEMM with MFMA(16,16,16, Float16) atom, `SharedAllocator`, `make_tiled_copy` + `partition_S/D`, deferred `make_fragment_C` accumulator, single-emit `mma_atom`/`thr_mma`/`copy_atom_float16` at kernel top, pipelined K-loop with `fx.gpu.barrier()` framing, `thr_mma.partition_C` epilogue. |
| `TileKernels/tile_kernels/transpose/batched_transpose_kernel.py` | Per-thread imperative code preserved verbatim (`tid`, `row`/`col`, unrolled `j`/`k` loops, hand-written swizzle math), tensor refs rewritten to `x_g[...]` / `out_g[...]`, closure constants (`block_x`, `block_k=4`, `num_threads=256`, …) promoted to `Constexpr[int]` kernel args, `T.dynamic` names threaded through as `fx.Int32` args. |

## Hard-fail diagnostics

Verified for `T.tma_copy`, `T.atomic_add`, `T.reduce_sum`, `T.cluster_sync` (NVIDIA-only), and missing-kernel files — each surfaces `file:line:col` plus the offending symbol.

## Known v1 limitations

Documented in README:

- Software pipelining defers to a plain loop with a `# TODO` comment.
- `dtype.bytes` passes through unchanged (TileLang's `T.dtype` has it; FlyDSL's `Constexpr[int]` does not).
- No reductions, atomics, FlashAttention-style kernels.
- Single arch (gfx942 CDNA3 MFMA, wave64).
- One kernel per file.

## Test status

All 7 stdlib-only tests pass via `python3 tests/run_tests.py`:

- `test_parse_gemm`, `test_parse_transpose` — KernelIR shape assertions.
- `test_translate_gemm_smoke`, `test_translate_transpose_smoke` — output contains expected idioms; `py_compile` accepts it.
- `test_generated_source_imports_under_stub` — emitted modules import cleanly under the conftest flydsl stub.
- `test_matches_golden` — byte-for-byte stable against `tests/expected/*.flydsl.py`.
- `test_unsupported_construct_raises` — `T.tma_copy` hard-fails with `file:line:col` and the symbol name.

## To validate on hardware

Copy `tests/expected/example_gemm.flydsl.py` (or the transpose golden) into a built FlyDSL/ROCm environment on a gfx942 box and run against a torch reference — the test suite covers structure and import-ability, not runtime correctness.
