# tilelang2flydsl

Source-to-source translator from TileLang kernels to FlyDSL kernels.

Reads a Python file containing a `@tilelang.jit` (or `@T.prim_func` inside a factory) kernel,
emits a Python file containing a `@flyc.kernel` device function and a `@flyc.jit` host wrapper
that target AMD `gfx942` (CDNA3, MFMA, wave64).

The translator never imports `tilelang` or `flydsl` at runtime — it works purely on Python
AST via the standard library.

## Usage

```bash
python -m tilelang2flydsl <input.py> -o <output.py>
```

Or via the console entry point:

```bash
tilelang2flydsl <input.py> -o <output.py>
```

## Supported (v1)

- `@tilelang.jit` direct kernels and `@tilelang.jit(...)` + inner `@T.prim_func` factory kernels.
- `T.Kernel` (1D / 2D / 3D grid) with `threads=` and `as (bx, by, …)` bindings.
- `T.alloc_shared`, `T.alloc_fragment`, `T.alloc_local`.
- `T.copy` (global ↔ shared, fragment → global), `T.clear`, `T.fill`.
- `T.gemm` on shared buffers + fragment accumulator → CDNA3 MFMA.
- `T.Pipelined` (lowers to a plain loop; pipelining deferred).
- `T.Parallel` (with optional `loop_layout=T.Fragment(forward_fn=…)`).
- `T.unroll`, `T.vectorized`, `T.serial`.
- `T.get_thread_binding`, `T.sync_threads`, `T.ceildiv`, `T.assume`.
- `T.const`, `T.dynamic` symbolic dims; `T.Tensor` / `T.StridedTensor` annotations.
- Per-thread scalar reads/writes preserved verbatim.

## Out of scope (v1)

- Software pipelining (`num_stages` is recorded as a TODO comment).
- Architecture dispatch (gfx950 / gfx1250 / WMMA).
- `T.reduce_*`, `T.atomic_*`, FlashAttention-style kernels.
- NVIDIA-only intrinsics (`T.tma_*`, `T.wgmma_*`, clusters, descriptors).
- `T.use_swizzle` and other `T.annotate_*` hints (silently dropped with a comment).
- Multi-kernel files (one `@tilelang.jit` per input).

Anything else raises a hard error with `file:line:col` and the offending symbol.
