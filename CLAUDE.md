# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workspace Layout

This directory is **not** a single project — it contains three independent git repositories side-by-side. Always `cd` into the relevant subdirectory before running builds/tests; none of them share a top-level toolchain.

| Path | What it is | Upstream |
|---|---|---|
| `tilelang/` | The Tile Language compiler: Python DSL + C++ runtime layered on Apache TVM. Targets CUDA, HIP/ROCm, CPU, Metal, WebGPU. | `tile-ai/tilelang` |
| `TileKernels/` | DeepSeek's library of LLM-oriented kernels (MoE routing, FP8/FP4 quant, engram, mHC) **built on top of** an installed `tilelang>=0.1.9`. SM90/SM100 only. | `deepseek-ai/TileKernels` |
| `flydsl/` | Unrelated sister project — a Python DSL + MLIR compiler stack for AMD GPU kernels (ROCm/HIP via Fly/FlyROCDL dialects). Has its own detailed `flydsl/CLAUDE.md`; read it before touching anything inside `flydsl/`. | — |

`tilelang/` and `TileKernels/` are coupled (TileKernels imports tilelang), but `flydsl/` is an entirely separate stack. Don't cross-reference symbols between flydsl and the other two.

## tilelang/ — Compiler

### What it is
A Pythonic DSL that lowers tile-level kernels through TVM TIR to CUDA (NVRTC), HIP, CPU, Metal, and WebGPU. Vendored deps live in `tilelang/3rdparty/` (TVM, CUTLASS, Composable Kernel, HIP headers). Build is `scikit-build-core` + CMake; the package ships TVM + CUTLASS inside the wheel.

### Key directories
- `tilelang/tilelang/language/` — Public DSL surface (`T.Kernel`, `T.copy`, `T.gemm`, `T.Pipelined`, `T.Parallel`, `T.alloc_shared`, layout/fragment APIs). User-facing primitives live here.
- `tilelang/tilelang/jit/` — `@tilelang.jit` decorator, `JITKernel`, adapter shims (Cython wrapper at `jit/adapter/cython/cython_wrapper.pyx`).
- `tilelang/tilelang/transform/` (Python) + `tilelang/src/transform/` (C++) — TIR passes. The C++ side is where the heavy compiler work happens: `layout_inference.cc`, `inject_pipeline.cc`, `lower_tile_op.cc`, `loop_vectorize.cc`, `merge_shared_memory_allocations.cc`, etc.
- `tilelang/src/op/` — Tile-op C++ implementations: `gemm.cc`, `copy.cc`, `reduce.cc`, `parallel.cc`, `atomic_*.cc`, `gemm_sp.cc` (2:4 sparse).
- `tilelang/src/target/`, `src/backend/`, `src/tl_templates/` — Code generation; CUDA/HIP/CPU/Metal/WebGPU backends and the C++/CUDA templates injected into generated source.
- `tilelang/tilelang/autotuner/`, `tilelang/tilelang/carver/` — Search/tuning infrastructure.
- `tilelang/tilelang/layout/` — `Layout`, `Fragment`, swizzle helpers.
- `tilelang/testing/python/` — Test suites organized by subsystem (`language/`, `layout/`, `transform/`, `kernel/`, `cuda/`, `amd/`, `metal/`, `webgpu/`, `cpu/`, `autotune/`, …).
- `tilelang/examples/` — ~50 example kernels (gemm, flash_attention, deepseek_mla, deepseek_nsa, dequantize_gemm, fusedmoe, blocksparse_attention, …) — these double as integration tests via `examples/conftest.py`.

### Compilation flow (mental model)
`@tilelang.jit` traces the Python kernel into TVM TIR → Python passes in `tilelang/transform/` legalize/annotate → C++ passes in `src/transform/` do layout inference, pipelining, tile-op lowering, memory rewrites → target backend in `src/target/` emits CUDA/HIP/etc. source → NVRTC (or host nvcc/hipcc) compiles → `JITKernel` wraps the resulting binary with a torch-tensor calling convention via `tvm-ffi`. Layout decisions for shared memory and fragments are inferred, not hand-specified — `layout_inference.cc` is the central pass.

### Build & install (editable)
```bash
cd tilelang
uv venv --seed .venv && source .venv/bin/activate
uv pip install --requirements requirements-dev.txt
python3 -m pip install --no-build-isolation --verbose --editable .
pre-commit install --install-hooks
```
Building requires CMake ≥3.26 and the system deps listed in the README (`gcc libtinfo-dev zlib1g-dev build-essential libedit-dev libxml2-dev`). The first build is long because it compiles the vendored TVM submodule under `3rdparty/tvm`.

### Lint & format
```bash
bash format.sh            # ruff + clang-format on changed files (vs upstream main)
bash format.sh --all      # all files
pre-commit run --all-files
```
Style: ruff (line-length 140, `py310` target, configured in `pyproject.toml`), clang-format for C++/CUDA. `examples/**` and `testing/**` are exempted from `UP`/`FA` rules — do not "modernize" type hints there.

### Tests
```bash
python3 -m pytest testing                                  # full suite
python3 -m pytest testing/python/language -v               # one subsystem
python3 -m pytest testing/python/kernel/test_tilelang_kernel_gemm.py -v
python3 -m pytest -m "not perf and not slow"               # default exclusion of perf/slow
python3 -m pytest -m perf                                  # explicit benchmark runs
```
Pytest markers `perf` and `slow` are registered in `pyproject.toml` and skipped by default. Examples also run as tests via `examples/conftest.py`.

## TileKernels/ — Downstream kernels

### What it is
A standalone PyTorch package of optimized kernels for LLM workloads, authored with TileLang. Depends on `tilelang>=0.1.9` and `torch>=2.10`. Hardware target: NVIDIA SM90 (H100) / SM100 only — kernels are not portable to other archs.

### Layout
```
tile_kernels/
├── moe/        # Top-k gating, expert routing, fused expansion/reduction
├── quant/      # FP8/FP4/E5M6 cast + fused SwiGLU-quant
├── transpose/  # Batched transpose
├── engram/     # Engram gate forward/backward + fused RMSNorm, weight-grad reduction
├── mhc/        # Manifold HyperConnection (Sinkhorn norm, mix split/apply)
├── modeling/   # torch.autograd.Function wrappers composing low-level kernels into trainable layers
├── torch/      # Pure-PyTorch reference impls used as test oracles
└── testing/    # Shared test/benchmark utilities
tests/<group>/  # Mirrors tile_kernels/<group>/, one test file per kernel
```

Each kernel file under `tile_kernels/<group>/` has a counterpart in `tests/<group>/test_*.py` that compares against the matching reference in `tile_kernels/torch/`. New kernels follow the same pattern: implement, write a torch oracle, write a pytest checking close-ness, and (where relevant) a `--run-benchmark` path.

### Install
```bash
cd TileKernels
pip install -e ".[dev]"          # picks up tilelang from the installed env; build it from ../tilelang first if developing in tandem
```

### Tests
```bash
pytest tests/transpose/test_transpose.py -n 4                 # correctness only, 4 workers
pytest tests/transpose/test_transpose.py --run-benchmark      # correctness + benchmark
TK_FULL_TEST=1 pytest -n 4 --count 2                          # pressure / repeated runs (uses pytest-repeat)
```
Custom pytest plugins live at `tests/pytest_benchmark_plugin.py` (the `--run-benchmark` flag) and `tests/pytest_random_plugin.py`. Single-quoted strings only — `tool.ruff` enforces `Q000` with `inline-quotes = single`, line length 150.

## flydsl/ — Separate AMD MLIR DSL

Has its own deep `flydsl/CLAUDE.md` covering build (`scripts/build_llvm.sh`, `scripts/build.sh`), env vars (`FLYDSL_*`), test tiers, GPU arch matrix (`gfx942`/`gfx950`/`gfx1250`), and kernel-authoring conventions specific to that stack. **Always read that file first when working inside `flydsl/`.** Do not apply its conventions to `tilelang/` or `TileKernels/`.

## Cross-cutting notes

- Each repo has its own `.git`, `pyproject.toml`, lint config, and `pre-commit`. Run formatters/tests from inside the relevant subdir; running from the workspace root will silently miss the right `git rev-parse --show-toplevel`.
- `tilelang`'s wheel ships a vendored TVM (`tilelang/3rdparty/tvm`). When TileKernels behavior depends on a tilelang change, you must rebuild tilelang and reinstall — `pip install -e` on TileKernels alone does not rebuild the TVM/C++ side.
- All three projects target GPU code generation but have **disjoint** target stacks: tilelang→TVM→CUDA/HIP; TileKernels→tilelang (SM90/SM100 only); flydsl→custom MLIR→ROCDL (AMD only). A pattern that works in one stack is rarely transferable to another.
