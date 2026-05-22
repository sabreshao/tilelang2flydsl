"""Runnable hand-port for tile_kernels/moe/topk_gate_kernel.py.

Algorithm (per token):
    1. Load scores[token, :] into per-thread register array (32 threads,
       num_aligned_experts // 32 elements each). Out-of-range positions get -inf.
    2. Repeat num_topk times:
       a. Wave-reduce-max across 32 threads to find the current largest score.
       b. Wave-reduce-min on indices where score == max (ties broken by
          smallest index — matches stable_topk).
       c. Thread 0 writes that index to topk_idx[token, k].
       d. Mask out that position (-inf) so the next iteration finds the next max.

FlyDSL specifics:
- NUM_THREADS=32. On CDNA3 the hardware wave is 64; we pass mask=32 to
  shuffle_xor so the reduction stays within the active half.
- Per-thread register array as a Python list (fully unrolled at trace time).
- Int min via ``(a <= b).select(a, b)`` (no direct .minimumi on ArithValue).
- Int32 → Int64 cast via ``arith.index_cast(T.i64, x)`` for the output write.

Validation: matches torch.sort(descending=True, stable=True)[:num_topk] semantics,
i.e. ties broken toward the smaller expert index — exactly what
tile_kernels/torch/topk.py::stable_topk computes.
"""
import math
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith
from flydsl.expr.typing import T


def _align(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


NUM_EXPERTS = 32
NUM_TOPK = 4
NUM_THREADS = 32
WAVE_MASK = 32  # shuffle_xor mask: confine to first 32 lanes of the wave
NUM_ALIGNED_EXPERTS = _align(NUM_EXPERTS, NUM_THREADS)
ELEMS_PER_THREAD = NUM_ALIGNED_EXPERTS // NUM_THREADS

NUM_TOKENS = 16
NEG_INF = float("-inf")
INT32_MAX_VAL = 2**31 - 1


@flyc.kernel
def topk_gate_kernel(scores: fx.Tensor, topk_idx: fx.Tensor):
    pid = fx.block_idx.x
    tid = fx.thread_idx.x

    scores_g = fx.rocdl.make_buffer_tensor(scores, max_size=False)
    topk_idx_g = fx.rocdl.make_buffer_tensor(topk_idx, max_size=False)

    neg_inf = fx.Float32(NEG_INF)
    int_max = fx.Int32(INT32_MAX_VAL)

    # Per-thread register arrays (Python lists — fully unrolled at trace time).
    scores_reg = [neg_inf for _ in range(ELEMS_PER_THREAD)]
    idx_reg = [fx.Int32(0) for _ in range(ELEMS_PER_THREAD)]

    # Load scores; mask out invalid lanes with -inf.
    for j in fx.range_constexpr(ELEMS_PER_THREAD):
        i = tid + j * NUM_THREADS
        in_range = i < NUM_EXPERTS
        safe_i = in_range.select(i, fx.Int32(0))
        v = scores_g[pid, safe_i]
        scores_reg[j] = in_range.select(v, neg_inf)
        idx_reg[j] = i

    for k in fx.range_constexpr(NUM_TOPK):
        # 1. Per-thread max of register slice.
        local_max = scores_reg[0]
        for j in fx.range_constexpr(1, ELEMS_PER_THREAD):
            local_max = local_max.maximumf(scores_reg[j])
        # 2. Wave-reduce max across NUM_THREADS=32 lanes.
        w = local_max
        for sh in fx.range_constexpr(int(math.log2(WAVE_MASK))):
            off = WAVE_MASK // (2 << sh)
            peer = w.shuffle_xor(off, WAVE_MASK)
            w = w.maximumf(peer)
        amax = w
        # 3. Per-thread min of indices matching amax.
        local_min = int_max
        for j in fx.range_constexpr(ELEMS_PER_THREAD):
            match = (scores_reg[j] == amax).select(idx_reg[j], int_max)
            local_min = (local_min <= match).select(local_min, match)
        # 4. Wave-reduce min across 32 lanes.
        m = local_min
        for sh in fx.range_constexpr(int(math.log2(WAVE_MASK))):
            off = WAVE_MASK // (2 << sh)
            peer = m.shuffle_xor(off, WAVE_MASK)
            m = (m <= peer).select(m, peer)
        min_idx = m
        # 5. Single-thread write (every lane has the same value, but only emit one store).
        if tid == 0:
            topk_idx_g[pid, k] = min_idx.extsi(T.i64)
        # 6. Mask the chosen position out for the next iteration.
        for j in fx.range_constexpr(ELEMS_PER_THREAD):
            scores_reg[j] = (idx_reg[j] == min_idx).select(neg_inf, scores_reg[j])


@flyc.jit
def topk_gate(scores: fx.Tensor, topk_idx: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    topk_gate_kernel(scores, topk_idx).launch(
        grid=(NUM_TOKENS, 1, 1), block=(NUM_THREADS, 1, 1), stream=stream,
    )


def torch_reference(scores: torch.Tensor, num_topk: int) -> torch.Tensor:
    """stable_topk: argsort descending, stable (ties → smaller index)."""
    _, sorted_idx = torch.sort(scores, dim=1, descending=True, stable=True)
    return sorted_idx[:, :num_topk].contiguous().to(torch.int64)


if __name__ == "__main__":
    torch.manual_seed(0)
    scores = torch.randn(NUM_TOKENS, NUM_EXPERTS, dtype=torch.float32).cuda()
    # Force some ties to exercise the smallest-index tie-breaker.
    scores[0, 3] = scores[0, 7] = 1.5
    scores[5, 1] = scores[5, 11] = 2.0

    topk_idx = torch.zeros(NUM_TOKENS, NUM_TOPK, dtype=torch.int64).cuda()
    topk_gate(scores, topk_idx, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    expected = torch_reference(scores, NUM_TOPK)
    ok = torch.equal(topk_idx, expected)
    print(f"topk_gate ok={ok}")
    if not ok:
        diff = (topk_idx != expected).any(dim=1).nonzero().flatten().tolist()
        print(f"  diff rows: {diff[:5]} of {NUM_TOKENS}")
        print(f"  emitted[0]:  {topk_idx[0].tolist()}")
        print(f"  expected[0]: {expected[0].tolist()}")
