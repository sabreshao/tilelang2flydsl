"""Runnable hand-port for tile_kernels/moe/reduce_fused_kernel.py.

Per-token weighted gather-and-sum:
    For each token t, for each k in 0..num_topk:
        pos = token_topk_to_pos[t, k]
        if pos >= 0:  # masked
            reduced[t, :] += x[pos, :] * weight(t, k)
    out[t, :] = reduced[t, :] * sf_global

Validation oracle: tile_kernels/torch/reduce_fused.py semantics, inlined here
so the test is self-contained.

Block layout: one block per token, NUM_THREADS=128 threads, each thread holds
ELEMS_PER_THREAD = hidden // NUM_THREADS register accumulators (striped:
thread t handles positions {t, t+NUM_THREADS, t+2*NUM_THREADS, ...}).

The TileLang `if pos >= 0` runtime guard is lowered as predication
(mask = 0 when invalid) — flydsl single-exit constraints make runtime `if`
with accumulator writes awkward; masking is equivalent and simpler.
"""
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

HIDDEN = 256
NUM_TOPK = 4
NUM_THREADS = 128
ELEMS_PER_THREAD = HIDDEN // NUM_THREADS  # 2

NUM_TOKENS = 16
NUM_EXPANDED_TOKENS = NUM_TOKENS * NUM_TOPK + 3  # some unused slots


@flyc.kernel
def reduce_fused_kernel(
    x: fx.Tensor,
    topk_weights: fx.Tensor,
    token_topk_to_pos: fx.Tensor,
    out: fx.Tensor,
    sf: fx.Tensor,
    x_sf: fx.Tensor,
):
    pid = fx.block_idx.x
    tid = fx.thread_idx.x

    x_g = fx.rocdl.make_buffer_tensor(x, max_size=False)
    tw_g = fx.rocdl.make_buffer_tensor(topk_weights, max_size=False)
    pos_g = fx.rocdl.make_buffer_tensor(token_topk_to_pos, max_size=False)
    out_g = fx.rocdl.make_buffer_tensor(out, max_size=False)
    sf_g = fx.rocdl.make_buffer_tensor(sf, max_size=False)
    xsf_g = fx.rocdl.make_buffer_tensor(x_sf, max_size=False)

    # Per-thread register accumulators (Python list — fully unrolled by tracing).
    acc = [fx.Float32(0.0) for _ in range(ELEMS_PER_THREAD)]

    sf_val = sf_g[0]

    for k in fx.range_constexpr(NUM_TOPK):
        pos = pos_g[pid, k]
        # Mask = 1 when pos >= 0, else 0. Index with clamped pos to avoid OOB.
        valid_f = (pos >= 0).select(fx.Float32(1.0), fx.Float32(0.0))
        safe_pos = (pos >= 0).select(pos, fx.Int32(0))

        w = tw_g[pid, k] * xsf_g[safe_pos] * valid_f

        for j in fx.range_constexpr(ELEMS_PER_THREAD):
            col = tid + j * NUM_THREADS
            acc[j] = acc[j] + x_g[safe_pos, col] * w

    for j in fx.range_constexpr(ELEMS_PER_THREAD):
        col = tid + j * NUM_THREADS
        out_g[pid, col] = acc[j] * sf_val


@flyc.jit
def reduce_fused(
    x: fx.Tensor,
    topk_weights: fx.Tensor,
    token_topk_to_pos: fx.Tensor,
    out: fx.Tensor,
    sf: fx.Tensor,
    x_sf: fx.Tensor,
    stream: fx.Stream = fx.Stream(None),
):
    reduce_fused_kernel(x, topk_weights, token_topk_to_pos, out, sf, x_sf).launch(
        grid=(NUM_TOKENS, 1, 1), block=(NUM_THREADS, 1, 1), stream=stream,
    )


def torch_reference(x, topk_weights, token_topk_to_pos, sf, x_sf):
    """Inlined oracle (matches tile_kernels/torch/reduce_fused.py semantics)."""
    num_tokens, num_topk = token_topk_to_pos.shape
    _, hidden = x.shape
    reduced = torch.zeros((num_tokens, hidden), dtype=torch.float32, device=x.device)
    valid = token_topk_to_pos >= 0
    for k in range(num_topk):
        pos_k = token_topk_to_pos[:, k]
        mask_k = valid[:, k]
        safe_pos = pos_k.clamp(min=0)
        rows = x[safe_pos].float()
        s = topk_weights[:, k].clone()
        s = s * x_sf[safe_pos]
        result = rows * s.unsqueeze(1) + reduced
        reduced = torch.where(mask_k.unsqueeze(1), result, reduced)
    reduced = reduced * sf[0].item()
    return reduced


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(NUM_EXPANDED_TOKENS, HIDDEN, dtype=torch.float32).cuda()
    topk_weights = torch.rand(NUM_TOKENS, NUM_TOPK, dtype=torch.float32).cuda()
    # Mostly-valid positions, with some -1 sentinels.
    token_topk_to_pos = torch.randint(0, NUM_EXPANDED_TOKENS, (NUM_TOKENS, NUM_TOPK), dtype=torch.int32).cuda()
    token_topk_to_pos[0, 1] = -1
    token_topk_to_pos[3, 2] = -1
    sf = torch.tensor([0.5], dtype=torch.float32).cuda()
    x_sf = torch.rand(NUM_EXPANDED_TOKENS, dtype=torch.float32).cuda()

    out = torch.zeros(NUM_TOKENS, HIDDEN, dtype=torch.float32).cuda()
    reduce_fused(x, topk_weights, token_topk_to_pos, out, sf, x_sf, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    expected = torch_reference(x, topk_weights, token_topk_to_pos, sf, x_sf)
    diff = (out - expected).abs().max().item()
    rtol_ok = torch.allclose(out, expected, atol=1e-4, rtol=1e-4)
    print(f"reduce_fused ok={rtol_ok}  max_diff={diff:.6f}")
