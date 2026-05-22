"""Runnable hand-port for tests/fixtures/batched_transpose_kernel.py.

Simplified: direct gmem→gmem element-wise transpose, no LDS staging or swizzle.
The TileLang fixture LDS-stages + swizzles for coalesced writes; this v1 drops
that optimization in favor of correctness simplicity. Validates against
torch.permute on MI300X.

Each block handles one (pid_batch, pid_x_tile, pid_y_tile) of size (BX, BY).
Each thread handles ELEMS_PER_THREAD elements of the tile.
"""
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

BLOCK_X = 128
BLOCK_Y = 128
NUM_THREADS = 256
ELEMS_PER_THREAD = (BLOCK_X * BLOCK_Y) // NUM_THREADS  # 64

NUM_BATCHES = 4
SHAPE_X = 256
SHAPE_Y = 512


@flyc.kernel
def transpose_kernel(x: fx.Tensor, out: fx.Tensor):
    pid_y = fx.block_idx.x
    pid_x = fx.block_idx.y
    pid_b = fx.block_idx.z
    tid = fx.thread_idx.x

    x_g = fx.rocdl.make_buffer_tensor(x, max_size=False)
    out_g = fx.rocdl.make_buffer_tensor(out, max_size=False)

    for k in fx.range_constexpr(ELEMS_PER_THREAD):
        idx = tid + k * NUM_THREADS  # linear index in (BY, BX) tile
        i = idx // BLOCK_X  # row within tile (along Y in output)
        j = idx % BLOCK_X   # col within tile (along X in output)
        # x[batch, pid_x*BX + j, pid_y*BY + i] → out[batch, pid_y*BY + i, pid_x*BX + j]
        v = x_g[pid_b, pid_x * BLOCK_X + j, pid_y * BLOCK_Y + i]
        out_g[pid_b, pid_y * BLOCK_Y + i, pid_x * BLOCK_X + j] = v


@flyc.jit
def transpose(x: fx.Tensor, out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    transpose_kernel(x, out).launch(
        grid=(SHAPE_Y // BLOCK_Y, SHAPE_X // BLOCK_X, NUM_BATCHES),
        block=(NUM_THREADS, 1, 1),
        stream=stream,
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(NUM_BATCHES, SHAPE_X, SHAPE_Y, dtype=torch.float16).cuda()
    out = torch.zeros(NUM_BATCHES, SHAPE_Y, SHAPE_X, dtype=torch.float16).cuda()
    transpose(x, out, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    expected = x.permute(0, 2, 1).contiguous()
    diff = (out.float() - expected.float()).abs().max().item()
    ok = torch.equal(out, expected)
    print(f"transpose ok={ok}  max_diff={diff}")
