import flydsl.compiler as flyc
import flydsl.expr as fx
BLOCK_X = 128
BLOCK_Y = 128
NUM_THREADS = 256
ELEMS_PER_THREAD = BLOCK_X * BLOCK_Y // NUM_THREADS
NUM_BATCHES = 4
SHAPE_X = 256
SHAPE_Y = 512

@flyc.kernel
def batched_transpose_kernel(x: fx.Tensor, out: fx.Tensor):
    pid_y = fx.block_idx.x
    pid_x = fx.block_idx.y
    pid_batch = fx.block_idx.z
    tid = fx.thread_idx.x
    x_g = fx.rocdl.make_buffer_tensor(x, max_size=False)
    out_g = fx.rocdl.make_buffer_tensor(out, max_size=False)
    for k_iter in fx.range_constexpr(ELEMS_PER_THREAD):
        idx = tid + k_iter * NUM_THREADS
        i = idx // BLOCK_X
        j = idx % BLOCK_X
        v = x_g[pid_batch, pid_x * BLOCK_X + j, pid_y * BLOCK_Y + i]
        out_g[pid_batch, pid_y * BLOCK_Y + i, pid_x * BLOCK_X + j] = v

@flyc.jit
def batched_transpose_kernel_launch(x: fx.Tensor, out: fx.Tensor, stream: fx.Stream=fx.Stream(None)):
    batched_transpose_kernel(x, out).launch(grid=(SHAPE_Y // BLOCK_Y, SHAPE_X // BLOCK_X, NUM_BATCHES), block=(NUM_THREADS, 1, 1), stream=stream)
