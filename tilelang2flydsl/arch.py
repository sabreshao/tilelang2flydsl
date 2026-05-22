"""gfx942 (CDNA3) MFMA / copy parameter selection.

These tables are intentionally small. They cover the cases needed by the v1
GEMM and transpose translations; new combinations should be added explicitly
rather than guessed at.
"""

from __future__ import annotations


# (A_dtype, B_dtype, accum_dtype) -> (M, N, K, fly_input_dtype_name)
MFMA_SHAPES_GFX942: dict[tuple[str, str, str], tuple[int, int, int, str]] = {
    ("Float16", "Float16", "Float32"): (16, 16, 16, "Float16"),
    ("Float16", "Float16", "Float16"): (16, 16, 16, "Float16"),
    ("BFloat16", "BFloat16", "Float32"): (16, 16, 16, "BFloat16"),
    ("Float32", "Float32", "Float32"): (16, 16, 4, "Float32"),
    ("Int8", "Int8", "Int32"): (16, 16, 32, "Int8"),
}


def pick_mfma(a_dtype: str, b_dtype: str, c_dtype: str) -> tuple[int, int, int, str]:
    key = (a_dtype, b_dtype, c_dtype)
    if key in MFMA_SHAPES_GFX942:
        return MFMA_SHAPES_GFX942[key]
    # Default to fp16 fp16 -> fp32 with a warning; caller decides if that's OK.
    return MFMA_SHAPES_GFX942[("Float16", "Float16", "Float32")]


# Default TV (thread-value) layout for a default `make_tiled_copy` over a
# `(block_rows, block_cols)` shared tile with `threads` total threads. We just
# spread threads contiguously over the inner dim. Good enough for v1.
def default_copy_tv_layout(threads: int, block_rows: int, block_cols: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return ((threads_per_row, threads_per_col), (vals_per_thread_row, vals_per_thread_col)).

    Picks a row-major thread mapping that covers the tile when multiplied with
    the per-thread vector length. This is a placeholder; production code will
    want arch-tuned variants.
    """
    if threads >= block_cols:
        threads_per_row = block_cols
        threads_per_col = max(threads // block_cols, 1)
    else:
        threads_per_row = threads
        threads_per_col = 1
    vals_per_thread_row = max(block_rows // threads_per_col, 1)
    vals_per_thread_col = max(block_cols // threads_per_row, 1)
    return (threads_per_row, threads_per_col), (vals_per_thread_row, vals_per_thread_col)


# Copy atom width selected by element byte size. Used for the BufferCopy* atom.
def buffer_copy_atom_name(elem_bytes: int) -> str:
    # 128b (16B) is the largest reliable vector load on CDNA3. 8B / 4B for smaller dtypes.
    if elem_bytes >= 2:
        return "BufferCopy128b"
    if elem_bytes == 1:
        return "BufferCopy64b"
    return "BufferCopy32b"
