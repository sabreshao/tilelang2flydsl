"""Catalogue of recognised TileLang symbols, organised by handler.

The parser uses these sets to dispatch on calls like `T.copy(...)`.
Anything in `NVIDIA_ONLY` or `UNSUPPORTED_FOR_NOW` raises a hard error
naming the symbol and pointing to the source location.
"""

from __future__ import annotations

# Call-style constructs that map directly to a lowering handler.
SUPPORTED_CALLS: set[str] = {
    # Allocation
    "alloc_shared",
    "alloc_fragment",
    "alloc_local",
    "alloc_var",
    # Data movement / fills
    "copy",
    "clear",
    "fill",
    # GEMM
    "gemm",
    # Loops (used as `for x in T.foo(...)`)
    "Pipelined",
    "Parallel",
    "unroll",
    "Unroll",
    "vectorized",
    "Vectorized",
    "serial",
    "Serial",
    # Sync / scheduling
    "sync_threads",
    # Helpers / assertions
    "assume",
    "ceildiv",
    "get_thread_binding",
    # Types appearing on annotations
    "Tensor",
    "StridedTensor",
    "Fragment",
    # Symbolic shape declarations
    "const",
    "dynamic",
    # Kernel
    "Kernel",
    # dtype factory
    "dtype",
}

# Loop-shaped iterators (used as `for v in T.<name>(...)`)
LOOP_CALLS: set[str] = {
    "Pipelined",
    "Parallel",
    "unroll",
    "Unroll",
    "vectorized",
    "Vectorized",
    "serial",
    "Serial",
}

# dtype attributes on `T.` (e.g. `T.float16`).
DTYPES: dict[str, str] = {
    "float16": "Float16",
    "float32": "Float32",
    "bfloat16": "BFloat16",
    "float64": "Float64",
    "int8": "Int8",
    "uint8": "UInt8",
    "int16": "Int16",
    "uint16": "UInt16",
    "int32": "Int32",
    "uint32": "UInt32",
    "int64": "Int64",
    "uint64": "UInt64",
    "bool": "Bool",
}

# Annotations we silently drop with a comment.
DROPPABLE_ANNOTATIONS: set[str] = {
    "use_swizzle",
    "annotate_layout",
    "annotate_safe_value",
    "annotate_l2_hit_ratio",
    "annotate_min_blocks_per_sm",
    "annotate_restrict_buffers",
}

# Symbols we know are NVIDIA-only and have no AMD equivalent.
NVIDIA_ONLY: set[str] = {
    "tma_copy",
    "tma_load",
    "tma_store",
    "tma_store_arrive",
    "tma_gather4",
    "tma_scatter4",
    "create_tma_descriptor",
    "alloc_descriptor",
    "alloc_tmem",
    "wgmma_gemm",
    "tcgen05_gemm",
    "warpgroup_arrive",
    "warpgroup_commit_batch",
    "warpgroup_wait",
    "cluster_arrive",
    "cluster_wait",
    "cluster_sync",
    "block_rank_in_cluster",
    "copy_cluster",
    "mbarrier_wait_parity",
    "mbarrier_arrive",
    "mbarrier_expect_tx",
    "mbarrier_arrive_expect_tx",
    "ptx_arrive_cluster_barrier",
}

# Symbols within TileLang's surface that v1 doesn't yet handle but might later.
UNSUPPORTED_FOR_NOW: set[str] = {
    "reduce",
    "reduce_sum",
    "reduce_max",
    "reduce_min",
    "reduce_abssum",
    "reduce_absmax",
    "reduce_bitand",
    "reduce_bitor",
    "warp_reduce_sum",
    "warp_reduce_max",
    "warp_reduce_min",
    "warp_reduce_bitand",
    "warp_reduce_bitor",
    "cumsum",
    "atomic_add",
    "atomic_max",
    "atomic_min",
    "atomic_load",
    "atomic_store",
    "atomic_addx2",
    "atomic_addx4",
    "Persistent",
    "gemm_sp",
    "async_copy",
    "transpose",
    "im2col",
    "alloc_barrier",
    "barrier_arrive",
    "barrier_wait",
    "shfl_xor",
    "shfl_down",
    "shfl_up",
    "get_lane_idx",
    "get_warp_idx",
    "print",
    "device_assert",
}


def classify_symbol(name: str) -> str:
    """Return one of: 'supported', 'droppable', 'nvidia_only', 'unsupported', 'unknown'."""
    if name in SUPPORTED_CALLS:
        return "supported"
    if name in DROPPABLE_ANNOTATIONS:
        return "droppable"
    if name in NVIDIA_ONLY:
        return "nvidia_only"
    if name in UNSUPPORTED_FOR_NOW:
        return "unsupported"
    return "unknown"
