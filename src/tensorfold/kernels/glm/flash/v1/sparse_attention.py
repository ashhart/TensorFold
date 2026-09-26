"""GLM-5.3-Flash sparse MLA decode rows: attention that reads the chosen latent keys straight from the cache.

Ported from mlx-vlm's ``indexed_sparse_attention`` (``mlx_vlm/models/sparse_attention.py``) as extended by mlx-vlm
PR #2245 ("Fix GLM-5.3 cached decode batch invariance", by raullenchai, validated on Vontra/GLM-5.3-Flash-MLX-4bit-MTP:
exact at 45/45 layers, all 154,880 logits), MIT License, Copyright (c) 2025 Prince Canuma. The kernel scores each
query's selected keys by index, with an fp32 online softmax and the value sum in the same pass, so no gathered
[keys, 512] copy is made (#2245: 8,192 cached / 2,048 selected, gather + SDPA 1.06 ms -> 0.73 ms). One threadgroup
per (row, head): a row's result does not depend on the other rows, so a verify window keeps one-row bits.

TensorFold's changes: the latent cache is one shared key/value tensor [capacity, 512] (NoPE MLA: keys are values),
read with the capacity as a runtime stride (#2245's PHYSICAL_LENGTH) so a growing cache does not compile a new
kernel every 256 positions; queries come as [R, H, 512] and outputs go out as [R, H, 512]; indices [R, TOPK] with -1
padding (TOPK = index_topk + kpool - 1: 2,048 chosen block keys plus up to 3 tail keys).
"""

from __future__ import annotations

import hashlib
from typing import Any

import mlx.core as mx

_SOURCE = r"""
  const uint gid = threadgroup_position_in_grid.y;          // row * HEADS + head
  const uint simd_gid = simdgroup_index_in_threadgroup;
  const uint simd_lid = thread_index_in_simdgroup;
  constexpr int SIMD_GROUPS = 32;
  constexpr int SIMD_WIDTH = 32;
  constexpr int qk_per_thread = QK_DIM / SIMD_WIDTH;
  constexpr int v_per_thread = QK_DIM / SIMD_WIDTH;
  typedef float U;
  thread U q[qk_per_thread];
  thread U o[v_per_thread];
  threadgroup U outputs[SIMD_GROUPS * SIMD_WIDTH];
  threadgroup U max_scores[SIMD_GROUPS];
  threadgroup U sum_exp_scores[SIMD_GROUPS];
  const int row = int(gid) / HEADS;
  const int key_length = int(meta[0]);
  const size_t stride = size_t(QK_DIM);
  const device bfloat* qptr = queries + size_t(gid) * QK_DIM + int(simd_lid) * qk_per_thread;
  device bfloat* optr = out + size_t(gid) * QK_DIM + int(simd_gid) * v_per_thread;
  const U s = U(scale[0]);
  for (int i = 0; i < qk_per_thread; i++) q[i] = s * static_cast<U>(qptr[i]);
  for (int i = 0; i < v_per_thread; i++) o[i] = 0;
  const int indices_offset = row * TOPK;
  U max_score = -3.4028234663852886e38f;
  U sum_exp_score = 0;
  for (int selected_idx = int(simd_gid); selected_idx < TOPK; selected_idx += SIMD_GROUPS) {
    const int key_pos = int(indices[indices_offset + selected_idx]);
    const bool valid = key_pos >= 0 && key_pos < key_length;
    U score = -3.4028234663852886e38f;
    if (valid) {
      const device bfloat* kptr = keys + size_t(key_pos) * stride + int(simd_lid) * qk_per_thread;
      score = 0;
      for (int j = 0; j < qk_per_thread; j++) score += q[j] * static_cast<U>(kptr[j]);
      score = simd_sum(score);
    }
    const U new_max = max(max_score, score);
    const U factor = fast::exp(max_score - new_max);
    const U exp_score = valid ? fast::exp(score - new_max) : U(0);
    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;
    if (valid) {
      const device bfloat* vptr = keys + size_t(key_pos) * stride + int(simd_lid) * v_per_thread;
      for (int j = 0; j < v_per_thread; j++) o[j] = o[j] * factor + exp_score * static_cast<U>(vptr[j]);
    } else {
      for (int j = 0; j < v_per_thread; j++) o[j] = o[j] * factor;
    }
  }
  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[simd_lid];
  const U new_max = simd_max(max_score);
  const U factor = fast::exp(max_score - new_max);
  const U total_sum = simd_sum(sum_exp_scores[simd_lid] * factor);
  for (int i = 0; i < v_per_thread; i++) {
    outputs[simd_lid * SIMD_WIDTH + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * SIMD_WIDTH + simd_lid] * factor);
    o[i] = total_sum == 0 ? U(0) : (o[i] / total_sum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (simd_lid == 0) {
    for (int i = 0; i < v_per_thread; i++) optr[i] = static_cast<bfloat>(o[i]);
  }
"""

_kernel_obj: dict[str, Any] = {}


def sources() -> dict[str, str]:
    return {"indexed_sparse_attention": _SOURCE}


def metal() -> bool:
    return mx.default_device() == mx.gpu and mx.metal.is_available()


def _kernel() -> Any:
    kernel = _kernel_obj.get("k")
    if kernel is None:
        digest = hashlib.sha256(_SOURCE.encode()).hexdigest()[:10]
        kernel = mx.fast.metal_kernel(name=f"tf_glm5_indexed_sparse_attention_{digest}",
                                      input_names=["queries", "keys", "indices", "scale", "meta"],
                                      output_names=["out"], source=_SOURCE)
        _kernel_obj["k"] = kernel
    return kernel


def indexed_attention(queries: mx.array, keys: mx.array, indices: mx.array, key_length: int,
                      scale: float) -> mx.array:
    """queries [R, H, 512] bf16 (latent queries), keys [capacity, 512] bf16 (the cache: keys are values),
    indices [R, TOPK] int32 (-1: none) -> [R, H, 512] bf16: each (row, head) attends over its indexed keys."""

    rows, heads, dim = queries.shape
    topk = int(indices.shape[-1])
    if not metal():
        return indexed_attention_ops(queries, keys, indices, key_length, scale)
    return _kernel()(inputs=[mx.contiguous(queries), keys, mx.contiguous(indices.astype(mx.int32)),
                             mx.array([scale], dtype=mx.float32), mx.array([int(key_length)], dtype=mx.int32)],
                     template=[("QK_DIM", int(dim)), ("TOPK", topk), ("HEADS", int(heads))],
                     grid=(1024, rows * heads, 1), threadgroup=(1024, 1, 1),
                     output_shapes=[(rows, heads, dim)], output_dtypes=[mx.bfloat16])[0]


def indexed_attention_ops(queries: mx.array, keys: mx.array, indices: mx.array, key_length: int,
                          scale: float) -> mx.array:
    """The same with MLX ops, one row at a time (Linux / CPU)."""

    outs = []
    for r in range(int(queries.shape[0])):
        idx = indices[r]
        valid = (idx >= 0) & (idx < key_length)
        k = mx.take(keys, mx.where(valid, idx, 0), axis=0).astype(mx.float32)            # [T, 512]
        s = (queries[r].astype(mx.float32) * scale) @ k.T                                 # [H, T]
        s = mx.where(valid[None], s, -mx.inf)
        p = mx.softmax(s, axis=-1)
        outs.append((p @ k).astype(mx.bfloat16)[None])
    return mx.concatenate(outs)
