"""Retired single-stream tree attention sources. See README.md."""

from typing import Sequence
import mlx.core as mx
from tensorfold.kernels.qwen.dense.v1.lane_attention import (
    MAX_QUERIES, TILE, CHUNK, TILES_PER_GROUP, _partial, _HEADER,
)

_kernels = {}

def _kernel(name):
    if name not in _kernels:
        source, inputs, outputs, contiguous = {
            "tail": (_TAIL, ["QB", "K", "V", "scale", "dims", "paths", "depths", "POA", "PMA", "PLA"], ["PO", "PM", "PL"], False),
            "tree_merge": (_TREE_MERGE, ["POA", "PMA", "PLA", "POB", "PMB", "PLB", "dims"], ["OUT"], True),
        }[name]
        _kernels[name] = mx.fast.metal_kernel(name="legacy_tree_" + name, source=source, header=_HEADER,
                            input_names=inputs, output_names=outputs, ensure_row_contiguous=contiguous)
    return _kernels[name]

def lane_tree_sdpa(queries: mx.array, keys: mx.array, values: mx.array, scale: float,
                   parents: Sequence[int]) -> mx.array:
    """Attention for a draft tree whose W nodes are the cache's last W key rows.

    Node v (query row v) attends to the committed keys [0, P) and its own path, with
    every key at its logical position P + depth: the bits of serial decoding along
    that path. Chunks wholly inside [0, P) run through the shared kernel; the chunks
    holding the window run per node with the node's keys gathered into their slots.
    """

    from tensorfold.kernels.qwen.dense.v1.lane_tree import MAX_DEPTH, tree_paths

    _, H, W, D = (int(s) for s in queries.shape)
    HKV, L = int(keys.shape[1]), int(keys.shape[2])
    if D != 256 or H % HKV or W > MAX_QUERIES or len(parents) != W:
        raise ValueError(f"lane_tree_sdpa: unsupported shape q={queries.shape} k={keys.shape}")
    G = H // HKV
    P = L - W
    depths, paths = tree_paths(parents)
    PT = (P // TILE) * TILE                           # committed keys in whole tiles: the shared kernel's
    CA = -(-PT // CHUNK)                              # its chunks (the last may continue in the tail kernel)
    last = P + max(depths)                            # deepest logical key
    NCB = last // CHUNK - PT // CHUNK + 1             # chunks the tail kernel works in
    sc = mx.array([float(scale)], dtype=mx.float32)
    # shared part (all rows see every key of the committed chunks)
    R = G * W
    RP = 16 * ((R + 15) // 16)
    SGA = RP // 16
    SG = min(SGA, TILES_PER_GROUP)
    qA = queries.reshape(HKV, G, W, D).transpose(0, 2, 1, 3).reshape(HKV, R, D)
    if RP != R:
        qA = mx.concatenate([qA, mx.zeros((HKV, RP - R, D), dtype=queries.dtype)], axis=1)
    qA = mx.contiguous(qA)
    if CA > 0:
        dimsA = mx.array([PT, CA, W, 0, SGA], dtype=mx.int32)
        poA, pmA, plA = _partial(qA, keys, values, sc, dimsA, G=G, D=D, SG=SG, SGA=SGA, HKV=HKV, nch=CA, RP=RP)
    else:
        poA = pmA = plA = mx.zeros((1,), dtype=mx.float32)
    # per-node part: [HKV, W, 16 rows (G valid), D] queries
    qB = queries.reshape(HKV, G, W, D).transpose(0, 2, 1, 3)
    qB = mx.contiguous(mx.concatenate([qB, mx.zeros((HKV, W, 16 - G, D), dtype=queries.dtype)], axis=2))
    flat = [0] * (W * MAX_DEPTH)
    for node, path in enumerate(paths):
        flat[node * MAX_DEPTH: node * MAX_DEPTH + len(path)] = [row for row in path]
    dimsB = mx.array([L, P, PT, NCB, W, RP, CA], dtype=mx.int32)
    paths_mx = mx.array(flat, dtype=mx.int32)
    depths_mx = mx.array(depths, dtype=mx.int32)
    poB, pmB, plB = _kernel("tail")(
        inputs=[qB, keys, values, sc, dimsB, paths_mx, depths_mx, poA, pmA, plA],
        template=[("G", G), ("D", D), ("CK", CHUNK), ("TK", TILE), ("MAXD", MAX_DEPTH)],
        grid=(HKV * 32, NCB, W), threadgroup=(32, 1, 1),
        output_shapes=[(HKV * NCB * W * 16 * D,), (HKV * NCB * W * 16,), (HKV * NCB * W * 16,)],
        output_dtypes=[mx.float32, mx.float32, mx.float32])
    return _kernel("tree_merge")(
        inputs=[poA, pmA, plA, poB, pmB, plB, dimsB], template=[("G", G), ("D", D), ("CK", CHUNK)],
        grid=(HKV * 32, R, 1), threadgroup=(32, 1, 1),
        output_shapes=[(1, H, W, D)], output_dtypes=[mx.bfloat16])[0]



_TAIL = r"""
  const ushort lane = thread_index_in_simdgroup;
  const uint hk = threadgroup_position_in_grid.x;              // key head
  const uint cb = threadgroup_position_in_grid.y;              // tail chunk (from the first chunk holding the window)
  const uint node = threadgroup_position_in_grid.z;            // tree node
  const int P = dims[1], PT = dims[2], NCB = dims[3], W = dims[4], RPA = dims[5], CA = dims[6];
  const int depth = depths[node];
  const int nmax = P + depth + 1;                              // logical keys 0 .. P + depth
  const short qid = lane >> 2;
  const short fm = (qid & 4) | ((lane >> 1) & 3);
  const short fn = ((qid & 2) | (lane & 1)) * 4;
  const int r0 = fm, r1 = fm + 8;                              // query rows: the node's heads (G of 16)
  const int n0 = r0 < G ? nmax : 0;
  const int n1 = r1 < G ? nmax : 0;
  threadgroup half myP[16 * TK];
  threadgroup bfloat KV[32 * D];                               // 32 keys at a time, or TK keys' half rows of values
  const device bfloat* kbase = (const device bfloat*)K + (int64_t)hk * K_strides[1];
  const device bfloat* vbase = (const device bfloat*)V + (int64_t)hk * V_strides[1];
  const int64_t kstep = K_strides[2], vstep = V_strides[2];
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tQ((device bfloat*)QB + ((int64_t)hk * W + node) * 16 * D, dextents<int32_t, 2>(D, 16));
  tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline> tK32(KV, dextents<int32_t, 2>(D, 32));
  tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline> tVh(KV, dextents<int32_t, 2>(128, TK));
  tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tP(myP, dextents<int32_t, 2>(TK, 16));
  // scores 32 keys at a time: each score equals the TK-key op's bit for bit (tested), and 32 keys
  // of K fit the 16 KB buffer that TK keys would overflow
  constexpr auto dS = matmul2d_descriptor(16, 32, D, false, true, false, matmul2d_descriptor::mode::multiply);
  constexpr auto dO = matmul2d_descriptor(16, 128, TK, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<dS, execution_simdgroup> opS;
  matmul2d<dO, execution_simdgroup> opO;
  auto Olo = opO.template get_destination_cooperative_tensor<decltype(tP), decltype(tVh), float>();
  auto Ohi = opO.template get_destination_cooperative_tensor<decltype(tP), decltype(tVh), float>();
  for (int i = 0; i < 64; i++) { Olo[i] = 0.0f; Ohi[i] = 0.0f; }
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
  // keys [0, PT) went through the shared kernel; the first chunk holding PT continues from the
  // state it left there (same tiles, same order, same arithmetic: the bits of one pass)
  const int c0 = PT / CK;
  const int c = c0 + int(cb);
  const int kbeg = max(c * CK, PT);
  const int kend = min((c + 1) * CK, nmax);
  if (cb == 0 && PT > c0 * CK) {
    const int64_t baseA = ((int64_t)hk * CA + c0) * RPA + node * G;
    for (int q = 0; q < 16; q++) {
      const int row = fm + (q & 1) * 8;
      if (row >= G) continue;
      const auto src = POA + (baseA + row) * D + (q >> 1) * 16 + fn;   // a placeholder is tiny: constant space
      for (int j = 0; j < 4; j++) { Olo[4 * q + j] = src[j]; Ohi[4 * q + j] = src[128 + j]; }
    }
    if (r0 < G) { m0 = PMA[baseA + r0]; l0 = PLA[baseA + r0]; }
    if (r1 < G) { m1 = PMA[baseA + r1]; l1 = PLA[baseA + r1]; }
  }
  for (int kt = kbeg; kt < kend; kt += TK) {
    float sraw[TK / 2];
    for (int h = 0; h < TK / 32; h++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint e = lane; e < 32 * D / 8; e += 32) {           // logical slot -> physical row
        const int row = int(e) / (D / 8), col = (int(e) % (D / 8)) * 8;
        const int q = kt + h * 32 + row;
        int phys = -1;
        if (q < P) phys = q;
        else if (q < nmax) phys = P + paths[node * MAXD + (q - P)];
        ((threadgroup vec<bfloat, 8>*)KV)[e] = phys >= 0 ? *(const device vec<bfloat, 8>*)(kbase + phys * kstep + col) : vec<bfloat, 8>(0);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      auto S = opS.template get_destination_cooperative_tensor<decltype(tQ), decltype(tK32), float>();
      opS.run(tQ, tK32, S);
      for (int i = 0; i < 16; i++) sraw[h * 16 + i] = S[i];
    }
    float s[TK / 2];
    for (int i = 0; i < TK / 2; i++) {                         // 8 elements per 16-key block: 4 of row fm, then 4 of fm + 8
      const int key = kt + (i >> 3) * 16 + fn + (i & 3);
      s[i] = key < ((i & 4) ? n1 : n0) ? sraw[i] * scale[0] : -INFINITY;
    }
    float x0 = -INFINITY, x1 = -INFINITY;
    for (int i = 0; i < TK / 2; i++) { if (i & 4) x1 = max(x1, s[i]); else x0 = max(x0, s[i]); }
    x0 = max(x0, simd_shuffle_xor(x0, 1)); x0 = max(x0, simd_shuffle_xor(x0, 8));
    x1 = max(x1, simd_shuffle_xor(x1, 1)); x1 = max(x1, simd_shuffle_xor(x1, 8));
    const float nm0 = max(m0, x0), nm1 = max(m1, x1);
    const float f0 = (x0 == -INFINITY) ? 1.0f : fast::exp(m0 - nm0);
    const float f1 = (x1 == -INFINITY) ? 1.0f : fast::exp(m1 - nm1);
    float p[TK / 2];
    for (int i = 0; i < TK / 2; i++) p[i] = (s[i] == -INFINITY) ? 0.0f : fast::exp(s[i] - ((i & 4) ? nm1 : nm0));
    float y0 = 0.0f, y1 = 0.0f;
    for (int b = 0; b < TK / 16; b++) {
      y0 += (p[b * 8] + p[b * 8 + 1]) + (p[b * 8 + 2] + p[b * 8 + 3]);
      y1 += (p[b * 8 + 4] + p[b * 8 + 5]) + (p[b * 8 + 6] + p[b * 8 + 7]);
    }
    y0 += simd_shuffle_xor(y0, 1); y0 += simd_shuffle_xor(y0, 8);
    y1 += simd_shuffle_xor(y1, 1); y1 += simd_shuffle_xor(y1, 8);
    if (x0 != -INFINITY) { l0 = l0 * f0 + y0; m0 = nm0; }
    if (x1 != -INFINITY) { l1 = l1 * f1 + y1; m1 = nm1; }
    for (int f = 0; f < TK / 16; f++)
      for (int i = 0; i < 4; i++) {
        myP[fm * TK + f * 16 + fn + i] = half(p[f * 8 + i]);
        myP[(fm + 8) * TK + f * 16 + fn + i] = half(p[f * 8 + 4 + i]);
      }
    for (int i = 0; i < 64; i++) { const float f = (i & 4) ? f1 : f0; Olo[i] *= f; Ohi[i] *= f; }
    for (int hv = 0; hv < 2; hv++) {                           // values: TK keys x 128 columns at a time
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint e = lane; e < TK * 128 / 8; e += 32) {
        const int row = int(e) / 16, col = hv * 128 + (int(e) % 16) * 8;
        const int q = kt + row;
        int phys = -1;
        if (q < P) phys = q;
        else if (q < nmax) phys = P + paths[node * MAXD + (q - P)];
        ((threadgroup vec<bfloat, 8>*)KV)[e] = phys >= 0 ? *(const device vec<bfloat, 8>*)(vbase + phys * vstep + col) : vec<bfloat, 8>(0);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (hv == 0) opO.run(tP, tVh, Olo);
      else opO.run(tP, tVh, Ohi);
    }
  }
  const int64_t base = (((int64_t)hk * NCB + cb) * W + node) * 16;
  for (int q = 0; q < 16; q++) {
    device float* dst = PO + (base + fm + (q & 1) * 8) * D + (q >> 1) * 16 + fn;
    *(device float4*)dst = float4(Olo[4 * q], Olo[4 * q + 1], Olo[4 * q + 2], Olo[4 * q + 3]);
    *(device float4*)(dst + 128) = float4(Ohi[4 * q], Ohi[4 * q + 1], Ohi[4 * q + 2], Ohi[4 * q + 3]);
  }
  if ((lane & 9) == 0) {
    PM[base + r0] = m0; PL[base + r0] = l0;
    PM[base + r1] = m1; PL[base + r1] = l1;
  }
"""

_TREE_MERGE = r"""
  const uint lane = thread_index_in_simdgroup;
  const uint hk = threadgroup_position_in_grid.x;
  const uint r = threadgroup_position_in_grid.y;               // node * G + g
  const int PT = dims[2], NCB = dims[3], W = dims[4], RPA = dims[5], CA = dims[6];
  const int CT = PT / CK;                                      // chunks the shared kernel finished
  constexpr int DP = D / 32;
  const int node = r / G, g = r % G;
  float m = -INFINITY, l = 0.0f, o[DP];
  for (int i = 0; i < DP; i++) o[i] = 0.0f;
  for (int c = 0; c < CT; c++) {                               // committed chunks, in order
    const int64_t row = ((int64_t)hk * CA + c) * RPA + r;
    const float mc = PMA[row];
    if (mc == -INFINITY) continue;
    const float lc = PLA[row];
    const float nm = max(m, mc);
    const float f1 = fast::exp(m - nm), f2 = fast::exp(mc - nm);
    l = l * f1 + lc * f2;
    for (int i = 0; i < DP; i++) o[i] = o[i] * f1 + POA[row * D + lane * DP + i] * f2;
    m = nm;
  }
  for (int c = 0; c < NCB; c++) {                              // then the window's chunks
    const int64_t row = (((int64_t)hk * NCB + c) * W + node) * 16 + g;
    const float mc = PMB[row];
    if (mc == -INFINITY) continue;
    const float lc = PLB[row];
    const float nm = max(m, mc);
    const float f1 = fast::exp(m - nm), f2 = fast::exp(mc - nm);
    l = l * f1 + lc * f2;
    for (int i = 0; i < DP; i++) o[i] = o[i] * f1 + POB[row * D + lane * DP + i] * f2;
    m = nm;
  }
  const int h = hk * G + g;
  for (int i = 0; i < DP; i++) OUT[((int64_t)h * W + node) * D + lane * DP + i] = static_cast<bfloat>(o[i] / l);
"""
