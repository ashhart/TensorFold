"""Row-exact 4-bit matvecs use the same independent per-row arithmetic for serial decoding, verify windows and grouped experts."""

from __future__ import annotations

import hashlib
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from tensorfold.kernels import threads
from tensorfold.kernels.inputs import MIN_ELEMENTS, ints, padded

MAX_ROWS = 16
RPS = 4                 # output rows a simdgroup

_HEADER = r"""
// MLX's 4-bit qmv_fast inner loop (quantized.h: load_vector, qdot). A lane's 16 inputs, pre-divided by 1, 16, 256,
// 4096 so the masked nibbles need no shift, and their sum with each run of 4 summed in bf16 first (MLX's
// x[i] + x[i + 1] + x[i + 2] + x[i + 3] on bfloat16_t).
inline float tf_load16(const device bfloat* x, thread float* xt) {
  float sum = 0.0f;
  for (int i = 0; i < 16; i += 4) {
    const bfloat a = x[i], b = x[i + 1], c = x[i + 2], d = x[i + 3];
    sum += float(bfloat(float(bfloat(float(bfloat(float(a) + float(b))) + float(c))) + float(d)));
    xt[i] = float(a); xt[i + 1] = float(b) / 16.0f; xt[i + 2] = float(c) / 256.0f; xt[i + 3] = float(d) / 4096.0f;
  }
  return sum;
}
// the lane's 16 inputs times its 8 bytes of one weight row: scale * sum(x q) + bias * sum(x)
inline float tf_qdot16(const device uint8_t* w, const thread float* xt, float scale, float bias, float sum) {
  const device uint16_t* ws = (const device uint16_t*)w;
  float accum = 0.0f;
  for (int i = 0; i < 4; i++)
    accum += (xt[4 * i] * (ws[i] & 0x000f) + xt[4 * i + 1] * (ws[i] & 0x00f0) +
              xt[4 * i + 2] * (ws[i] & 0x0f00) + xt[4 * i + 3] * (ws[i] & 0xf000));
  return scale * accum + sum * bias;
}
// One input row x [K] times RPS weight rows (w: the first row's words, rows K / 2 bytes apart; sc, bi: its group
// scales and biases, rows K / GS apart), fp32 over the simdgroup. Lane l takes inputs 16 l .. 16 l + 15 of each
// 512-input step, then lanes below (K % 512) / 16 one more 16-input chunk; the lane sums add up in simd_sum.
// Reads nothing but its own row of x.
template <int K, int GS, int RPS>
inline void tf_rowdot(const device uint8_t* w, const device bfloat* sc, const device bfloat* bi,
                      const device bfloat* x, uint lane, thread float* acc) {
  constexpr int KB = K / 2;
  constexpr int KG = K / GS;
  constexpr int FULL = K / 512 * 512;
  w += lane * 8;
  sc += lane / (GS / 16);
  bi += lane / (GS / 16);
  x += lane * 16;
  for (int j = 0; j < RPS; j++) acc[j] = 0.0f;
  for (int k0 = 0; k0 < FULL; k0 += 512) {
    float xt[16];
    const float sum = tf_load16(x, xt);
    for (int j = 0; j < RPS; j++) acc[j] += tf_qdot16(w + j * KB, xt, float(sc[j * KG]), float(bi[j * KG]), sum);
    w += 256; sc += 512 / GS; bi += 512 / GS; x += 512;
  }
  if (FULL < K && int(lane) < (K - FULL) / 16) {
    float xt[16];
    const float sum = tf_load16(x, xt);
    for (int j = 0; j < RPS; j++) acc[j] += tf_qdot16(w + j * KB, xt, float(sc[j * KG]), float(bi[j * KG]), sum);
  }
  for (int j = 0; j < RPS; j++) acc[j] = simd_sum(acc[j]);
}
"""

HEADER = _HEADER             # the qmv loop, for kernels that must give rows.qmv's bits

_QMV = r"""
  // Threadgroup: one simdgroup per input row (x) for BLK blocks of RPS output rows (y). The row count is a launch
  // dimension only: every simdgroup runs the same instructions over its own row.
  const uint lane = thread_index_in_simdgroup;
  const int r = int(thread_position_in_threadgroup.x) / 32;
  const int row0 = int(thread_position_in_grid.y) * RPS;
  float acc[RPS];
  tf_rowdot<K, GS, RPS>((const device uint8_t*)W + size_t(row0) * (K / 2), S + size_t(row0) * (K / GS),
                        B + size_t(row0) * (K / GS), X + size_t(r) * K, lane, acc);
  if (lane == 0)
    for (int j = 0; j < RPS; j++) OUT[size_t(r) * N + row0 + j] = bfloat(acc[j]);
"""

# Each expert group walks its MEMBERS in order, preserving per-pair bits regardless of grouping; groups past UCOUNT[0] are empty.
_GROUP_HEAD = r"""
  const uint lane = thread_index_in_simdgroup;
  const int u = int(threadgroup_position_in_grid.z);
  if (u >= UCOUNT[0]) return;
  const size_t e = size_t(UIDS[u]);
  const int row0 = (int(threadgroup_position_in_grid.y) * SG + int(simdgroup_index_in_threadgroup)) * RPS;
  const size_t at = e * N + size_t(row0);
  const int first = START[u], last = START[u] + COUNT[u];
"""

_EXPERT_UP = _GROUP_HEAD + r"""
  // fc1 and mlx_lm's relu2 on its bf16 output: bf16(max(bf16(sum), 0)^2)
  #pragma clang loop unroll(disable)
  for (int m = first; m < last; m++) {
    const int p = MEMBERS[m];
    float acc[RPS];
    tf_rowdot<K, GS, RPS>((const device uint8_t*)W + at * (K / 2), S + at * (K / GS), B + at * (K / GS),
                          X + size_t(p / TOPK) * K, lane, acc);
    if (lane == 0)
      for (int j = 0; j < RPS; j++) {
        const float h = metal::max(float(bfloat(acc[j])), 0.0f);
        ACT[size_t(p) * N + row0 + j] = bfloat(h * h);
      }
  }
"""

_EXPERT_DOWN = _GROUP_HEAD + r"""
  // fc2 over each member pair's activation (bf16 out)
  #pragma clang loop unroll(disable)
  for (int m = first; m < last; m++) {
    const int p = MEMBERS[m];
    float acc[RPS];
    tf_rowdot<K, GS, RPS>((const device uint8_t*)W + at * (K / 2), S + at * (K / GS), B + at * (K / GS),
                          X + size_t(p) * K, lane, acc);
    if (lane == 0)
      for (int j = 0; j < RPS; j++) Y[size_t(p) * N + row0 + j] = bfloat(acc[j]);
  }
"""

# route and group in one launch: simdgroup s routes rows s, s + T / 32, ... exactly as the route kernel does (one
# simdgroup a row), keeping each row's K expert ids in threadgroup memory; then the group kernel's phase builds the
# tables from them (thread e counts the pairs of expert e).
_ROUTE_GROUP = r"""
  const uint t = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const int R = rows[0];
  const int P = R * K;
  threadgroup uint ids[MAXP];
  threadgroup int sg_pairs[T / 32], sg_used[T / 32];
  for (int r = int(sg); r < R; r += T / 32) {
    float sel[NE / 32], prob[NE / 32];
    for (int j = 0; j < NE / 32; j++) {
      const int e = int(lane) + 32 * j;
      const float g = float(G[r * NE + e]);
      prob[j] = 1.0f / (1.0f + metal::exp(-g));
      sel[j] = prob[j] + bias[e];
    }
    float total = 0.0f;
    float picked[K];
    for (int k = 0; k < K; k++) {
      float best = -INFINITY;
      int best_e = 1 << 20;
      for (int j = 0; j < NE / 32; j++) {
        const int e = int(lane) + 32 * j;
        if (sel[j] > best) { best = sel[j]; best_e = e; }
      }
      const float top = simd_max(best);
      const int winner = simd_min(best == top ? best_e : (1 << 20));   // ties: the lowest expert id
      float p = 0.0f;
      for (int j = 0; j < NE / 32; j++) {
        if (int(lane) + 32 * j == winner) { p = prob[j]; sel[j] = -INFINITY; }
      }
      p = simd_sum(p);
      picked[k] = p;
      total += p;
      if (lane == 0) { IDX[r * K + k] = uint(winner); ids[r * K + k] = uint(winner); }
    }
    if (lane == 0) {
      const float denominator = total + 1e-20f;
      for (int k = 0; k < K; k++) WT[r * K + k] = picked[k] / denominator * scaling[0];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int e = int(t);
  int count = 0;
  if (e < NE)
    for (int p = 0; p < P; p++) count += int(ids[p]) == e ? 1 : 0;
  const int used = count > 0 ? 1 : 0;
  const int pairs_before = simd_prefix_exclusive_sum(count);
  const int used_before = simd_prefix_exclusive_sum(used);
  if (lane == 31) { sg_pairs[sg] = pairs_before + count; sg_used[sg] = used_before + used; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int start = pairs_before, u = used_before;
  for (uint q = 0; q < sg; q++) { start += sg_pairs[q]; u += sg_used[q]; }
  if (used) {
    UIDS[u] = uint(e);
    START[u] = start;
    COUNT[u] = count;
    int m = start;
    for (int p = 0; p < P; p++)
      if (int(ids[p]) == e) MEMBERS[m++] = p;
  }
  if (int(t) == T - 1) UCOUNT[0] = u + used;
"""

_GROUP = r"""
  // One threadgroup of T >= E threads: thread e counts the pairs that picked expert e; the used experts, in
  // increasing id, get groups u = 0, 1, ...: UIDS[u] = e, START[u] / COUNT[u] = its run in MEMBERS, where its
  // pairs sit in increasing order; UCOUNT[0] = the number of groups.
  const uint t = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const int P = pairs[0];
  threadgroup uint ids[MAXP];
  threadgroup int sg_pairs[T / 32], sg_used[T / 32];
  for (int p = int(t); p < P; p += T) ids[p] = IDS[p];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int e = int(t);
  int count = 0;
  if (e < E)
    for (int p = 0; p < P; p++) count += int(ids[p]) == e ? 1 : 0;
  const int used = count > 0 ? 1 : 0;
  const int pairs_before = simd_prefix_exclusive_sum(count);
  const int used_before = simd_prefix_exclusive_sum(used);
  if (lane == 31) { sg_pairs[sg] = pairs_before + count; sg_used[sg] = used_before + used; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int start = pairs_before, u = used_before;
  for (uint q = 0; q < sg; q++) { start += sg_pairs[q]; u += sg_used[q]; }
  if (used) {
    UIDS[u] = uint(e);
    START[u] = start;
    COUNT[u] = count;
    int m = start;
    for (int p = 0; p < P; p++)
      if (int(ids[p]) == e) MEMBERS[m++] = p;
  }
  if (int(t) == T - 1) UCOUNT[0] = u + used;
"""

_kernels: dict[Any, Any] = {}


def _kernel(name: str, source: str, inputs: list[str], outputs: list[str], header: str = _HEADER) -> Any:
    kernel = _kernels.get(name)
    if kernel is None:
        digest = hashlib.sha256((header + source).encode()).hexdigest()[:16]
        kernel = mx.fast.metal_kernel(name=f"{name}_{digest}", input_names=inputs, output_names=outputs,
                                      source=source, header=header)
        _kernels[name] = kernel
    return kernel


def fits(weight: mx.array, scales: mx.array, group_size: int, bits: int, mode: str = "affine") -> bool:
    """Require affine 4-bit weights [..., N, K / 8], bf16 scales, groups of 32/64/128, K divisible by 64 and N divisible by 8."""

    return (bits == 4 and group_size in (32, 64, 128) and mode == "affine" and scales.dtype == mx.bfloat16
            and weight.dtype == mx.uint32 and (int(weight.shape[-1]) * 8) % 64 == 0 and int(weight.shape[-2]) % 8 == 0)


def qmv(x: mx.array, weight: mx.array, scales: mx.array, biases: mx.array, group_size: int) -> mx.array:
    """Multiply bf16 x [..., K] by 4-bit W.T [K, N], returning bf16 [..., N] with each row's bits independent of other rows."""

    shape = x.shape
    dims = int(shape[-1])
    x2 = x.reshape(-1, dims)
    rows = int(x2.shape[0])
    n = int(weight.shape[0])
    if rows > MAX_ROWS:
        # rows are independent: a longer input (several streams' windows) runs in calls of MAX_ROWS rows
        parts = [qmv(x2[i:i + MAX_ROWS], weight, scales, biases, group_size) for i in range(0, rows, MAX_ROWS)]
        return mx.concatenate(parts).reshape(*shape[:-1], n)
    if not 1 <= rows <= MAX_ROWS or n % (2 * RPS) or dims % 64:
        raise ValueError(f"rows.qmv: needs 1 to {MAX_ROWS} rows, N % {2 * RPS} == 0, K % 64 == 0 "
                         f"(R {rows}, N {n}, K {dims})")
    blocks = 2 if rows <= 8 else 1
    kernel = _kernels.get(("qmv", dims, n, int(group_size)))
    if kernel is None:
        consts = (("K", dims), ("N", n), ("GS", int(group_size)), ("RPS", RPS))
        source = "".join(f"  constexpr int {k} = {v};\n" for k, v in consts) + _QMV
        # up to MAX_ROWS rows in one threadgroup: the pipeline reserves them on every GPU
        kernel = _kernels[("qmv", dims, n, int(group_size))] = _kernel(
            f"nemotron_rows_qmv_{dims}_{n}_{group_size}", source, ["X", "W", "S", "B"], ["OUT"],
            _HEADER + threads.reserve(32 * MAX_ROWS))
    out = kernel(inputs=[x2, weight, scales, biases], grid=(32 * rows, n // RPS, 1), threadgroup=(32 * rows, blocks, 1),
                 output_shapes=[(rows, n)], output_dtypes=[mx.bfloat16])[0]
    return out.reshape(*shape[:-1], n)


# Grouping shares each expert's weight reads from this many rows while preserving the same per-pair bits.
GROUP_ROWS = 8
MAX_GROUP_PAIRS = 1024          # pairs one grouping pass takes (it keeps the ids in threadgroup memory)
_pair_tables: dict[int, tuple[mx.array, mx.array, mx.array, mx.array]] = {}
_pair_counts: dict[int, mx.array] = {}


def _pairs(count: int) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """A group for each pair: (START, COUNT, MEMBERS, UCOUNT) for ``count`` pairs, UIDS being the pairs' experts."""

    tables = _pair_tables.get(count)
    if tables is None:
        order = ints(range(count))
        tables = (order, ints([1] * count), order, mx.array([count], dtype=mx.int32))
        mx.eval(*tables)
        _pair_tables[count] = tables
    return tables


def group(ids: mx.array, experts: int) -> tuple[mx.array, ...]:
    """Return (UIDS, START, COUNT, MEMBERS, UCOUNT) from uint32 expert ids [P], ordering groups by expert id and each group's pairs by input order."""

    pairs = int(ids.shape[0])
    if pairs > MAX_GROUP_PAIRS:
        raise ValueError(f"rows.group: at most {MAX_GROUP_PAIRS} pairs")
    threads = max(32, -(-experts // 32) * 32)
    count = _pair_counts.get(pairs)
    if count is None:
        count = _pair_counts[pairs] = mx.array([pairs], dtype=mx.int32)
    kernel = _kernel("nemotron_rows_expert_group", _GROUP, ["IDS", "pairs"], ["UIDS", "START", "COUNT", "MEMBERS",
                                                                            "UCOUNT"])
    size = max(pairs, MIN_ELEMENTS)                 # the tables are the expert kernels' inputs
    return tuple(kernel(inputs=[padded(ids), count], template=[("E", experts), ("T", threads),
                                                               ("MAXP", MAX_GROUP_PAIRS)],
                        grid=(threads, 1, 1), threadgroup=(threads, 1, 1),
                        output_shapes=[(size,), (size,), (size,), (size,), (1,)],
                        output_dtypes=[mx.uint32, mx.int32, mx.int32, mx.int32, mx.int32]))


ROUTE_GROUP = True              # grouped windows route and group in one launch
ROUTE_THREADS = 512


def route_group(logits: mx.array, bias: mx.array, top_k: int, scaling: mx.array
                ) -> tuple[mx.array, mx.array, tuple[mx.array, ...]]:
    """The route kernel's ids [R, K] and weights [R, K] plus the group kernel's tables, from one launch (R * K <= MAX_GROUP_PAIRS)."""

    rows, experts_count = int(logits.shape[0]), int(logits.shape[1])
    pairs = rows * int(top_k)
    if pairs > MAX_GROUP_PAIRS or experts_count % 32 or experts_count > ROUTE_THREADS:
        raise ValueError(f"rows.route_group: at most {MAX_GROUP_PAIRS} pairs and {ROUTE_THREADS} experts (a multiple of 32)")
    count = _row_counts.get(rows)
    if count is None:
        count = _row_counts[rows] = mx.array([rows], dtype=mx.int32)
    # constants in the source, not template arguments: the thread reservation attaches only to a plain kernel
    consts = (("NE", experts_count), ("K", int(top_k)), ("T", ROUTE_THREADS), ("MAXP", MAX_GROUP_PAIRS))
    source = "".join(f"  constexpr int {k} = {v};\n" for k, v in consts) + _ROUTE_GROUP
    kernel = _kernel(f"nemotron_rows_route_group_{experts_count}_{int(top_k)}", source,
                     ["G", "bias", "scaling", "rows"], ["IDX", "WT", "UIDS", "START", "COUNT", "MEMBERS", "UCOUNT"],
                     _HEADER + threads.reserve(ROUTE_THREADS))
    size = max(pairs, MIN_ELEMENTS)
    out = kernel(inputs=[logits, bias, scaling, count],
                 grid=(ROUTE_THREADS, 1, 1), threadgroup=(ROUTE_THREADS, 1, 1),
                 output_shapes=[(rows, int(top_k)), (rows, int(top_k)), (size,), (size,), (size,), (size,), (1,)],
                 output_dtypes=[mx.uint32, mx.float32, mx.uint32, mx.int32, mx.int32, mx.int32, mx.int32])
    return out[0], out[1], tuple(out[2:])


def experts(table: Any, x: mx.array, indices: mx.array, *, simdgroups: int = 2, grouped: bool | None = None,
            tables: tuple[mx.array, ...] | None = None) -> mx.array:
    """Run SwitchMLP ``table`` for ``indices`` [R, k] on bf16 x [R, D], returning bf16 [R, k, D] with identical per-pair bits whether grouped or alone."""

    fc1, fc2 = table.fc1, table.fc2
    rows, dims = int(x.shape[0]), int(x.shape[-1])
    top_k = int(indices.shape[-1])
    count = int(fc1["weight"].shape[0])
    hidden, out = int(fc1["weight"].shape[1]), int(fc2["weight"].shape[1])
    block = RPS * simdgroups
    if hidden % block or out % block or dims % 64 or hidden % 64:
        raise ValueError(f"rows.experts: needs widths a multiple of {block} and 64")
    ids = indices.reshape(-1)
    if ids.dtype != mx.uint32:
        ids = ids.astype(mx.uint32)
    pairs = rows * top_k
    uids, start, counts, members, used, groups = _grouping(ids, rows, count, grouped, tables)
    inputs = ["X", "UIDS", "START", "COUNT", "MEMBERS", "UCOUNT", "W", "S", "B"]
    up = _kernel("nemotron_rows_expert_up", _EXPERT_UP, inputs, ["ACT"])
    act = up(inputs=[x.reshape(rows, dims), uids, start, counts, members, used, fc1["weight"], fc1["scales"],
                     fc1["biases"]],
             template=[("K", dims), ("N", hidden), ("GS", int(fc1.group_size)), ("RPS", RPS), ("SG", simdgroups),
                       ("TOPK", top_k)],
             grid=(32 * simdgroups, hidden // block, groups), threadgroup=(32 * simdgroups, 1, 1),
             output_shapes=[(pairs, hidden)], output_dtypes=[mx.bfloat16])[0]
    down = _kernel("nemotron_rows_expert_down", _EXPERT_DOWN, inputs, ["Y"])
    y = down(inputs=[act, uids, start, counts, members, used, fc2["weight"], fc2["scales"], fc2["biases"]],
             template=[("K", hidden), ("N", out), ("GS", int(fc2.group_size)), ("RPS", RPS), ("SG", simdgroups)],
             grid=(32 * simdgroups, out // block, groups), threadgroup=(32 * simdgroups, 1, 1),
             output_shapes=[(pairs, out)], output_dtypes=[mx.bfloat16])[0]
    return y.reshape(rows, top_k, out)


_row_counts: dict[int, mx.array] = {}


def _grouping(ids: mx.array, rows: int, count: int, grouped: bool | None,
              tables: tuple[mx.array, ...] | None = None) -> tuple:
    """(UIDS, START, COUNT, MEMBERS, UCOUNT, groups) for the pairs' ids: by expert (``tables`` when route_group made them), or a group a pair."""

    pairs = int(ids.shape[0])
    if grouped is None:
        grouped = rows >= GROUP_ROWS
    if grouped and pairs <= MAX_GROUP_PAIRS:
        uids, start, counts, members, used = tables if tables is not None else group(ids, count)
        return uids, start, counts, members, used, min(pairs, count)
    (start, counts, members, used), uids = _pairs(pairs), padded(ids)
    return uids, start, counts, members, used, pairs


class RowLinear(nn.QuantizedLinear):
    """A 4-bit linear: calls of up to ``qmv_rows`` rows (a shared round's) run ``qmv``, longer ones MLX's kernel."""

    qmv_rows = MAX_ROWS             # ``install`` raises it to the model's ``batch_rows``

    def __call__(self, x: mx.array) -> mx.array:
        rows = x.size // x.shape[-1]
        if (rows > self.qmv_rows or x.dtype != mx.bfloat16 or (rows == 1 and getattr(self, "mlx_one_row", False))):
            return super().__call__(x)
        y = qmv(x, self["weight"], self["scales"], self["biases"], self.group_size)
        if "bias" in self:
            y = y + self["bias"]
        return y


def matches_mlx(linear: Any, *, seed: int = 0, trials: int = 64) -> bool:
    """Check bit equality with MLX over ``trials`` random rows; only K divisible by 512 shares its one-row loop, and sparse bf16 differences need many trials."""

    k = int(linear["weight"].shape[1]) * 8
    if k % 512:
        return False
    x = (mx.random.normal((trials, k), key=mx.random.key(seed)) * 0.5).astype(mx.bfloat16)
    ours = mx.concatenate([qmv(x[i:i + MAX_ROWS], linear["weight"], linear["scales"], linear["biases"],
                               linear.group_size) for i in range(0, trials, MAX_ROWS)])
    theirs = mx.concatenate([mx.quantized_matmul(x[i:i + 1], linear["weight"], scales=linear["scales"],
                                                 biases=linear["biases"], transpose=True,
                                                 group_size=linear.group_size, bits=linear.bits)
                             for i in range(trials)])
    return bool(mx.array_equal(ours, theirs).item())


def linears(nemotron: Any) -> list[Any]:
    """The 4-bit linears a Nemotron-H decode step calls (``NemotronH``: its model and ``fused`` decode)."""

    found = []
    fused = nemotron.fused
    for i, layer in enumerate(nemotron.model.layers):
        mixer = layer.mixer
        if layer.block_type == "M":
            found += [mixer.in_proj, mixer.out_proj]
        elif layer.block_type == "*":
            found += [fused.qkv[i][0]] if i in fused.qkv else [mixer.q_proj, mixer.k_proj, mixer.v_proj]
            found.append(mixer.o_proj)
        elif layer.block_type == "E" and getattr(mixer, "shared_experts", None) is not None:
            found += [mixer.shared_experts.up_proj, mixer.shared_experts.down_proj]
    found.append(nemotron.model.lm_head)
    return found


def install(nemotron: Any, *, mlx_one_row: bool = False) -> dict[str, int]:
    """Install and compile row-exact linears and experts, allowing MLX for one row only when ``mlx_one_row`` and ``matches_mlx`` both hold; return counts."""

    covered = mlx_rows = 0
    first: dict[tuple[int, int, int], Any] = {}
    for linear in linears(nemotron):
        mode = getattr(linear, "mode", "affine")
        if not (isinstance(linear, nn.QuantizedLinear)
                and fits(linear["weight"], linear["scales"], linear.group_size, linear.bits, mode)):
            continue
        linear.__class__ = RowLinear
        object.__setattr__(linear, "qmv_rows", int(nemotron.batch_rows))   # MLX's kernel would change a row's bits
        covered += 1
        first.setdefault((int(linear["weight"].shape[0]), int(linear["weight"].shape[1]), int(linear.group_size)),
                         linear)
        same = bool(mlx_one_row) and matches_mlx(linear)
        object.__setattr__(linear, "mlx_one_row", same)
        mlx_rows += int(same)
    tables = [layer.mixer.switch_mlp for layer in nemotron.model.layers if layer.block_type == "E"]
    for table in tables:
        for fc in (table.fc1, table.fc2):
            if not fits(fc["weight"], fc["scales"], fc.group_size, fc.bits, getattr(fc, "mode", "affine")):
                raise ValueError("rows.install: an expert table does not have the layout the kernels read")
    # compile every variant now, not inside the load-time window check
    warm = [qmv(mx.zeros((2, int(m["weight"].shape[1]) * 8), dtype=mx.bfloat16), m["weight"], m["scales"],
                m["biases"], m.group_size) for m in first.values()]
    if tables:
        dims = int(tables[0].fc1["weight"].shape[-1]) * 8
        ids = mx.zeros((2, int(nemotron.args.num_experts_per_tok)), dtype=mx.uint32)
        x = mx.zeros((2, dims), dtype=mx.bfloat16)
        warm.append(experts(tables[0], x, ids))
        experts_count = int(tables[0].fc1["weight"].shape[0])
        if experts_count % 32 == 0 and experts_count <= ROUTE_THREADS:
            logits = mx.zeros((2, experts_count), dtype=mx.bfloat16)
            idx, wt, made = route_group(logits, mx.zeros((experts_count,), dtype=mx.float32),
                                        int(nemotron.args.num_experts_per_tok), mx.ones((1,), dtype=mx.float32))
            warm += [idx, wt, *made]
    mx.eval(warm)
    return {"linears": covered, "mlx_one_row": mlx_rows, "shapes": len(first), "expert_tables": len(tables)}


__all__ = ["MAX_ROWS", "RowLinear", "experts", "fits", "install", "linears", "matches_mlx", "qmv", "route_group"]
