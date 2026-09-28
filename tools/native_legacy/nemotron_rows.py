"""Row-exact 4-bit matvecs for Nemotron-H on Macs without the M5's tensor units (M1 to M4).

MLX 0.32 sums a row of a quantized matmul one way when it rides alone (``qmv``) and another way from 2 rows on
(``qmv_wide`` on gen-15 GPUs and later), so on an M3 Ultra no verify window reproduced one-row decoding. Here the
decode path's 4-bit projections (Mamba in/out, the stacked q/k/v, o_proj, the shared expert, the head) and the
routed experts run through kernels in which each input row is its own simdgroup running one row's loop. Every row
executes the same instructions whatever the row count, and reads nothing of the other rows, so its bits cannot
depend on the rows beside it. Serial decoding goes through the same kernels: they define the reference drafted
rounds reproduce.

    qmv        x [R, K] @ W.T, 4-bit weights in groups of 32, 64 or 128, R <= 16, K a multiple of 64: lane l takes
               inputs 16 l .. 16 l + 15 of each 512-input step (MLX's qmv_fast loop), then, when K % 512 != 0,
               lanes below (K % 512) / 16 take one more 16-input chunk
    experts    mlx_lm's SwitchMLP (fc1, relu squared, fc2) for R <= 16 rows of top-k slots: each (row, slot) pair
               is one simdgroup per 4 output rows running qmv's loop over its expert's rows; bf16 where mlx_lm
               stores bf16 (fc1's output, relu squared, fc2's output)

M3 Ultra, MLX 0.32.0, 2k context, 2026-09-26: windows of 2 to 16 rows reproduce one-row steps (MLX's kernels:
none). Wall ms of a forward (hidden + head) at 1 / 2 / 3 / 4 / 8 / 16 rows: 4.8 / 6.3 / 7.9 / 9.5 / 15.9 / 27.9,
MLX's 5.0 / 6.5 / 7.9 / 9.4 / 14.7 / 25.0. The experts beat MLX's gather at every width (its loop for K = 2,688
and 1,856 reads 8 inputs a lane); the dense projections lose from 3 rows on (0.45 ms at 3, 3.1 at 16), where
MLX's qmv_wide unpacks each weight once for up to 5 rows and ``qmv`` once a row.
"""

from __future__ import annotations

import hashlib
from typing import Any

import mlx.core as mx
import mlx.nn as nn

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

# Routed experts, one (row, slot) pair a threadgroup column: pair p is row p / TOPK's slot p % TOPK and uses expert
# IDS[p]; simdgroup g of threadgroup (b, p) takes the expert's output rows RPS (SG b + g) .. + RPS - 1.
_EXPERT_UP = r"""
  // fc1 and mlx_lm's relu2 on its bf16 output: bf16(max(bf16(sum), 0)^2)
  const uint lane = thread_index_in_simdgroup;
  const int p = int(threadgroup_position_in_grid.z);
  const size_t e = size_t(IDS[p]);
  const int row0 = (int(threadgroup_position_in_grid.y) * SG + int(simdgroup_index_in_threadgroup)) * RPS;
  const size_t at = e * N + size_t(row0);
  float acc[RPS];
  tf_rowdot<K, GS, RPS>((const device uint8_t*)W + at * (K / 2), S + at * (K / GS), B + at * (K / GS),
                        X + size_t(p / TOPK) * K, lane, acc);
  if (lane == 0)
    for (int j = 0; j < RPS; j++) {
      const float h = metal::max(float(bfloat(acc[j])), 0.0f);
      ACT[size_t(p) * N + row0 + j] = bfloat(h * h);
    }
"""

_EXPERT_DOWN = r"""
  // fc2 over the pair's activation (bf16 out)
  const uint lane = thread_index_in_simdgroup;
  const int p = int(threadgroup_position_in_grid.z);
  const size_t e = size_t(IDS[p]);
  const int row0 = (int(threadgroup_position_in_grid.y) * SG + int(simdgroup_index_in_threadgroup)) * RPS;
  const size_t at = e * N + size_t(row0);
  float acc[RPS];
  tf_rowdot<K, GS, RPS>((const device uint8_t*)W + at * (K / 2), S + at * (K / GS), B + at * (K / GS),
                        X + size_t(p) * K, lane, acc);
  if (lane == 0)
    for (int j = 0; j < RPS; j++) Y[size_t(p) * N + row0 + j] = bfloat(acc[j]);
"""

_kernels: dict[str, Any] = {}


def _kernel(name: str, source: str, inputs: list[str], outputs: list[str]) -> Any:
    kernel = _kernels.get(name)
    if kernel is None:
        digest = hashlib.sha256((_HEADER + source).encode()).hexdigest()[:16]
        kernel = mx.fast.metal_kernel(name=f"{name}_{digest}", input_names=inputs, output_names=outputs,
                                      source=source, header=_HEADER)
        _kernels[name] = kernel
    return kernel


def fits(weight: mx.array, scales: mx.array, group_size: int, bits: int, mode: str = "affine") -> bool:
    """Whether a 4-bit matrix [..., N, K / 8] has the layout the kernels read: groups of 32, 64 or 128, bf16 scales,
    K a multiple of 64, N a multiple of 8."""

    return (bits == 4 and group_size in (32, 64, 128) and mode == "affine" and scales.dtype == mx.bfloat16
            and weight.dtype == mx.uint32 and (int(weight.shape[-1]) * 8) % 64 == 0 and int(weight.shape[-2]) % 8 == 0)


def qmv(x: mx.array, weight: mx.array, scales: mx.array, biases: mx.array, group_size: int) -> mx.array:
    """x [..., K] bf16 (1 to MAX_ROWS rows) @ W.T for 4-bit weights [N, K / 8] -> [..., N] bf16; a row's bits do not
    depend on the other rows or on how many there are."""

    shape = x.shape
    dims = int(shape[-1])
    x2 = x.reshape(-1, dims)
    rows = int(x2.shape[0])
    n = int(weight.shape[0])
    if not 1 <= rows <= MAX_ROWS or n % (2 * RPS) or dims % 64:
        raise ValueError(f"rows.qmv: needs 1 to {MAX_ROWS} rows, N % {2 * RPS} == 0, K % 64 == 0 "
                         f"(R {rows}, N {n}, K {dims})")
    blocks = 2 if rows <= 8 else 1
    kernel = _kernel("nemotron_rows_qmv", _QMV, ["X", "W", "S", "B"], ["OUT"])
    out = kernel(inputs=[x2, weight, scales, biases],
                 template=[("K", dims), ("N", n), ("GS", int(group_size)), ("RPS", RPS)],
                 grid=(32 * rows, n // RPS, 1), threadgroup=(32 * rows, blocks, 1),
                 output_shapes=[(rows, n)], output_dtypes=[mx.bfloat16])[0]
    return out.reshape(*shape[:-1], n)


def experts(table: Any, x: mx.array, indices: mx.array, *, simdgroups: int = 2) -> mx.array:
    """mlx_lm's SwitchMLP ``table`` (fc1, relu2, fc2) on x [R, D] bf16 for each row's experts ``indices`` [R, k]:
    [R, k, D] bf16. Each (row, slot) pair is computed on its own, the same way at any R (R <= MAX_ROWS)."""

    fc1, fc2 = table.fc1, table.fc2
    rows, dims = int(x.shape[0]), int(x.shape[-1])
    top_k = int(indices.shape[-1])
    hidden, out = int(fc1["weight"].shape[1]), int(fc2["weight"].shape[1])
    block = RPS * simdgroups
    if rows > MAX_ROWS or hidden % block or out % block or dims % 64 or hidden % 64:
        raise ValueError(f"rows.experts: needs at most {MAX_ROWS} rows and widths a multiple of {block} and 64")
    ids = indices.reshape(-1)
    if ids.dtype != mx.uint32:
        ids = ids.astype(mx.uint32)
    pairs = rows * top_k
    up = _kernel("nemotron_rows_expert_up", _EXPERT_UP, ["X", "IDS", "W", "S", "B"], ["ACT"])
    act = up(inputs=[x.reshape(rows, dims), ids, fc1["weight"], fc1["scales"], fc1["biases"]],
             template=[("K", dims), ("N", hidden), ("GS", int(fc1.group_size)), ("RPS", RPS), ("SG", simdgroups),
                       ("TOPK", top_k)],
             grid=(32 * simdgroups, hidden // block, pairs), threadgroup=(32 * simdgroups, 1, 1),
             output_shapes=[(pairs, hidden)], output_dtypes=[mx.bfloat16])[0]
    down = _kernel("nemotron_rows_expert_down", _EXPERT_DOWN, ["X", "IDS", "W", "S", "B"], ["Y"])
    y = down(inputs=[act, ids, fc2["weight"], fc2["scales"], fc2["biases"]],
             template=[("K", hidden), ("N", out), ("GS", int(fc2.group_size)), ("RPS", RPS), ("SG", simdgroups)],
             grid=(32 * simdgroups, out // block, pairs), threadgroup=(32 * simdgroups, 1, 1),
             output_shapes=[(pairs, out)], output_dtypes=[mx.bfloat16])[0]
    return y.reshape(rows, top_k, out)


class RowLinear(nn.QuantizedLinear):
    """A 4-bit linear whose calls of 1 to MAX_ROWS rows run ``qmv`` (one-row calls MLX's kernel when
    ``mlx_one_row`` is set: its bits equal ``qmv``'s there); longer inputs (prompts) MLX's quantized matmul."""

    def __call__(self, x: mx.array) -> mx.array:
        rows = x.size // x.shape[-1]
        if (rows > MAX_ROWS or x.dtype != mx.bfloat16 or (rows == 1 and getattr(self, "mlx_one_row", False))):
            return super().__call__(x)
        y = qmv(x, self["weight"], self["scales"], self["biases"], self.group_size)
        if "bias" in self:
            y = y + self["bias"]
        return y


def matches_mlx(linear: Any, *, seed: int = 0, trials: int = 64) -> bool:
    """Whether ``qmv`` on one row equals MLX's one-row quantized matmul bit for bit for this weight, over ``trials``
    random rows. Only a K that is a multiple of 512 can: there MLX's one-row kernel runs the same loop
    (``qmv_fast``); for other K it runs 8 inputs a lane in 256-input steps and sums in another order. A bf16
    output rarely shows an fp32 difference (a few in 10^5 outputs flip), so a few rows prove nothing: 2026-09-26
    on the M3 Ultra, 16 of Nemotron's non-512 projections passed a 3-row check and then broke drafted windows."""

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
    """Route a ``NemotronH``'s decode step through the row-exact kernels: its 4-bit linears become ``RowLinear``,
    and its fused decode's routed experts run ``experts``. One-row calls (serial decoding) go through ``qmv`` too,
    unless ``mlx_one_row`` and ``matches_mlx``. Every kernel variant is compiled here. Returns counts."""

    covered = mlx_rows = 0
    first: dict[tuple[int, int, int], Any] = {}
    for linear in linears(nemotron):
        mode = getattr(linear, "mode", "affine")
        if not (isinstance(linear, nn.QuantizedLinear)
                and fits(linear["weight"], linear["scales"], linear.group_size, linear.bits, mode)):
            continue
        linear.__class__ = RowLinear
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
    nemotron.fused.experts_fn = experts
    # compile every variant now, not inside the load-time window check
    warm = [qmv(mx.zeros((2, int(m["weight"].shape[1]) * 8), dtype=mx.bfloat16), m["weight"], m["scales"],
                m["biases"], m.group_size) for m in first.values()]
    if tables:
        dims = int(tables[0].fc1["weight"].shape[-1]) * 8
        ids = mx.zeros((2, int(nemotron.args.num_experts_per_tok)), dtype=mx.uint32)
        warm.append(experts(tables[0], mx.zeros((2, dims), dtype=mx.bfloat16), ids))
    mx.eval(warm)
    return {"linears": covered, "mlx_one_row": mlx_rows, "shapes": len(first), "expert_tables": len(tables)}


__all__ = ["MAX_ROWS", "RowLinear", "experts", "fits", "install", "linears", "matches_mlx", "qmv"]
