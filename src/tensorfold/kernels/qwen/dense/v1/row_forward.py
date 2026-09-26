"""The lane decoder for GPUs without tensor units (M1 to M4): a row-exact matmul and the lane glue.

On an M3 Ultra a one-row step of Qwen3.8-27B through mlx_lm's layers runs ~1,590 kernels: 497 matvecs and ~1,090
small ones (norms, residual adds, the recurrent layers' conv, gates and scales), each 3-5 us when the next waits on
it. That glue costs 4.3 ms of a 24.8 ms step (26 Sep 2026). This forward runs the M5 lane decoder's structure
(``lane_tree.tree_forward``) without tensor units, 10 kernels a recurrent layer instead of 26:

    add_norm    residual add + RMSNorm
    in          [qkv | z | b | a] in one matmul (one stacked weight; the members become views of it)
    gdn_pre     conv + SiLU + q/k norms and scales + g + beta, reading ``in``'s rows in place (and the conv tail)
    recurrence  ``lane_tree``'s (mlx_lm's step arithmetic), which for a chain also writes the state after its last
                row
    gdn_post    gated RMSNorm, reading z in place
    out         out_proj
    add_norm, [gate | up] in one matmul, SiLU(gate) * up, down

The matmul is pluggable (``BACKEND``): ``row_qmv`` (MLX's one-row qmv loop run per row; its one-row bits are MLX's
where ``row_qmv.matches_mlx`` says so) or any kernel with the same call and the property that a row's bits never
depend on the rows beside it. Every row of every call is then a function of its own inputs only, so a verify
window's rows reproduce one-row steps bit for bit. The glue is lane_glue's arithmetic (it differs from mlx_lm's
ops by 1-5 bf16 ulps), so serial decoding, verify windows and prompt prefill all go through this forward: it is the
reference drafted rounds reproduce, as ``tree_forward`` is on the M5. A window kept whole (every one-row round) is
committed from what its recurrence wrote; a partly kept one is replayed (``commit``).

Attention runs query by query through MLX's one-query kernel over exactly the keys the serial step at that position
sees (``exact_attention``), so windows are chains. ``TF_ROW_ATTENTION=1`` switches to ``row_attention`` (each row
attends to the committed keys and its own path in chunks fixed by absolute position), which takes draft trees.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, Callable, Sequence

import mlx.core as mx


class Backend:
    """A row-exact 4-bit matmul: ``qmm(x, weight, scales, biases, group_size)`` for up to ``max_rows`` rows.

    ``gate_up_act(x, weight, scales, biases, group_size)`` (optional): SiLU(gate) * up straight from a stacked
    [gate; up] weight, with the bits of ``qmm`` followed by ``mlp_act``.

    ``variant(x, weight, scales, biases, group_size, *, norm=None, epilogue="plain", res=None)`` (optional): the
    matmul with the glue around it folded in. ``norm=(parts, weight, eps)``: ``x`` is the residual stream and each
    row is RMSNorm-ed on load, its scale from ``parts`` (the row's partial sums of squares, added in a fixed order);
    ``epilogue="act"``: SiLU(gate) * up of a stacked [gate; up] weight; ``epilogue="residual"``: returns
    (h, parts) with h = res + x @ W.T and the partial sums of squares of h's rows (partial t over outputs
    [8 t, 8 t + 8)). With it, the forward runs no norm kernels after the first layer's."""

    def __init__(self, name: str, qmm: Callable[..., mx.array], max_rows: int, fits: Callable[[Any], bool],
                 one_row: Callable[..., mx.array] | None = None,
                 gate_up_act: Callable[..., mx.array] | None = None,
                 variant: Callable[..., Any] | None = None,
                 prepare: Callable[[list[tuple[mx.array, mx.array, mx.array]]], None] | None = None) -> None:
        self.name, self.qmm, self.max_rows, self.fits, self.one_row = name, qmm, int(max_rows), fits, one_row
        self.gate_up_act = gate_up_act
        self.variant = variant
        # called once with every (weight, scales, biases) the forward multiplies by, stacks included
        self.prepare = prepare

    def __call__(self, x: mx.array, weight: mx.array, scales: mx.array, biases: mx.array, group_size: int) -> mx.array:
        rows = x.size // int(x.shape[-1])
        if rows == 1 and self.one_row is not None:
            return self.one_row(x, weight, scales, biases, group_size)
        return self.qmm(x, weight, scales, biases, group_size)


def row_qmv_backend() -> Backend:
    """``row_qmv`` for 1-8 rows; a one-row call through MLX's own kernel where its bits are row_qmv's."""

    from tensorfold.kernels.qwen.dense.v1 import row_qmv

    def mlx_one_row(x: mx.array, w: mx.array, s: mx.array, b: mx.array, gs: int) -> mx.array:
        return mx.quantized_matmul(x, w, s, b, transpose=True, group_size=gs, bits=4)

    return Backend("row_qmv", row_qmv.qmv, row_qmv.MAX_ROWS, row_qmv.fits,
                   one_row=mlx_one_row if row_qmv.mlx_one_row else None, gate_up_act=row_qmv_gate_up_act,
                   variant=row_qmv_variant if FOLD_NORMS else None)


# Attention through ``row_attention`` (exact for chains and draft trees; one-row steps use it too, so it is the
# reference), which lets DFlash2 draft trees; off: MLX's one-query kernel query by query (``exact_attention``,
# chains only). Off by default: on an M3 Ultra (26 Sep 2026) trees with it lost 10-19% against chains
# (62.7 / 46.1 / 61.0 / 46.4 against 69.3 / 56.7 / 69.7 / 57.0 tok/s): the kernel runs 1.4-1.9x MLX's, the
# tree recurrence's 16-slot variant is slower, and at 2-4 rows a tree drafts fewer right tokens than a chain.
ROW_ATTENTION = os.environ.get("TF_ROW_ATTENTION", "0") == "1"

# The norms folded into row_qmv (``Backend.variant``). Off: on an M3 Ultra (26 Sep 2026) the norm on load cost more
# than the 128 norm kernels it removes, because every simdgroup normalizes the inputs it reads again (one-row
# round 23.18 -> 23.66 ms, 4-row window 44.1 -> 50.3 ms). A matmul that stages each input block once per
# threadgroup can take it (TF_ROW_FOLD=1 to A/B).
FOLD_NORMS = os.environ.get("TF_ROW_FOLD", "0") == "1"

_LOAD16N = r"""
// row_qmv's load16 of the RMSNorm-ed input x = bf16(w * (h * inv)): the same pre-scaling and bf16 sum
inline float load16n(const device bfloat* h, const device bfloat* nw, float inv, thread float* xt) {
  float sum = 0.0f;
  for (int i = 0; i < 16; i += 4) {
    const bfloat a = bfloat(float(nw[i]) * (float(h[i]) * inv));
    const bfloat b = bfloat(float(nw[i + 1]) * (float(h[i + 1]) * inv));
    const bfloat c = bfloat(float(nw[i + 2]) * (float(h[i + 2]) * inv));
    const bfloat d = bfloat(float(nw[i + 3]) * (float(h[i + 3]) * inv));
    sum += float(bfloat(float(bfloat(float(bfloat(float(a) + float(b))) + float(c))) + float(d)));
    xt[i] = float(a); xt[i + 1] = float(b) / 16.0f; xt[i + 2] = float(c) / 256.0f; xt[i + 3] = float(d) / 4096.0f;
  }
  return sum;
}
"""

_FINAL_LOOP = """  for (int r = 0; r < R; r++)
    for (int j = 0; j < RPS; j++) {
      const float v = simd_sum(acc[r][j]);
      if (lane == 0) OUT[r * N + row0 + j] = bfloat(v);
    }
"""


def _variant_source(norm: bool, epilogue: str) -> str:
    """row_qmv's source with the norm on load and/or an epilogue; each output row's arithmetic is row_qmv's."""

    from tensorfold.kernels.qwen.dense.v1 import row_qmv
    from tensorfold.kernels.qwen.dense.v1.lane_fuse import _replace_once

    src = row_qmv._SOURCE
    if epilogue == "act":
        src = _gate_up_act_source()
    elif epilogue == "residual":
        src = _replace_once(src, _FINAL_LOOP, """  // h = res + y (bf16, as mlx_lm's residual add) and the partial sum of squares of this threadgroup's outputs
  threadgroup float part[SG][R];
  for (int r = 0; r < R; r++) {
    float ss = 0.0f;
    for (int j = 0; j < RPS; j++) {
      const float v = simd_sum(acc[r][j]);
      const bfloat h = bfloat(float(RES[r * N + row0 + j]) + float(bfloat(v)));
      if (lane == 0) OUT[r * N + row0 + j] = h;
      ss = fma(float(h), float(h), ss);
    }
    if (lane == 0) part[g][r] = ss;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (g == 0 && lane == 0)
    for (int r = 0; r < R; r++) {
      float total = 0.0f;
      for (int s2 = 0; s2 < SG; s2++) total += part[s2][r];
      PO[r * (N / (SG * RPS)) + int(threadgroup_position_in_grid.y)] = total;
    }
""")
    elif epilogue != "plain":
        raise ValueError(f"unknown epilogue {epilogue!r}")
    # one row: MLX's qmv_fast order (the input first, then each weight row's words, scale and bias with its qdot);
    # the same arithmetic as the rows-inner loop below it, which several rows take
    loop = src[src.index("  for (int k0 = 0; k0 < K; k0 += 512) {"):src.index("    w += 256; sc += 512 / GS; bi += 512 / GS;\n  }\n")]
    loop += "    w += 256; sc += 512 / GS; bi += 512 / GS;\n  }\n"
    src = _replace_once(src, loop, """  if (R == 1) {
    for (int k0 = 0; k0 < K; k0 += 512) {
      float xt[16];
      const float sum = load16(X + k0 + lane * 16, xt);
      for (int j = 0; j < RPS; j++) {
        const device uint16_t* wp = (const device uint16_t*)(w + j * KB);
        uint16_t ws1[4];
        for (int i = 0; i < 4; i++) ws1[i] = wp[i];
        acc[0][j] += qdot16w(ws1, xt, float(sc[j * KG]), float(bi[j * KG]), sum);
      }
      w += 256; sc += 512 / GS; bi += 512 / GS;
    }
  } else {
""" + loop + "  }\n")
    if norm:
        src = _replace_once(src, "  if (R == 1) {\n", """  // each input row's RMSNorm scale from the T partial sums of squares its producer wrote, added in a fixed order
  float inv[R];
  for (int r = 0; r < R; r++) {
    float part_sum = 0.0f;
    for (int t = int(lane); t < T; t += 32) part_sum += PART[r * T + t];
    inv[r] = metal::rsqrt(simd_sum(part_sum) / float(K) + eps[0]);
  }
  if (R == 1) {
""")
        src = _replace_once(src, "const float sum = load16(X + r * K + k0 + lane * 16, xt);",
                            "const float sum = load16n(X + r * K + k0 + lane * 16, NW + k0 + lane * 16, inv[r], xt);")
        src = _replace_once(src, "const float sum = load16(X + k0 + lane * 16, xt);",
                            "const float sum = load16n(X + k0 + lane * 16, NW + k0 + lane * 16, inv[0], xt);")
    return src


_variants: dict[tuple[bool, str], Any] = {}


def _variant_kernel(norm: bool, epilogue: str) -> Any:
    key = (norm, epilogue)
    if key not in _variants:
        from tensorfold.kernels.qwen.dense.v1.row_qmv import _HEADER

        source = _variant_source(norm, epilogue)
        header = _HEADER + (_LOAD16N if norm else "")
        inputs = ["X", "W", "S", "B"] + (["NW", "PART", "eps"] if norm else []) + (["RES"] if epilogue == "residual"
                                                                                   else [])
        outputs = ["OUT"] + (["PO"] if epilogue == "residual" else [])
        name = f"row_qmv_{int(norm)}{epilogue}_" + hashlib.sha256((header + source).encode()).hexdigest()[:16]
        _variants[key] = mx.fast.metal_kernel(name=name, input_names=inputs, output_names=outputs, source=source,
                                              header=header)
    return _variants[key]


def row_qmv_variant(x: mx.array, weight: mx.array, scales: mx.array, biases: mx.array, group_size: int, *,
                    norm: tuple[mx.array, mx.array, float] | None = None, epilogue: str = "plain",
                    res: mx.array | None = None) -> Any:
    """``Backend.variant`` for row_qmv (see there)."""

    from tensorfold.kernels.qwen.dense.v1 import row_qmv

    shape = x.shape
    K = int(shape[-1])
    x2 = x.reshape(-1, K)
    rows = int(x2.shape[0])
    N = int(weight.shape[0])
    template = [("K", K), ("N", N), ("R", rows), ("GS", int(group_size)), ("RPS", row_qmv.RPS), ("SG", row_qmv.SG)]
    inputs = [x2, weight, scales, biases]
    if norm is not None:
        parts, nw, eps = norm
        T = int(parts.shape[-1])
        inputs += [nw, parts.reshape(rows, T), _eps(eps)]
        template.append(("T", T))
    if epilogue == "act":
        nh = N // 2
        template.append(("NH", nh))
        return _variant_kernel(norm is not None, epilogue)(
            inputs=inputs, template=template, grid=(64, nh // row_qmv.RPS, 1), threadgroup=(64, 1, 1),
            output_shapes=[(rows, nh)], output_dtypes=[mx.bfloat16])[0].reshape(*shape[:-1], nh)
    if epilogue == "residual":
        inputs.append(res.reshape(rows, N))
        h, parts_out = _variant_kernel(norm is not None, epilogue)(
            inputs=inputs, template=template, grid=(64, N // (row_qmv.SG * row_qmv.RPS), 1), threadgroup=(64, 1, 1),
            output_shapes=[(rows, N), (rows, N // (row_qmv.SG * row_qmv.RPS))],
            output_dtypes=[mx.bfloat16, mx.float32])
        return h.reshape(*shape[:-1], N), parts_out.reshape(*shape[:-1], -1)
    return _variant_kernel(norm is not None, epilogue)(
        inputs=inputs, template=template, grid=(64, N // (row_qmv.SG * row_qmv.RPS), 1), threadgroup=(64, 1, 1),
        output_shapes=[(rows, N)], output_dtypes=[mx.bfloat16])[0].reshape(*shape[:-1], N)


# row_qmv with SiLU(gate) * up as its epilogue: simdgroup 0 of a threadgroup computes RPS gate rows, simdgroup 1 the
# same up rows (NH rows later in the stacked [gate; up] weight), each exactly as row_qmv computes a row; the up
# values reach simdgroup 0 through threadgroup memory. One kernel instead of the matmul and ``mlp_act``.
def _gate_up_act_source() -> str:
    from tensorfold.kernels.qwen.dense.v1 import row_qmv
    from tensorfold.kernels.qwen.dense.v1.lane_fuse import _replace_once

    src = _replace_once(row_qmv._SOURCE,
                        "const int row0 = int(threadgroup_position_in_grid.y) * (SG * RPS) + int(g) * RPS;",
                        "const int row0 = int(g) * NH + int(threadgroup_position_in_grid.y) * RPS;")
    return _replace_once(src, """  for (int r = 0; r < R; r++)
    for (int j = 0; j < RPS; j++) {
      const float v = simd_sum(acc[r][j]);
      if (lane == 0) OUT[r * N + row0 + j] = bfloat(v);
    }
""", """  threadgroup float ups[R][RPS];
  for (int r = 0; r < R; r++)
    for (int j = 0; j < RPS; j++) {
      const float v = simd_sum(acc[r][j]);
      if (lane == 0 && g == 1) ups[r][j] = float(bfloat(v));
      acc[r][j] = v;
    }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (g == 0 && lane == 0)
    for (int r = 0; r < R; r++)
      for (int j = 0; j < RPS; j++) {
        const float gf = float(bfloat(acc[r][j]));
        OUT[r * NH + row0 + j] = bfloat(gf / (1.0f + metal::exp(-gf)) * ups[r][j]);
      }
""")


_gate_up_kernel: Any = None


def row_qmv_gate_up_act(x: mx.array, weight: mx.array, scales: mx.array, biases: mx.array, group_size: int
                        ) -> mx.array:
    """SiLU(gate) * up for x [..., K] (up to row_qmv.MAX_ROWS rows) and a stacked [gate; up] weight [2 NH, K / 8]."""

    global _gate_up_kernel
    from tensorfold.kernels.qwen.dense.v1 import row_qmv

    if _gate_up_kernel is None:
        from tensorfold.kernels.qwen.dense.v1.row_qmv import _HEADER

        source = _gate_up_act_source()
        name = "row_qmv_gate_up_act_" + hashlib.sha256((_HEADER + source).encode()).hexdigest()[:16]
        _gate_up_kernel = mx.fast.metal_kernel(name=name, input_names=["X", "W", "S", "B"], output_names=["OUT"],
                                               source=source, header=_HEADER)
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    rows, dims = int(x2.shape[0]), int(x2.shape[1])
    nh = int(weight.shape[0]) // 2
    if nh % row_qmv.RPS or row_qmv.SG != 2:
        raise ValueError("gate_up_act: needs whole blocks of RPS rows and two simdgroups a threadgroup")
    out = _gate_up_kernel(inputs=[x2, weight, scales, biases],
                          template=[("K", dims), ("N", 2 * nh), ("NH", nh), ("R", rows), ("GS", int(group_size)),
                                    ("RPS", row_qmv.RPS), ("SG", 2)],
                          grid=(64, nh // row_qmv.RPS, 1), threadgroup=(64, 1, 1),
                          output_shapes=[(rows, nh)], output_dtypes=[mx.bfloat16])[0]
    return out.reshape(*shape[:-1], nh)


def simd_qmm_backend() -> Backend | None:
    """The row-exact simdgroup-matrix matmul (``simd_qmm``: 1-16 rows, every row the bits of a one-row call), when
    the package has it. Its one-row calls are its own (they define the reference), never MLX's."""

    try:
        from tensorfold.kernels.qwen.dense.v1 import simd_qmm
    except ImportError:
        return None
    def prepare(weights: list[tuple[mx.array, mx.array, mx.array]]) -> None:
        # per shape, one-row calls take the MMA kernel where its scalar kernel's bits are not the MMA rows'
        seen: set[tuple[int, int]] = set()
        for w, s, b in weights:
            shape = (int(w.shape[0]), int(w.shape[1]) * 8)
            if shape not in seen:
                seen.add(shape)
                if not simd_qmm.check(w, s, b):
                    simd_qmm.mma_one_row.add(shape)

    # windows and prompt chains of up to 16 rows (the rows attention takes query by query)
    return Backend("simd_qmm", simd_qmm.qmm, min(int(simd_qmm.MAX_ROWS), 16), simd_qmm.fits, prepare=prepare)


def choose_backend() -> Backend:
    """``TF_ROW_MATMUL``: "simd_qmm" (default: the simdgroup matmul, windows of 2-8 rows at about one row's cost
    on an M3 Ultra) or "row_qmv" (MLX's one-row loop per row). Falls back to row_qmv where simd_qmm is missing."""

    if os.environ.get("TF_ROW_MATMUL", "simd_qmm") == "simd_qmm":
        backend = simd_qmm_backend()
        if backend is not None:
            return backend
        print("[tensorfold] simd_qmm is not in this package: row_qmv instead", flush=True)
    return row_qmv_backend()


BACKEND: Backend | None = None


# -- stacked projections ----------------------------------------------------------------------------

GROUPS: dict[str, tuple[str, ...]] = {
    "in": ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"),     # recurrent layers
    "gu": ("gate_proj", "up_proj"),                                     # MLPs
}
_ATTR = "_row_forward_stacks"


class Stack:
    """Weights of projections that read the same rows, concatenated along the outputs; the members hold views."""

    __slots__ = ("weight", "scales", "biases", "group_size", "sizes", "members", "held")

    def __init__(self, members: Sequence[Any]) -> None:
        self.members = tuple(members)
        self.group_size = int(members[0].group_size)
        self.sizes = tuple(int(m["weight"].shape[0]) for m in members)
        self.weight = mx.concatenate([m["weight"] for m in members], axis=0)
        self.scales = mx.concatenate([m["scales"] for m in members], axis=0)
        self.biases = mx.concatenate([m["biases"] for m in members], axis=0)
        mx.eval(self.weight, self.scales, self.biases)
        offset = 0
        for m, n in zip(members, self.sizes):
            m.weight = self.weight[offset:offset + n]
            m.scales = self.scales[offset:offset + n]
            m.biases = self.biases[offset:offset + n]
            offset += n
        mx.eval([a for m in members for a in (m["weight"], m["scales"], m["biases"])])
        self.held = tuple(m["weight"] for m in members)

    def valid(self) -> bool:
        return all(m["weight"] is w for m, w in zip(self.members, self.held))


def _stackable(members: Sequence[Any], backend: Backend) -> bool:
    import mlx.nn as nn

    if not all(isinstance(m, nn.QuantizedLinear) and backend.fits(m) and "bias" not in m for m in members):
        return False
    k8 = {int(m["weight"].shape[1]) for m in members}
    gs = {int(m.group_size) for m in members}
    return len(k8) == 1 and len(gs) == 1


def stack_of(parent: Any, kind: str) -> Stack | None:
    stacks = parent.__dict__.get(_ATTR)
    if stacks is None:
        return None
    stack = stacks.get(kind)
    return stack if stack is not None and stack.valid() else None


def build(model: Any, backend: Backend) -> dict[str, int]:
    """Stack every group now: {kind: groups stacked}. No weight is stored twice."""

    counts = {kind: 0 for kind in GROUPS}
    for _, module in model.named_modules():
        for kind, names in GROUPS.items():
            members = [getattr(module, name, None) for name in names]
            if any(m is None for m in members) or stack_of(module, kind) is not None:
                continue
            if not _stackable(members, backend):
                continue
            stacks = module.__dict__.setdefault(_ATTR, {})
            stacks[kind] = Stack(members)
            counts[kind] += 1
    mx.clear_cache()          # the replaced arrays' buffers would otherwise sit in MLX's buffer cache
    return counts


def project(module: Any, x: mx.array) -> mx.array:
    """One projection through the backend (any bias added after)."""

    y = BACKEND(x, module["weight"], module["scales"], module["biases"], module.group_size)
    if "bias" in module:
        y = y + module["bias"]
    return y


def project_stack(stack: Stack, x: mx.array) -> mx.array:
    return BACKEND(x, stack.weight, stack.scales, stack.biases, stack.group_size)


# -- the glue ---------------------------------------------------------------------------------------------
#
# lane_glue's arithmetic, without what only the M5's lane matmul reads (the 64-group input sums) or pads (rows to
# a multiple of 16), reading the stacked projections' rows in place. The recurrent layer's glue also writes the conv
# tail and its recurrence the state after a chain's last row, so a window kept whole needs no replay.

_NORM = r"""
  // residual add + RMSNorm of row m: one threadgroup of K / 16 threads, thread t holds [16 t, 16 t + 16); the
  // row's sum of squares is each thread's sequential fma over its 16, then simd_sum, then the simdgroups in order
  const uint t = thread_position_in_threadgroup.x;
  const uint m = threadgroup_position_in_grid.y;
  constexpr int E = 16;
  constexpr int TPG = K / E;
  threadgroup float red[TPG / 32];
  const int base = int(m) * K + int(t) * E;
  float hv[E];
  float ss = 0.0f;
  for (int i = 0; i < E; i++) {
    bfloat h = H[base + i];
    RESIDUAL_ADD
    hv[i] = float(h);
    ss = fma(hv[i], hv[i], ss);
  }
  ss = simd_sum(ss);
  if (thread_index_in_simdgroup == 0) red[simdgroup_index_in_threadgroup] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (int i = 0; i < TPG / 32; i++) total += red[i];
  const float inv = metal::rsqrt(total / float(K) + eps[0]);
  for (int i = 0; i < E; i++) XO[base + i] = bfloat(float(Wt[int(t) * E + i]) * (hv[i] * inv));
"""

_GDN_POST = r"""
  // one simdgroup per (row m, v head): SiLU(z) * RMSNorm(y) * w, z read in place from the [qkv | z | b | a] rows
  const uint lane = thread_index_in_simdgroup;
  const uint hv = threadgroup_position_in_grid.y;
  const uint m = threadgroup_position_in_grid.z;
  constexpr int PER = DV / 32;
  float yv[PER];
  float ss = 0.0f;
  for (int j = 0; j < PER; j++) {
    yv[j] = float(Y[(m * NV + hv) * DV + lane * PER + j]);
    ss += yv[j] * yv[j];
  }
  ss = simd_sum(ss);
  const float inv = metal::rsqrt(ss / float(DV) + eps[0]);
  for (int j = 0; j < PER; j++) {
    const int d = int(lane) * PER + j;
    const float x = float(bfloat(float(NW[d]) * (yv[j] * inv)));
    const float zf = float(Z[m * ZS + ZO + hv * DV + d]);
    OUT[m * NV * DV + hv * DV + d] = bfloat(zf / (1.0f + metal::exp(-zf)) * x);
  }
"""

_ROW_PARTS = r"""
  // the partial sums of squares of row m as a residual epilogue writes them: partial t covers [8 t, 8 t + 8), two
  // sequential fma runs of 4 (one a simdgroup there) added in order
  const uint t = thread_position_in_grid.x;
  const uint m = thread_position_in_grid.y;
  if (t >= uint(K / 8)) return;
  float total = 0.0f;
  for (int s2 = 0; s2 < 2; s2++) {
    float ss = 0.0f;
    for (int j = 0; j < 4; j++) {
      const float h = float(H[m * K + t * 8 + s2 * 4 + j]);
      ss = fma(h, h, ss);
    }
    total += ss;
  }
  PO[m * (K / 8) + t] = total;
"""

_MLP_ACT = r"""
  // SiLU(gate) * up over [gate | up] rows of 2N
  const uint i = thread_position_in_grid.x;
  const uint m = thread_position_in_grid.y;
  if (i >= uint(N)) return;
  const float gf = float(GU[m * 2 * N + i]);
  HOUT[m * N + i] = bfloat(gf / (1.0f + metal::exp(-gf)) * float(GU[m * 2 * N + N + i]));
"""

def _gdn_pre_source() -> str:
    """lane_glue's gdn_pre reading the stacked [qkv | z | b | a] rows in place (qkv at 0, b at BO, a at AO, rows ZS
    apart), which also writes the conv tail after the window's last row (for a chain: the next conv state)."""

    from tensorfold.kernels.qwen.dense.v1 import lane_glue
    from tensorfold.kernels.qwen.dense.v1.lane_fuse import _replace_once

    pre = _replace_once(lane_glue._GDN_PRE, "float(QKV[(row - (TAPS - 1)) * C + c])",
                        "float(QKV[(row - (TAPS - 1)) * ZS + c])")
    pre = _replace_once(pre, "float(Ain[w * NV + hv])", "float(Ain[w * ZS + AO + hv])")
    pre = _replace_once(pre, "float(Bin[w * NV + hv])", "float(Bin[w * ZS + BO + hv])")
    return pre + """
  // the conv tail after the last row: the last TAPS - 1 rows of [conv state; window rows], this lane's channels
  if (int(w) == nodes[0] - 1)
    for (int r = 0; r < TAPS - 1; r++) {
      const int row = windows[w * TAPS + 1 + r];
      for (int j = 0; j < PER; j++) {
        const int c = c0 + int(lane) * PER + j;
        CO[r * C + c] = row < TAPS - 1 ? CS[row * C + c] : QKV[(row - (TAPS - 1)) * ZS + c];
      }
    }
"""


def _tree_source() -> str:
    """lane_tree's recurrence over window nodes, which for a chain also writes the state after its last row."""

    from tensorfold.kernels.qwen.dense.v1 import lane_tree

    return lane_tree._TREE_SOURCE + """
        if (CHAIN) {
          auto o_state = state_out + (hv_idx * Dv + dv_idx) * Dk;
          for (int i = 0; i < n_per_t; ++i) o_state[n_per_t * dk_idx + i] = states[0][i];
        }
"""


_SPECS: dict[str, tuple[str, str, list[str], list[str]]] = {
    "norm": (_NORM.replace("RESIDUAL_ADD", "h = bfloat(float(h) + float(R[base + i]));\n    HO[base + i] = h;"), "",
             ["H", "R", "Wt", "eps"], ["HO", "XO"]),
    "norm_nores": (_NORM.replace("RESIDUAL_ADD", ""), "", ["H", "Wt", "eps"], ["XO"]),
    "gdn_post": (_GDN_POST, "", ["Y", "Z", "NW", "eps"], ["OUT"]),
    "mlp_act": (_MLP_ACT, "", ["GU"], ["HOUT"]),
    "row_parts": (_ROW_PARTS, "", ["H"], ["PO"]),
    "gdn_pre": (_gdn_pre_source(), "", ["QKV", "CS", "CW", "windows", "Ain", "Bin", "ALOG", "DT", "nodes"],
                ["Q", "Kout", "Vout", "G", "BETA", "CO"]),
    "tree": (_tree_source(), "", ["q", "k", "v", "g", "beta", "state_in", "parents", "nodes"], ["y", "state_out"]),
}


_kernels: dict[str, tuple[str, Any]] = {}


def sources() -> dict[str, str]:
    out = {name: header + source for name, (source, header, _, _) in _SPECS.items()}
    out["gate_up_act"] = _gate_up_act_source()
    return out


def _kernel(name: str) -> Any:
    hit = _kernels.get(name)
    if hit is None:
        source, header, inputs, outputs = _SPECS[name]
        digest = hashlib.sha256((header + source).encode()).hexdigest()[:16]
        kernel = mx.fast.metal_kernel(name=f"row_forward_{name}_{digest}", input_names=inputs, output_names=outputs,
                                      source=source, header=header)
        hit = _kernels[name] = (source, kernel)
    return hit[1]


_consts: dict[Any, mx.array] = {}


def _const(key: Any, make: Callable[[], mx.array]) -> mx.array:
    if key not in _consts:
        _consts[key] = make()
    return _consts[key]


def _eps(eps: float) -> mx.array:
    return _const(("eps", float(eps)), lambda: mx.array([float(eps)], dtype=mx.float32))


def add_norm(hidden: mx.array, residual: mx.array | None, weight: mx.array, eps: float) -> tuple[mx.array, mx.array]:
    """(h, x): h = hidden + residual (hidden when residual is None), x = RMSNorm(h) * weight; (1, M, K) bf16."""

    lead = hidden.shape[:-1]
    K = int(hidden.shape[-1])
    M = hidden.size // K
    if K % 512 or K > 16384:
        raise ValueError(f"norm: the hidden size must be a multiple of 512 up to 16384, got {K}")
    common = dict(template=[("K", K)], grid=(K // 16, M, 1), threadgroup=(K // 16, 1, 1))
    if residual is None:
        x = _kernel("norm_nores")(inputs=[hidden.reshape(M, K), weight, _eps(eps)], output_shapes=[(M, K)],
                                  output_dtypes=[mx.bfloat16], **common)[0]
        return hidden, x.reshape(*lead, K)
    h, x = _kernel("norm")(inputs=[hidden.reshape(M, K), residual.reshape(M, K), weight, _eps(eps)],
                           output_shapes=[(M, K), (M, K)], output_dtypes=[mx.bfloat16, mx.bfloat16], **common)
    return h.reshape(*lead, K), x.reshape(*lead, K)


def row_parts(h: mx.array) -> mx.array:
    """The partial sums of squares of each row of ``h`` (..., K) as ``Backend.variant``'s residual epilogue writes
    them: (..., K / 8) fp32."""

    K = int(h.shape[-1])
    M = h.size // K
    return _kernel("row_parts")(inputs=[h.reshape(M, K)], template=[("K", K)], grid=(K // 8, M, 1),
                                threadgroup=(min(256, K // 8), 1, 1), output_shapes=[(M, K // 8)],
                                output_dtypes=[mx.float32])[0].reshape(*h.shape[:-1], K // 8)


def gdn_post(rec: mx.array, y: mx.array, weight: mx.array, eps: float, *, zo: int) -> mx.array:
    """SiLU(z) * RMSNorm(rec) * weight per head, z read in place from the stacked rows ``y`` (z at column ``zo``)."""

    _, W, nv, dv = (int(s) for s in rec.shape)
    zs = int(y.shape[-1])
    return _kernel("gdn_post")(
        inputs=[rec, y.reshape(W, zs), weight, _eps(eps)], template=[("NV", nv), ("DV", dv), ("ZS", zs), ("ZO", zo)],
        grid=(32, nv, W), threadgroup=(32, 1, 1), output_shapes=[(1, W, nv * dv)], output_dtypes=[rec.dtype])[0]


def mlp_act(gu: mx.array) -> mx.array:
    """SiLU(gate) * up over [gate | up] rows (..., 2N) -> (..., N)."""

    N2 = int(gu.shape[-1])
    N = N2 // 2
    W = gu.size // N2
    return _kernel("mlp_act")(
        inputs=[gu.reshape(W, N2)], template=[("N", N)], grid=(256 * (-(-N // 256)), W, 1), threadgroup=(256, 1, 1),
        output_shapes=[(W, N)], output_dtypes=[gu.dtype])[0].reshape(*gu.shape[:-1], N)


def gdn_pre(y: mx.array, conv_state: mx.array, conv_weight: mx.array, windows: mx.array, a_log: mx.array,
            dt_bias: mx.array, *, nk: int, nv: int, dk: int, dv: int) -> tuple[mx.array, ...]:
    """q, k [1, W, nk, dk], v [1, W, nv, dv], g [1, W, nv] fp32, beta [1, W, nv] and the conv tail [1, taps - 1, C]
    after the last row, from the stacked [qkv | z | b | a] rows ``y``."""

    zs = int(y.shape[-1])
    W = y.size // zs
    C = 2 * nk * dk + nv * dv
    taps = int(conv_weight.shape[1])
    if zs != C + nv * dv + 2 * nv or dk != dv or dk % 32:
        raise ValueError(f"gdn_pre: [qkv | z | b | a] rows of {C + nv * dv + 2 * nv} and head dims multiple of 32 "
                         f"expected, got {zs}")
    y2 = y.reshape(W, zs)
    nodes = _const(("nodes", W), lambda: mx.array([W], dtype=mx.int32))
    return tuple(_kernel("gdn_pre")(
        inputs=[y2, conv_state.reshape(taps - 1, C), conv_weight.reshape(C, taps), windows, y2, y2, a_log, dt_bias,
                nodes],
        template=[("NK", nk), ("NV", nv), ("DK", dk), ("DV", dv), ("TAPS", taps), ("ZS", zs),
                  ("AO", C + nv * dv + nv), ("BO", C + nv * dv)],
        grid=(32, 2 * nk + nv, W), threadgroup=(32, 1, 1),
        output_shapes=[(1, W, nk, dk), (1, W, nk, dk), (1, W, nv, dv), (1, W, nv), (1, W, nv), (1, taps - 1, C)],
        output_dtypes=[y.dtype, y.dtype, y.dtype, mx.float32, y.dtype, y.dtype]))


def gated_delta(q: mx.array, k: mx.array, v: mx.array, g: mx.array, beta: mx.array, state: mx.array,
                parents: Sequence[int]) -> tuple[mx.array, mx.array]:
    """``lane_tree.gated_delta_tree`` (each node's output, the recurrence walked from ``state`` along its path),
    plus for a chain the state after its last row."""

    from tensorfold.kernels.qwen.dense.v1 import lane_tree

    _, W, Hk, Dk = (int(s) for s in k.shape)
    Hv, Dv = int(v.shape[2]), int(v.shape[3])
    lane_tree.tree_paths(parents)                                   # validates the parent order
    chain = _chain(parents)
    if W > (lane_tree.MAX_DEPTH if chain else lane_tree.MAX_TREE):
        raise ValueError(f"window of {W} rows: trees take up to {lane_tree.MAX_TREE}, chains {lane_tree.MAX_DEPTH}")
    maxw = 1 if chain else (16 if W <= 16 else lane_tree.MAX_TREE)
    parents_a = _const(("parents", tuple(parents)), lambda: mx.array(list(parents), dtype=mx.int32))
    nodes = _const(("nodes", W), lambda: mx.array([W], dtype=mx.int32))
    y, state_out = _kernel("tree")(
        inputs=[q, k, v, g, beta, state, parents_a, nodes],
        template=[("InT", q.dtype), ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv), ("MAXW", maxw), ("CHAIN", chain)],
        grid=(32, Dv, Hv), threadgroup=(32, 4, 1),
        output_shapes=[(1, W, Hv, Dv), tuple(state.shape)], output_dtypes=[q.dtype, mx.float32])
    return y, state_out


# -- the layers --------------------------------------------------------------------------------------

def _chain(parents: Sequence[int]) -> bool:
    return list(parents) == list(range(-1, len(parents) - 1))


class _Normed:
    """A layer's RMSNorm-ed input rows: materialized (``x``), or, with a backend that normalizes on load, implied by
    the residual stream ``h``, its rows' partial sums of squares and the norm's weight."""

    __slots__ = ("x", "h", "parts", "weight", "eps")

    def __init__(self, *, x: mx.array | None = None, h: mx.array | None = None, parts: mx.array | None = None,
                 weight: mx.array | None = None, eps: float = 0.0) -> None:
        self.x, self.h, self.parts, self.weight, self.eps = x, h, parts, weight, eps

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.x if self.x is not None else self.h).shape

    def project(self, weight: mx.array, scales: mx.array, biases: mx.array, group_size: int) -> mx.array:
        if self.x is not None:
            return BACKEND(self.x, weight, scales, biases, group_size)
        return BACKEND.variant(self.h, weight, scales, biases, group_size, norm=(self.parts, self.weight, self.eps))

    def gate_up_act(self, weight: mx.array, scales: mx.array, biases: mx.array, group_size: int) -> mx.array:
        if self.x is None:
            return BACKEND.variant(self.h, weight, scales, biases, group_size, norm=(self.parts, self.weight, self.eps),
                                   epilogue="act")
        if BACKEND.gate_up_act is not None:
            return BACKEND.gate_up_act(self.x, weight, scales, biases, group_size)
        return mlp_act(BACKEND(self.x, weight, scales, biases, group_size))


def _in(module: Any, inp: _Normed) -> mx.array:
    y = inp.project(module["weight"], module["scales"], module["biases"], module.group_size)
    if "bias" in module:
        y = y + module["bias"]
    return y


def _attention(attn: Any, inp: _Normed, cache: Any, parents: Sequence[int], positions: list[int],
               record: list[Any]) -> mx.array:
    """Qwen3-Next attention for window rows (per-row RoPE positions, query-by-query exact attention): the input of
    o_proj."""

    from tensorfold.kernels.qwen.dense.v1 import exact_attention, row_attention

    if not ROW_ATTENTION and not _chain(parents):
        raise NotImplementedError("row_forward: draft trees need row_attention")
    B, L, _ = inp.shape
    H, nkv = attn.num_attention_heads, attn.num_key_value_heads
    q_proj_output = _in(attn.q_proj, inp)
    D = int(attn.head_dim) if hasattr(attn, "head_dim") else int(q_proj_output.shape[-1]) // (2 * H)
    gate = q_proj_output.reshape(B, L, H, 2 * D)[..., D:].reshape(B, L, -1)
    # RMSNorm per head row over [q_h | gate_h] halves, then the q halves (rows are their own: no copy of a slice)
    queries = attn.q_norm(q_proj_output.reshape(B, L, 2 * H, D))[:, :, 0::2]
    keys = attn.k_norm(_in(attn.k_proj, inp).reshape(B, L, nkv, -1))
    values = _in(attn.v_proj, inp).reshape(B, L, nkv, -1)
    queries = queries.transpose(0, 2, 1, 3)
    keys = keys.transpose(0, 2, 1, 3)
    values = values.transpose(0, 2, 1, 3)
    pos = mx.array(positions, dtype=mx.int32)
    queries = attn.rope(queries.transpose(2, 1, 0, 3), offset=pos).transpose(2, 1, 0, 3)
    keys = attn.rope(keys.transpose(2, 1, 0, 3), offset=pos).transpose(2, 1, 0, 3)
    record.append(("kv", keys, values))
    keys, values = cache.update_and_fetch(keys, values)
    if ROW_ATTENTION:
        # the cache's whole buffers: the window's rows sit at start + row
        start = int(cache.offset) - L
        output = row_attention.row_sdpa(queries, cache.keys, cache.values, attn.scale, start, parents)
    else:
        output = exact_attention.exact_sdpa(queries, keys, values, cache, attn.scale, "causal" if L > 1 else None)
    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
    return output * mx.sigmoid(gate)


def _gdn(gdn: Any, inp: _Normed, cache: Any, parents: Sequence[int], windows: mx.array, record: list[Any]
         ) -> mx.array:
    """The recurrent layer for window rows: the input of out_proj."""

    B = int(inp.shape[0])
    stack = stack_of(gdn, "in")
    if stack is None:
        raise RuntimeError("row_forward: the recurrent layer's projections are not stacked (build(model) first)")
    y = inp.project(stack.weight, stack.scales, stack.biases, stack.group_size)     # [qkv | z | b | a]
    n_keep = gdn.conv_kernel_size - 1
    conv_state = cache[0] if cache[0] is not None else mx.zeros((B, n_keep, gdn.conv_dim), dtype=y.dtype)
    state = cache[1]
    if state is None:
        state = mx.zeros((B, gdn.num_v_heads, gdn.head_v_dim, gdn.head_k_dim), dtype=mx.float32)
    heads = dict(nk=gdn.num_k_heads, nv=gdn.num_v_heads, dk=gdn.head_k_dim, dv=gdn.head_v_dim)
    q, k, v, g, beta, conv_out = gdn_pre(y, conv_state, gdn.conv1d.weight, windows, gdn.A_log, gdn.dt_bias, **heads)
    rec, state_out = gated_delta(q, k, v, g, beta, mx.contiguous(state), parents)
    record.append(("gdn", q, k, v, g, beta, state, (conv_state, y, gdn.conv_dim, state_out, conv_out, _chain(parents)),
                   n_keep))
    return gdn_post(rec, y, gdn.norm.weight, gdn.norm.eps, zo=gdn.conv_dim)


def commit(cache: list[Any], record: list[Any], path: Sequence[int], window: int, start: int) -> None:
    """``lane_tree.commit_tree`` for this forward's record: a chain kept whole takes the state and conv tail its
    recurrence wrote (no replay); otherwise the kept rows are replayed as there."""

    from tensorfold.kernels.qwen.dense.v1 import lane_tree

    keep = len(path)
    whole = keep == window and list(path) == list(range(keep))
    if not whole:
        # the conv tail rows come from [conv state; window rows]: built only for a partly kept window
        fixed = []
        for kind, *entry in record:
            if kind == "gdn":
                conv_state, y, C, _, _, _ = entry[6]
                entry = entry[:6] + [mx.concatenate([conv_state, y[..., :C]], axis=1)[0]] + entry[7:]
            fixed.append((kind, *entry))
        lane_tree.commit_tree(cache, fixed, path, window, start)
        return
    j = 0
    for item in cache:
        kind, *entry = record[j]
        j += 1
        if hasattr(item, "keys") and hasattr(item, "values"):
            if kind != "kv":
                raise RuntimeError(f"record entry {j - 1} is {kind!r}, the cache has an attention layer")
            item.trim(window - keep)
            continue
        if kind != "gdn":
            raise RuntimeError(f"record entry {j - 1} is {kind!r}, the cache has a recurrent layer")
        conv_state, y, C, state_out, conv_out, chain = entry[6]
        if not chain:
            raise RuntimeError("a tree window kept whole is not a path")
        item[1] = state_out
        item[0] = conv_out
        item.advance(keep)
    if j != len(record):
        raise RuntimeError(f"recorded {len(record)} layers, cache has {j}")


def forward(core: Any, head: Any, tokens: Sequence[int], parents: Sequence[int], cache: list[Any], start: int, *,
            pipeline_layers: int = 4, last_only: bool = False, first_alone: bool = True) -> tuple[mx.array, list[Any]]:
    """Logits [1, W, V] for a window whose root sits at ``start``; the record ``lane_tree.commit_tree`` keeps.

    ``lane_tree.tree_forward``'s contract: attention layers append the W rows to their caches, recurrent layers
    leave their state alone and record what the commit replays; DFlash tap hooks get the layer outputs.
    """

    from tensorfold.kernels.qwen.dense.v1 import lane_tree

    if len(tokens) > BACKEND.max_rows:
        raise ValueError(f"row_forward: {len(tokens)} rows, the {BACKEND.name} matmul takes up to {BACKEND.max_rows}")
    positions = [start + d for d in lane_tree.tree_paths(parents)[0]]
    ids = (tokens.reshape(1, -1).astype(mx.uint32) if isinstance(tokens, mx.array)      # lazy ids (timing chains)
           else mx.array([list(tokens)], dtype=mx.uint32))
    hidden = core.embed_tokens(ids)
    fold = BACKEND.variant is not None                # norms folded into the matmuls
    parts = row_parts(hidden) if fold else None       # the first layer's input: its own partial sums
    record: list[Any] = []
    layers = list(core.layers)
    windows = None
    pending: mx.array | None = None                   # (separate norms) the last MLP's output, not yet added
    tapped: Any = None

    def out(module: Any, x: mx.array) -> None:
        """hidden += x @ module.T: the residual add (and the next norm's partial sums) in the matmul, or pending."""
        nonlocal hidden, parts, pending
        if fold:
            hidden, parts = BACKEND.variant(x, module["weight"], module["scales"], module["biases"], module.group_size,
                                            epilogue="residual", res=hidden)
        else:
            pending = project(module, x)

    def normed(norm: Any) -> _Normed:
        nonlocal hidden, pending
        if fold:
            return _Normed(h=hidden, parts=parts, weight=norm.weight, eps=norm.eps)
        hidden, x = add_norm(hidden, pending, norm.weight, norm.eps)
        pending = None
        return _Normed(x=x)

    for index, (layer, item) in enumerate(zip(layers, cache)):
        inner = getattr(layer, "_layer", layer)
        inp = normed(inner.input_layernorm)
        if tapped is not None:
            tapped[0][tapped[1]] = hidden
        if getattr(inner, "is_linear", False):
            if windows is None:
                windows = lane_tree._conv_windows(parents, inner.linear_attn.conv_kernel_size - 1)
            out(inner.linear_attn.out_proj, _gdn(inner.linear_attn, inp, item, parents, windows, record))
        else:
            out(inner.self_attn.o_proj, _attention(inner.self_attn, inp, item, parents, positions, record))
        inp = normed(inner.post_attention_layernorm)
        mlp = inner.mlp
        stack = stack_of(mlp, "gu")
        if stack is not None:
            act = inp.gate_up_act(stack.weight, stack.scales, stack.biases, stack.group_size)
        else:
            act = mlp_act(mx.concatenate([_in(mlp.gate_proj, inp), _in(mlp.up_proj, inp)], axis=-1))
        out(mlp.down_proj, act)
        storage = getattr(layer, "_storage", None)
        tapped = (storage, layer._idx) if storage is not None else None
        if pipeline_layers and ((index + 1) % pipeline_layers == 0 or (index == 0 and first_alone)) \
                and index + 1 < len(layers):
            mx.async_eval(hidden, parts if fold else pending)
    inp = normed(core.norm)
    if tapped is not None:
        tapped[0][tapped[1]] = hidden
    sink = lane_tree.HIDDEN_SINK          # a proposer that drafts from the rows' post-norm hidden states
    if sink is not None:
        sink.append(inp.x if inp.x is not None else add_norm(hidden, None, core.norm.weight, core.norm.eps)[1])
        if len(sink) > 1024:
            del sink[0]
    if last_only:
        inp = (_Normed(x=inp.x[:, -1:]) if inp.x is not None
               else _Normed(h=inp.h[:, -1:], parts=inp.parts[:, -1:], weight=inp.weight, eps=inp.eps))
    return inp.project(head["weight"], head["scales"], head["biases"], head.group_size), record


def fits(model: Any, backend: Backend) -> bool:
    """Whether every projection and the head take the backend's layout."""

    import mlx.nn as nn

    language_model = getattr(model, "language_model", model)
    head = getattr(language_model, "lm_head", None)
    if not isinstance(head, nn.QuantizedLinear) or not backend.fits(head):
        return False
    for layer in language_model.model.layers:
        inner = layer.linear_attn if getattr(layer, "is_linear", False) else layer.self_attn
        for _, module in list(inner.named_modules()) + list(layer.mlp.named_modules()):
            if isinstance(module, nn.QuantizedLinear) and not backend.fits(module):
                return False
    return True


def install(model: Any, backend: Backend | None = None) -> dict[str, int]:
    """Use ``backend`` (default ``row_qmv``) and stack the model's projection groups. Idempotent."""

    global BACKEND
    BACKEND = backend or choose_backend()
    stacked = build(model, BACKEND)
    if BACKEND.prepare is not None:
        import mlx.nn as nn

        weights = [(m["weight"], m["scales"], m["biases"]) for _, m in model.named_modules()
                   if isinstance(m, nn.QuantizedLinear)]
        for _, module in model.named_modules():
            for stack in module.__dict__.get(_ATTR, {}).values():
                weights.append((stack.weight, stack.scales, stack.biases))
        BACKEND.prepare(weights)
    return stacked


__all__ = ["BACKEND", "Backend", "GROUPS", "Stack", "build", "fits", "forward", "install", "project",
           "project_stack", "row_qmv_backend", "sources"]
