"""Retired row kernels, preserved for independent native diagnostics. See README.md."""
from __future__ import annotations
import hashlib
from typing import Any, Callable, Sequence
import mlx.core as mx

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

    from tools.native_legacy import row_qmv
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
        from tools.native_legacy.row_qmv import _HEADER

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

    from tools.native_legacy import row_qmv

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
    from tools.native_legacy import row_qmv
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
    from tools.native_legacy import row_qmv

    if _gate_up_kernel is None:
        from tools.native_legacy.row_qmv import _HEADER

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
    "norm_nores": (_NORM.replace("    RESIDUAL_ADD\n", ""), "", ["H", "Wt", "eps"], ["XO"]),
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



def _chain(parents: Sequence[int]) -> bool:
    return list(parents) == list(range(-1, len(parents) - 1))
