"""M5 decode steps with fewer launches: kernels that also write the next lane matmul's input group sums, same bits."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from tensorfold.kernels.nemotron.lightning.v1.sources import _GROUP_NORM
from tensorfold.kernels.qwen.dense.v1 import lane_qmm

# group_norm, also writing its output's 64-input sums (XS [XD / 64, MP], lane_qmm XSUM's order; padded rows zero)
_GROUP_NORM_XS = _GROUP_NORM.replace(
    "  const uint r = threadgroup_position_in_grid.y;\n",
    "  const uint r = threadgroup_position_in_grid.y;\n"
    "  constexpr int SUB = GS / 64;\n"
    "  if (int(r) >= dims[0]) {\n"
    "    for (int s = int(thread_position_in_threadgroup.x); s < SUB; s += GS / 4) XS[(int(grp) * SUB + s) * MP + int(r)] = 0.0f;\n"
    "    return;\n"
    "  }\n"
    "  threadgroup bfloat outv[GS];\n").replace(
    "    OUT[base + i] = bfloat(float(W[c]) * float(bfloat(v[i] * scale)));\n  }\n",
    "    const bfloat o = bfloat(float(W[c]) * float(bfloat(v[i] * scale)));\n"
    "    OUT[base + i] = o;\n"
    "    outv[int(t) * 4 + i] = o;\n"
    "  }\n"
    "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "  if (int(t) < SUB) {\n"
    "    float acc = 0.0f;\n"
    "    for (int i = 0; i < 64; i++) acc += float(outv[int(t) * 64 + i]);\n"
    "    XS[(int(grp) * SUB + int(t)) * MP + int(r)] = acc;\n"
    "  }\n")
assert _GROUP_NORM_XS.count("outv") == 3

# lane_qmm's 64-wide kernel with mlx_lm's relu2 on its bf16 output, plus that output's group sums: its 64 columns
# are one group of the next projection's inputs
_STORE = """  if (slice == 0)
    for (int i = 0; i < CAP; i++) {
      const int m = rb + erow[i], n = n0 + ecol[i];
      if (m < M) Y[m * N + n] = static_cast<bfloat>(C[i]);
    }
"""
assert _STORE in lane_qmm._COOP
_COOP_RELU2_XS = lane_qmm._COOP.replace(_STORE, """  threadgroup bfloat tile[16 * TMR * 64];
  if (slice == 0)
    for (int i = 0; i < CAP; i++) {
      const float h = metal::max(float(static_cast<bfloat>(C[i])), 0.0f);
      const bfloat y = static_cast<bfloat>(h * h);
      tile[erow[i] * 64 + ecol[i]] = y;
      const int m = rb + erow[i], n = n0 + ecol[i];
      if (m < M) Y[m * N + n] = y;
    }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int tid = int(thread_position_in_threadgroup.x);
  if (tid < 16 * TMR && rb + tid < MP) {
    float acc = 0.0f;
    if (rb + tid < M) for (int i = 0; i < 64; i++) acc += float(tile[tid * 64 + i]);
    XSO[(n0 / 64) * MP + rb + tid] = acc;
  }
""")

_baked: dict[str, Any] = {}


def _kernel(name: str, body: str, inputs: list[str], outputs: list[str]) -> Any:
    found = _baked.get(name)
    if found is None:
        found = _baked[name] = lane_qmm._Baked(name, body, inputs, outputs)
    return found


def group_norm_sums(x: mx.array, weight: mx.array, eps: mx.array, group: int) -> tuple[mx.array, mx.array]:
    """``kernels.group_norm`` and its output's group sums [D / 64, MP] for the lane matmul that reads it."""

    rows, dims = int(x.shape[0]), int(x.shape[-1])
    mp = 16 * -(-rows // 16)
    kernel = _kernel("nemotron_group_norm_xs", _GROUP_NORM_XS, ["X", "W", "eps", "dims"], ["OUT", "XS"])
    out, xs = kernel(inputs=[x, weight, eps, lane_qmm._mdims(rows, mp)], template=[("XD", dims), ("GS", group), ("MP", mp)],
                     grid=((group // 4) * (dims // group), mp, 1), threadgroup=(group // 4, 1, 1),
                     output_shapes=[(rows, dims), (dims // 64, mp)], output_dtypes=[mx.bfloat16, mx.float32])
    return out, xs


def takes_relu2(linear: Any) -> bool:
    """A lane-tiled 64-wide 4-bit projection (``lane_qmm.install(wide=True)``) the relu2 kernel reads."""

    return (getattr(linear, "_lane_tiled", False) and getattr(linear, "_lane_nt", 0) == 64 and linear.bits == 4
            and getattr(linear, "_lane_sbt", None) is not None and "bias" not in linear)


def up_relu2(x: mx.array, xs: mx.array, linear: Any) -> tuple[mx.array, mx.array]:
    """relu2(linear(x)) as bf16 [M, N] with its group sums [N / 64, MP]: one launch for lane_matmul + relu2 + XSUM."""

    k = int(x.shape[-1])
    m = int(x.reshape(-1, k).shape[0])
    n = int(linear["weight"].shape[0])
    mp = 16 * ((m + 15) // 16)
    block = mp if mp <= lane_qmm.ROW_BLOCK else lane_qmm.ROW_BLOCK
    edge = int(mp % block != 0)
    sk = lane_qmm.split_k(n, k)
    kernel = _kernel("nemotron_lane_up_relu2", _COOP_RELU2_XS, ["X", "XS", "Wq", "SBt", "mdims"], ["Y", "XSO"])
    y, xso = kernel(inputs=[x.reshape(m, k), xs, linear["weight"], linear._lane_sbt, lane_qmm._mdims(m, mp)],
                    template=[("TMR", block // 16), ("N", n), ("K", k), ("SK", sk), ("GS", int(linear.group_size)),
                              ("EDGE", edge)],
                    grid=((n // 64) * 64 * sk, -(-mp // block), 1), threadgroup=(64 * sk, 1, 1),
                    output_shapes=[(m, n), (n // 64, mp)], output_dtypes=[mx.bfloat16, mx.float32])
    return y, xso


__all__ = ["group_norm_sums", "takes_relu2", "up_relu2"]
