"""A prompt chunk's forward: the reference blocks with hyper-connections in the fused decode's arithmetic."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import numpy as np

from tensorfold.kernels.qwen.flash_next.v1 import base
from tensorfold.kernels.qwen.flash_next.v1.hc import RINV, hc_norm

_HC_NORMED = r"""
  // Thread (e, r): element e of row r's streams: bf16((h * rinv(stream)) * scale), kernels.hc_project's normed input.
  const int e = int(thread_position_in_grid.x);
  const int r = int(thread_position_in_grid.y);
  constexpr int W = S * D;
  const float rv = stream_rinv(SSP, r, e / D, D / 256, S, D, eps[0]);
  NORMED[size_t(r) * W + e] = bfloat((float(HN[size_t(r) * W + e]) * rv) * NW[e]);
"""

_HC_ACT = r"""
  // Thread (c, r): output c of the down + inject rows, / S, then SiLU (c < LOW) or the gate 2 sigmoid (hc_project's).
  const int c = int(thread_position_in_grid.x);
  const int r = int(thread_position_in_grid.y);
  const float v4 = float(bfloat(float(DN[size_t(r) * ND + c]) / float(S)));
  if (c < LOW) ACT[size_t(r) * LOW + c] = bfloat(bsilu(v4));
  else INJ[size_t(r) * S + (c - LOW)] = bfloat(2.0f * bsig(v4));
"""

_HC_MIX = r"""
  // Thread (d, r): dim d of row r's block input: mean over streams of bf16(sigmoid(up) * normed).
  const int d = int(thread_position_in_grid.x);
  const int r = int(thread_position_in_grid.y);
  constexpr int W = S * D;
  float total = 0.0f;
  for (int s = 0; s < S; s++) {
    const size_t e = size_t(r) * W + s * D + d;
    total += float(bfloat(bsig(float(UP[e])) * float(NORMED[e])));
  }
  MIXED[size_t(r) * D + d] = bfloat(total / float(S));
"""

_HEADER = base.QDOT_HEADER + RINV


def _qmm(x: mx.array, w: Any) -> mx.array:
    from tensorfold.kernels.qwen.flash_next.v1 import prefill_mm

    return prefill_mm.matmul(x, w.weight, w.scales, w.biases)


def hyper_connection(hc: Any, h: mx.array, pending: tuple[mx.array, mx.array] | None, *, streams: int,
                     eps: mx.array) -> tuple[mx.array, mx.array, mx.array | None]:
    """h [R, S*D] and the pending (branch, inject gates) or None -> (h written back, block input, inject gates)."""

    rows, wide = h.shape
    dims = wide // streams
    if pending is None:
        h_new, ssp = hc_norm(h, streams=streams)
    else:
        h_new, ssp = hc_norm(h, streams=streams, write_back="plain", branch=(pending[0],), inject=pending[1])
    normed_k = base.kernel("tf_hc_normed", _HC_NORMED, ["HN", "SSP", "NW", "eps"], ["NORMED"], header=_HEADER)
    normed = normed_k(inputs=[h_new, ssp, hc.scale, eps], template=[("S", streams), ("D", dims)],
                      grid=(wide, rows, 1), threadgroup=(256, 1, 1), output_shapes=[(rows, wide)],
                      output_dtypes=[mx.bfloat16])[0]
    dn = _qmm(normed, hc.down)                                          # [R, LOW (+ S inject rows)]
    nd = int(dn.shape[1])
    low = hc.low
    act_k = base.kernel("tf_hc_act", _HC_ACT, ["DN"], ["ACT", "INJ"], header=_HEADER)
    act, inj = act_k(inputs=[dn], template=[("S", streams), ("LOW", low), ("ND", nd)],
                     grid=(nd, rows, 1), threadgroup=(min(nd, 256), 1, 1),
                     output_shapes=[(rows, low), (rows, streams)], output_dtypes=[mx.bfloat16, mx.bfloat16])
    up = _qmm(act, hc.up)                                               # [R, S*D]
    mix_k = base.kernel("tf_hc_mix", _HC_MIX, ["UP", "NORMED"], ["MIXED"], header=_HEADER)
    mixed = mix_k(inputs=[up, normed], template=[("S", streams), ("D", dims)],
                  grid=(dims, rows, 1), threadgroup=(256, 1, 1), output_shapes=[(rows, dims)],
                  output_dtypes=[mx.bfloat16])[0]
    return h_new, mixed, (inj if nd > low else None)


QUEUE_LAYERS = 2                    # layers a slice; MLX holds a queued slice's buffers, so this bounds memory


def hidden(model: Any, tokens: np.ndarray, cache: list[Any]) -> mx.array:
    """Qwen4Exp.hidden for a prompt chunk (batch 1); the same graph and bits however it is sliced."""

    fused = model.__dict__["fused"]
    streams = model.args.hc_count
    eps = fused.eps
    h = model.model.embed_tokens(mx.array(tokens.astype(np.int32)))[0]
    h = mx.tile(h, (1, streams))                                        # [L, S*D]
    pending = None
    queued = None
    for i, (layer, c) in enumerate(zip(model.layers, cache)):
        entry = fused.layers[i]
        if "ple" in layer:
            if pending is not None:
                h = hc_norm(h, streams=streams, write_back="plain", branch=(pending[0],), inject=pending[1])[0]
                pending = None
            h = h + layer.ple(h[None], tokens, c)[0]
        h, mixed, inj = hyper_connection(entry["attn_hc"], h, pending, streams=streams, eps=eps)
        mixer = layer.linear_attn if layer.is_linear else layer.self_attn
        branch = mixer(mixed[None], c)[0]
        h, mixed, inj2 = hyper_connection(entry["mlp_hc"], h, (branch, inj), streams=streams, eps=eps)
        pending = (layer.mlp(mixed[None])[0], inj2)
        if QUEUE_LAYERS and (i + 1) % QUEUE_LAYERS == 0:
            step = (h, *pending)
            mx.async_eval(*step)
            if queued is not None:
                mx.eval(*queued)
            queued = step
    h, mixed, _ = hyper_connection(fused.mixer, h, pending, streams=streams, eps=eps)
    model.__dict__["last_streams"] = h                                  # [L, S*D]: the streams before the mixer
    return mixed[None]
