"""Every Flash Next kernel input stays on one side of MLX's constant/device size line (8 elements) at 1-9 rows and
streams, so each kernel name has one source (GPU, real dims, 32 experts)."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

if not mx.metal.is_available():
    pytest.skip("needs a Metal GPU", allow_module_level=True)

from tensorfold.kernels import inputs  # noqa: E402
from tensorfold.kernels.qwen.flash_next.v1 import attention, base, embed, experts, gdn, hc  # noqa: E402

D, S, LOW, NE, TOPK, NI = 2560, 4, 320, 32, 10, 640
NK, NV, DK, DV, TAPS = 16, 48, 128, 128, 4
C = 2 * NK * DK + NV * DV
PW = C + NV * DV + 2 * NV


class _Linear:
    """Random 4-bit group-32 weights [rows, cols] (a stack of ``stack`` when given)."""

    def __init__(self, rng, rows, cols, stack=None):
        lead = (stack,) if stack else ()
        self.weight = mx.array(rng.integers(0, 2**32, size=lead + (rows, cols // 8), dtype=np.uint32))
        self.scales = mx.array((0.02 * rng.random(lead + (rows, cols // 32))).astype(np.float32)).astype(mx.bfloat16)
        self.biases = mx.array((0.01 * rng.normal(size=lead + (rows, cols // 32))).astype(np.float32)).astype(mx.bfloat16)


def _bf16(rng, *shape, scale=0.3):
    return mx.array((scale * rng.normal(size=shape)).astype(np.float32)).astype(mx.bfloat16)


@pytest.fixture
def sizes(monkeypatch):
    """Each kernel call's input sizes, by kernel name (the kernel cache emptied so every kernel is recorded)."""

    seen: dict[str, list[list[int]]] = {}
    real = mx.fast.metal_kernel

    def recording(**spec):
        kernel = real(**spec)

        def call(*, inputs, **kwargs):
            seen.setdefault(spec["name"], []).append([int(a.size) for a in inputs])
            return kernel(inputs=inputs, **kwargs)

        return call

    monkeypatch.setattr(mx.fast, "metal_kernel", recording)
    monkeypatch.setattr(base, "_kernels", {})
    return seen


def test_inputs_keep_one_side_of_eight(sizes):
    rng = np.random.default_rng(0)
    eps = mx.array([1e-6], dtype=mx.float32)
    hd_, hu_ = _Linear(rng, LOW + S, S * D), _Linear(rng, S * D, LOW)
    hc_down = base.QWeights(hd_.weight, hd_.scales, hd_.biases)
    hc_up = base.QWeights(hu_.weight, hu_.scales, hu_.biases)
    scale = mx.array((1.0 + 0.1 * rng.normal(size=(S * D,))).astype(np.float32))
    gate, up, down = _Linear(rng, NI, D, NE), _Linear(rng, NI, D, NE), _Linear(rng, D, NI, NE)
    shared = (_Linear(rng, NI, D), _Linear(rng, NI, D), _Linear(rng, D, NI))
    router_rows = _bf16(rng, NE + 1, D, scale=0.05)
    emb = _Linear(rng, 100, D)
    conv_w, a_log, dt, nw = _bf16(rng, C, TAPS), _bf16(rng, NV), _bf16(rng, NV), _bf16(rng, DV)
    conv = _bf16(rng, TAPS - 1, C)
    ssm = mx.array((0.1 * rng.normal(size=(NV, DV, DK))).astype(np.float32))
    keys, values = _bf16(rng, 1, 2, 256, 256), _bf16(rng, 1, 2, 256, 256)
    outs = []
    for rows in range(1, 10):
        h = _bf16(rng, rows, S * D)
        hn, ssp = hc.hc_norm(h, streams=S, write_back="plain", branch=(_bf16(rng, rows, D),),
                             inject=mx.ones((max(rows, 2), S), dtype=mx.bfloat16))
        x, inj = hc.hc_project(hn, ssp, hc_down, hc_up, scale, eps=eps, streams=S, low=LOW)
        logits = experts.router(x, router_rows)
        act, picks, weights = experts.expert_gateup(x, logits, TOPK, NE, gate, up, shared=shared[:2])
        y = experts.expert_down_y(act, picks, down, shared[2])
        outs.append(hc.hc_norm(hn, streams=S, write_back="grouped", branch=(y, weights, logits), inject=inj)[0])
        outs.append(embed.embed_rows(list(range(rows)), emb, tile=S))
        outs.append(embed.rms_norm_rows(h, scale, eps, group=D))
        outs.append(gdn.gdn_step(_bf16(rng, rows, PW), conv, ssm, conv_w, a_log, dt, nw, eps, nk=NK, nv=NV,
                                 dk=DK, dv=DV)[0])
        q = _bf16(rng, rows, 24, 256)
        counts = [100 + r for r in range(rows)]
        outs.append(attention.attention_rows(q, keys, values, counts, None, [False] * rows, 0.0625))
        streams = min(rows, base.MAX_STREAMS)
        lengths = [1] * (streams - 1) + [rows - streams + 1]
        srow = [b for b, n in enumerate(lengths) for _ in range(n)]
        outs.append(attention.attention_rows_multi(q, [keys] * streams, [values] * streams, srow, counts, None,
                                                   [False] * rows, 0.0625))
        outs.append(gdn.gdn_step_multi(_bf16(rng, rows, PW), [conv] * streams, [ssm] * streams, lengths, conv_w,
                                       a_log, dt, nw, eps, nk=NK, nv=NV, dk=DK, dv=DV)[0])
    mx.eval(*outs)
    mixed = {}
    for name, calls in sizes.items():
        for i in range(len(calls[0])):
            if len({c[i] < inputs.MIN_ELEMENTS for c in calls}) > 1:
                mixed.setdefault(name, []).append(i)
    assert not mixed, mixed
