"""A tiny DSpine fixture from Tencent's dspine_mlx (bridge, serial tables, greedy tail): fixture.py OUT.safetensors."""
import sys
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn

from dspine_mlx.bridge import DSpineCodeBridge, build_serial_tables
from dspine_mlx.propose import RefinementTail
from dspine_mlx.target import HeadAccess

H, C, R, V, T, K, L = 128, 64, 64, 96, 6, 4, 2
mx.random.seed(7)


def rnd(*shape, scale=1.0):
    return (mx.random.normal(shape) * scale).astype(mx.bfloat16)


cfg = SimpleNamespace(hidden_size=H, num_hidden_layers=L, block_size=T, vocab_size=V, feature_layers=[1, 2],
                      bridge=SimpleNamespace(code_size=C, rank=R, eps=1e-6))
b = DSpineCodeBridge(cfg)
b.prev.weight = rnd(R, C, scale=0.2)
b.gate.weight = rnd(R, C, scale=0.2)
b.gate.bias = rnd(R, scale=0.5)
b.predecessor_gate.weight = rnd(R, C, scale=0.2)
b.out.weight = mx.zeros((C, R), dtype=mx.bfloat16)
b.read_weight = rnd(C, H, scale=0.1)
b.early_read_weights = rnd(L - 1, C, H, scale=0.1)
b.feature_write_weights = rnd(L, H, R, scale=0.05)
b.codebook = rnd(V, C, scale=0.15)
b.independent_write_version = mx.array(1, dtype=mx.int64)
b.terminal_disabled_version = mx.array(1, dtype=mx.int64)
b.prepare()
norm0 = nn.RMSNorm(H, eps=1e-6)
norm0.weight = rnd(H, scale=0.3) + 1
norm1 = nn.RMSNorm(H, eps=1e-6)
norm1.weight = rnd(H, scale=0.3) + 1
anchor = 17
ids = mx.array([[anchor] + [3] * (T - 1)], dtype=mx.int32)
x0 = rnd(1, T, H, scale=2.0)
normed0, out0 = b(x0, None, ids, 0, norm0)  # an early layer's write
x1 = rnd(1, T, H, scale=2.0)
stash = {}
rows, out1 = b(x1, None, ids, L - 1, norm1, stash)  # the last write, stashed
head = nn.Linear(H, V, bias=False)
head.weight = rnd(V, H, scale=0.2)
access = HeadAccess(head)
tg, tsv = build_serial_tables(head_rows=access.rows_range, vocab_size=V, norm_w=norm1.weight,
                              write_w=b.feature_write_weights[-1], pv_w=b.packed_weight[R:], codebook=b.codebook, chunk=40)
draft = SimpleNamespace(head_topk=K, selected_prefix_bridge=b, norm=norm1)
tail = RefinementTail(draft, access, tg, tsv)
rows1 = rows[0, 1:]
drafts, cand = tail.greedy(rows1, stash, anchor)
logits = access.matmul(rows1)
mx.save_safetensors(sys.argv[1], {
    "prev": b.prev.weight, "gate": b.gate.weight, "gate_bias": b.gate.bias, "pred": b.predecessor_gate.weight,
    "read_last": b.read_weight, "read_early": b.early_read_weights, "write": b.feature_write_weights,
    "codebook": b.codebook, "norm0": norm0.weight, "norm1": norm1.weight, "head": head.weight,
    "anchor": mx.array([anchor], dtype=mx.int32), "x0": x0[0], "x1": x1[0], "normed0": normed0[0], "out0": out0[0],
    "normed1": rows[0], "out1": out1[0], "stash_raw": stash["raw"], "stash_rms": stash["rms"].astype(mx.float32),
    "stash_packed": stash["packed"], "tg": tg, "tsv": tsv, "logits": logits, "candidates": cand, "drafts": drafts})
print("drafts", drafts.tolist(), "cand", cand.tolist())
