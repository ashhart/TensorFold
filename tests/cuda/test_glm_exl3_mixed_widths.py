"""EXL3 routed experts of mixed widths, as GLM's loader builds them (``experts.prepare``, a width per expert tensor):
every slot's output in a mixed layer equals, bit for bit, its output in a layer whose experts all share its widths.

Synthetic at GLM-5.3-Flash's rank shapes; with ``TENSORFOLD_GLM_EXL3_MIXED=<checkpoint dir>``, also on the first
routed layer of that checkpoint whose experts hold more than one width (skipped otherwise)."""

from __future__ import annotations

import os
import re
from collections import defaultdict
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")

MODEL = os.environ.get("TENSORFOLD_GLM_EXL3_MIXED", "")
LIMIT = 10.0                                              # GLM-5.3-Flash's swiglu_limit
PROJ = ("gate", "up", "down")


def _routed(ex, x, pick):
    from tensorfold.cuda.exl3 import experts

    R, slots = pick.shape
    s = experts.Scratch(ex, R, slots)
    return experts.routed(x, pick, None, ex, s, None, R, LIMIT, act_mode=experts.ACT_BF16).clone()


def _equal_to_uniform_layers(gate, up, down, rows=(1, 8, 64), slots=9, seed=0):
    """Each set of experts sharing (gate, up, down) widths as its own layer: the same bits on the same slots."""

    from tensorfold.cuda.exl3 import experts

    E, D = len(gate), gate[0][0].shape[0] * 16
    mixed = experts.prepare(gate, up, down, "mcg")
    sets = defaultdict(list)
    for e in range(E):
        sets[tuple(experts.k2_of(m[e][0]) for m in (gate, up, down))].append(e)
    assert len(sets) > 1 and mixed.k2_gu[0] < mixed.k2_gu[1]
    g = torch.Generator().manual_seed(seed)
    for R in rows:
        x = torch.randn((R, D), generator=g).to(torch.bfloat16).cuda()
        routed = [torch.randperm(E, generator=g)[:slots - 1] for _ in range(R)]       # E, last: the shared slot
        pick = torch.stack([torch.cat([r, torch.tensor([E])]) for r in routed]).to(torch.int32).cuda()
        want = _routed(mixed, x, pick)
        for k2s, members in sets.items():
            one = experts.prepare(*([m[e] for e in members] for m in (gate, up, down)), "mcg")
            assert one.k2_gu == (min(k2s[:2]), max(k2s[:2])) and one.k2_d == (k2s[2], k2s[2])
            local = torch.full((E + 1,), len(members), dtype=torch.int32, device="cuda")
            local[members] = torch.arange(len(members), dtype=torch.int32, device="cuda")
            got = _routed(one, x, local[pick.long()])
            live = torch.isin(pick.view(-1), torch.tensor(members, dtype=torch.int32, device="cuda"))
            assert torch.equal(got[live].view(torch.int32), want[live].view(torch.int32)), (R, k2s)


def test_synthetic_mixed_layer_equals_its_uniform_layers():
    E, D, I = 288, 4096, 1024                       # GLM-5.3-Flash: 288 routed experts, a rank's 1024 of 2048
    g = torch.Generator().manual_seed(7)
    widths = [(2, 2, 3), (3, 3, 3), (3, 3, 4), (4, 4, 4), (2, 3, 4), (4, 4, 2)]

    def triple(k, n, bits):
        t = torch.randint(-32768, 32768, (k // 16, n // 16, 16 * bits), dtype=torch.int16, generator=g)
        suh = ((torch.rand((k,), generator=g) + 0.5) * 0.02).half()
        svh = ((torch.rand((n,), generator=g) + 0.5) * 0.5).half()
        return t.cuda(), suh.cuda(), svh.cuda()

    gate, up, down = [], [], []
    for e in range(E):
        bg, bu, bd = widths[e % len(widths)]
        gate.append(triple(D, I, bg))
        up.append(triple(D, I, bu))
        down.append(triple(I, D, bd))
    _equal_to_uniform_layers(gate, up, down)


def _tensor(path: Path, name: str) -> torch.Tensor:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cuda") as f:
        return f.get_tensor(name).contiguous()


def _mixed_layer(model: Path):
    """The first routed layer holding more than one expert width: {(proj, e): (trellis, suh, svh)} on the GPU."""

    from tensorfold.cuda.exl3 import format as fmt

    ckpt = fmt.scan(model, read_markers=False)
    where = {name: path.name for path in sorted(model.glob("*.safetensors")) for name in fmt.read_header(path)}
    by_layer = defaultdict(dict)
    for prefix, grp in ckpt.groups.items():
        m = re.search(r"layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj$", prefix)
        if m:
            by_layer[int(m.group(1))][(m.group(3), int(m.group(2)))] = (prefix, grp)
    for layer in sorted(by_layer):
        parts = by_layer[layer]
        if len({grp.bits for _, grp in parts.values()}) > 1:
            out = {}
            for key, (prefix, _) in parts.items():
                names = [f"{prefix}.{p}" for p in ("trellis", "suh", "svh")]
                out[key] = tuple(_tensor(model / where[n], n) for n in names)
            return layer, out
    pytest.skip(f"{model} has no routed layer of mixed widths")


@pytest.mark.skipif(not MODEL or not Path(MODEL).is_dir(), reason="set TENSORFOLD_GLM_EXL3_MIXED to a checkpoint")
def test_checkpoint_mixed_layer_equals_its_uniform_layers():
    _, parts = _mixed_layer(Path(MODEL))
    E = 1 + max(e for _, e in parts)
    gate, up, down = ([parts[(p, e)] for e in range(E)] for p in PROJ)
    _equal_to_uniform_layers(gate, up, down, rows=(1, 8, 64), seed=1)
