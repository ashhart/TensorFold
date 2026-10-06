"""Gateless ReLU² EXL3 routed experts: synthetic and local Nemotron-H checkpoint tensors on CUDA."""

import math
import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")


def _projection(k, n, gen, input_scale):
    trellis = torch.randint(-32768, 32768, (k // 16, n // 16, 64), generator=gen,
                            dtype=torch.int32).short().cuda()
    suh = (torch.rand(k, generator=gen) * input_scale + input_scale).half().cuda()
    svh = (torch.rand(n, generator=gen) * 0.4 + 0.8).half().cuda()
    return trellis, suh, svh


def _layer(d=256, padded=256, logical=192, e=2, seed=16):
    g = torch.Generator().manual_seed(seed)
    up = [_projection(d, padded, g, 0.06) for _ in range(e)]
    down = [_projection(padded, d, g, 0.06) for _ in range(e)]
    return up, down


def _hadamard():
    i = torch.arange(128)
    parity = torch.tensor([v.bit_count() & 1 for v in range(128)])
    return torch.where(parity[i[:, None] & i[None, :]].bool(), -1.0, 1.0).cuda().double() / math.sqrt(128)


def _reference(x, ids, up, down, logical, *, mask=True, reconstruct=None):
    """Double-precision dense reference; decode of EXL3's Wq is independent of the routed GEMV."""
    from tensorfold.cuda.exl3 import experts

    h = _hadamard()

    def rot(v):
        return (v.reshape(-1, 128) @ h).reshape(-1)

    def project(v, triple):
        t, suh, svh = triple
        wq = reconstruct(t) if reconstruct else experts.dequant(t, "mul1")
        return rot(rot(v * suh.double()) @ wq.double()) * svh.double()

    y = torch.zeros((*ids.shape, x.shape[-1]), dtype=torch.float64, device="cuda")
    for r in range(ids.shape[0]):
        for s in range(ids.shape[1]):
            e = int(ids[r, s])
            if e >= len(up):
                continue
            u = project(x[r].double(), up[e]).clamp_min(0).square()
            if mask:
                u[logical:] = 0
            y[r, s] = project(u, down[e])
    return y


def test_relu2_masks_padded_intermediate_before_down_projection():
    from tensorfold.cuda.exl3 import experts

    up, down = _layer()
    ex = experts.prepare_nemotron(up, down, "mul1", intermediate_size=192)
    scratch = experts.NemotronScratch(ex, rows=2, slots=2)
    x = (torch.randn((2, 256), generator=torch.Generator().manual_seed(4)) * 0.6).bfloat16().cuda()
    ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device="cuda")
    got = experts.routed_nemotron(x, ids, ex, scratch, R=2).double()
    want = _reference(x, ids, up, down, 192)
    unmasked = _reference(x, ids, up, down, 192, mask=False)
    assert (want - unmasked).abs().max() > want.abs().max() * 0.05
    torch.testing.assert_close(got, want, rtol=0.02, atol=want.abs().max().item() * 0.012)


def test_skipped_ids_are_zero_even_when_scratch_was_previously_used():
    from tensorfold.cuda.exl3 import experts

    up, down = _layer(seed=18)
    ex = experts.prepare_nemotron(up, down, "mul1", intermediate_size=192)
    scratch = experts.NemotronScratch(ex, rows=3, slots=3)
    x = torch.randn((3, 256), generator=torch.Generator().manual_seed(12)).half().cuda()
    ids = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.int32, device="cuda")
    experts.routed_nemotron(x, ids, ex, scratch, R=3)
    ids = torch.tensor([[ex.count, 1, ex.count + 1], [0, ex.count, -1], [1, 0, 1]],
                       dtype=torch.int32, device="cuda")
    actual = experts.routed_nemotron(x, ids, ex, scratch, R=3)
    assert torch.count_nonzero(actual[0, 0]) == 0
    assert torch.count_nonzero(actual[0, 2]) == 0
    assert torch.count_nonzero(actual[1, 1]) == 0
    assert torch.count_nonzero(actual[1, 2]) == 0
    assert torch.isfinite(actual).all() and torch.count_nonzero(actual[0, 1]) > 0


def test_each_row_matches_the_same_row_alone_with_128_experts():
    from tensorfold.cuda.exl3 import experts

    up, down = _layer(e=128, seed=29)
    ex = experts.prepare_nemotron(up, down, "mul1", intermediate_size=192)
    scratch = experts.NemotronScratch(ex, rows=24, slots=7)
    g = torch.Generator().manual_seed(34)
    x = torch.randn((24, 256), generator=g).bfloat16().cuda()
    picks = torch.stack([torch.randperm(128, generator=g)[:6] for _ in range(24)]).int()
    picks = torch.cat((picks, torch.full((24, 1), 128, dtype=torch.int32)), dim=1).cuda().contiguous()
    full = experts.routed_nemotron(x, picks, ex, scratch, 24).clone()
    for r in range(24):
        alone = experts.routed_nemotron(x[r:r + 1], picks[r:r + 1], ex, scratch, 1).clone()
        assert torch.equal(full[r:r + 1], alone), r


def test_routed_nemotron_replays_in_cuda_graph():
    from tensorfold.cuda.exl3 import experts

    up, down = _layer(seed=33)
    ex = experts.prepare_nemotron(up, down, "mul1", intermediate_size=192)
    scratch = experts.NemotronScratch(ex, rows=2, slots=2)
    x = torch.randn((2, 256), device="cuda").half()
    ids = torch.tensor([[0, 2], [1, 0]], dtype=torch.int32, device="cuda")
    experts.routed_nemotron(x, ids, ex, scratch, 2)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = experts.routed_nemotron(x, ids, ex, scratch, 2)
    x.copy_(torch.randn_like(x))
    ids.copy_(torch.tensor([[1, 0], [2, 1]], dtype=torch.int32, device="cuda"))
    want = experts.routed_nemotron(x, ids, ex, scratch, 2).clone()
    captured.fill_(float("nan"))
    graph.replay()
    assert torch.equal(captured, want)


def test_real_quant_tensors_agree_with_exllamav3_reconstruction():
    """Actual calibrated 4bpw layer, logical 1856 / stored 1920, EXL3 decode by ExLlamaV3."""
    from tensorfold.cuda.exl3 import experts

    raw = os.environ.get("TENSORFOLD_NEMOTRON_EXL3_MODEL")
    if not raw:
        pytest.skip("set TENSORFOLD_NEMOTRON_EXL3_MODEL for real checkpoint test")
    assert raw is not None
    root = Path(raw).expanduser()
    files = sorted(root.glob("*.safetensors"))
    if not files:
        pytest.fail(f"calibrated Nemotron-H EXL3 checkpoint absent: {root}")
    safe_open = pytest.importorskip("safetensors").safe_open
    exllamav3_ext = pytest.importorskip("exllamav3.ext").exllamav3_ext
    prefix = "backbone.layers.1.mixer.experts"
    want_keys = {f"{prefix}.{e}.{proj}.{part}" for e in (0, 1) for proj in ("up_proj", "down_proj")
                 for part in ("trellis", "suh", "svh")}
    parts = {}
    for path in files:
        with safe_open(str(path), framework="pt") as f:
            for key in want_keys.intersection(f.keys()):
                parts[key] = f.get_tensor(key)
        if len(parts) == len(want_keys):
            break
    assert parts.keys() == want_keys

    def triples(proj):
        return [tuple(parts[f"{prefix}.{e}.{proj}.{part}"].cuda() for part in ("trellis", "suh", "svh"))
                for e in (0, 1)]

    up, down = triples("up_proj"), triples("down_proj")
    ex = experts.prepare_nemotron(up, down, "mul1", intermediate_size=1856)
    assert (ex.count, ex.dims, ex.width, ex.intermediate_size) == (2, 2688, 1920, 1856)
    scratch = experts.NemotronScratch(ex, rows=2, slots=3)
    x = (torch.randn((2, 2688), generator=torch.Generator().manual_seed(23)) * 0.4).bfloat16().cuda()
    ids = torch.tensor([[0, 1, 2], [1, 0, 3]], dtype=torch.int32, device="cuda")

    def reconstruct(t):
        w = torch.empty((16 * t.shape[0], 16 * t.shape[1]), dtype=torch.float16, device=t.device)
        exllamav3_ext.reconstruct(w, t, 4, False, True)
        return w

    for t, _, _ in up + down:
        assert torch.equal(experts.dequant(t, "mul1"), reconstruct(t))
    got = experts.routed_nemotron(x, ids, ex, scratch, 2).double()
    ref = _reference(x, ids, up, down, 1856, reconstruct=reconstruct)
    assert torch.count_nonzero(got[:, 2]) == 0
    torch.testing.assert_close(got, ref, rtol=0.02, atol=ref.abs().max().item() * 0.012)
