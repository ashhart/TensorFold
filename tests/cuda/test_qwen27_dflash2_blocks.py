"""DFlash2 blocks of several streams in one pass give each stream the candidates of its own block pass."""

import json

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from safetensors.torch import save_file  # noqa: E402

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen3_5.cuda.dflash2 import DFlash2  # noqa: E402
from tensorfold.families.qwen3_5.cuda.weights import Config, QLinear, Weights  # noqa: E402

H, V, RANK, INTER = 2048, 512, 512, 512


def _qlinear(gen, n, k):
    words = torch.randint(-(2**31), 2**31 - 1, (n, k // 8), generator=gen, device="cuda",
                          dtype=torch.int64).to(torch.int32)
    scales = (torch.rand(n, k // 64, generator=gen, device="cuda") * 0.003 + 0.001).bfloat16()
    return QLinear(words, scales, (torch.rand(n, k // 64, generator=gen, device="cuda") * 0.003 - 0.0015).bfloat16())


def _drafter(tmp_path):
    gen = torch.Generator().manual_seed(5)

    def r(*shape, s=0.02):
        return (torch.randn(*shape, generator=gen) * s).bfloat16()

    t = {"candidate_selector.hidden_projection.weight": r(RANK, H),
         "candidate_selector.predecessor_codebook": r(V, RANK, s=0.1),
         "candidate_selector.successor_codebook": r(V, RANK, s=0.1),
         "fc.weight": r(H, 5 * H), "hidden_norm.weight": torch.ones(H).bfloat16(), "norm.weight": torch.ones(H).bfloat16()}
    for i in range(2):
        p = f"layers.{i}."
        t |= {p + "self_attn.q_proj.weight": r(1024, H), p + "self_attn.k_proj.weight": r(256, H),
              p + "self_attn.v_proj.weight": r(256, H), p + "self_attn.o_proj.weight": r(H, 1024),
              p + "self_attn.q_norm.weight": torch.ones(128).bfloat16(),
              p + "self_attn.k_norm.weight": torch.ones(128).bfloat16(),
              p + "input_layernorm.weight": torch.ones(H).bfloat16(),
              p + "post_attention_layernorm.weight": torch.ones(H).bfloat16(),
              p + "mlp.gate_proj.weight": r(INTER, H), p + "mlp.up_proj.weight": r(INTER, H),
              p + "mlp.down_proj.weight": r(H, INTER)}
        for conv in ("attention_conv", "mlp_conv"):
            t[p + conv + ".base_kernel"] = r(2, 2, H, s=0.5)
            t[p + conv + ".kernel_projection.weight"] = r(4 * H // 16, H)
    save_file(t, str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps({
        "hidden_size": H, "head_dim": 128, "num_attention_heads": 8, "num_key_value_heads": 2, "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 10000000}, "num_hidden_layers": 2, "sliding_window": 64, "is_causal": False,
        "dflash_config": {"mask_token_id": V - 1, "conv_group_size": 16}}))
    g = torch.Generator(device="cuda").manual_seed(3)
    config = Config(hidden=H, intermediate=INTER, layers=0, heads=8, kv_heads=2, head_dim=128, vocab=V, k_heads=1,
                    v_heads=1, dk=128, dv=128, conv_kernel=4, interval=4, eps=1e-6, rope_dims=32,
                    rope_theta=10000000.0, eos=(0,))
    target = Weights(config, _qlinear(g, V, H), [], torch.ones(H, device="cuda", dtype=torch.bfloat16),
                     _qlinear(g, V, H), torch.ones(16, device="cuda"))
    return DFlash2(tmp_path, target)


def test_blocks_of_several_streams_equal_their_own(tmp_path):
    d = _drafter(tmp_path)
    gen = torch.Generator(device="cuda").manual_seed(9)
    snaps = []
    for n in (3, 40, 90):               # contexts shorter than, near and past the 63-row window
        d.restore(([None] * d.layers, [None] * d.layers, 0, 0))
        d.add_taps((torch.randn(n, 5 * H, generator=gen, device="cuda") * 0.5).bfloat16())
        snaps.append(d.snapshot())
    empty = ([None] * d.layers, [None] * d.layers, 0, 0)
    pendings = [7, 100, 42, 5]
    alone = []
    for snap, pending in zip(snaps, pendings):
        d.restore(snap)
        alone.append(d.launch_block(pending, 11))
    together = d.launch_blocks(snaps[:1] + [empty] + snaps[1:], [pendings[0], 9] + pendings[1:3], 11)
    assert together[1] is None
    sampling = Sampling(1234, 1.0, 20, 0.95)
    for a, b in zip(alone, [together[0]] + together[2:]):
        ids_a, floats_a = (t[a[1]:a[1] + a[2]] for t in a[0][:2])
        ids_b, floats_b = (t[b[1]:b[1] + b[2]] for t in b[0][:2])
        # the 16 candidates of each depth, in the order of their ids (``topk`` leaves them unsorted)
        order_a, order_b = ids_a.argsort(dim=1), ids_b.argsort(dim=1)
        assert torch.equal(ids_a.gather(1, order_a), ids_b.gather(1, order_b))
        assert torch.equal(floats_a[:, :16].gather(1, order_a), floats_b[:, :16].gather(1, order_b))
        assert torch.equal(floats_a[:, 16:], floats_b[:, 16:])
        assert d.finish_tree(a, 50, 11, sampling) == d.finish_tree(b, 50, 11, sampling)


def test_taps_of_several_streams_equal_their_own(tmp_path):
    d = _drafter(tmp_path)
    gen = torch.Generator(device="cuda").manual_seed(4)
    snaps, taps = [], []
    for n, extra in ((3, 2), (70, 5), (20, 1)):
        d.restore(([None] * d.layers, [None] * d.layers, 0, 0))
        d.add_taps((torch.randn(n, 5 * H, generator=gen, device="cuda") * 0.5).bfloat16())
        snaps.append(d.snapshot())
        taps.append((torch.randn(extra, 5 * H, generator=gen, device="cuda") * 0.5).bfloat16())
    alone = []
    for snap, t in zip(snaps, taps):
        d.restore(snap)
        d.add_taps(t)
        alone.append(d.snapshot())
    together = d.add_taps_streams(snaps, taps)
    for a, b in zip(alone, together):
        assert a[2:] == b[2:]
        assert all(torch.equal(x, y) for x, y in zip(a[0] + a[1], b[0] + b[1]))


def test_skipped_rows_leave_the_context_as_adding_them(tmp_path):
    """Prefill gives the drafter taps only for the rows its window keeps: skipping the rest ends in the same context."""

    d = _drafter(tmp_path)
    gen = torch.Generator(device="cuda").manual_seed(9)
    taps = (torch.randn(200, 5 * H, generator=gen, device="cuda") * 0.1).bfloat16()
    empty = ([None] * d.layers, [None] * d.layers, 0, 0)
    d.restore(empty)
    d.add_taps(taps[:40])
    d.add_taps(taps[40:])
    kc, vc, n, end = d.snapshot()
    d.restore(empty)
    d.add_taps(taps[:40])
    d.skip(200 - d.window - 40)
    d.add_taps(taps[200 - d.window:])
    kc2, vc2, n2, end2 = d.snapshot()
    assert (n, end) == (n2, end2) == (d.window, 200)
    assert all(torch.equal(a, b) for a, b in zip(kc + vc, kc2 + vc2))
