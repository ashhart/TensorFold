"""One DSpark stream's CUDA graphs: the draft step and the target's chain verify replay the eager bits."""

import json

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from safetensors.torch import save_file  # noqa: E402

from tensorfold.families.qwen3_5.cuda.chain_graphs import ChainGraphs  # noqa: E402
from tensorfold.families.qwen3_5.cuda.decode import clone_state, prefill  # noqa: E402
from tensorfold.families.qwen3_5.cuda.dspark import DSpark  # noqa: E402
from tensorfold.families.qwen3_5.cuda.forward import tree_forward  # noqa: E402
from tensorfold.families.qwen3_5.cuda.qmm_fast import prepare  # noqa: E402
from tensorfold.families.qwen3_5.cuda.weights import Attention, Config, GDN, Layer, QLinear, Weights  # noqa: E402

H, V, VD, R, INTER, BLOCK = 2048, 512, 256, 256, 512, 8


def _qlinear(gen, n, k):
    words = torch.randint(-(2**31), 2**31 - 1, (n, k // 8), generator=gen, device="cuda",
                          dtype=torch.int64).to(torch.int32)
    scales = (torch.rand(n, k // 64, generator=gen, device="cuda") * 0.003 + 0.001).bfloat16()
    return QLinear(words, scales, (torch.rand(n, k // 64, generator=gen, device="cuda") * 0.003 - 0.0015).bfloat16())


def _drafter(tmp_path):
    gen = torch.Generator().manual_seed(5)

    def r(*shape, s=0.02):
        return (torch.randn(*shape, generator=gen) * s).bfloat16()

    ones = torch.ones(H).bfloat16()
    t = {"embed_tokens.weight": r(V, H, s=0.5), "lm_head.weight": r(VD, H), "fc.weight": r(H, 2 * H),
         "hidden_norm.weight": ones.clone(), "norm.weight": ones.clone(),
         "d2t": torch.randint(0, V - VD, (VD,), generator=gen, dtype=torch.int64),
         "markov_head.markov_w1.weight": r(V, R, s=0.1), "markov_head.markov_w2.weight": r(VD, R, s=0.1),
         "confidence_head.proj.weight": r(1, H + R), "confidence_head.proj.bias": torch.zeros(1).bfloat16()}
    for i in range(2):
        p = f"layers.{i}."
        t |= {p + "self_attn.q_proj.weight": r(1024, H), p + "self_attn.k_proj.weight": r(256, H),
              p + "self_attn.v_proj.weight": r(256, H), p + "self_attn.o_proj.weight": r(H, 1024),
              p + "self_attn.q_norm.weight": torch.ones(128).bfloat16(),
              p + "self_attn.k_norm.weight": torch.ones(128).bfloat16(),
              p + "input_layernorm.weight": ones.clone(), p + "post_attention_layernorm.weight": ones.clone(),
              p + "mlp.gate_proj.weight": r(INTER, H), p + "mlp.up_proj.weight": r(INTER, H),
              p + "mlp.down_proj.weight": r(H, INTER)}
    save_file(t, str(tmp_path / "model.safetensors"))
    layer = {"hidden_size": H, "head_dim": 128, "num_attention_heads": 8, "num_key_value_heads": 2,
             "rms_norm_eps": 1e-6, "rope_theta": 10000000, "num_hidden_layers": 2, "sliding_window": 64,
             "layer_types": ["sliding_attention", "full_attention"]}
    (tmp_path / "config.json").write_text(json.dumps({
        "speculators_model_type": "dspark", "transformer_layer_config": layer, "sample_from_anchor": True,
        "mask_token_id": V - 1, "aux_hidden_state_layer_ids": [1, 2], "block_size": BLOCK}))
    g = torch.Generator(device="cuda").manual_seed(3)
    config = Config(hidden=H, intermediate=INTER, layers=0, heads=8, kv_heads=2, head_dim=128, vocab=V, k_heads=1,
                    v_heads=1, dk=128, dv=128, conv_kernel=4, interval=4, eps=1e-6, rope_dims=32,
                    rope_theta=10000000.0, eos=(0,))
    target = Weights(config, _qlinear(g, V, H), [], torch.ones(H, device="cuda", dtype=torch.bfloat16),
                     _qlinear(g, V, H), torch.ones(16, device="cuda"))
    return DSpark(tmp_path, target)


def test_the_draft_step_graph_replays_the_eager_chain(tmp_path):
    d = _drafter(tmp_path)
    gen = torch.Generator(device="cuda").manual_seed(9)
    for n, pending in ((3, 7), (40, 100), (90, 42), (5, 300)):   # contexts below, near and past the sliding window
        d.restore(([None] * d.layers, [None] * d.layers, 0, 0))
        d.add_taps((torch.randn(n, 2 * H, generator=gen, device="cuda") * 0.5).bfloat16())
        launched = d.launch_blocks([d.snapshot()], [pending], BLOCK)[0]
        eager_ids, eager_nlp = launched[0][0][0].clone(), launched[0][1][0].clone()
        eager = d.finish_tree(launched, d.context_len, BLOCK)[0]
        replayed = d.graph_chain(pending, BLOCK)                    # captured on the first call, replayed after
        assert torch.equal(d.graph["ids"][0], eager_ids), f"ids, context {n}"
        assert torch.equal(d.graph["nlp"][0], eager_nlp), f"scores, context {n}"
        assert replayed == eager


def _model():
    gen = torch.Generator(device="cuda").manual_seed(21)
    norm = torch.ones(128, device="cuda", dtype=torch.bfloat16)

    def q(n, k):
        return _qlinear(gen, n, k)

    gdn = GDN(q(384, 128), q(128, 128), q(1, 128), q(1, 128), q(128, 128),
              torch.randn(384, 4, generator=gen, device="cuda").bfloat16() * 0.1,
              torch.zeros(1, device="cuda"), torch.zeros(1, device="cuda"), norm)
    attn = Attention(q(2 * 2 * 128, 128), q(128, 128), q(128, 128), q(128, 2 * 128), norm, norm)
    layers = [Layer(True, norm, norm, gdn, None, q(128, 128), q(128, 128), q(128, 128)),
              Layer(False, norm, norm, None, attn, q(128, 128), q(128, 128), q(128, 128))]
    config = Config(hidden=128, intermediate=128, layers=2, heads=2, kv_heads=1, head_dim=128, vocab=256,
                    k_heads=1, v_heads=1, dk=128, dv=128, conv_kernel=4, interval=2, eps=1e-6, rope_dims=32,
                    rope_theta=10000000.0, eos=(0,))
    w = Weights(config, q(256, 128), layers, norm, q(256, 128), torch.ones(16, device="cuda"))
    w.tap_layers = (0, 1)
    prepare(w)
    return w


def test_the_chain_verify_graph_replays_the_eager_forward():
    w = _model()
    g = torch.Generator().manual_seed(5)
    st, first = prefill(w, torch.randint(1, 256, (300,), generator=g).tolist(), None)
    runner = ChainGraphs(w, 8192)
    runner.load(st, st.pos + 16)
    for width, seed in ((4, 1), (9, 2), (4, 3), (16, 4)):          # a width seen twice replays its first graph
        tokens = [first] + torch.randint(1, 256, (width - 1,), generator=torch.Generator().manual_seed(seed)).tolist()
        logits, _, taps = runner.verify(tokens)
        logits, taps = logits.clone(), taps.clone()
        ids = torch.tensor(tokens, dtype=torch.int32, device="cuda")
        want, _, want_taps = tree_forward(w, ids, list(range(-1, width - 1)), clone_state(st), capture_taps=True)
        assert torch.equal(logits, want), f"logits, width {width}"
        assert torch.equal(taps, want_taps), f"taps, width {width}"
