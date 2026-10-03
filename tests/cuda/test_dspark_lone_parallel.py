"""A lone DSpark stream under --parallel: its rounds replay the chain graphs and decode the eager rounds' tokens."""

import json

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from safetensors.torch import save_file  # noqa: E402

from tensorfold.cuda.streams import Stream  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen3_5.cuda.chain_graphs import ChainGraphs  # noqa: E402
from tensorfold.families.qwen3_5.cuda.decode import prefill, serial_decode  # noqa: E402
from tensorfold.families.qwen3_5.cuda.dspark import DSpark  # noqa: E402
from tensorfold.families.qwen3_5.cuda.multi import MultiDecoder  # noqa: E402
from tensorfold.families.qwen3_5.cuda.qmm_fast import prepare  # noqa: E402
from tensorfold.families.qwen3_5.cuda.weights import Attention, Config, GDN, Layer, QLinear, Weights  # noqa: E402

H, V, VD, R, INTER, BLOCK, CONTEXT, COUNT = 2048, 512, 256, 256, 512, 8, 1024, 40


def _qlinear(gen, n, k):
    words = torch.randint(-(2**31), 2**31 - 1, (n, k // 8), generator=gen, device="cuda",
                          dtype=torch.int64).to(torch.int32)
    scales = (torch.rand(n, k // 64, generator=gen, device="cuda") * 0.003 + 0.001).bfloat16()
    return QLinear(words, scales, (torch.rand(n, k // 64, generator=gen, device="cuda") * 0.003 - 0.0015).bfloat16())


def _target():
    gen = torch.Generator(device="cuda").manual_seed(21)
    norm = torch.ones(H, device="cuda", dtype=torch.bfloat16)
    hnorm = torch.ones(128, device="cuda", dtype=torch.bfloat16)

    def q(n, k):
        return _qlinear(gen, n, k)

    gdn = GDN(q(384, H), q(128, H), q(1, H), q(1, H), q(H, 128),
              torch.randn(384, 4, generator=gen, device="cuda").bfloat16() * 0.1,
              torch.zeros(1, device="cuda"), torch.zeros(1, device="cuda"), hnorm)
    attn = Attention(q(2 * 2 * 128, H), q(128, H), q(128, H), q(H, 2 * 128), hnorm, hnorm)
    layers = [Layer(True, norm, norm, gdn, None, q(INTER, H), q(INTER, H), q(H, INTER)),
              Layer(False, norm, norm, None, attn, q(INTER, H), q(INTER, H), q(H, INTER))]
    config = Config(hidden=H, intermediate=INTER, layers=2, heads=2, kv_heads=1, head_dim=128, vocab=V,
                    k_heads=1, v_heads=1, dk=128, dv=128, conv_kernel=4, interval=2, eps=1e-6, rope_dims=32,
                    rope_theta=10000000.0, eos=(0,))
    w = Weights(config, q(V, H), layers, norm, q(V, H), torch.ones(16, device="cuda"))
    w.tap_layers = (0, 1)
    prepare(w)
    return w


def _drafter(tmp_path, target):
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
    return DSpark(tmp_path, target)


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    w = _target()
    return w, _drafter(tmp_path_factory.mktemp("dspark"), w)


def _decode(w, d, requests, chain):
    """Each request's tokens from one decoder (streams admitted together), and the rows its chain graphs verified."""

    dec = MultiDecoder(w, d, context=CONTEXT)
    verified: list[int] = []
    if chain:
        dec.chain = ChainGraphs(w, CONTEXT + 32)
        real = dec.chain.verify
        dec.chain.verify = lambda tokens: verified.append(len(tokens)) or real(tokens)
    outs = []
    for prompt, sampling, count in requests:
        got: list[int] = []
        dec.admit(Stream(prompt, count, sampling, draft=True, emit=lambda new, got=got: got.extend(new)))
        outs.append(got)
    while dec.live():
        dec.finish(dec.round())
    return outs, verified


def _serial(w, prompt, sampling, count=COUNT):
    st, first = prefill(w, prompt, sampling)
    return serial_decode(w, st, first, count, sampling).tokens


PROMPT = torch.randint(1, V, (60,), generator=torch.Generator().manual_seed(3)).tolist()


@pytest.mark.parametrize("sampling", [None, Sampling(1234, 1.0, 20, 0.95)])
def test_a_lone_stream_replays_the_chain_graphs_with_the_eager_tokens(pair, sampling):
    w, d = pair
    (eager,), none = _decode(w, d, [(PROMPT, sampling, COUNT)], chain=False)
    (chained,), verified = _decode(w, d, [(PROMPT, sampling, COUNT)], chain=True)
    assert verified, "the lone stream's rounds did not replay the chain graphs"
    assert none == []
    assert chained == eager == _serial(w, PROMPT, sampling)


def test_the_stream_left_alone_takes_the_graphs_and_both_keep_their_serial_tokens(pair):
    w, d = pair
    requests = [(PROMPT, None, COUNT), (PROMPT[:20] + [7, 8, 9], Sampling(99, 0.8, 0, 1.0), 8)]
    outs, verified = _decode(w, d, requests, chain=True)
    assert outs == [_serial(w, p, s, n) for p, s, n in requests]
    assert verified, "the stream left alone did not replay the chain graphs"
