"""DeepSeek-V4.1's fp32 reference port: RoPE tables, QDQ, mHC, sparse attention, candidates, index-K, DSpark window."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

import dsv41_reference as ref
from dsv41_reference import FP32, MIRROR, Mode, Reference

from tensorfold.families.deepseek_v41.config import Config, yarn_bounds
from tensorfold.families.deepseek_v41.engram_hash import Hasher, TokenMap, primes

FIXTURES = Path(__file__).parent / "fixtures" / "deepseek_v41"
D, V = 64, 64
ROWS = int(primes((1,), 50, 1, 4).sum())        # one Engram layer, 3 columns of primes above 49
INF = float("inf")
TINY = {
    "hidden_size": D, "num_attention_heads": 2, "head_dim": 128, "q_lora_rank": 32, "o_groups": 2, "o_lora_rank": 32,
    "moe_intermediate_size": 64, "n_routed_experts": 4, "num_experts_per_tok": 2, "dspark_n_routed_experts": 4,
    "dspark_num_experts_per_tok": 2, "vocab_size": V, "num_hidden_layers": 5, "num_nextn_predict_layers": 3,
    "compress_ratios": [0, 2, 1, 1, 1, 0, 0, 0], "kv_source_layer_ids": [1, 2], "index_source_layer_ids": [1, 2, 3],
    "candidate_source_layer_id": 2, "candidate_topk_blocks": 2, "candidate_block_size": 4, "index_n_heads": 2,
    "index_head_dim": 128, "index_topk": 4, "engram_layer_ids": [0], "engram_num_embeddings": [ROWS],
    "engram_n_heads": 1, "engram_head_dim": 32, "engram_vocab_size": 50, "engram_compressed_vocab_size": V,
    "dspark_target_layer_ids": [2, 3, 4], "dspark_markov_rank": 16, "dspark_noise_token_id": V - 1,
}   # layers: swa + Engram, full ratio 2, full ratio 1 (candidates), reindex, reuse; three DSpark stages


def _cfg() -> Config:
    raw = json.loads((FIXTURES / "config.json").read_text())
    raw["text_config"].update(TINY)
    return Config.from_dict(raw)


def _weights(cfg: Config, seed: int = 0) -> dict[str, torch.Tensor]:
    """Random fp32 tensors under every checkpoint name the tiny reference reads."""

    gen = torch.Generator().manual_seed(seed)
    t: dict[str, torch.Tensor] = {}

    def lin(name: str, outs: int, ins: int) -> None:
        t[name] = torch.randn(outs, ins, generator=gen) * ins**-0.5

    def norm(name: str, n: int) -> None:
        t[name] = 1 + 0.1 * torch.randn(n, generator=gen)

    hd, hc, inter = cfg.head_dim, cfg.hc_mult, cfg.moe_intermediate_size
    lin("embed.weight", V, D)
    lin("lm_head.weight", V, D)
    norm("norm.weight", D)
    for layer, role in enumerate(cfg.roles):
        p = f"layers.{layer}" if role.mode != "dspark" else f"mtp.{layer - cfg.num_hidden_layers}"
        for kind in ("attn", "ffn"):
            t[f"{p}.hc_{kind}_fn"] = 0.1 * torch.randn((2 + hc) * hc, hc * D, generator=gen)
            t[f"{p}.hc_{kind}_base"] = 0.1 * torch.randn((2 + hc) * hc, generator=gen)
            t[f"{p}.hc_{kind}_scale"] = 0.5 + torch.rand(3, generator=gen)
            norm(f"{p}.{kind}_norm.weight", D)
        t[f"{p}.attn.attn_sink"] = torch.randn(cfg.num_attention_heads, generator=gen)
        lin(f"{p}.attn.wq_a.weight", cfg.q_lora_rank, D)
        norm(f"{p}.attn.q_norm.weight", cfg.q_lora_rank)
        lin(f"{p}.attn.wq_b.weight", cfg.num_attention_heads * hd, cfg.q_lora_rank)
        lin(f"{p}.attn.wkv.weight", hd, D)
        norm(f"{p}.attn.kv_norm.weight", hd)
        lin(f"{p}.attn.wo_a.weight", cfg.o_groups * cfg.o_lora_rank, cfg.num_attention_heads * hd // cfg.o_groups)
        lin(f"{p}.attn.wo_b.weight", D, cfg.o_groups * cfg.o_lora_rank)
        experts = cfg.n_routed_experts if role.mode != "dspark" else cfg.dspark_n_routed_experts
        lin(f"{p}.ffn.gate.weight", experts, D)
        t[f"{p}.ffn.gate.bias"] = 0.1 * torch.randn(experts, generator=gen)
        for name in [f"{p}.ffn.shared_experts"] + [f"{p}.ffn.experts.{e}" for e in range(experts)]:
            lin(f"{name}.w1.weight", inter, D)
            lin(f"{name}.w3.weight", inter, D)
            lin(f"{name}.w2.weight", D, inter)
        if role.kv_src == layer:
            lin(f"{p}.attn.compressor.wkv.weight", hd, D)
            lin(f"{p}.attn.compressor.wgate.weight", hd, D)
            norm(f"{p}.attn.compressor.norm.weight", hd)
            lin(f"{p}.attn.indexer.wk.weight", cfg.index_head_dim, hd)
            norm(f"{p}.attn.indexer.k_norm.weight", cfg.index_head_dim)
        if role.idx_src == layer:
            lin(f"{p}.attn.indexer.wq_b.weight", cfg.index_n_heads * cfg.index_head_dim, cfg.q_lora_rank)
            lin(f"{p}.attn.indexer.weights_proj.weight", cfg.index_n_heads, D)
        if role.engram:
            cols = (cfg.engram_max_ngram_size - 1) * cfg.engram_n_heads
            lin(f"{p}.engram.wkv.weight", D * (hc + 1), cols * cfg.engram_head_dim)
            t[f"{p}.engram.q_weight"] = 1 + 0.1 * torch.randn(hc, D, generator=gen)
            t[f"{p}.engram.k_weight"] = 1 + 0.1 * torch.randn(hc, D, generator=gen)
    lin("mtp.0.main_proj.weight", D, D * len(cfg.dspark_target_layer_ids))
    norm("mtp.0.main_norm.weight", D)
    last = f"mtp.{cfg.num_nextn_predict_layers - 1}"
    norm(f"{last}.norm.weight", D)
    lin(f"{last}.markov_head.embed.weight", V, cfg.dspark_markov_rank)
    lin(f"{last}.markov_head.head.weight", V, cfg.dspark_markov_rank)
    lin(f"{last}.confidence_head.proj.weight", 1, D + cfg.dspark_markov_rank)
    return t


def _reference(mode: Mode = FP32, seed: int = 0) -> Reference:
    cfg = _cfg()
    w = _weights(cfg, seed)
    table = torch.randn(ROWS, cfg.engram_head_dim, generator=torch.Generator().manual_seed(seed + 1))
    hasher = Hasher(cfg, TokenMap(np.arange(V, dtype=np.int32), V))
    return Reference(cfg, w.__getitem__, engram_rows=lambda layer, ids: table[ids], hasher=hasher, mode=mode)


def _model_py_freqs(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow):
    """M:369-389 as written."""

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        def corrected_dim(rotations):
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    freqs = torch.outer(torch.arange(seqlen), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def _bits(z: torch.Tensor) -> torch.Tensor:
    return torch.view_as_real(z).contiguous().view(torch.int32)


@pytest.mark.parametrize("original, base", [(0, 10000.0), (65536, 160000.0)])
def test_rope_tables_equal_model_py(original, base):
    ours = ref.freqs_cis(64, 4096, original, base, 16.0, 32.0, 1.0)
    assert torch.equal(_bits(ours), _bits(_model_py_freqs(64, 4096, original, base, 16.0, 32.0, 1.0)))


def test_yarn_keeps_dims_to_15_and_divides_from_25():
    assert yarn_bounds(64, 65536, 160000.0, 32.0, 1.0) == (15, 25)
    plain = 1.0 / (160000.0 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
    cis = _reference().cis("yarn", torch.tensor([1]))[0]
    assert torch.equal(_bits(cis[:16]), _bits(torch.polar(torch.ones(16), plain[:16])))
    assert torch.equal(_bits(cis[25:]), _bits(torch.polar(torch.ones(7), plain[25:] / 16)))
    assert not torch.isclose(cis[16:25], torch.polar(torch.ones(9), plain[16:25])).any()
    assert torch.equal(_reference().cis("local", torch.arange(7)), ref.freqs_cis(64, 7, 0, 10000.0, 16, 32, 1))


def _block(*values: float, n: int = 32) -> torch.Tensor:
    return torch.tensor(list(values) + [0.0] * (n - len(values)), dtype=torch.float32)


@pytest.mark.parametrize("values, expected", [
    ((), ()),
    ((1e-6, 1e-9), (4 * 2**-22, 2**-30)),           # amax floor 1e-4: s = 2^-22; 4.19 -> 4, 0.0042 -> 2^-8
    ((448.0, 3.0, 4.25, 4.75, 3 * 2**-9), (448.0, 3.0, 4.0, 5.0, 3 * 2**-9)),   # 448 * f32(1/448) == 1: s = 1
    ((450.0, 4.25, 3 * 2**-9), (448.0, 4.0, 2**-7)),   # past 448: s = 2; 225 -> 224, subnormal tie 1.5 -> 2
    ((896.0, -9.0), (896.0, -9.0)),                 # a power-of-two boundary keeps s = 2 (no +1)
    ((900.0, 1.0), (896.0, 1.0)),                   # s = 4: 225 -> 224, 0.25 exact
    ((0.4375, 2**-14), (0.4375, 2**-14)),           # 448 * 2^-10: s = 2^-10
])
def test_act_quant_hand_cases(values, expected):
    out = ref.act_quant(_block(*values))
    assert torch.equal(out, _block(*expected))
    assert ref.act_quant(_block(*values).bfloat16()).dtype == torch.bfloat16


@pytest.mark.parametrize("values, expected", [
    ((), ()),
    ((1e-38,), (2**-126,)),                          # floor 6 * 2^-126: s = 2^-126, 0.85 -> 1
    ((6.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.25, -5.0, 5.9, 0.5),
     (6.0, 0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, -0.0, -4.0, 6.0, 0.5)),   # 6 * f32(1/6) == 1: s = 1; ties to even
    ((6.5, 5.0), (6.0, 4.0)),                        # just past 6: s = 2, 3.25 -> 3, 2.5 -> 2
    ((3.0, 0.375), (3.0, 0.5)),                      # 3 * f32(1/6) == 0.5: s = 0.5, 0.75 -> 1
])
def test_fp4_e8m0_hand_cases(values, expected):
    out = ref.fp4_act_quant(_block(*values), 32)
    assert torch.equal(out, _block(*expected))
    assert torch.equal(torch.signbit(out), torch.signbit(_block(*expected)))


@pytest.mark.parametrize("values, expected", [
    ((), ()),
    ((0.75 * 2**-9,), (2**-9,)),                     # floor 6 * 2^-9: s = e4m3(2^-9), 0.75 -> 1
    ((7.0, 2.8125, 0.5625, 1.4), (6.75, 2.25, 0.5625, 1.125)),   # s = e4m3(7/6) = 1.125; 2.5 -> 2
    ((6.0, 1.25), (6.0, 1.0)),                       # s = 1
])
def test_fp4_e4m3_hand_cases(values, expected):
    assert torch.equal(ref.fp4_act_quant(_block(*values, n=16), 16), _block(*expected, n=16))


def test_e2m1_ties_go_to_even():
    v = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 0.26, 1.3, 4.9, 5.1])
    want = torch.tensor([0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 0.5, 1.5, 4.0, 6.0])
    assert torch.equal(ref.e2m1(v), want) and torch.equal(ref.e2m1(-v), -want)


def test_hc_split_sinkhorn_is_doubly_stochastic():
    gen = torch.Generator().manual_seed(3)
    mixes = torch.randn(50, 24, generator=gen)
    pre, post, comb = ref.hc_split_sinkhorn(mixes, 0.5 + torch.rand(3, generator=gen), torch.randn(24, generator=gen),
                                            4, 20, 1e-6)
    assert torch.allclose(comb.sum(-2), torch.ones(50, 4), atol=1e-5)     # the last of the 20 steps is a column one
    assert torch.allclose(comb.sum(-1), torch.ones(50, 4), atol=1e-2)
    assert bool((pre > 0).all() and (pre <= 1 + 1e-6).all() and (post >= 0).all() and (post <= 2).all())


def test_sparse_attn_with_sink_equals_softmax_in_fp64():
    gen = torch.Generator().manual_seed(4)
    q, kv, sink = torch.randn(4, 128, generator=gen), torch.randn(40, 128, generator=gen), torch.randn(4, generator=gen)
    scale = 128**-0.5
    logits = torch.cat([(q.double() @ kv.double().T) * scale, sink.double()[:, None]], -1)
    want = logits.softmax(-1)[:, :-1] @ kv.double()
    assert torch.allclose(ref.sparse_attn(q, kv, sink, scale, FP32).double(), want, rtol=1e-5, atol=1e-6)
    mirror = ref.sparse_attn(q.bfloat16().float(), kv.bfloat16().float(), sink, scale, MIRROR)
    assert torch.allclose(mirror.double(), want, rtol=2e-2, atol=2e-2)
    assert torch.equal(ref.sparse_attn(q, kv[:0], sink, scale, FP32), torch.zeros_like(q))


def test_candidate_blocks_pin_the_newest_partial_block():
    s = torch.tensor([9.0, 1, 1, 1, 2, 2, 2, 2, 5, 1, 1, 1, 5, 1, 1, 1, 0, 0, -INF, -INF])
    rows = torch.stack([s, torch.where(torch.arange(20) < 5, s, -INF)])
    mask = ref.select_candidate_blocks(rows, torch.tensor([18, 5]), 3, 4)
    # row 0: newest block 4 (positions 16, 17) pinned; then block 0; blocks 2 and 3 tie at 5 -> the lower one
    assert mask[0].tolist() == [i < 4 or 8 <= i < 12 or i >= 16 for i in range(20)]
    # row 1: two reachable blocks only, block 1 (position 4) pinned
    assert mask[1].tolist() == [i < 8 for i in range(20)]


def test_decode_indexer_at_an_even_position_reads_its_own_index_k():
    r = _reference()
    r.forward(list(range(3, 23)))                    # 20 tokens: the next position 20 is even
    st = r.state
    assert len(st.index_k[1]) == 10 and len(st.index_k[2]) == 20
    x = torch.randn(1, D, generator=torch.Generator().manual_seed(5))

    def lists(swap: int | None) -> tuple[list, torch.Tensor]:
        other = copy.deepcopy(r)
        if swap is not None:
            other.state.index_k[swap] = torch.randn_like(other.state.index_k[swap])
        out = other.attention(1, x, 20)
        assert len(other.state.index_k[1]) == 10      # an even position completes no ratio-2 group
        return [t.tolist() for t in other.state.lists], out

    own, out = lists(None)
    alias, alias_out = lists(2)                       # the ratio-1 owner's keys, which model.py's decode would read
    assert alias == own and torch.equal(alias_out, out)
    assert lists(1)[0] != own


@pytest.mark.parametrize("p", [3, 127, 300])
def test_dspark_block_rows_see_the_ring_to_p_and_all_block_rows(p, monkeypatch):
    r = _reference()
    n = r.cfg.num_hidden_layers
    ring = torch.randn(128, r.cfg.head_dim, generator=torch.Generator().manual_seed(p))
    if p < 127:
        ring[p + 1:] = float("nan")                  # slots no position up to p has written
    r.state.ring[n], r.state.pos = ring, p + 1
    calls, real = [], ref.sparse_attn
    monkeypatch.setattr(ref, "sparse_attn", lambda q, kv, *a: calls.append(kv) or real(q, kv, *a))
    out = r.dspark_attention(n, torch.randn(5, D, generator=torch.Generator().manual_seed(9)), p + 1)
    window = ring[[w % 128 for w in range(max(0, p - 127), p + 1)]]
    assert len(calls) == 5 and bool(torch.isfinite(out).all())
    for kv in calls:
        assert len(kv) == len(window) + 5 and torch.equal(kv[:len(window)], window)
        assert torch.equal(kv[len(window):], calls[0][len(window):])


@pytest.mark.parametrize("mode, tol", [(FP32, 1e-4), (MIRROR, 3e-2), (Mode(act_quant=True), 3e-2)])
def test_chunks_at_any_start_match_one_forward(mode, tol):
    tokens = [(5 * i + 1) % V for i in range(40)]
    whole = _reference(mode)
    logits, taps = whole.forward(tokens, all_logits=True)
    chunked = _reference(mode)
    pieces = [chunked.forward(tokens[a:b], all_logits=True)[0] for a, b in ((0, 17), (17, 18), (18, 19), (19, 40))]
    assert bool(torch.isfinite(logits).all()) and taps.shape == (40, 3 * D)
    assert (torch.cat(pieces) - logits).abs().max() <= tol * logits.abs().max()
    for name in ("comp", "index_k", "ring"):
        a, b = getattr(whole.state, name), getattr(chunked.state, name)
        assert a.keys() == b.keys() and all(torch.allclose(a[k], b[k], rtol=tol, atol=tol) for k in a)
    assert whole.state.tail.keys() == chunked.state.tail.keys() == set()   # 40 rows close every ratio-2 group


def test_engram_gate_at_zero_dot_is_copysign_not_sign():
    r = _reference()
    X = torch.zeros(1, 4, D)
    out = r.engram(0, X, torch.tensor([[1, 2, 3]]))
    value = (r.engram_rows(0, torch.tensor([[1, 2, 3]])).flatten(1) @ r.W("layers.0.engram.wkv.weight").T)[:, 4 * D:]
    assert torch.allclose(out, torch.sigmoid(torch.tensor(1e-3)) * value[:, None], rtol=1e-6, atol=1e-7)


def test_dspark_proposes_five_drafts_with_the_markov_chain():
    r = _reference()
    tokens = list(range(1, 31))
    _, taps = r.forward(tokens)
    r.absorb(taps, 0)
    drafts, logits, conf = r.propose(7)
    assert len(drafts) == 5 and logits.shape == (5, V) and conf.shape == (5,)
    assert drafts == [int(torch.argmax(row)) for row in logits]
    assert len(r.state.ring[r.cfg.num_hidden_layers + 2]) == 128
