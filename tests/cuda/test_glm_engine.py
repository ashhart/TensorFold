"""GLM-5.3-Flash's engine end to end on a tiny synthetic checkpoint (2 layers, 1 KDA + 1 DSA, MoE, MTP head) and a
one-layer synthetic DFlash2 drafter: drafted replies equal serial ones for every policy, including the default that
picks the drafter per round, and a prompt resumed from a kept state gives the reply a fresh prefill gives.

One GPU plays rank 0 of two: its all-gathers hand back two copies of its own partials, a stand-in with real shapes
and fixed bits (the numbers are not the two-rank model's, the equalities are the engine's)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.glm5_next.cuda import split  # noqa: E402

D, V, S = 512, 1024, 4
MOE = 256                 # expert width: each rank's half (128) a whole EXL3 Hadamard block
CONFIG = {
    "model_type": "glm5_next",
    "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
    "text_config": {
        "hidden_size": D, "num_hidden_layers": 2, "vocab_size": V, "rms_norm_eps": 1e-5,
        "num_attention_heads": 2, "q_lora_rank": 128, "kv_lora_rank": 128, "qk_nope_head_dim": 256,
        "qk_rope_head_dim": 0, "v_head_dim": 256,
        "linear_attn_config": {"num_heads": 2, "head_dim": 128, "short_conv_kernel_size": 4, "gate_lower_bound": -5.0},
        "n_routed_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": MOE, "n_shared_experts": 1,
        "intermediate_size": 256, "routed_scaling_factor": 2.5, "norm_topk_prob": True, "hc_mult": S,
        "hc_sinkhorn_iters": 20, "hc_eps": 1e-6, "index_n_heads": 2, "index_head_dim": 128, "index_topk": 2048,
        "index_kpool": 4, "swiglu_limit": 10.0, "layer_types": ["linear_attention", "full_attention"],
        "mlp_layer_types": ["dense", "sparse"], "eos_token_id": [1000], "num_nextn_predict_layers": 1,
    },
}


def _checkpoint(path, exl3: bool = False, mtp: bool = True) -> None:
    """The synthetic model as an MLX 4-bit checkpoint, or with ``exl3`` as an EXL3 one: routed experts as trellis
    tiles with their scales, every other weight BF16 (the layout of Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw).
    ``mtp=False`` leaves out the MTP layer (the last tensors written, so the others keep their values)."""

    rng = np.random.default_rng(3)
    tensors: list[tuple[str, str, list[int], np.ndarray]] = []

    def bf16(name: str, shape: list[int], scale: float = 0.05, offset: float = 0.0) -> None:
        x = torch.tensor(rng.standard_normal(shape) * scale + offset, dtype=torch.float32).to(torch.bfloat16)
        tensors.append((name, "BF16", shape, x.view(torch.uint16).numpy().view(np.uint8).reshape(-1)))

    def f32(name: str, shape: list[int], scale: float = 0.1, offset: float = 0.0) -> None:
        x = (rng.standard_normal(shape) * scale + offset).astype(np.float32)
        tensors.append((name, "F32", shape, x.view(np.uint8).reshape(-1)))

    def q4(name: str, n: int, k: int, scale: float = 0.01) -> None:
        if exl3:                  # the BF16 weight whose 4-bit version the MLX layout stores
            bf16(name + ".weight", [n, k], 4.6 * scale)
            return
        words = rng.integers(0, 2**32, size=(n, k // 8), dtype=np.uint64).astype(np.uint32)
        tensors.append((name + ".weight", "U32", [n, k // 8], words.view(np.uint8).reshape(-1)))
        bf16(name + ".scales", [n, k // 64], 0.0, scale)
        bf16(name + ".biases", [n, k // 64], 0.0, -7.5 * scale)

    def trellis(name: str, n: int, k: int) -> None:
        t = rng.integers(-2**15, 2**15, size=(k // 16, n // 16, 64)).astype(np.int16)
        tensors.append((name + ".trellis", "I16", [k // 16, n // 16, 64], t.view(np.uint8).reshape(-1)))
        for part, size, sc in (("suh", k, 0.03), ("svh", n, 0.03)):
            v = (rng.standard_normal(size) * sc).astype(np.float16)
            tensors.append((name + "." + part, "F16", [size], v.view(np.uint8).reshape(-1)))
        tensors.append((name + ".mcg", "I32", [1], np.array([0xCBAC1FED], dtype=np.uint32).view(np.uint8)))

    def dsa(p: str) -> None:
        q4(p + "self_attn.q_a_proj", 128, D)
        q4(p + "self_attn.kv_a_proj_with_mqa", 128, D)
        bf16(p + "self_attn.q_a_layernorm.weight", [128], 0.05, 1.0)
        bf16(p + "self_attn.kv_a_layernorm.weight", [128], 0.05, 1.0)
        q4(p + "self_attn.q_b_proj", 2 * 256, 128)
        q4(p + "self_attn.kv_b_proj", 2 * 512, 128)
        q4(p + "self_attn.o_proj", D, 2 * 256)
        q4(p + "self_attn.indexer.wk", 128, D)
        q4(p + "self_attn.indexer.weights_proj", 2, D)
        q4(p + "self_attn.indexer.wq_b", 2 * 128, 128)
        bf16(p + "self_attn.indexer.k_norm.weight", [128], 0.05, 1.0)
        bf16(p + "self_attn.indexer.k_norm.bias", [128])
        bf16(p + "self_attn.indexer.index_kpool_compress_gate", [128, D])
        bf16(p + "self_attn.indexer.index_kpool_compress_ape", [4, 128])

    def moe(p: str) -> None:
        bf16(p + "mlp.gate.weight", [8, D])
        f32(p + "mlp.gate.e_score_correction_bias", [8], 0.01)
        for e in [f"experts.{i}" for i in range(8)] + ["shared_experts"]:
            if exl3 and e != "shared_experts":
                trellis(p + f"mlp.{e}.gate_proj", MOE, D)
                trellis(p + f"mlp.{e}.up_proj", MOE, D)
                trellis(p + f"mlp.{e}.down_proj", D, MOE)
                continue
            q4(p + f"mlp.{e}.gate_proj", MOE, D)
            q4(p + f"mlp.{e}.up_proj", MOE, D)
            q4(p + f"mlp.{e}.down_proj", D, MOE)

    L = "model.language_model."
    q4(L + "embed_tokens", V, D, 0.02)
    bf16(L + "norm.weight", [D], 0.05, 1.0)
    q4("lm_head", V, D)
    for i in (0, 1):
        p = f"{L}layers.{i}."
        bf16(p + "input_layernorm.weight", [D], 0.05, 1.0)
        bf16(p + "post_attention_layernorm.weight", [D], 0.05, 1.0)
        for site in ("attn", "ffn"):
            bf16(p + f"hc_{site}_fn", [24, S * D], 0.01)
            f32(p + f"hc_{site}_base", [24])
            f32(p + f"hc_{site}_scale", [3], 0.1, 1.0)
    p = L + "layers.0.self_attn."
    for x in "qkv":
        q4(p + f"{x}_proj", 256, D)
        bf16(p + f"{x}_conv1d.weight", [256, 1, 4], 0.3)
    q4(p + "f_a_proj", 128, D)
    q4(p + "g_a_proj", 128, D)
    q4(p + "b_proj", 2, D)
    q4(p + "f_b_proj", 256, 128)
    q4(p + "g_b_proj", 256, 128)
    f32(p + "A_log", [2], 0.5)
    f32(p + "dt_bias", [256], 0.5)
    bf16(p + "o_norm.weight", [128], 0.05, 1.0)
    q4(p + "o_proj", D, 256)
    q4(L + "layers.0.mlp.gate_proj", 256, D)
    q4(L + "layers.0.mlp.up_proj", 256, D)
    q4(L + "layers.0.mlp.down_proj", D, 256)
    dsa(L + "layers.1.")
    moe(L + "layers.1.")
    if mtp:
        m = L + "layers.2."
        bf16(m + "enorm.weight", [D], 0.05, 1.0)
        bf16(m + "hnorm.weight", [D], 0.05, 1.0)
        q4(m + "eh_proj", D, 2 * D)
        bf16(m + "shared_head.norm.weight", [D], 0.05, 1.0)
        bf16(m + "input_layernorm.weight", [D], 0.05, 1.0)
        bf16(m + "post_attention_layernorm.weight", [D], 0.05, 1.0)
        dsa(m)
        moe(m)
    path.mkdir(parents=True, exist_ok=True)
    split.write(str(path / "model-00001-of-00001.safetensors"), tensors, {"format": "mlx"})
    config = json.loads(json.dumps(CONFIG))
    if not mtp:
        config["text_config"]["num_nextn_predict_layers"] = 0
    if exl3:
        del config["quantization"]
        config["quantization_config"] = {"quant_method": "exl3", "bits": 4, "codebook": "mcg", "head_bits": 16}
    (path / "config.json").write_text(json.dumps(config))


DRAFT = {
    "hidden_size": D, "head_dim": 128, "num_attention_heads": 8, "num_key_value_heads": 2, "rms_norm_eps": 1e-5,
    "rope_parameters": {"rope_theta": 10000.0}, "sliding_window": 2048, "is_causal": False,
    "intermediate_size": 256, "num_hidden_layers": 1,
    "dflash_config": {"mask_token_id": 1001, "conv_group_size": 16, "conv_kernel_size": 2, "block_size": 8,
                      "selector_rank": 16, "selector_top_k": 8, "target_layer_ids": [0, 1]},
}


def _drafter(path) -> None:
    rng = np.random.default_rng(4)
    tensors: list[tuple[str, str, list[int], np.ndarray]] = []

    def bf16(name: str, shape: list[int], scale: float = 0.05, offset: float = 0.0) -> None:
        x = torch.tensor(rng.standard_normal(shape) * scale + offset, dtype=torch.float32).to(torch.bfloat16)
        tensors.append((name, "BF16", shape, x.view(torch.uint16).numpy().view(np.uint8).reshape(-1)))

    H, KV, hd, inter = 8, 2, 128, 256
    bf16("fc.weight", [D, 2 * D], 0.03)
    bf16("hidden_norm.weight", [D], 0.05, 1.0)
    bf16("norm.weight", [D], 0.05, 1.0)
    bf16("candidate_selector.hidden_projection.weight", [16, D], 0.05)
    bf16("candidate_selector.predecessor_codebook", [V, 16], 0.3)
    bf16("candidate_selector.successor_codebook", [V, 16], 0.3)
    p = "layers.0."
    bf16(p + "self_attn.q_proj.weight", [H * hd, D], 0.03)
    bf16(p + "self_attn.k_proj.weight", [KV * hd, D], 0.03)
    bf16(p + "self_attn.v_proj.weight", [KV * hd, D], 0.03)
    bf16(p + "self_attn.o_proj.weight", [D, H * hd], 0.03)
    bf16(p + "self_attn.q_norm.weight", [hd], 0.05, 1.0)
    bf16(p + "self_attn.k_norm.weight", [hd], 0.05, 1.0)
    bf16(p + "mlp.gate_proj.weight", [inter, D], 0.03)
    bf16(p + "mlp.up_proj.weight", [inter, D], 0.03)
    bf16(p + "mlp.down_proj.weight", [D, inter], 0.03)
    for conv in ("attention_conv", "mlp_conv"):
        bf16(p + conv + ".base_kernel", [2, 2, D], 0.1, 0.5)
        bf16(p + conv + ".kernel_projection.weight", [4 * D // 16, D], 0.02)
    bf16(p + "input_layernorm.weight", [D], 0.05, 1.0)
    bf16(p + "post_attention_layernorm.weight", [D], 0.05, 1.0)
    path.mkdir(parents=True, exist_ok=True)
    split.write(str(path / "model.safetensors"), tensors, {"format": "pt"})
    (path / "config.json").write_text(json.dumps(DRAFT))


class _TwoCopies:
    """Rank 0 of two on one GPU: every all-gather returns this rank's input twice."""

    rank, world = 0, 2

    def all_gather(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        n = send.numel()
        flat = recv.view(-1)
        flat[:n].copy_(send.reshape(-1))
        flat[n:2 * n].copy_(send.reshape(-1))

    def barrier(self) -> None:
        torch.cuda.synchronize()


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    path = tmp_path_factory.mktemp("glm")
    _checkpoint(path)
    return GlmEngine(path, rank=0, master="", port=0, comm=_TwoCopies())


@pytest.fixture(scope="module")
def engine_f(tmp_path_factory):
    """The same model with the DFlash2 drafter loaded, so the default policy chooses between two drafters."""

    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    path = tmp_path_factory.mktemp("glm_f")
    _checkpoint(path / "model")
    _drafter(path / "dflash2")
    return GlmEngine(path / "model", rank=0, master="", port=0, drafter=path / "dflash2", comm=_TwoCopies())


@pytest.fixture(scope="module")
def engine_x(tmp_path_factory):
    """The model as an EXL3 checkpoint (trellis experts, BF16 elsewhere), with the drafter."""

    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    path = tmp_path_factory.mktemp("glm_x")
    _checkpoint(path / "model", exl3=True)
    _drafter(path / "dflash2")
    return GlmEngine(path / "model", rank=0, master="", port=0, drafter=path / "dflash2", comm=_TwoCopies())


def _generate(engine, prompt, sampling, *, draft=True, policy=None, tokens=24):
    out: list[int] = []
    engine.request.policy = policy
    engine.request.stop_eos = False
    stats = engine.generate(list(prompt), tokens, sampling, lambda new: out.extend(new), draft=draft)
    return out, stats


def _state(e) -> list[torch.Tensor]:
    """What a prompt leaves: the KDA states and conv windows, the attention and MTP cache rows below the position."""

    st = e.st
    return [st.rec[st.cur[0]], st.conv] + [x[:st.pos] for x in st.kc + st.vc] + [st.mtp_kc[:st.mtp_len],
                                                                                 st.mtp_vc[:st.mtp_len]]


def test_prompt_chunks_leave_the_same_state(engine):
    """Any prompt chunking leaves bit-identical states and first tokens; decode windows land within bf16 rounding."""

    from tensorfold.families.glm5_next.cuda.decode import Engine, prefill
    from tensorfold.families.glm5_next.cuda.forward import commit, forward

    w = engine.w
    prompt = [int(t) for t in np.random.default_rng(5).integers(0, 1000, size=100)]
    ref = Engine(w, capacity=2560, max_rows=8, prefill_rows=100)
    first = prefill(ref, prompt, None)
    want = [t.clone() for t in _state(ref)]
    prefill(ref, prompt, None, mtp=False)               # the head would reuse the logits buffer
    logits = ref.pbuf.logits[:1].float().clone()
    for rows in (7, 16):
        e = Engine(w, capacity=2560, max_rows=8, prefill_rows=rows)
        assert prefill(e, prompt, None) == first, rows
        assert all(torch.equal(a, b) for a, b in zip(_state(e), want)), rows
    dec = Engine(w, capacity=2560, max_rows=8, prefill_rows=16)
    dec.reset()
    for s0 in range(0, len(prompt), 8):
        chunk = prompt[s0:s0 + 8]
        last = forward(w, dec.st, dec.buf, chunk)[len(chunk) - 1:len(chunk)].float()
        commit(w, dec.st, dec.buf, len(chunk), len(chunk))
    a, b = want[0], dec.st.rec[dec.st.cur[0]]
    assert float((a - b).abs().max()) <= 2e-2 * float(b.abs().max())
    ka, kb = want[2].float(), dec.st.kc[0][:len(prompt)].float()
    assert float((ka - kb).abs().max()) <= 2e-2 * float(kb.abs().max())
    assert float(torch.nn.functional.cosine_similarity(logits, last, dim=1)) > 0.999


@pytest.mark.parametrize("sampling", [Sampling(1234, 1.0, 20, 0.95), None], ids=["sampled", "greedy"])
def test_drafted_replies_equal_serial(engine, sampling):
    prompt = list(np.random.default_rng(5).integers(0, 1000, size=37))
    serial, stats = _generate(engine, prompt, sampling, draft=False)
    assert len(serial) == 24 and stats["drafts"] is False
    for policy in (None, "auto", "1", "2", "3", "c3:0.35", "a:0.6:0.85"):
        drafted, stats = _generate(engine, prompt, sampling, policy=policy)
        assert drafted == serial, policy
        assert stats["rounds"] >= 1 and stats["min_rows"] >= 2, (policy, stats)     # every round a window


@pytest.mark.parametrize("sampling", [Sampling(1234, 1.0, 20, 0.95), None], ids=["sampled", "greedy"])
def test_drafter_choice_equals_serial(engine_f, sampling):
    """Every policy with both drafters loaded, and the per-round choice made to switch often."""

    prompt = list(np.random.default_rng(6).integers(0, 1000, size=41))
    serial, _ = _generate(engine_f, prompt, sampling, draft=False, tokens=40)
    seen = set()
    from tensorfold.families.glm5_next.cuda.engine import encode_policy

    assert engine_f._effective(encode_policy("auto")) == encode_policy("auto")        # MLX weights: the choice
    for policy in (None, "auto:1:2:0", "auto:1:1:0", "auto:2:3:0.5", "f3", "fc5:0.3", "2", "c3:0.35"):
        drafted, stats = _generate(engine_f, prompt, sampling, policy=policy, tokens=40)
        assert drafted == serial, policy
        seen.update(stats.get("drafters", ""))
    assert seen == {"m", "f"}


def test_drafter_choice_resumes(engine_f):
    """After a reply from both drafters, a prompt carrying it resumes from the prompt's end with every drafter."""

    sampling = Sampling(11, 1.0, 20, 0.95)
    rng = np.random.default_rng(12)
    first = list(rng.integers(0, 1000, size=30))
    unrelated = list(rng.integers(0, 1000, size=9))
    reply, stats = _generate(engine_f, first, sampling, policy="auto:1:1:0", tokens=30)
    assert set(stats["drafters"]) == {"m", "f"}
    after = first + reply + [21, 22]
    for policy in ("auto:1:1:0", "auto", "2", "f3"):
        warm, stats = _generate(engine_f, after, sampling, policy=policy)
        assert stats["cached"] == len(first), policy
        _generate(engine_f, unrelated, sampling)
        cold, stats = _generate(engine_f, after, sampling, policy=policy)
        assert stats["cached"] == 0 and warm == cold, policy
        _generate(engine_f, first, sampling, policy="auto:1:1:0", tokens=30)      # the prompt's state again


@pytest.mark.parametrize("sampling", [Sampling(7, 1.0, 20, 0.95), None], ids=["sampled", "greedy"])
def test_resumed_prompts_equal_fresh_prefills(engine, sampling):
    rng = np.random.default_rng(9)
    first = list(rng.integers(0, 1000, size=70))
    unrelated = list(rng.integers(0, 1000, size=12))
    reply, _ = _generate(engine, first, sampling)
    after_reply = first + reply + [5, 6, 7]
    warm, stats = _generate(engine, after_reply, sampling)
    assert stats["cached"] == len(first)                # the reply prefills again
    _generate(engine, unrelated, sampling)              # a fresh prefill: every kept state goes
    cold, stats = _generate(engine, after_reply, sampling)
    assert stats["cached"] == 0 and warm == cold
    _generate(engine, first, sampling)
    after_prompt = first + [11, 12, 13]
    warm, stats = _generate(engine, after_prompt, sampling, policy="2")
    assert stats["cached"] == len(first)
    _generate(engine, unrelated, sampling)
    cold, stats = _generate(engine, after_prompt, sampling, policy="2")
    assert stats["cached"] == 0 and warm == cold
    serial, _ = _generate(engine, after_prompt, sampling, draft=False)
    assert serial == cold


@pytest.mark.parametrize("sampling", [Sampling(4321, 1.0, 20, 0.95), None], ids=["sampled", "greedy"])
def test_exl3_checkpoint_drafted_equals_serial(engine_x, sampling):
    """An EXL3 checkpoint through the same engine: every policy's reply equals serial decoding."""

    from tensorfold.families.glm5_next.cuda.engine import EXL3_AUTO, encode_policy
    from tensorfold.families.glm5_next.cuda.exl3_mm import Exl3Experts

    assert isinstance(engine_x.w.layers[1].moe.experts, Exl3Experts) and engine_x.w.layers[1].moe.shared is not None
    assert engine_x._effective(encode_policy("auto")) == encode_policy(EXL3_AUTO)       # the default drafts DFlash2
    assert engine_x._effective(encode_policy("auto:1:1:0")) == encode_policy("auto:1:1:0")
    prompt = list(np.random.default_rng(8).integers(0, 1000, size=45))
    serial, _ = _generate(engine_x, prompt, sampling, draft=False, tokens=32)
    for policy in (None, "auto:1:1:0", "f3", "fc5:0.3", "2", "c3:0.35", "a:0.6:0.85"):
        drafted, stats = _generate(engine_x, prompt, sampling, policy=policy, tokens=32)
        assert drafted == serial, policy


def test_exl3_checkpoint_resumes(engine_x):
    sampling = Sampling(21, 1.0, 20, 0.95)
    rng = np.random.default_rng(22)
    first = list(rng.integers(0, 1000, size=70))
    unrelated = list(rng.integers(0, 1000, size=9))
    reply, _ = _generate(engine_x, first, sampling, policy="auto:1:1:0", tokens=20)
    after = first + reply + [31, 32]
    warm, stats = _generate(engine_x, after, sampling)
    assert stats["cached"] == len(first)
    _generate(engine_x, unrelated, sampling)
    cold, stats = _generate(engine_x, after, sampling)
    assert stats["cached"] == 0 and warm == cold


def test_exl3_prompt_chunks_past_128_rows_leave_the_same_state(engine_x):
    """An EXL3 checkpoint's 300-row prompt chunk leaves the bits 64-row chunks leave, and drafted equals serial after
    it; 0.3.5.1's BF16 projections refused any prompt chunk past 128 rows."""

    from tensorfold.families.glm5_next.cuda.decode import Engine, prefill

    prompt = [int(t) for t in np.random.default_rng(128).integers(0, 1000, size=300)]
    runs = []
    for rows in (2048, 64):
        e = Engine(engine_x.w, capacity=2560, max_rows=8, prefill_rows=rows)
        runs.append((prefill(e, prompt, None), [t.clone() for t in _state(e)]))
        del e
    (a, want), (b, got) = runs
    assert a == b and all(torch.equal(x, y) for x, y in zip(want, got))
    sampling = Sampling(29, 1.0, 20, 0.95)
    serial, _ = _generate(engine_x, prompt, sampling, draft=False)
    drafted, _ = _generate(engine_x, prompt, sampling)
    assert drafted == serial


@pytest.fixture(scope="module")
def engine_n(tmp_path_factory):
    """A checkpoint without the MTP head, with the drafter: every policy drafts with DFlash2."""

    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    path = tmp_path_factory.mktemp("glm_n")
    _checkpoint(path / "model", mtp=False)
    _drafter(path / "dflash2")
    with pytest.raises(ValueError, match="no MTP head"):
        GlmEngine(path / "model", rank=0, master="", port=0, comm=_TwoCopies())
    return GlmEngine(path / "model", rank=0, master="", port=0, drafter=path / "dflash2", comm=_TwoCopies())


@pytest.mark.parametrize("sampling", [Sampling(77, 1.0, 20, 0.95), None], ids=["sampled", "greedy"])
def test_no_mtp_head_drafts_with_dflash2(engine_n, sampling):
    """Without the MTP head the default and every MTP spec run as DFlash2 drafts: still windows, still serial's
    tokens, never one token a round."""

    from tensorfold.families.glm5_next.cuda.engine import DFLASH_POLICY, encode_policy

    assert engine_n.w.mtp is None
    assert engine_n._effective(encode_policy("auto")) == encode_policy(DFLASH_POLICY)
    assert engine_n._effective(encode_policy("2")) == encode_policy("f2")
    assert engine_n._effective(encode_policy("0")) == encode_policy("0")
    prompt = list(np.random.default_rng(13).integers(0, 1000, size=39))
    serial, _ = _generate(engine_n, prompt, sampling, draft=False, tokens=32)
    for policy in (None, "auto", "2", "c3:0.35", "a:0.6:0.85", "f3"):
        drafted, stats = _generate(engine_n, prompt, sampling, policy=policy, tokens=32)
        assert drafted == serial, policy
        assert stats["min_rows"] >= 2 and "m" not in stats.get("drafters", ""), (policy, stats)


@pytest.fixture(scope="module")
def engine_long(tmp_path_factory):
    """The model with a context past the dense limit (2,051 tokens), so rows attend to DSA-selected tokens."""

    from tensorfold.families.glm5_next.cuda.engine import GlmEngine

    path = tmp_path_factory.mktemp("glm_long")
    _checkpoint(path)
    return GlmEngine(path, rank=0, master="", port=0, comm=_TwoCopies(), context=2600)


@pytest.mark.parametrize("sampling", [Sampling(99, 1.0, 20, 0.95), None], ids=["sampled", "greedy"])
def test_drafted_replies_equal_serial_past_the_dense_limit(engine_long, sampling):
    """Past 2,051 tokens a window's rows must score like serial steps (2 index heads, not GLM-5.3-Flash's 32)."""

    prompt = list(np.random.default_rng(11).integers(0, 1000, size=2100))
    serial, _ = _generate(engine_long, prompt, sampling, draft=False, tokens=32)
    for policy in (None, "2", "c3:0.35"):
        drafted, _ = _generate(engine_long, prompt, sampling, policy=policy, tokens=32)
        assert drafted == serial, policy


def test_long_prompt_chunks_leave_the_same_state(engine_long):
    """Past the dense limit, 2,048-row prompt chunks leave the bits 64-row chunks leave; 0.3.5.1's sparse attention
    lost rows 128 and up of a chunk, and GLM answered "!!!!" past 2,051 tokens (#53)."""

    from tensorfold.families.glm5_next.cuda.decode import Engine, prefill

    prompt = [int(t) for t in np.random.default_rng(53).integers(0, 1000, size=2400)]
    runs = []
    for rows in (2048, 64):
        e = Engine(engine_long.w, capacity=2600, max_rows=8, prefill_rows=rows, long_context=True)
        first = prefill(e, prompt, None)
        index = []
        for i, (ik, ig, pk) in enumerate(e.st.index):           # the MTP layer's indexer caches come last
            k = e.st.mtp_len if i == len(e.st.index) - 1 else e.st.pos
            index += [ik[:k], ig[:k], pk[:k // 4]]
        runs.append((first, [t.clone() for t in _state(e) + index]))
        del e
    (a, want), (b, got) = runs
    assert a == b
    assert len(want) == len(got) and all(torch.equal(x, y) for x, y in zip(want, got))
