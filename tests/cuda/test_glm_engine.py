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
CONFIG = {
    "model_type": "glm5_next",
    "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
    "text_config": {
        "hidden_size": D, "num_hidden_layers": 2, "vocab_size": V, "rms_norm_eps": 1e-5,
        "num_attention_heads": 2, "q_lora_rank": 128, "kv_lora_rank": 128, "qk_nope_head_dim": 256,
        "qk_rope_head_dim": 0, "v_head_dim": 256,
        "linear_attn_config": {"num_heads": 2, "head_dim": 128, "short_conv_kernel_size": 4, "gate_lower_bound": -5.0},
        "n_routed_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 128, "n_shared_experts": 1,
        "intermediate_size": 256, "routed_scaling_factor": 2.5, "norm_topk_prob": True, "hc_mult": S,
        "hc_sinkhorn_iters": 20, "hc_eps": 1e-6, "index_n_heads": 2, "index_head_dim": 128, "index_topk": 2048,
        "index_kpool": 4, "swiglu_limit": 10.0, "layer_types": ["linear_attention", "full_attention"],
        "mlp_layer_types": ["dense", "sparse"], "eos_token_id": [1000], "num_nextn_predict_layers": 1,
    },
}


def _checkpoint(path) -> None:
    rng = np.random.default_rng(3)
    tensors: list[tuple[str, str, list[int], np.ndarray]] = []

    def bf16(name: str, shape: list[int], scale: float = 0.05, offset: float = 0.0) -> None:
        x = torch.tensor(rng.standard_normal(shape) * scale + offset, dtype=torch.float32).to(torch.bfloat16)
        tensors.append((name, "BF16", shape, x.view(torch.uint16).numpy().view(np.uint8).reshape(-1)))

    def f32(name: str, shape: list[int], scale: float = 0.1, offset: float = 0.0) -> None:
        x = (rng.standard_normal(shape) * scale + offset).astype(np.float32)
        tensors.append((name, "F32", shape, x.view(np.uint8).reshape(-1)))

    def q4(name: str, n: int, k: int, scale: float = 0.01) -> None:
        words = rng.integers(0, 2**32, size=(n, k // 8), dtype=np.uint64).astype(np.uint32)
        tensors.append((name + ".weight", "U32", [n, k // 8], words.view(np.uint8).reshape(-1)))
        bf16(name + ".scales", [n, k // 64], 0.0, scale)
        bf16(name + ".biases", [n, k // 64], 0.0, -7.5 * scale)

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
            q4(p + f"mlp.{e}.gate_proj", 128, D)
            q4(p + f"mlp.{e}.up_proj", 128, D)
            q4(p + f"mlp.{e}.down_proj", D, 128)

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
    (path / "config.json").write_text(json.dumps(CONFIG))


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


def _generate(engine, prompt, sampling, *, draft=True, policy=None, tokens=24):
    out: list[int] = []
    engine.request.policy = policy
    engine.request.stop_eos = False
    stats = engine.generate(list(prompt), tokens, sampling, lambda new: out.extend(new), draft=draft)
    return out, stats


@pytest.mark.parametrize("sampling", [Sampling(1234, 1.0, 20, 0.95), None], ids=["sampled", "greedy"])
def test_drafted_replies_equal_serial(engine, sampling):
    prompt = list(np.random.default_rng(5).integers(0, 1000, size=37))
    serial, stats = _generate(engine, prompt, sampling, draft=False)
    assert len(serial) == 24 and stats["drafts"] is False
    for policy in (None, "auto", "1", "2", "3", "c3:0.35", "a:0.6:0.85"):
        drafted, stats = _generate(engine, prompt, sampling, policy=policy)
        assert drafted == serial, policy
        assert stats["rounds"] >= 1


@pytest.mark.parametrize("sampling", [Sampling(1234, 1.0, 20, 0.95), None], ids=["sampled", "greedy"])
def test_drafter_choice_equals_serial(engine_f, sampling):
    """Every policy with both drafters loaded, and the per-round choice made to switch often."""

    prompt = list(np.random.default_rng(6).integers(0, 1000, size=41))
    serial, _ = _generate(engine_f, prompt, sampling, draft=False, tokens=40)
    seen = set()
    for policy in (None, "auto:1:2:0", "auto:1:1:0", "auto:2:3:0.5", "f3", "fc5:0.3", "2", "c3:0.35"):
        drafted, stats = _generate(engine_f, prompt, sampling, policy=policy, tokens=40)
        assert drafted == serial, policy
        seen.update(stats.get("drafters", ""))
    assert seen == {"m", "f"}


def test_drafter_choice_resumes(engine_f):
    """A reply drafted with both drafters leaves a state that both drafters resume from."""

    sampling = Sampling(11, 1.0, 20, 0.95)
    rng = np.random.default_rng(12)
    first = list(rng.integers(0, 1000, size=30))
    unrelated = list(rng.integers(0, 1000, size=9))
    reply, stats = _generate(engine_f, first, sampling, policy="auto:1:1:0", tokens=30)
    assert set(stats["drafters"]) == {"m", "f"}
    after = first + reply + [21, 22]
    for policy in ("auto:1:1:0", "auto", "2", "f3"):
        warm, stats = _generate(engine_f, after, sampling, policy=policy)
        assert stats["cached"] >= len(first) + len(reply) - 1, policy
        _generate(engine_f, unrelated, sampling)
        cold, stats = _generate(engine_f, after, sampling, policy=policy)
        assert stats["cached"] == 0 and warm == cold, policy
        _generate(engine_f, first, sampling, policy="auto:1:1:0", tokens=30)      # the state after the reply again


@pytest.mark.parametrize("sampling", [Sampling(7, 1.0, 20, 0.95), None], ids=["sampled", "greedy"])
def test_resumed_prompts_equal_fresh_prefills(engine, sampling):
    rng = np.random.default_rng(9)
    first = list(rng.integers(0, 1000, size=70))       # more than one 64-row prefill chunk
    unrelated = list(rng.integers(0, 1000, size=12))
    reply, _ = _generate(engine, first, sampling)
    after_reply = first + reply + [5, 6, 7]
    warm, stats = _generate(engine, after_reply, sampling)
    assert stats["cached"] >= len(first) + len(reply) - 1
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
