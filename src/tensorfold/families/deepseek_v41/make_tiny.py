"""A tiny synthetic DeepSeek-V4.1-Flash checkpoint in the converted source layout (Q3 affine g64 + BF16 exceptions).

Every tensor name, dtype and shape follows the layout dossier: no ``model.`` prefix, ``layers.N`` blocks,
uint32-packed affine ``.weight`` + ``.scales``/``.biases`` triples (bits 3, group 64), BF16 exceptions for
norms / sinks / hc (flat ``hc_attn_fn`` names) / gate / engram q,k weights / compressor / indexer.wk / markov
heads, engram tables with per-row scales, and one MTP stage under ``mtp.0``. Small dims but the real ratios
and modes (both compress ratios, one engram layer, one MTP stage). This seeds the family's test suite
(plan s6.1). ``hidden_size`` and every quantized ``ins`` stay multiples of 64 so the affine format fits.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx

BITS, GROUP = 3, 64

D, VOCAB = 128, 320
HEADS, HEAD_DIM, ROPE = 8, 64, 16
Q_RANK, O_RANK, O_GROUPS = 64, 64, 2
INTER, EXPERTS, TOPK = 64, 6, 2
ENG_DIM, ENG_HEADS, ENG_NGRAM = 64, 3, 4
MTP_INTER, MTP_EXPERTS, MTP_TOPK = 64, 4, 2
MARKOV_RANK = 64
MTP_STAGES = 3               # num_nextn_predict_layers: mtp.0/1/2, the taps 37-39 pattern in miniature

# layer 0 window-only, 1 engram + ratio 2 (a kv/index source), 2 ratio 2 reuse, 3 ratio 1 kv/index/candidate
# source, 4 ratio 1 reindex, 5 ratio 1 reuse; mtp.0/1/2 window-only. DSpark taps at layers 4, 5.
RATIOS = [0, 2, 2, 1, 1, 1]
KV_SOURCES = [1, 3]
INDEX_SOURCES = [1, 3, 4]
CANDIDATE_SOURCE = 3
ENGRAM_LAYER = 1
ENGRAM_VOCAB = 400          # engram_vocab_size: primes drawn above this
PRIMES = []                 # filled by prime_layout(): 3 n-gram orders x 3 heads, drawn above ENGRAM_VOCAB
TAPS = [4, 5]               # dspark_target_layer_ids


def _primes() -> list[int]:
    """The 9 smallest unused primes above ``engram_vocab_size - 1``, drawn in (order, head) order."""

    from tensorfold.families.deepseek_v41.engram import find_next_prime

    seen: set[int] = set()
    primes: list[int] = []
    for _ in range(ENG_NGRAM - 1):
        current = ENGRAM_VOCAB - 1
        for _ in range(ENG_HEADS):
            current = find_next_prime(current, seen)
            seen.add(current)
            primes.append(current)
    return primes


def TEXT_dict() -> dict:
    return {
        "model_type": "deepseek_v41_text", "hidden_size": D, "num_hidden_layers": 6, "vocab_size": VOCAB,
        "rms_norm_eps": 1e-20, "num_attention_heads": HEADS, "num_key_value_heads": 1, "head_dim": HEAD_DIM,
        "qk_rope_head_dim": ROPE, "q_lora_rank": Q_RANK, "o_lora_rank": O_RANK, "o_groups": O_GROUPS,
        "sliding_window": 16, "compress_ratios": RATIOS, "kv_source_layer_ids": KV_SOURCES,
        "index_source_layer_ids": INDEX_SOURCES, "candidate_source_layer_id": CANDIDATE_SOURCE,
        "candidate_block_size": 4, "candidate_topk_blocks": 3, "rope_theta": 10000.0,
        "compress_rope_theta": 160000.0,
        "rope_scaling": {"type": "yarn", "factor": 16, "original_max_position_embeddings": 65536,
                         "beta_fast": 32, "beta_slow": 1},
        "index_n_heads": 2, "index_head_dim": 64, "index_topk": 6, "n_routed_experts": EXPERTS,
        "n_shared_experts": 1, "num_experts_per_tok": TOPK, "moe_intermediate_size": INTER,
        "norm_topk_prob": True,
        "scoring_func": "sqrtsoftplus", "scoring_topk_method": "noaux_tc", "routed_scaling_factor": 1.5,
        "swiglu_limit": 10.0, "hc_mult": 4, "hc_eps": 1e-6, "hc_sinkhorn_iters": 20,
        "num_nextn_predict_layers": MTP_STAGES, "dspark_target_layer_ids": TAPS, "dspark_block_size": 5,
        "dspark_noise_token_id": 300, "dspark_num_experts": MTP_EXPERTS,
        "dspark_experts_per_token": MTP_TOPK, "dspark_markov_rank": MARKOV_RANK,
        "engram_layer_ids": [ENGRAM_LAYER], "engram_num_embeddings": [sum(_primes())],
        "engram_max_ngram_size": ENG_NGRAM,
        "engram_vocab_size": ENGRAM_VOCAB, "engram_compressed_vocab_size": VOCAB,
        "engram_pad_token_id": 2, "engram_n_heads": ENG_HEADS, "engram_head_dim": ENG_DIM,
        "eos_token_id": 1, "bos_token_id": 0,
    }


def TOP_dict() -> dict:
    return {"model_type": "deepseek_v41", "text_config": TEXT_dict(), "architectures":
            ["DeepseekV41ForCausalLM"],
            "quantization": {"bits": BITS, "group_size": GROUP, "mode": "affine"},
            "eos_token_id": 1, "bos_token_id": 0, "pad_token_id": 2, "image_token_id": 319}


def _q(t: dict, name: str, outs: int, ins: int, scale: float = 0.08, bits: int = BITS) -> None:
    if ins % GROUP:
        raise ValueError(f"{name}: {ins} inputs do not fit groups of {GROUP}")
    w = (scale * mx.random.normal((outs, ins))).astype(mx.bfloat16)
    if bits:
        t[f"{name}.weight"], t[f"{name}.scales"], t[f"{name}.biases"] = mx.quantize(w, group_size=GROUP, bits=bits)
    else:
        t[f"{name}.weight"] = w


def _norm(t: dict, name: str, n: int) -> None:
    t[name] = (1.0 + 0.1 * mx.random.normal((n,))).astype(mx.bfloat16)


def _hc(t: dict, prefix: str, mixes: int) -> None:
    t[f"{prefix}_fn"] = (0.02 * mx.random.normal((mixes, 4 * D))).astype(mx.float32)
    t[f"{prefix}_base"] = (0.1 * mx.random.normal((mixes,))).astype(mx.float32)
    t[f"{prefix}_scale"] = (1.0 + 0.1 * mx.random.normal((3,))).astype(mx.float32)


def _block(t: dict, p: str, ratio: int, kv_source: bool, index_source: bool, engram: bool, c: dict,
           experts: int, topk: int) -> None:
    _hc(t, f"{p}.hc_attn", 24)
    _hc(t, f"{p}.hc_ffn", 24)
    _norm(t, f"{p}.attn_norm.weight", D)
    _norm(t, f"{p}.ffn_norm.weight", D)
    a = f"{p}.attn"
    _q(t, f"{a}.wq_a", c["q_lora_rank"], D)
    _norm(t, f"{a}.q_norm.weight", c["q_lora_rank"])
    _q(t, f"{a}.wq_b", HEADS * HEAD_DIM, c["q_lora_rank"])
    _q(t, f"{a}.wkv", HEAD_DIM, D)
    _norm(t, f"{a}.kv_norm.weight", HEAD_DIM)
    _q(t, f"{a}.wo_a", O_GROUPS * O_RANK, HEADS * HEAD_DIM // O_GROUPS)
    _q(t, f"{a}.wo_b", D, O_GROUPS * O_RANK)
    t[f"{a}.attn_sink"] = (0.5 * mx.random.normal((HEADS,))).astype(mx.float32)
    if kv_source:
        cp = f"{a}.compressor"
        _q(t, f"{cp}.wkv", HEAD_DIM, D, 0.3, bits=0)              # BF16 exception
        if ratio > 1:
            _q(t, f"{cp}.wgate", HEAD_DIM, D, 0.3, bits=0)        # BF16 exception, promoted to fp32
        _norm(t, f"{cp}.norm.weight", HEAD_DIM)
    if index_source:
        ix = f"{a}.indexer"
        _q(t, f"{ix}.wq_b", c["index_n_heads"] * c["index_head_dim"], c["q_lora_rank"], 0.3)
        _q(t, f"{ix}.weights_proj", c["index_n_heads"], D, 0.3)
        if kv_source:
            _q(t, f"{ix}.wk", c["index_head_dim"], HEAD_DIM, 0.3, bits=0)        # BF16 exception
            _norm(t, f"{ix}.k_norm.weight", c["index_head_dim"])
    f = f"{p}.ffn"
    t[f"{f}.gate.weight"] = (0.1 * mx.random.normal((experts, D))).astype(mx.bfloat16)
    t[f"{f}.gate.bias"] = (0.05 * mx.random.normal((experts,))).astype(mx.float32)
    t[f"{f}.gate.bias_vl"] = (0.05 * mx.random.normal((experts,))).astype(mx.float32)
    for e in range(experts):
        for part in ("w1", "w3", "w2"):
            outs = c["moe_intermediate_size"] if part != "w2" else D
            ins = D if part != "w2" else c["moe_intermediate_size"]
            _q(t, f"{f}.experts.{e}.{part}", outs, ins)
    _q(t, f"{f}.shared_experts.w1", c["moe_intermediate_size"], D)
    _q(t, f"{f}.shared_experts.w3", c["moe_intermediate_size"], D)
    _q(t, f"{f}.shared_experts.w2", D, c["moe_intermediate_size"])
    if engram:
        g = f"{p}.engram"
        rows = c["engram_num_embeddings"][0]
        _q(t, f"{g}.embed", rows, c["engram_head_dim"], 0.1)
        _q(t, f"{g}.wkv", (1 + 4) * D, ENG_HEADS * (ENG_NGRAM - 1) * c["engram_head_dim"], 0.05)
        t[f"{g}.q_weight"] = (0.1 * mx.random.normal((4, D))).astype(mx.bfloat16)     # BF16 exceptions
        t[f"{g}.k_weight"] = (0.1 * mx.random.normal((4, D))).astype(mx.bfloat16)


def _layer(t: dict, i: int, c: dict) -> None:
    _block(t, f"layers.{i}", RATIOS[i], i in KV_SOURCES, i in INDEX_SOURCES, i == ENGRAM_LAYER, c,
           c["n_routed_experts"], c["num_experts_per_tok"])


def _mtp_stage0(t: dict, c: dict, last: int) -> None:
    """Stage 0 plus the heads, drawn in the M1/M2 checkpoint's exact order (its RNG stream is the spec).

    The heads belong to the LAST stage (official: ``n_mtp_layers - 1``); the draws keep the M1/M2
    sequence — block, main_proj, main_norm, norm, markov embed, markov head, confidence — so every
    backbone and mtp.0 tensor stays bit-identical to the pre-M3 checkpoint.
    """
    _block(t, "mtp.0", 0, False, False, False, c, c["dspark_num_experts"], c["dspark_experts_per_token"])
    _q(t, "mtp.0.main_proj", D, len(TAPS) * D, 0.05)
    _norm(t, "mtp.0.main_norm.weight", D)
    _norm(t, f"mtp.{last}.norm.weight", D)
    _q(t, f"mtp.{last}.markov_head.embed", VOCAB, MARKOV_RANK, 0.05)    # per-row affine, gathered on lookup
    _q(t, f"mtp.{last}.markov_head.head", VOCAB, MARKOV_RANK, 0.05)
    t[f"mtp.{last}.confidence_head.proj.weight"] = (0.05 * mx.random.normal(
        (1, D + MARKOV_RANK))).astype(mx.bfloat16)


def _mtp_extra(t: dict, c: dict, last: int) -> None:
    """Stages 1 .. last: plain DSpark blocks appended after the backbone's draws (heads sit on ``last``)."""
    for stage in range(1, last + 1):
        _block(t, f"mtp.{stage}", 0, False, False, False, c, c["dspark_num_experts"],
               c["dspark_experts_per_token"])


def make_arrays(seed: int = 7) -> dict[str, mx.array]:
    mx.random.seed(seed)
    text = TEXT_dict()
    t: dict[str, mx.array] = {}
    for i in range(text["num_hidden_layers"]):
        _layer(t, i, text)
    # the M1/M2 draw order through the root norm is kept bit-for-bit; the extra stages extend it
    _mtp_stage0(t, text, MTP_STAGES - 1)
    _q(t, "embed", VOCAB, D, 0.06)
    _q(t, "head", VOCAB, D, 0.06)
    _norm(t, "norm.weight", D)
    _mtp_extra(t, text, MTP_STAGES - 1)
    return t


def write_checkpoint(folder: Path, seed: int = 7) -> Path:
    """The tiny checkpoint: shards + config.json (+ the 3 MTP stages in the same layout, drafting from M3)."""
    folder.mkdir(parents=True, exist_ok=True)
    arrays = make_arrays(seed)
    mx.save_safetensors(str(folder / "model.safetensors"), arrays)
    index = {name: "model.safetensors" for name in arrays}
    (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
    (folder / "config.json").write_text(json.dumps(TOP_dict(), indent=1))
    return folder


if __name__ == "__main__":
    import sys

    out = write_checkpoint(Path(sys.argv[1] if len(sys.argv) > 1 else "ds41-tiny"))
    print(f"wrote {out}")
