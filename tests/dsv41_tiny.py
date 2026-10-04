"""A tiny DeepSeek-V4.1-Flash checkpoint in the pack's layout: its names, dtypes, shards, index, config and tokenizer."""

from __future__ import annotations

import copy
import json
import math
import struct
from pathlib import Path

import numpy as np
import pytest

from tensorfold.families.deepseek_v41.engram_hash import primes

PACK_CONFIG = Path(__file__).parent / "fixtures" / "deepseek_v41" / "config.json"
VOCAB = 1024
# 0-1 as the pack; a pair that normalizes alike sits before the pad, so the pad's Engram id is not its token id
FIXED = ("<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>", "A", "a", "<｜▁pad▁｜>", "<unk>", "<｜User｜>",
         "<｜Assistant｜>", "<｜System｜>", "<｜latest_reminder｜>", "<think>", "</think>", "<｜deepseek_image｜>",
         "｜DSML｜")
SPECIAL = FIXED[:2] + FIXED[4:6]
TWINS = 64                  # words w0..w63 also appear capitalized: Engram's token map folds each pair
PAD = FIXED.index("<｜▁pad▁｜>")
TEXT = {
    "vocab_size": VOCAB, "hidden_size": 512, "moe_intermediate_size": 256, "num_hidden_layers": 8,
    "num_attention_heads": 4, "head_dim": 512, "qk_rope_head_dim": 64, "q_lora_rank": 256, "o_lora_rank": 256,
    "o_groups": 2, "n_routed_experts": 8, "num_experts_per_tok": 2, "n_shared_experts": 1,
    # swa, swa+Engram, full r2, reuse r2+Engram, full r1 (candidates), reuse r1, reindex r1, reuse r1; 3 DSpark stages
    "compress_ratios": [0, 0, 2, 2, 1, 1, 1, 1, 0, 0, 0], "kv_source_layer_ids": [2, 4],
    "index_source_layer_ids": [2, 4, 6], "candidate_source_layer_id": 4, "index_n_heads": 4, "index_head_dim": 128,
    "index_topk": 16, "candidate_topk_blocks": 4, "candidate_block_size": 8, "engram_layer_ids": [1, 3],
    "engram_vocab_size": 1000, "engram_head_dim": 64, "engram_pad_token_id": PAD,
    "engram_compressed_vocab_size": VOCAB - 1 - TWINS, "dspark_target_layer_ids": [5, 6, 7],
    "dspark_markov_rank": 64, "dspark_n_routed_experts": 8, "dspark_num_experts_per_tok": 2,
    "dspark_noise_token_id": FIXED.index("<｜System｜>"),
}
SHARDS = ("embed", "layers 0-3", "layers 4-7", "head", "dspark", "engram 1", "engram 3")
MCG = 0xCBAC1FED - (1 << 32)        # the mcg codebook's multiplier, as the pack's signed I32 marker
E4M3_RMS = 100                      # about the RMS of e4m3 values drawn from every non-NaN byte alike
E2M1_RMS = 2.9


def vocabulary() -> list[str]:
    """The 1024 tokens in id order."""

    words = VOCAB - len(FIXED) - TWINS
    return [*FIXED, *(f"w{i}" for i in range(words)), *(f"W{i}" for i in range(TWINS))]


def config() -> dict:
    """config.json: the pack's, with the tiny dims, roles, vocabulary and Engram tables."""

    out = copy.deepcopy(json.loads(PACK_CONFIG.read_text()))
    out.pop("vision_config")
    t = out["text_config"]
    t.update(TEXT)
    sizes = primes(tuple(t["engram_layer_ids"]), t["engram_vocab_size"], t["engram_n_heads"],
                   t["engram_max_ngram_size"])
    t["engram_num_embeddings"] = [int(n) for n in sizes.sum(axis=(1, 2))]
    out.update(pad_token_id=PAD, image_token_id=FIXED.index("<｜deepseek_image｜>"))
    out["quantization_config"]["mtp_experts_start_layer"] = t["num_hidden_layers"]
    return out


def write_tokenizer(path: Path) -> None:
    """A WordLevel tokenizer.json over ``vocabulary()``, the chat markers matched before whitespace splitting."""

    from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers

    tokens = vocabulary()
    tok = Tokenizer(models.WordLevel({t: i for i, t in enumerate(tokens)}, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.add_special_tokens([AddedToken(t, normalized=False) for t in SPECIAL])
    tok.add_tokens([AddedToken(t, normalized=False) for t in FIXED[6:]])
    tok.save(str(path))


class _Tensors:
    """Random tensors by name, each in a shard, drawn in insertion order from one seeded generator."""

    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.shards: dict[str, dict[str, tuple[str, list[int], np.ndarray]]] = {s: {} for s in SHARDS}

    def put(self, shard: str, name: str, dtype: str, shape: list[int], data: np.ndarray) -> None:
        self.shards[shard][name] = (dtype, list(shape), data)

    def e4m3(self, shape) -> np.ndarray:
        """e4m3 bytes, never 0x7F or 0xFF (NaN)."""

        b = self.rng.integers(0, 254, size=shape, dtype=np.uint8)
        return b + (b >= 0x7F).astype(np.uint8)

    def e8m0(self, shape, rms: float, k: int, band: int) -> np.ndarray:
        """Powers of two around 1 / (rms * sqrt(k)), so a K-long dot product keeps its input's scale."""

        centre = 127 - round(math.log2(rms * math.sqrt(k)))
        return self.rng.integers(centre - band, centre + band + 1, size=shape, dtype=np.uint8)

    def bf16(self, shape, std: float, mean: float = 0.0) -> np.ndarray:
        x = (self.rng.standard_normal(shape, dtype=np.float32) * std + mean).view(np.uint32)
        return ((x + 0x7FFF + ((x >> 16) & 1)) >> 16).astype("<u2")       # round to nearest even

    def f32(self, shape, std: float, mean: float = 0.0) -> np.ndarray:
        return (self.rng.standard_normal(shape, dtype=np.float32) * std + mean).astype("<f4")

    def fp8(self, shard: str, name: str, n: int, k: int) -> None:
        """A DeepSeek FP8 weight: e4m3 [N, K] and an E8M0 scale per 32x32 block."""

        self.put(shard, name + ".weight", "F8_E4M3", [n, k], self.e4m3((n, k)))
        blocks = [-(-n // 32), -(-k // 32)]
        self.put(shard, name + ".scale", "F8_E8M0", blocks, self.e8m0(blocks, E4M3_RMS, k, 1))

    def linear(self, shard: str, name: str, n: int, k: int) -> None:
        self.put(shard, name, "BF16", [n, k], self.bf16((n, k), 1 / math.sqrt(k)))

    def norm(self, shard: str, name: str, n: int) -> None:
        self.put(shard, name, "BF16", [n], self.bf16((n,), 0.1, 1.0))

    def exl3(self, shard: str, name: str, k: int, n: int) -> None:
        """An EXL3 2-bit mcg projection W [K, N]: any int16 is a trellis; suh and svh in the pack's ranges."""

        self.put(shard, name + ".trellis", "I16", [k // 16, n // 16, 32],
                 self.rng.integers(-32768, 32768, size=(k // 16, n // 16, 32), dtype=np.int16).astype("<i2"))
        for part, size, lo, hi in (("suh", k, 0.005, 0.025), ("svh", n, 0.27, 1.06)):
            sign = self.rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=size)
            self.put(shard, f"{name}.{part}", "F16", [size],
                     (sign * self.rng.uniform(lo, hi, size=size)).astype("<f2"))
        self.put(shard, name + ".mcg", "I32", [], np.array(MCG, dtype="<i4"))

    def mxfp4(self, shard: str, name: str, n: int, k: int) -> None:
        """A DeepSeek MXFP4 matrix: e2m1 pairs (low nibble first) [N, K/2] and an E8M0 per 32 inputs of a row."""

        self.put(shard, name + ".weight", "I8", [n, k // 2], self.rng.integers(0, 256, size=(n, k // 2), dtype=np.uint8))
        self.put(shard, name + ".scale", "F8_E8M0", [n, k // 32], self.e8m0((n, k // 32), E2M1_RMS, k, 2))


def _block(g: _Tensors, shard: str, p: str, t: dict, experts: int) -> None:
    """One backbone layer or DSpark stage: attention, mHC, norms, router and shared expert."""

    d, heads = t["hidden_size"], t["num_attention_heads"] * t["head_dim"]
    q, groups, inter = t["q_lora_rank"], t["o_groups"], t["moe_intermediate_size"] * t["n_shared_experts"]
    g.fp8(shard, p + "attn.wq_a", q, d)
    g.norm(shard, p + "attn.q_norm.weight", q)
    g.fp8(shard, p + "attn.wq_b", heads, q)
    g.fp8(shard, p + "attn.wkv", t["head_dim"], d)
    g.norm(shard, p + "attn.kv_norm.weight", t["head_dim"])
    g.fp8(shard, p + "attn.wo_a", groups * t["o_lora_rank"], heads // groups)
    g.fp8(shard, p + "attn.wo_b", d, groups * t["o_lora_rank"])
    g.put(shard, p + "attn.attn_sink", "F32", [t["num_attention_heads"]], g.f32(t["num_attention_heads"], 1.0))
    g.norm(shard, p + "attn_norm.weight", d)
    g.norm(shard, p + "ffn_norm.weight", d)
    hc = t["hc_mult"]
    mixes = (2 + hc) * hc
    for sub in ("attn", "ffn"):
        g.put(shard, f"{p}hc_{sub}_fn", "F32", [mixes, hc * d], g.f32((mixes, hc * d), 1 / math.sqrt(hc * d)))
        g.put(shard, f"{p}hc_{sub}_base", "F32", [mixes], g.f32(mixes, 0.5))
        g.put(shard, f"{p}hc_{sub}_scale", "F32", [3], g.rng.uniform(0.5, 1.5, size=3).astype("<f4"))
    g.linear(shard, p + "ffn.gate.weight", experts, d)
    for bias in ("bias", "bias_vl"):
        g.put(shard, f"{p}ffn.gate.{bias}", "F32", [experts], g.f32(experts, 0.1))
    g.fp8(shard, p + "ffn.shared_experts.w1", inter, d)
    g.fp8(shard, p + "ffn.shared_experts.w3", inter, d)
    g.fp8(shard, p + "ffn.shared_experts.w2", d, inter)


def _layer(g: _Tensors, layer: int, t: dict) -> None:
    shard, p = SHARDS[1 + layer * 2 // t["num_hidden_layers"]], f"layers.{layer}."
    d, hd, inter = t["hidden_size"], t["head_dim"], t["moe_intermediate_size"]
    _block(g, shard, p, t, t["n_routed_experts"])
    if layer in t["kv_source_layer_ids"]:
        g.linear(shard, p + "attn.compressor.wkv.weight", hd, d)
        if t["compress_ratios"][layer] == 2:                 # ratio 1 pools nothing, so it has no gate
            g.linear(shard, p + "attn.compressor.wgate.weight", hd, d)
        g.norm(shard, p + "attn.compressor.norm.weight", hd)
        g.linear(shard, p + "attn.indexer.wk.weight", t["index_head_dim"], hd)
        g.norm(shard, p + "attn.indexer.k_norm.weight", t["index_head_dim"])
    if layer in t["index_source_layer_ids"]:
        g.fp8(shard, p + "attn.indexer.wq_b", t["index_n_heads"] * t["index_head_dim"], t["q_lora_rank"])
        g.linear(shard, p + "attn.indexer.weights_proj.weight", t["index_n_heads"], d)
    for e in range(t["n_routed_experts"]):
        for w, (k, n) in (("w1", (d, inter)), ("w2", (inter, d)), ("w3", (d, inter))):
            g.exl3(shard, f"{p}ffn.experts.{e}.{w}", k, n)


def _engram(g: _Tensors, layer: int, rows: int, t: dict) -> None:
    """One Engram layer's shard: its hash table of e4m3 rows with E8M0 per 32 bytes, wkv and the q/k weights."""

    shard, p = f"engram {layer}", f"layers.{layer}.engram."
    d, hd, hc = t["hidden_size"], t["engram_head_dim"], t["hc_mult"]
    g.put(shard, p + "embed.weight", "F8_E4M3", [rows, hd], g.e4m3((rows, hd)))
    g.put(shard, p + "embed.scale", "F8_E8M0", [rows, hd // 32],
          g.rng.integers(116, 124, size=(rows, hd // 32), dtype=np.uint8))
    g.fp8(shard, p + "wkv", (hc + 1) * d, (t["engram_max_ngram_size"] - 1) * t["engram_n_heads"] * hd)
    g.linear(shard, p + "q_weight", hc, d)
    g.linear(shard, p + "k_weight", hc, d)


def _dspark(g: _Tensors, t: dict) -> None:
    d, inter, stages = t["hidden_size"], t["moe_intermediate_size"], t["num_nextn_predict_layers"]
    for stage in range(stages):
        p = f"mtp.{stage}."
        _block(g, "dspark", p, t, t["dspark_n_routed_experts"])
        for e in range(t["dspark_n_routed_experts"]):
            for w, (n, k) in (("w1", (inter, d)), ("w2", (d, inter)), ("w3", (inter, d))):
                g.mxfp4("dspark", f"{p}ffn.experts.{e}.{w}", n, k)
    g.fp8("dspark", "mtp.0.main_proj", d, len(t["dspark_target_layer_ids"]) * d)
    g.norm("dspark", "mtp.0.main_norm.weight", d)
    last, rank, vocab = f"mtp.{stages - 1}.", t["dspark_markov_rank"], t["vocab_size"]
    g.norm("dspark", last + "norm.weight", d)
    g.put("dspark", last + "markov_head.embed.weight", "BF16", [vocab, rank], g.bf16((vocab, rank), 1.0))
    g.linear("dspark", last + "markov_head.head.weight", vocab, rank)
    g.linear("dspark", last + "confidence_head.proj.weight", 1, d + rank)


def tensors(cfg: dict, seed: int) -> dict[str, dict[str, tuple[str, list[int], np.ndarray]]]:
    """Every tensor by shard, then name: (dtype, shape, little-endian data)."""

    t, g = cfg["text_config"], _Tensors(seed)
    d, vocab = t["hidden_size"], t["vocab_size"]
    g.put("embed", "embed.weight", "BF16", [vocab, d], g.bf16((vocab, d), 1.0))
    for layer in range(t["num_hidden_layers"]):
        _layer(g, layer, t)
    for layer, rows in zip(t["engram_layer_ids"], t["engram_num_embeddings"]):
        _engram(g, layer, rows, t)
    _dspark(g, t)
    g.norm("head", "norm.weight", d)
    g.put("head", "lm_head.weight", "F8_E4M3", [vocab, d], g.e4m3((vocab, d)))
    g.put("head", "lm_head.weight_scale", "U8", [vocab, d // 32], g.e8m0((vocab, d // 32), E4M3_RMS, d, 1))
    return g.shards


def _write_shard(path: Path, entries: dict[str, tuple[str, list[int], np.ndarray]]) -> None:
    """A safetensors file, tensors in name order."""

    header, at = {}, 0
    for name in sorted(entries):
        dtype, shape, data = entries[name]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [at, at + data.nbytes]}
        at += data.nbytes
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw)
        for name in sorted(entries):
            f.write(np.ascontiguousarray(entries[name][2]).tobytes())


def write_tiny(path: str | Path, seed: int = 0) -> Path:
    """Write the tiny checkpoint into ``path``: the same seed writes the same bytes."""

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    cfg = config()
    (path / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
    write_tokenizer(path / "tokenizer.json")
    weight_map = {}
    for i, (shard, entries) in enumerate(tensors(cfg, seed).items(), 1):
        file = f"model-{i:05d}-of-{len(SHARDS):05d}.safetensors"
        _write_shard(path / file, entries)
        weight_map.update(dict.fromkeys(entries, file))
    index = {"metadata": {"total_size": 0}, "weight_map": dict(sorted(weight_map.items()))}
    (path / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    return path


_WRITTEN: dict[int, Path] = {}


@pytest.fixture(scope="session")
def tiny_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The seed-0 tiny checkpoint, written once a session however many modules import this fixture."""

    if 0 not in _WRITTEN:
        pytest.importorskip("tokenizers")
        _WRITTEN[0] = write_tiny(tmp_path_factory.mktemp("dsv41_tiny"))
    return _WRITTEN[0]
