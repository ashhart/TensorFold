"""A tiny modelopt (NVFP4) checkpoint for the loader's tests: the real format, the real layout.

The main layers' routed experts are FP4 (packed E2M1 words, fp8e4m3 block scales, per-tensor scale);
everything else is BF16, the MTP layer's stacked experts included — the Swift checkpoint's exclusions.
The n-gram tables are the MLX layout (words/scales/biases), bitwise passthroughs. Shapes are the real
Flash Next head sizes with the layers and experts cut to two each, so a whole load fits a test."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import torch

from tensorfold.families.qwen4_exp.cuda import nvfp4
from tensorfold.families.qwen4_exp.cuda.ngram import NGram

DTYPE_NAMES = {torch.bfloat16: "BF16", torch.float32: "F32", torch.uint8: "U8", torch.int32: "I32",
               torch.float8_e4m3fn: "F8_E4M3", torch.int64: "I64"}


def tiny_ngram(vocab: int) -> NGram:
    """The tiny checkpoint's n-gram constants and table size, derived with the params ``_config`` writes
    (one head a step over a three-gram table, the config's seed and vocab base): the real checkpoint ships
    its own, and the loader's ``NGram.check`` compares them value for value."""

    return NGram(vocab=vocab, ngram_size=3, heads_per_ngram=1, vocab_base=vocab, divisor=1, shards=1,
                 seed=1234, eos=0, embed_dim=64, ple_index=0)


def _quant_rows(rows: torch.Tensor, rng: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, float]:
    """A bf16 [N, K] matrix as the checkpoint's FP4 arrays, ModelOpt's recipe: ``value = code * e4m3 * scale_2``,
    ``scale_2`` the tensor's max over 6 * 448 (here rounded up to a power of two, so every product is exact)."""

    n, k = rows.shape
    mags = torch.tensor(nvfp4._E2M1, dtype=torch.float32)
    w = rows.to(torch.float32)
    g = w.reshape(n, k // 16, 16)
    amax = w.abs().amax()
    scale2 = torch.pow(torch.tensor(2.0), torch.ceil(torch.log2(amax / (6.0 * 448.0))))
    s_blk = (g.abs().amax(dim=-1, keepdim=True) / 6.0 / scale2).clamp(2.0 ** -9, 448.0)   # the block scale
    e4m3 = s_blk.squeeze(-1).to(torch.float8_e4m3fn)
    q = (g / (e4m3.float().unsqueeze(-1) * scale2)).clamp(-6.0, 6.0)          # the code's value (approximate)
    code = (q.abs().unsqueeze(-1) - mags).abs().argmin(dim=-1).to(torch.int32) \
        + (q < 0).to(torch.int32) * 8
    code = code.reshape(n, k)
    words = (code[:, 1::2] << 4 | code[:, 0::2]).to(torch.uint8)
    return words, e4m3, float(scale2)


def write(dir: Path, *, layers: int = 2, experts: int = 2, vocab: int = 256, hidden: int = 256,
          heads: int = 2, kv_heads: int = 2, hd: int = 64, nk: int = 8, nv: int = 24, dk: int = 128, dv: int = 128,
          moe_width: int = 128, shared_width: int = 64, streams: int = 4, low: int = 64,
          ple: bool = True, mtp: bool = True, seed: int = 0, prefix: str = "", ple_bf16: bool = False,
          mxfp8: bool = False, ple_nvfp4: bool = False, centred: bool = False, fp8block: bool = False) -> Path:
    """``mxfp8``: DeltaNet, attention and shared-expert linears in MXFP8 and ``ple_nvfp4``: NVFP4 n-gram rows, as
    local-inference-lab's export stores them; ``centred``: RMSNorm weights stored around 0 (the model adds 1);
    ``fp8block``: ModelOpt ``FP8_PB_WO`` (e4m3 with an fp32 ``weight_scale_inv`` per 128x128 block) for the DeltaNet
    qkv/z/out and attention q/k/v/o projections and lm_head, the rest bf16 (so stacks mix block FP8 and bf16)."""
    dir.mkdir(parents=True, exist_ok=True)
    rng = torch.Generator().manual_seed(seed)

    def rand(*shape, scale: float = 0.02, dtype=torch.bfloat16) -> torch.Tensor:
        return (torch.randn(*shape, generator=rng) * scale).to(dtype)

    def norm(n: int) -> torch.Tensor:
        return rand(n, scale=0.05, dtype=torch.float32) + (0.0 if centred else 1.0)

    entries: dict[str, dict] = {}
    blobs: list[torch.Tensor] = []

    def add(name: str, t: torch.Tensor) -> None:
        if prefix and name.startswith("model."):        # the published checkpoint's language-model group
            name = prefix + name[len("model."):]        # (its lm_head and mtp stay at the top level)
        t = t.contiguous()
        entries[name] = {"dtype": DTYPE_NAMES[t.dtype], "shape": list(t.shape),
                         "data_offsets": [0, 0]}                                      # patched on write
        blobs.append(t)

    def linear(name: str, n: int, k: int, *, fp4: bool, mx: bool = False, blk: bool = False) -> None:
        w = rand(n, k)
        if blk and fp8block:                               # e4m3 with an fp32 scale per 128x128 block
            nb = -(-n // 128)
            g = torch.zeros(nb * 128, k)
            g[:n] = w.float()
            g = g.view(nb, 128, k // 128, 128)
            s = (g.abs().amax(dim=(1, 3)).clamp_min(1e-12) / 448.0)                 # [nb, K/128]
            codes = (g / s[:, None, :, None]).view(nb * 128, k)[:n].to(torch.float8_e4m3fn)
            add(name + ".weight", codes)
            add(name + ".weight_scale_inv", s.float())
        elif mx and mxfp8:                                   # e4m3 with a power-of-two scale every 32 inputs
            g = w.float().view(n, k // 32, 32)
            e = torch.ceil(torch.log2(g.abs().amax(-1).clamp_min(1e-30) / 448.0)).clamp(-127, 127)
            add(name + ".weight", (g / torch.pow(2.0, e)[..., None]).view(n, k).to(torch.float8_e4m3fn))
            add(name + ".weight_scale", (e + 127).to(torch.uint8))
        elif fp4:
            words, s8, s2 = _quant_rows(w, rng)
            add(name + ".weight", words)
            add(name + ".weight_scale", s8.view(torch.uint8).view(torch.float8_e4m3fn))
            add(name + ".weight_scale_2", torch.tensor(s2))
            add(name + ".input_scale", torch.tensor(1.0))
        else:
            add(name + ".weight", w)

    cfg = _config(layers, experts, vocab, hidden, heads, kv_heads, hd, nk, nv, dk, dv, moe_width,
                  shared_width, streams, low, ple)
    (dir / "config.json").write_text(json.dumps(cfg, indent=1))

    add("model.embed_tokens.weight", rand(vocab, hidden))
    for i in range(layers):
        b = f"model.layers.{i}"
        for hc in ("attn_hyper_connection", "mlp_hyper_connection"):
            linear(f"{b}.{hc}.input_mix_weight_down", low, streams * hidden, fp4=False)
            linear(f"{b}.{hc}.input_mix_weight_up", streams * hidden, low, fp4=False)
            linear(f"{b}.{hc}.block_inject_weight", low, streams * hidden, fp4=False)
            add(f"{b}.{hc}.hc_norm.weight", norm(streams * hidden))
        linear(f"{b}.mlp.gate", experts, hidden, fp4=False)
        add(f"{b}.mlp.shared_expert_gate.weight", rand(1, hidden))
        for proj, n_, k_ in (("gate_proj", moe_width, hidden), ("up_proj", moe_width, hidden),
                             ("down_proj", hidden, moe_width)):
            linear(f"{b}.mlp.shared_expert.{proj}", n_, k_, fp4=False, mx=True)
        for e in range(experts):
            for proj, n_, k_ in (("gate_proj", moe_width, hidden), ("up_proj", moe_width, hidden),
                                 ("down_proj", hidden, moe_width)):
                linear(f"{b}.mlp.experts.{e}.{proj}", n_, k_, fp4=True)
        for proj, n_, k_ in (("in_proj_qkv", 2 * nk * dk + nv * dv, hidden),
                             ("in_proj_z", nv * dv, hidden),
                             ("in_proj_b", nv, hidden), ("in_proj_a", nv, hidden),
                             ("out_proj", hidden, nv * dv)):
            linear(f"{b}.linear_attn.{proj}", n_, k_, fp4=False, mx=True, blk=proj in ("in_proj_qkv", "in_proj_z",
                                                                                     "out_proj"))
        add(f"{b}.linear_attn.conv1d.weight", rand(2 * nk * dk + nv * dv, 4))
        add(f"{b}.linear_attn.A_log", rand(nv, dtype=torch.float32) - 4.0)
        add(f"{b}.linear_attn.dt_bias", rand(nv, dtype=torch.float32))
        add(f"{b}.linear_attn.norm.weight", rand(dv) + 1.0)
        for proj, n_, k_ in (("q_proj", 2 * heads * hd, hidden), ("k_proj", kv_heads * hd, hidden),
                             ("v_proj", kv_heads * hd, hidden),
                             ("o_proj", hidden, heads * hd),
                             ("indexer.index_qk_proj", (4 + 1) * 128, hidden)):
            linear(f"{b}.self_attn.{proj}", n_, k_, fp4=False, mx=True, blk=not proj.startswith("indexer"))
        for nm, size in (("q_norm", hd), ("k_norm", hd), ("indexer.q_layernorm", 128), ("indexer.k_layernorm", 128)):
            add(f"{b}.self_attn.{nm}.weight", norm(size))
        if ple and i == 1:
            linear(f"{b}.ple.key_proj", streams * hidden, 64, fp4=False)
            linear(f"{b}.ple.value_proj", hidden, 64, fp4=False)
            for nm in ("norm_key", "norm_query", "norm_conv"):
                add(f"{b}.ple.{nm}.weight", norm(streams * hidden))
            add(f"{b}.ple.conv1d.weight", rand(streams * hidden, 4))
            ng = tiny_ngram(vocab)
            # the hashing constants the checkpoint ships (the loader's check compares them) and the
            # one-shard table they index: the MLX 4-bit layout, rows = the derived table size
            add(f"{b}.ple.ple_embedding.layer_multipliers", torch.as_tensor(ng.multipliers))
            add(f"{b}.ple.ple_embedding.ngram_heads_offsets", torch.as_tensor(ng.head_offsets))
            add(f"{b}.ple.ple_embedding.ngram_heads_vocab_sizes", torch.as_tensor(ng.head_sizes))
            heads_rows = int(ng.rows)
            dims = int(ng.dims)
            if ple_bf16:                                   # the published revision: plain bf16 rows, no scales
                add(f"{b}.ple.ple_embedding.ngram_embedding.shard_0.weight", rand(heads_rows, dims))
            elif ple_nvfp4:                                # codes, e4m3 a 16 values, one table scale
                words, s8, s2 = _quant_rows(rand(heads_rows, dims), rng)
                add(f"{b}.ple.ple_embedding.ngram_embedding.shard_0.weight", words)
                add(f"{b}.ple.ple_embedding.ngram_embedding.shard_0.weight_scale",
                    s8.view(torch.uint8).view(torch.float8_e4m3fn))
                add(f"{b}.ple.ple_embedding.ngram_embedding.weight_scale_2", torch.tensor([s2]))
            else:
                # a shard row is one head's embedding at ``dims`` 4-bit values: dims/8 int32 words and
                # dims/32 fp16 scales/biases (``host_table.HostTable`` reads the words as int32 and the
                # scales as int16, and the engine's PLE buffer row is the same width)
                words = torch.randint(0, 256, (heads_rows, dims // 8), generator=rng, dtype=torch.int32)
                scales = rand(heads_rows, dims // 32)
                biases = rand(heads_rows, dims // 32)
                add(f"{b}.ple.ple_embedding.ngram_embedding.shard_0.weight", words)
                add(f"{b}.ple.ple_embedding.ngram_embedding.shard_0.scales", scales)
                add(f"{b}.ple.ple_embedding.ngram_embedding.shard_0.biases", biases)
    linear("model.hyper_connection_mixer.input_mix_weight_down", low, streams * hidden, fp4=False)
    linear("model.hyper_connection_mixer.input_mix_weight_up", streams * hidden, low, fp4=False)
    add("model.hyper_connection_mixer.hc_norm.weight", norm(streams * hidden))
    linear("lm_head", vocab, hidden, fp4=False, blk=True)
    if mtp:
        add("mtp.pre_fc_norm_embedding.weight", norm(hidden))
        add("mtp.pre_fc_norm_hidden.weight", norm(streams * hidden))
        linear("mtp.fc_embedding", hidden, hidden, fp4=False)
        linear("mtp.fc_hidden", hidden, hidden, fp4=False)
        for hc in ("attn_hyper_connection", "mlp_hyper_connection"):
            linear(f"mtp.layers.0.{hc}.input_mix_weight_down", low, streams * hidden, fp4=False)
            linear(f"mtp.layers.0.{hc}.input_mix_weight_up", streams * hidden, low, fp4=False)
            linear(f"mtp.layers.0.{hc}.block_inject_weight", low, streams * hidden, fp4=False)
            add(f"mtp.layers.0.{hc}.hc_norm.weight", norm(streams * hidden))
        linear("mtp.layers.0.mlp.gate", experts, hidden, fp4=False)
        add("mtp.layers.0.mlp.shared_expert_gate.weight", rand(1, hidden))
        for proj, n_, k_ in (("gate_proj", moe_width, hidden), ("up_proj", moe_width, hidden),
                             ("down_proj", hidden, moe_width)):
            linear(f"mtp.layers.0.mlp.shared_expert.{proj}", n_, k_, fp4=False)
        add("mtp.layers.0.mlp.experts.gate_up_proj", rand(experts, 2 * moe_width, hidden))
        add("mtp.layers.0.mlp.experts.down_proj", rand(experts, hidden, moe_width))
        for proj, n_, k_ in (("q_proj", 2 * heads * hd, hidden), ("k_proj", kv_heads * hd, hidden),
                             ("v_proj", kv_heads * hd, hidden), ("o_proj", hidden, heads * hd),
                             ("indexer.index_qk_proj", (4 + 1) * 128, hidden)):
            linear(f"mtp.layers.0.self_attn.{proj}", n_, k_, fp4=False)
        for nm, size in (("q_norm", hd), ("k_norm", hd), ("indexer.q_layernorm", 128), ("indexer.k_layernorm", 128)):
            add(f"mtp.layers.0.self_attn.{nm}.weight", norm(size))
        linear("mtp.hyper_connection_mixer.input_mix_weight_down", low, streams * hidden, fp4=False)
        linear("mtp.hyper_connection_mixer.input_mix_weight_up", streams * hidden, low, fp4=False)
        add("mtp.hyper_connection_mixer.hc_norm.weight", norm(streams * hidden))

    shard = dir / "model-00001-of-00001.safetensors"
    offset = 0
    for name, t in zip(entries, blobs):
        entries[name]["data_offsets"] = [offset, offset + t.numel() * t.element_size()]
        offset += t.numel() * t.element_size()
    header = json.dumps({"__metadata__": {"format": "pt"}, **entries}, separators=(",", ":")).encode()

    def raw(t: torch.Tensor) -> bytes:
        return (t.reshape(1).contiguous() if t.dim() == 0 else t).view(torch.uint8).numpy().tobytes()

    with open(shard, "wb") as f:
        f.write(struct.pack("<Q", len(header)))
        f.write(header)
        for t in blobs:
            f.write(raw(t))
    (dir / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": offset}, "weight_map": {n: shard.name for n in entries}}))
    return dir


def _config(layers, experts, vocab, hidden, heads, kv_heads, hd, nk, nv, dk, dv, moe_width, shared_width,
            streams, low, ple) -> dict:
    return {
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "model_type": "qwen4_exp",
        "text_config": {
            "hidden_size": hidden, "num_hidden_layers": layers, "vocab_size": vocab,
            "rms_norm_eps": 1e-6, "num_attention_heads": heads, "num_key_value_heads": kv_heads,
            "head_dim": hd, "layer_types": (["linear_attention", "full_attention"] * (layers // 2) +
                                            (["linear_attention"] if layers % 2 else [])),
            "rope_parameters": {"rope_theta": 10_000_000, "partial_rotary_factor": 0.25},
            "linear_num_key_heads": nk, "linear_num_value_heads": nv,
            "linear_key_head_dim": dk, "linear_value_head_dim": dv, "linear_conv_kernel_dim": 4,
            "num_experts": experts, "num_experts_per_tok": min(2, experts),
            "moe_intermediate_size": moe_width, "shared_expert_intermediate_size": shared_width,
            "hc_count": streams, "hc_lowrank": low,
            "indexer_n_heads": 4, "indexer_head_dim": 128, "indexer_budget": 128, "indexer_compress_ratio": 4,
            "ple_layer_ids": ([2] if ple and layers >= 2 else []), "ple_embed_dim": 64,
            "ple_conv_kernel_size": 4, "ngram_size": 3, "heads_per_ngram": 1,
            "ngram_vocab_size_base": vocab, "make_ngram_vocab_size_divisible_by": 1, "split_ngram_parts": 1,
            "seed": 1234, "dtype": "bfloat16",
            "bos_token_id": 0, "eos_token_id": [0],
        },
        "quantization": {"quant_method": "modelopt", "quant_algo": "NVFP4", "group_size": 16,
                         "config_groups": {"group_0": {"weights": {"num_bits": 4, "group_size": 16}}},
                         "exclude_modules": ["lm_head", "model.embed_tokens"]},
        "bos_token_id": 0, "eos_token_id": [0],
    }
