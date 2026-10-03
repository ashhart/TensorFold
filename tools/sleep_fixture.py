"""Small random affine Qwen checkpoint for lifecycle plumbing, not model-quality qualification."""

import json
from pathlib import Path


def create(directory: Path, *, layers: int = 2, hidden: int = 128) -> Path:
    import torch
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, models, pre_tokenizers

    directory.mkdir(parents=True, exist_ok=False)
    gen = torch.Generator().manual_seed(21)
    tensors = {}

    def linear(name, n, k):
        tensors[name + ".weight"] = torch.randint(-(2**31), 2**31 - 1, (n, k // 8),
                                                   generator=gen, dtype=torch.int64).int()
        tensors[name + ".scales"] = (torch.rand(n, k // 64, generator=gen) * .003 + .001).bfloat16()
        tensors[name + ".biases"] = (torch.rand(n, k // 64, generator=gen) * .003 - .0015).bfloat16()

    def norm(name, n=128):
        tensors[name] = torch.ones(n, dtype=torch.bfloat16)

    interval = 2 if layers == 2 else 4
    for i in range(layers):
        prefix = f"model.layers.{i}."
        norm(prefix + "input_layernorm.weight", hidden)
        norm(prefix + "post_attention_layernorm.weight", hidden)
        if (i + 1) % interval:
            for key, rows in (("in_proj_qkv", 384), ("in_proj_z", 128), ("in_proj_b", 1),
                              ("in_proj_a", 1)):
                linear(prefix + "linear_attn." + key, rows, hidden)
            linear(prefix + "linear_attn.out_proj", hidden, 128)
            tensors[prefix + "linear_attn.conv1d.weight"] = (
                torch.randn(384, 4, generator=gen) * .1).bfloat16()
            for key in ("A_log", "dt_bias"):
                tensors[prefix + "linear_attn." + key] = torch.zeros(1)
            norm(prefix + "linear_attn.norm.weight")
        else:
            for key, n, k in (("q_proj", 512, hidden), ("k_proj", 128, hidden),
                              ("v_proj", 128, hidden), ("o_proj", hidden, 256)):
                linear(prefix + "self_attn." + key, n, k)
            norm(prefix + "self_attn.q_norm.weight")
            norm(prefix + "self_attn.k_norm.weight")
        for key in ("gate_proj", "up_proj"):
            linear(prefix + "mlp." + key, 128, hidden)
        linear(prefix + "mlp.down_proj", hidden, 128)
    linear("model.embed_tokens", 256, hidden)
    linear("lm_head", 256, hidden)
    norm("model.norm.weight", hidden)
    save_file(tensors, str(directory / "model.safetensors"))
    (directory / "config.json").write_text(json.dumps(dict(
        model_type="qwen3_5", hidden_size=hidden, intermediate_size=128, num_hidden_layers=layers,
        num_attention_heads=2, num_key_value_heads=1, head_dim=128, vocab_size=256,
        linear_num_key_heads=1, linear_num_value_heads=1, linear_key_head_dim=128,
        linear_value_head_dim=128, linear_conv_kernel_dim=4, full_attention_interval=interval,
        rms_norm_eps=1e-6, partial_rotary_factor=.25, rope_theta=10000000,
        eos_token_id=0, max_position_embeddings=2048, quantization=dict(bits=4, group_size=64))))
    tok = Tokenizer(models.WordLevel({f"t{i}": i for i in range(256)}, unk_token="t0"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(str(directory / "tokenizer.json"))
    (directory / "tokenizer_config.json").write_text(json.dumps({
        "chat_template": "{% for message in messages %}{{ message['content'] }} {% endfor %}"}))
    return directory


def create_draft(directory: Path) -> Path:
    """Tiny DFlash2 with the real five-tap interface; its target must have 64 layers."""

    import torch
    from safetensors.torch import save_file

    directory.mkdir(parents=True, exist_ok=False)
    gen = torch.Generator().manual_seed(5)

    def random(*shape, scale=.02):
        return (torch.randn(*shape, generator=gen) * scale).bfloat16()

    hidden, rank, intermediate = 2048, 512, 512
    tensors = {"candidate_selector.hidden_projection.weight": random(rank, hidden),
               "candidate_selector.predecessor_codebook": random(256, rank, scale=.1),
               "candidate_selector.successor_codebook": random(256, rank, scale=.1),
               "fc.weight": random(hidden, 5 * hidden), "hidden_norm.weight": torch.ones(hidden).bfloat16(),
               "norm.weight": torch.ones(hidden).bfloat16()}
    for key, n, k in (("self_attn.q_proj", 1024, hidden), ("self_attn.k_proj", 256, hidden),
                      ("self_attn.v_proj", 256, hidden), ("self_attn.o_proj", hidden, 1024),
                      ("mlp.gate_proj", intermediate, hidden), ("mlp.up_proj", intermediate, hidden),
                      ("mlp.down_proj", hidden, intermediate)):
        tensors["layers.0." + key + ".weight"] = random(n, k)
    for key in ("self_attn.q_norm", "self_attn.k_norm", "input_layernorm", "post_attention_layernorm"):
        tensors["layers.0." + key + ".weight"] = torch.ones(128 if "self_attn" in key else hidden).bfloat16()
    for conv in ("attention_conv", "mlp_conv"):
        tensors[f"layers.0.{conv}.base_kernel"] = random(2, 2, hidden, scale=.5)
        tensors[f"layers.0.{conv}.kernel_projection.weight"] = random(4 * hidden // 16, hidden)
    save_file(tensors, str(directory / "model.safetensors"))
    (directory / "config.json").write_text(json.dumps(dict(
        hidden_size=hidden, intermediate_size=intermediate, head_dim=128, num_attention_heads=8,
        num_key_value_heads=2, rms_norm_eps=1e-6, rope_parameters=dict(rope_theta=10000000),
        num_hidden_layers=1, sliding_window=64, is_causal=False,
        dflash_config=dict(mask_token_id=255, conv_group_size=16, block_size=8))))
    return directory
