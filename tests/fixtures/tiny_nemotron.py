"""Build a tiny on-disk nemotron_h model for mini-scale exactness tests."""
from __future__ import annotations
import json
from pathlib import Path
import mlx.core as mx


def tiny_config() -> dict:
    # 6 layers exercising every block type: M, E, *, M, E, -
    pattern = ["mamba", "moe", "attention", "mamba", "moe", "mlp"]
    return {
        "model_type": "nemotron_h",
        "architectures": ["NemotronHForCausalLM"],
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": len(pattern),
        "max_position_embeddings": 256,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "attention_bias": False,
        "mamba_num_heads": 8,
        "mamba_head_dim": 8,
        "mamba_proj_bias": False,
        "ssm_state_size": 16,
        "conv_kernel": 4,
        "n_groups": 2,
        "mlp_bias": False,
        "layer_norm_epsilon": 1e-5,
        "use_bias": False,
        "use_conv_bias": True,
        "layers_block_type": pattern,
        "moe_intermediate_size": 64,
        "moe_shared_expert_intermediate_size": 128,
        "moe_latent_size": 32,
        "n_group": 2,
        "topk_group": 2,
        "n_routed_experts": 8,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "norm_topk_prob": True,
        "routed_scaling_factor": 1.0,
        "time_step_limit": [0.0, float("inf")],
    }


# Quantization group size for the quantized toy. mx.quantize supports only
# {32, 64, 128}; 32 is the largest that divides EVERY quantizable input dim in
# tiny_config() (smallest is moe_latent_size=32 on switch_mlp.fc1), so
# nn.quantize(..., group_size=32, bits=4) succeeds on every module + embedding.
_QUANT_GROUP_SIZE = 32
_QUANT_BITS = 4
_QUANT_MODE = "affine"


def build_tiny_nemotron(out_dir: str | Path, *, quantize: bool = False) -> Path:
    """Instantiate mlx_lm's nemotron_h, randomise, and save in HF on-disk layout.

    Saves routed experts STACKED (``mixer.switch_mlp.fc1/fc2``) and conv1d in HF
    orientation, matching the REAL on-disk Nemotron-Ultra format: the routed
    experts already live as the model's own stacked SwitchMLP params, so the
    model's own ``sanitize()`` is a no-op for them (its per-expert stacking loop
    only fires when ``experts.0.up_proj.weight`` is present, which it is not).

    When ``quantize=True`` the model is 4-bit affine quantized (group_size 32)
    BEFORE saving — so ``switch_mlp.fc1/fc2`` and the other Linears land on disk
    as ``{weight,scales,biases}`` — and a ``quantization`` block is written into
    config.json so ``build_mlx_model_shell`` reconstructs a quantized shell. This
    mirrors the real 4-bit Ultra, whose switch_mlp is stacked AND quantized.
    """
    from mlx.utils import tree_flatten
    from mlx_lm.models.nemotron_h import Model, ModelArgs

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = tiny_config()

    mx.random.seed(0)
    model = Model(ModelArgs.from_dict(cfg))

    # Force NON-CONTIGUOUS routing so global->local remap is a genuine remap
    # (not the identity {0,1}). Each MoE gate gets a STRUCTURED weight where
    # expert e responds to its own hidden-feature block, plus a correction bias
    # that nudges experts {2,5,7} in. With n_group=2/topk_group=2 both groups
    # stay live, so top_k=2 picks from the biased high-index experts. For the
    # fixed test token sequence this makes every MoE layer route to the
    # non-contiguous union {2,5,7}. Stock and paged forwards share these gate
    # weights, so token-exactness is unaffected — only WHICH experts route
    # changes, which is exactly what the paged remap must handle.
    import numpy as np

    n_experts = cfg["n_routed_experts"]
    hidden = cfg["hidden_size"]
    block = hidden // n_experts
    gate_weight = np.zeros((n_experts, hidden), dtype=np.float32)
    for e in range(n_experts):
        gate_weight[e, e * block : (e + 1) * block] = 4.0
    # Bias that selects the non-contiguous set {2, 5, 7}.
    gate_bias = np.zeros((n_experts,), dtype=np.float32)
    gate_bias[[2, 5, 7]] = 2.0
    for layer_idx, block_type in enumerate(cfg["layers_block_type"]):
        if block_type != "moe":
            continue
        gate = model.backbone.layers[layer_idx].mixer.gate
        gate.weight = mx.array(gate_weight)
        gate.e_score_correction_bias = mx.array(gate_bias)

    if quantize:
        from mlx import nn

        # 4-bit affine quantize EVERY quantizable Linear/SwitchLinear. The gate
        # (MoEGate.weight / e_score_correction_bias) is a bare parameter, not a
        # Linear, so it is left in full precision — exactly like the real model.
        nn.quantize(
            model,
            group_size=_QUANT_GROUP_SIZE,
            bits=_QUANT_BITS,
            mode=_QUANT_MODE,
        )
        cfg = dict(cfg)
        cfg["quantization"] = {
            "group_size": _QUANT_GROUP_SIZE,
            "bits": _QUANT_BITS,
            "mode": _QUANT_MODE,
        }

    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    params = dict(tree_flatten(model.parameters()))

    # Convert in-memory params to HF on-disk names. Routed experts stay STACKED
    # (the model already holds them as switch_mlp.fc1/fc2.{weight,scales,biases});
    # we do NOT un-stack into per-expert tensors. Only conv1d needs the inverse
    # of sanitize()'s HF-orientation moveaxis.
    disk: dict[str, mx.array] = {}
    for name, arr in params.items():
        if name.endswith("conv1d.weight") and arr.shape[-1] == 1:
            disk[name] = arr.moveaxis(1, 2)  # inverse of sanitize's moveaxis(2,1)
        else:
            disk[name] = arr

    mx.save_safetensors(str(out / "model.safetensors"), disk)
    index = {"metadata": {"total_size": sum(a.nbytes for a in disk.values())},
             "weight_map": {k: "model.safetensors" for k in disk}}
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    return out


def build_tiny_nemotron_quantized(out_dir: str | Path) -> Path:
    """Convenience: 4-bit affine quantized stacked toy (see ``build_tiny_nemotron``)."""
    return build_tiny_nemotron(out_dir, quantize=True)
