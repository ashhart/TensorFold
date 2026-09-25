"""Nemotron-H's MTP head (one attention block + one MoE block) for drafting the token after next.

The released BF16 checkpoint carries ``mtp.layers.0`` (enorm, hnorm, eh_proj and an attention block) and
``mtp.layers.1`` (a 128-expert MoE block and a final norm); the MLX 4-bit conversion dropped them.
``convert`` quantizes them like the main model (4-bit affine, group 64; router, norms and correction
bias kept) into one safetensors file. At position i the head reads the main model's hidden state h_i
and the embedding of token i+1 and predicts token i+2 through the main model's LM head:

    x = eh_proj([enorm(embed(token_{i+1})), hnorm(h_i)])
    x = x + attention(norm0(x))        # over the head's own cache of earlier positions
    x = x + moe(norm1(x))
    logits = lm_head(final_layernorm(x))
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

DEFAULT_DIR = Path.home() / ".cache" / "tensorfold" / "mtp" / "nemotron-3.5-lightning"


class _AttentionBlock(nn.Module):
    def __init__(self, args: Any) -> None:
        super().__init__()
        from mlx_lm.models.nemotron_h import NemotronHAttention

        d = args.hidden_size
        self.enorm = nn.RMSNorm(d, eps=args.layer_norm_epsilon)
        self.hnorm = nn.RMSNorm(d, eps=args.layer_norm_epsilon)
        self.eh_proj = nn.Linear(2 * d, d, bias=False)
        self.norm = nn.RMSNorm(d, eps=args.layer_norm_epsilon)
        self.mixer = NemotronHAttention(args)


class _MoEBlock(nn.Module):
    def __init__(self, args: Any) -> None:
        super().__init__()
        from mlx_lm.models.nemotron_h import NemotronHMoE

        d = args.hidden_size
        self.norm = nn.RMSNorm(d, eps=args.layer_norm_epsilon)
        self.mixer = NemotronHMoE(args)
        self.final_layernorm = nn.RMSNorm(d, eps=args.layer_norm_epsilon)


class NemotronMTP(nn.Module):
    def __init__(self, args: Any) -> None:
        super().__init__()
        self.layers = [_AttentionBlock(args), _MoEBlock(args)]

    def make_cache(self) -> Any:
        from mlx_lm.models.cache import KVCache

        return KVCache()

    def __call__(self, hidden: mx.array, next_embeddings: mx.array, cache: Any, *,
                 tail: int | None = None) -> mx.array | None:
        """hidden [1, R, D] (main model, position i), next_embeddings [1, R, D] (token i+1) -> [1, R, D] normed.

        Every row extends the head's cache (its attention block); ``tail`` rows (the last ones, all if None)
        go on through the MoE block, the rest are only context: [1, tail, D], or None for tail 0.
        """

        first, second = self.layers
        x = first.eh_proj(mx.concatenate([first.enorm(next_embeddings), first.hnorm(hidden)], axis=-1))
        rows = x.shape[1]
        x = x + first.mixer(first.norm(x), mask="causal" if rows > 1 else None, cache=cache)
        if tail is not None:
            if tail == 0:
                return None
            x = x[:, rows - tail:]
        x = x + second.mixer(second.norm(x))
        return second.final_layernorm(x)


def convert(shard: Path, args: Any, out: Path, *, group_size: int = 64, bits: int = 4) -> Path:
    """The MTP tensors of the BF16 shard -> a 4-bit MLX safetensors file with this module's names."""

    raw = {k[len("mtp."):]: v for k, v in mx.load(str(shard)).items() if k.startswith("mtp.")}
    experts: dict[str, dict[int, mx.array]] = {"up_proj": {}, "down_proj": {}}
    weights: dict[str, mx.array] = {}
    for name, value in raw.items():
        found = re.match(r"layers\.1\.mixer\.experts\.(\d+)\.(up_proj|down_proj)\.weight$", name)
        if found:
            experts[found.group(2)][int(found.group(1))] = value
        else:
            weights[name] = value
    count = int(args.n_routed_experts)
    weights["layers.1.mixer.switch_mlp.fc1.weight"] = mx.stack([experts["up_proj"][e] for e in range(count)])
    weights["layers.1.mixer.switch_mlp.fc2.weight"] = mx.stack([experts["down_proj"][e] for e in range(count)])
    mtp = NemotronMTP(args)
    mtp.load_weights(list(weights.items()), strict=True)
    nn.quantize(mtp, group_size=group_size, bits=bits,
                class_predicate=lambda path, module: hasattr(module, "to_quantized") and not path.endswith("gate"))
    mx.eval(mtp.parameters())
    from mlx.utils import tree_flatten

    out.parent.mkdir(parents=True, exist_ok=True)
    flat = dict(tree_flatten(mtp.parameters()))
    mx.save_safetensors(str(out), flat, metadata={
        "format": "tensorfold-nemotron-mtp", "quantization": json.dumps({"group_size": group_size, "bits": bits}),
        "source": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 model-00014-of-00014.safetensors"})
    return out


def load(path: Path, args: Any) -> NemotronMTP:
    weights, meta = mx.load(str(path), return_metadata=True)
    quant = json.loads(meta.get("quantization", '{"group_size": 64, "bits": 4}'))
    mtp = NemotronMTP(args)
    nn.quantize(mtp, group_size=int(quant["group_size"]), bits=int(quant["bits"]),
                class_predicate=lambda p, m: hasattr(m, "to_quantized") and f"{p}.scales" in weights)
    mtp.load_weights(list(weights.items()), strict=True)
    mx.eval(mtp.parameters())
    return mtp
