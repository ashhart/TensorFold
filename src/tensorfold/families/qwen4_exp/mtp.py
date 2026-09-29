"""Draft the token after next from the preceding residual streams and next-token embedding, chaining output streams."""

from __future__ import annotations

from dataclasses import replace

import mlx.core as mx
import mlx.nn as nn

from tensorfold.families.qwen4_exp.model import (
    AttentionCache,
    CenteredRMSNorm,
    Config,
    DecoderLayer,
    HyperConnection,
)


class FlashMTP(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        d = cfg.hidden_size
        self.streams = cfg.hc_count
        self.pre_fc_norm_embedding = CenteredRMSNorm(d, cfg.rms_norm_eps)
        self.pre_fc_norm_hidden = CenteredRMSNorm(cfg.hc_count * d, cfg.rms_norm_eps)
        self.fc_embedding = nn.Linear(d, d, bias=False)
        self.fc_hidden = nn.Linear(d, d, bias=False)
        layer_cfg = replace(cfg, num_hidden_layers=1, layer_types=["sparse_attention"], ple_layer_ids=[])
        self.layers = [DecoderLayer(layer_cfg, 0)]
        self.hyper_connection_mixer = HyperConnection(cfg, combine=False)

    def make_cache(self) -> AttentionCache:
        return AttentionCache()

    def __call__(self, token_embed: mx.array, streams: mx.array, cache: AttentionCache) -> tuple[mx.array, mx.array]:
        """token_embed [B, L, D], streams [B, L, S*D] -> (mixed hidden [B, L, D] for the head, output streams)."""

        batch, length, wide = streams.shape
        dims = wide // self.streams
        e = self.fc_embedding(self.pre_fc_norm_embedding(token_embed))
        hs = self.fc_hidden(self.pre_fc_norm_hidden(streams).reshape(batch, length, self.streams, dims))
        x = (e[..., None, :] + hs).reshape(batch, length, wide)
        x = self.layers[0](x, None, cache)
        return self.hyper_connection_mixer(x), x


PREFIXES = ("language_model.mtp.", "mtp.")          # where checkpoints keep the head (mlx-lm, oMLX conversions)


def sanitize(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """The checkpoint's MTP tensors under this module's names."""

    out = {}
    for name, value in weights.items():
        for prefix in PREFIXES:
            if name.startswith(prefix):
                out[name[len(prefix):]] = value
    return out


def load(model_dir, cfg: Config) -> FlashMTP:
    """The MTP head from the checkpoint's shards (each module in its config.json format, its router bf16)."""

    from pathlib import Path

    import json

    from tensorfold.families.qwen4_exp.model import norms_stored_around_one, quant_params

    config = json.loads((Path(model_dir) / "config.json").read_text())
    weights: dict[str, mx.array] = {}
    for path in sorted(Path(model_dir).glob("model*.safetensors")):
        found = {k: v for k, v in mx.load(str(path), stream=mx.cpu).items() if k.startswith(PREFIXES)}
        weights.update(found)
    weights = sanitize(weights)
    mx.eval(list(weights.values()))          # read before any GPU work uses them (see model.load)
    head = FlashMTP(cfg)
    nn.quantize(head, group_size=cfg.group_size, bits=cfg.bits,
                class_predicate=lambda p, m: (hasattr(m, "to_quantized") and f"{p}.scales" in weights
                                              and quant_params(config, f"mtp.{p}")))
    # the same storage convention as the main model's centred norms (checked on its hc_norm anchors)
    main = {}
    for path in sorted(Path(model_dir).glob("model*.safetensors")):
        main.update({k[len("language_model."):]: v for k, v in mx.load(str(path), stream=mx.cpu).items()
                     if k.startswith("language_model.model.layers.") and k.endswith("attn_hyper_connection.hc_norm.weight")})
    if norms_stored_around_one(main):
        for path_, module in head.named_modules():
            if isinstance(module, CenteredRMSNorm) and f"{path_}.weight" in weights:
                weights[f"{path_}.weight"] = weights[f"{path_}.weight"].astype(mx.float32) - 1.0
    head.load_weights(list(weights.items()), strict=True)
    mx.eval(head.parameters())
    return head
