"""The backbone from the checkpoint: tensors read shard by shard, each layer built and evaluated in turn."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx

from tensorfold.families.glm5_next import layouts
from tensorfold.families.glm5_next.config import BITS, GROUPS, Config, quant_formats, unreadable
from tensorfold.families.glm5_next.kda import KDA
from tensorfold.families.glm5_next.linear import Dense, Q, QSplit, bf16_if_exact, one_format
from tensorfold.families.glm5_next.mla import MLA
from tensorfold.families.glm5_next.mlp import DenseMLP, MoE
from tensorfold.families.glm5_next.model import GLM5, HC, Layer


class Weights:
    """The language model's tensors by short name in either layout (``layouts``), read shard by shard when asked."""

    def __init__(self, model_dir: Path, mtp_layer: int | None = None) -> None:
        index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
        self.dir = model_dir
        self.mtp_layer = mtp_layer
        self.layout = layouts.detect(index)
        self.where: dict[str, str] = {}
        for name, shard in index.items():
            short = self._short(name)
            if short is not None:
                self.where[short] = shard
        self.default, stated = quant_formats(json.loads((model_dir / "config.json").read_text()))
        self.overrides = {layouts.canonical(key, mtp_layer) or key: fmt for key, fmt in stated.items()}
        self._shard: tuple[str, dict[str, mx.array]] | None = None
        self._cache: dict[str, dict[str, mx.array]] = {}

    def _short(self, name: str) -> str | None:
        return layouts.canonical(name, self.mtp_layer)

    def has(self, name: str) -> bool:
        return name in self.where

    def get(self, name: str) -> mx.array:
        shard = self.where[name]
        loaded = self._cache.get(shard)
        if loaded is None:
            raw = mx.load(str(self.dir / shard))
            loaded = {}
            for full, value in raw.items():
                short = self._short(full)
                if short is not None:
                    loaded[short] = value
            self._cache = {shard: loaded}          # one shard at a time: arrays already taken stay alive
        return loaded[name]

    def q(self, prefix: str) -> Q:
        """A quantized linear at the format the config states for it, checked against its shapes."""

        fmt = self.overrides.get(prefix, self.default)
        if unreadable(fmt):
            stored = "unquantized" if fmt is None else f"{fmt[0]}-bit {fmt[2]} in groups of {fmt[1]}"
            raise ValueError(f"{prefix}: stored {stored}; GLM-5.3-Flash's Mac engine reads MLX affine weights of "
                             f"{', '.join(map(str, BITS))} bits in groups of {', '.join(map(str, GROUPS))}")
        try:
            return Q(self.get(f"{prefix}.weight"), self.get(f"{prefix}.scales"), self.get(f"{prefix}.biases"),
                     bits=fmt[0], group=fmt[1])
        except ValueError as exc:
            raise ValueError(f"{prefix}: {exc}") from None

    def linear(self, prefix: str) -> Q | Dense:
        """A linear as stored: quantized (``Q``) when it has scales, else the bf16 matrix (``Dense``)."""

        if self.has(f"{prefix}.scales"):
            return self.q(prefix)
        return Dense(self.get(f"{prefix}.weight"))


def _materialize(*arrays: Any) -> None:
    flat: list[mx.array] = []
    for a in arrays:
        if isinstance(a, (Q, QSplit, Dense)):
            flat += a.arrays()
        elif isinstance(a, mx.array):
            flat.append(a)
    mx.eval(*flat)


def _placeholder(w: Weights, m: str, proj: str) -> Q:
    """A routed projection's format and shapes with no experts: a stream's pool holds them (``stream.attach``)."""

    name = f"{m}.switch_mlp.{proj}" if w.has(f"{m}.switch_mlp.{proj}.weight") else f"{m}.experts.0.{proj}"
    q = w.q(name)                                        # lazy: its bytes are never read
    return Q(*(mx.zeros((0, *a.shape[-2:]), dtype=a.dtype) for a in q.arrays()), bits=q.bits, group=q.group)


def load_layer(w: Weights, i: int, cfg: Config, *, plain: bool = False, stream: bool = False) -> Layer:
    p = f"layers.{i}"
    attn_prefix = f"{p}.self_attn"
    if w.has(f"{attn_prefix}.q_a_proj.weight"):
        names = ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "o_proj", "indexer.wq_b", "indexer.wk",
                 "indexer.weights_proj"]
        # kv_b_proj as stored (vontra), or the absorbed pair the mlxlm layout keeps instead
        names += ["kv_b_proj"] if w.has(f"{attn_prefix}.kv_b_proj.weight") else ["embed_q", "unembed_out"]
        aw: dict[str, Any] = {n: w.q(f"{attn_prefix}.{n}") for n in names}
        for n in ("q_a_layernorm", "kv_a_layernorm"):
            aw[n] = w.get(f"{attn_prefix}.{n}.weight")
        for n in ("indexer.k_norm.weight", "indexer.k_norm.bias", "indexer.index_kpool_compress_ape",
                  "indexer.index_kpool_compress_gate"):
            aw[n] = w.get(f"{attn_prefix}.{n}")
        attn: Any = MLA(aw, cfg)
        _materialize(attn.x_proj, attn.qr_proj, attn.q_a, attn.q_b, attn.kv_a, attn.o_proj, attn.wk, attn.wv, attn.iq,
                     attn.ik_proj, attn.iw,
                     attn.q_norm, attn.kv_norm, attn.ik_norm_w, attn.ik_norm_b, attn.ape, attn.igate)
    else:
        names = ["q_proj", "k_proj", "v_proj", "f_a_proj", "f_b_proj", "g_a_proj", "g_b_proj", "b_proj", "o_proj"]
        aw = {n: w.q(f"{attn_prefix}.{n}") for n in names}
        aw["o_norm"] = w.get(f"{attn_prefix}.o_norm.weight")
        if w.has(f"{attn_prefix}.conv1d.weight"):                        # mlxlm: one conv over q | k | v
            aw["conv1d"] = w.get(f"{attn_prefix}.conv1d.weight")
        else:
            for n in ("q_conv1d", "k_conv1d", "v_conv1d"):
                aw[n] = w.get(f"{attn_prefix}.{n}.weight")
        aw["A_log"] = w.get(f"{attn_prefix}.A_log")
        aw["dt_bias"] = w.get(f"{attn_prefix}.dt_bias")
        attn = KDA(aw, cfg)
        _materialize(attn.in_proj, attn.f_b, attn.g_b, attn.o_proj, attn.conv_w, attn.A, attn.dt_bias, attn.o_norm)
    m = f"{p}.mlp"
    if w.has(f"{m}.gate.weight"):
        shared = None
        if w.has(f"{m}.shared_experts.gate_proj.weight"):
            shared = DenseMLP(w.q(f"{m}.shared_experts.gate_proj"), w.q(f"{m}.shared_experts.up_proj"),
                              w.q(f"{m}.shared_experts.down_proj"), cfg.swiglu_limit)
        stacked = []
        for proj in ("gate_proj", "up_proj", "down_proj"):
            if stream:
                stacked.append(_placeholder(w, m, proj))
                continue
            if w.has(f"{m}.switch_mlp.{proj}.weight"):
                q = w.q(f"{m}.switch_mlp.{proj}")
                _materialize(q)             # read now: a lazy load first touched in a server thread has no CPU stream
                stacked.append(q)
                continue
            parts = [w.q(f"{m}.experts.{e}.{proj}") for e in range(cfg.n_routed_experts)]
            bits, group = one_format(parts)
            q = Q(mx.stack([x.weight for x in parts]), mx.stack([x.scales for x in parts]),
                  mx.stack([x.biases for x in parts]), bits=bits, group=group)
            _materialize(q)
            stacked.append(q)
        mlp: Any = MoE(bf16_if_exact(w.get(f"{m}.gate.weight")), w.get(f"{m}.gate.e_score_correction_bias"), *stacked,
                       shared, cfg)
        _materialize(mlp.router, mlp.bias, mlp.scale_arr, mlp.limit_arr, mlp.router_packed,
                     *(mlp.shared.gate_up, mlp.shared.down) if shared else ())
    else:
        mlp = DenseMLP(w.q(f"{m}.gate_proj"), w.q(f"{m}.up_proj"), w.q(f"{m}.down_proj"), cfg.swiglu_limit)
        _materialize(mlp.gate_up, mlp.down)
    in_norm = w.get(f"{p}.input_layernorm.weight")
    post_norm = w.get(f"{p}.post_attention_layernorm.weight")
    attn_hc = ffn_hc = None
    if not plain:
        attn_hc = HC(bf16_if_exact(w.get(f"{p}.hc_attn_fn")), w.get(f"{p}.hc_attn_base"), w.get(f"{p}.hc_attn_scale"),
                     cfg)
        ffn_hc = HC(bf16_if_exact(w.get(f"{p}.hc_ffn_fn")), w.get(f"{p}.hc_ffn_base"), w.get(f"{p}.hc_ffn_scale"), cfg)
        _materialize(attn_hc.fn, attn_hc.base, attn_hc.scale, ffn_hc.fn, ffn_hc.base, ffn_hc.scale,
                     attn_hc.fn_packed, ffn_hc.fn_packed)
    _materialize(in_norm, post_norm)
    return Layer(attn, mlp, in_norm, post_norm, attn_hc, ffn_hc, cfg)


def load_backbone(model_dir: Path, *, layers: int | None = None, stream: bool = False) -> GLM5:
    """The backbone, layer by layer; ``layers``: only the first that many; ``stream``: routed experts left on disk."""

    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    cfg = Config.from_dict(config)
    w = Weights(model_dir, mtp_layer=cfg.num_hidden_layers)
    count = cfg.num_hidden_layers if layers is None else min(int(layers), cfg.num_hidden_layers)
    layers = [load_layer(w, i, cfg, stream=stream) for i in range(count)]
    embed = w.q("embed_tokens")
    lm_head = w.q("lm_head")
    norm = w.get("norm.weight")
    _materialize(embed, lm_head, norm)
    model = GLM5(cfg, embed, layers, norm, lm_head)
    model.weights = w                                                    # the MTP head reads its layer from it
    return model


def load(model_dir: Path, *, ssd_experts: float | None = None) -> tuple[GLM5, Any]:
    """The backbone and tokenizer; ``ssd_experts``: stream routed experts into a GPU pool of that many GiB."""

    from tensorfold.families.tokenizer import load_tokenizer

    from tensorfold.families.glm5_next.prompts import GlmTokenizer

    model = load_backbone(Path(model_dir), stream=bool(ssd_experts))
    if ssd_experts:
        from tensorfold.families.glm5_next import stream

        stream.attach(model, Path(model_dir), float(ssd_experts))
    tokenizer = GlmTokenizer(load_tokenizer(Path(model_dir), eos_token_ids=model.args.eos_token_id or None))
    return model, tokenizer

