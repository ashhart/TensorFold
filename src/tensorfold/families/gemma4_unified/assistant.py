"""Gemma 4's MTP assistant (``gemma4_unified_assistant``): a chain from the target's caches and one hidden row."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import mlx.core as mx
import mlx.nn as nn


class AssistantSlot:
    """A stream's last kept row for the assistant (its cache list's last entry); never stored with a prefix."""

    keys = None
    stored = False

    def __init__(self) -> None:
        self.hidden: mx.array | None = None      # [1, 1, D] the target's final-normed row
        self.position = 0                        # that row's absolute position
        self.token = 0                           # the token after it (the round's pending token)

    @property
    def state(self) -> list[Any]:
        return []

    @property
    def nbytes(self) -> int:
        return 0

    def __copy__(self) -> "AssistantSlot":
        return AssistantSlot()                   # a copy drafts again from its own next forward


class Assistant:
    """The assistant model and its chain drafting against a ``Gemma4Unified`` target."""

    # drafts a round the engine may ask for (it learns the depth that pays); the head was trained for 3 (block 4)
    default_drafts = 6

    def __init__(self, text: Any, pre: nn.Module, post: nn.Module, config: dict[str, Any]) -> None:
        self.text = text                         # mlx_lm gemma4_text.Model: embed_tokens (tied head), layers, norm
        self.pre, self.post = pre, post
        self.config = config
        self.window = int(text.args.sliding_window)
        self.target: Any = None
        self.kinds = [layer.layer_type for layer in text.model.layers]

    # -- loading -------------------------------------------------------------------------------------------------
    @classmethod
    def load(cls, path: Path) -> "Assistant":
        from mlx_lm.models import gemma4_text

        config = json.loads((Path(path) / "config.json").read_text())
        if config.get("model_type") not in ("gemma4_unified_assistant", "gemma4_assistant"):
            raise ValueError(f"{path} is not a Gemma 4 assistant (model_type {config.get('model_type')!r})")
        if config.get("use_ordered_embeddings"):
            raise ValueError("this Gemma 4 assistant uses ordered (centroid-masked) embeddings, which TensorFold "
                             "does not read yet")
        text_cfg = dict(config["text_config"])
        # built unshared (every layer reads the target's keys), no per-layer inputs, no double-wide MLP
        text_cfg.update(num_kv_shared_layers=0, hidden_size_per_layer_input=0, vocab_size_per_layer_input=0,
                        use_double_wide_mlp=False, final_logit_softcapping=None)
        text = gemma4_text.Model(gemma4_text.ModelArgs.from_dict(text_cfg))
        for layer in text.model.layers:
            attn = layer.self_attn
            for name in ("k_proj", "v_proj", "k_norm", "v_norm"):
                if name in attn:
                    del attn[name]
            attn.has_kv = False
        hidden, backbone = int(text_cfg["hidden_size"]), int(config["backbone_hidden_size"])
        holder = nn.Module()
        holder.text = text
        holder.pre_projection = nn.Linear(2 * backbone, hidden, bias=False)
        holder.post_projection = nn.Linear(hidden, backbone, bias=False)
        weights: dict[str, mx.array] = {}
        for file in sorted(Path(path).glob("*.safetensors")):
            weights.update(mx.load(str(file)))
        weights = {(k if k.startswith(("pre_projection", "post_projection")) else f"text.{k}"): v
                   for k, v in weights.items() if not k.startswith("lm_head") and "token_ordering" not in k}
        quant = config.get("quantization") or config.get("quantization_config")
        if quant:
            nn.quantize(holder, group_size=int(quant.get("group_size", 64)), bits=int(quant["bits"]),
                        mode=str(quant.get("mode", "affine")),
                        class_predicate=lambda p, m: hasattr(m, "to_quantized") and f"{p}.scales" in weights)
        holder.load_weights(list(weights.items()), strict=True)
        mx.eval(holder.parameters())
        from tensorfold.families.gemma4.model import realize

        realize(holder)                 # proportional RoPE's private ``_freqs`` too: the engine thread can't load them
        return cls(text, holder.pre_projection, holder.post_projection, config)

    def describe(self) -> str:
        q = self.config.get("quantization") or {}
        return f"{len(self.kinds)} layers, {q.get('bits', 16)}-bit"

    def bind(self, target: Any) -> None:
        """The target whose embedding and caches the assistant reads; its layer kinds must cover the assistant's."""

        self.target = target
        layers = target.text.model.layers
        self.sources = {}
        for kind in set(self.kinds):
            found = [i for i, layer in enumerate(layers) if layer.layer_type == kind]
            if not found:
                raise ValueError(f"the target has no {kind} layer for the assistant to read")
            self.sources[kind] = found[-1]                  # the last layer of each kind (HF's shared_kv_states)
        width = int(target.text.args.hidden_size)
        if int(self.config["backbone_hidden_size"]) != width:
            raise ValueError(f"the assistant was trained on a {self.config['backbone_hidden_size']}-wide target, "
                             f"this one is {width}")

    def slot(self) -> AssistantSlot:
        return AssistantSlot()

    # -- drafting ------------------------------------------------------------------------------------------------
    def _shared(self, layers: list[Any], position: int) -> dict[str, tuple[mx.array, mx.array]]:
        """Each kind's keys and values [1, Hk, S, D] at positions up to ``position`` (the window for sliding)."""

        out = {}
        for kind, index in self.sources.items():
            cache = layers[index]
            end = position + 1
            if kind == "sliding_attention":
                begin = max(0, end - self.window)
                out[kind] = (cache._ordered(cache.ring_keys, begin, end), cache._ordered(cache.ring_values, begin, end))
            else:
                out[kind] = (cache.keys[..., :end, :], cache.values[..., :end, :])
        return out

    def _embed(self, tokens: mx.array) -> mx.array:
        backbone = self.target.backbone
        return backbone.embed_tokens(tokens) * backbone.embed_scale

    def _step(self, x: mx.array, shared: dict[str, tuple[mx.array, mx.array]], offset: mx.array
              ) -> tuple[mx.array, mx.array]:
        """One assistant forward [B, 1, 2 * backbone] -> (projected hidden [B, 1, backbone], logits [B, 1, V])."""

        h = self.pre(x)
        for layer, kind in zip(self.text.model.layers, self.kinds):
            h, _, _ = layer(h, None, None, per_layer_input=None, shared_kv=shared[kind], offset=offset)
        h = self.text.model.norm(h)
        return self.post(h), self.text.model.embed_tokens.as_linear(h)

    @staticmethod
    def _pick(logits: mx.array, sampling: Any, position: int) -> mx.array:
        """The draft at ``position``: the argmax, or the draw keyed as the target keys its own at that position."""

        from tensorfold.engine.gpu_sampling import sample

        flat = logits.reshape(-1, logits.shape[-1])
        if sampling is None:
            return mx.argmax(flat, axis=-1).astype(mx.uint32)
        return sample(flat.astype(mx.float32), sampling, [int(position)]).astype(mx.uint32)

    def draft(self, layers: list[Any], hidden: mx.array, token: int, row_position: int, position: int, sampling: Any,
              count: int) -> mx.array:
        """``count`` drafts for positions ``position`` .. after ``token`` (lazy: they run with the next verify)."""

        shared = self._shared(layers, int(row_position))
        offset = mx.array(int(row_position))
        tok = mx.array([[int(token)]], dtype=mx.uint32)
        h = hidden.reshape(1, 1, -1)
        out = []
        for step in range(int(count)):
            x = mx.concatenate([self._embed(tok), h.astype(mx.bfloat16)], axis=-1)
            h, logits = self._step(x, shared, offset)
            picked = self._pick(logits, sampling, int(position) + step)
            out.append(picked)
            tok = picked.reshape(1, 1)
        return mx.concatenate(out)

    def draft_many(self, layers: Sequence[list[Any]], slots: Sequence[AssistantSlot], positions: Sequence[int],
                   samplings: Sequence[Any], counts: Sequence[int]) -> list[mx.array]:
        """``draft`` for several streams in one assistant forward a step: keys padded to the longest, masked."""

        if len(slots) == 1:
            s = slots[0]
            return [self.draft(layers[0], s.hidden, s.token, s.position, positions[0], samplings[0], counts[0])]
        from tensorfold.engine.gpu_sampling import sample_rows

        streams = len(slots)
        shared, masks = {}, {}
        per = [self._shared(l, int(s.position)) for l, s in zip(layers, slots)]
        for kind in self.sources:
            lengths = [int(p[kind][0].shape[2]) for p in per]
            longest = max(lengths)
            keys, values = [], []
            for p, n in zip(per, lengths):
                k, v = p[kind]
                if n < longest:
                    pad = [(0, 0), (0, 0), (0, longest - n), (0, 0)]
                    k, v = mx.pad(k, pad), mx.pad(v, pad)
                keys.append(k)
                values.append(v)
            shared[kind] = (mx.concatenate(keys), mx.concatenate(values))
            if min(lengths) < longest:
                inside = mx.arange(longest)[None, :] < mx.array(lengths)[:, None]
                masks[kind] = mx.where(inside, 0.0, -mx.inf).astype(mx.bfloat16)[:, None, None, :]
            else:
                masks[kind] = None
        offset = mx.array([int(s.position) for s in slots])
        tok = mx.array([[int(s.token)] for s in slots], dtype=mx.uint32)
        h = mx.concatenate([s.hidden.reshape(1, 1, -1) for s in slots])
        out = []
        for step in range(max(int(c) for c in counts)):
            x = mx.concatenate([self._embed(tok), h.astype(mx.bfloat16)], axis=-1)
            h = self.pre(x)
            for layer, kind in zip(self.text.model.layers, self.kinds):
                h, _, _ = layer(h, masks[kind], None, per_layer_input=None, shared_kv=shared[kind], offset=offset)
            h = self.text.model.norm(h)
            logits = self.text.model.embed_tokens.as_linear(h).reshape(streams, -1)
            h = self.post(h)
            if all(sm is None for sm in samplings):
                picked = mx.argmax(logits, axis=-1).astype(mx.uint32)
            else:
                picked = sample_rows(logits.astype(mx.float32), list(samplings),
                                     [int(p) + step for p in positions]).astype(mx.uint32)
            out.append(picked)
            tok = picked.reshape(streams, 1)
        drafted = mx.stack(out, axis=1)                      # [streams, steps]
        return [drafted[i, :int(c)] for i, c in enumerate(counts)]


__all__ = ["Assistant", "AssistantSlot"]
