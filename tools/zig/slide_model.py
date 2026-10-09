"""Stock mlx-lm access for the Sliding Weights oracle: chat ids, greedy answers and a recording forward."""
from __future__ import annotations

import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import create_attention_mask, create_ssm_mask

USER_HEAD = "<|im_start|>user\n"


def open_model(path: str):
    return load(path)


def chat_ids(tok, user: str, system: str | None = None, answer: str | None = None) -> list[int]:
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tok.encode(text + (answer or ""), add_special_tokens=False)


def ask(model, tok, user: str, system: str | None = None, max_tokens: int = 48) -> str:
    return generate(model, tok, chat_ids(tok, user, system), max_tokens=max_tokens).strip()


def layers(model):
    return model.language_model.model.layers


def trace(model, ids: list[int], keep: tuple[int, ...]):
    """Residual after every layer, and gated shared-expert keys at the layers in keep, for one sequence."""
    text = model.language_model.model
    h = text.embed_tokens(mx.array(ids)[None])
    fa, ssm = create_attention_mask(h, None), create_ssm_mask(h, None)
    resid, keys = [], {}
    for i, layer in enumerate(text.layers):
        x = layer.input_layernorm(h)
        h = h + (layer.linear_attn(x, ssm, None) if layer.is_linear else layer.self_attn(x, fa, None))
        x = layer.post_attention_layernorm(h)
        if i in keep:
            se = layer.mlp.shared_expert
            keys[i] = (mx.sigmoid(layer.mlp.shared_expert_gate(x)) * swiglu(se.gate_proj(x), se.up_proj(x)))[0]
        h = h + layer.mlp(x)
        resid.append(h[0])
    return resid, keys


def parity(model, ids: list[int]) -> float:
    """Max abs gap between trace and the model's own forward, so model-code drift fails loudly."""
    text = model.language_model.model
    ours = text.norm(trace(model, ids, ())[0][-1][None]).astype(mx.float32)
    return float(mx.max(mx.abs(ours - text(mx.array(ids)[None]).astype(mx.float32))))
