"""Check k3_reference against Moonshot's modeling_kimi_linear.py on small random shapes (torch on the CPU)."""
import argparse
import importlib
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch

import k3_reference as ref
from k3_synth import bf16, mxfp4_decode, synthetic

FLA_SRC = None


def _naive():
    """FLA's own torch reference for the delta rule, loaded from the downloaded source (no Triton needed)."""
    spec = importlib.util.spec_from_file_location("fla_naive_kda", Path(FLA_SRC) / "fla/ops/kda/naive.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.naive_recurrent_kda


class ShortConvolution(torch.nn.Conv1d):
    """Stand-in for fla's ShortConvolution: depthwise causal conv, silu, cache of the last K inputs."""

    def __init__(self, hidden_size, kernel_size, activation="silu", bias=False, **_):
        super().__init__(hidden_size, hidden_size, kernel_size, groups=hidden_size, bias=bias, padding=kernel_size - 1)

    def forward(self, x, cache=None, output_final_state=False, cu_seqlens=None, **_):
        B, T, D = x.shape
        K = self.kernel_size[0]
        prev = cache.transpose(1, 2)[:, 1:] if cache is not None else x.new_zeros(B, K - 1, D)
        full = torch.cat([prev.to(x.dtype), x], 1).float()
        w = self.weight.float().reshape(D, K)
        y = sum(full[:, j:j + T] * w[:, j] for j in range(K))
        y = (y * torch.sigmoid(y)).to(x.dtype)
        state = full[:, -K:].to(x.dtype).transpose(1, 2).contiguous()
        return y, (state if output_final_state else None)


class FusedRMSNormGated(torch.nn.Module):
    """Stand-in for fla's gated RMS norm: fp32 norm, weight, times sigmoid(g), back to x's dtype."""

    def __init__(self, hidden_size, eps=1e-5, activation="sigmoid", **_):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x, g):
        xf = x.float()
        y = xf / torch.sqrt((xf * xf).mean(-1, keepdim=True) + self.eps) * self.weight.float()
        return (y * torch.sigmoid(g.float())).to(x.dtype)


def kda_op(q, k, v, g, beta, A_log, dt_bias, initial_state=None, lower_bound=None, **_):
    """fla's fused_recurrent_kda in-kernel steps (l2 norm, gate, beta sigmoid) around FLA's naive recurrence."""
    H, D = q.shape[-2], q.shape[-1]
    qf, kf = q.float(), k.float()
    qf = qf / torch.sqrt((qf * qf).sum(-1, keepdim=True) + 1e-6)
    kf = kf / torch.sqrt((kf * kf).sum(-1, keepdim=True) + 1e-6)
    gk = lower_bound * torch.sigmoid(A_log.float()[:H].exp().view(H, 1) * (g.float() + dt_bias.float().view(H, D)))
    o, state = _naive()(qf, kf, v.float(), gk, torch.sigmoid(beta.float()), initial_state=initial_state,
                        output_final_state=True)
    return o.to(v.dtype), state


def install_fla_stubs():
    """Register the fla names the official file imports; the delta rule itself is FLA's naive reference."""
    mods = {n: types.ModuleType(n) for n in ("fla", "fla.modules", "fla.ops", "fla.ops.kda", "fla.ops.utils",
                                              "fla.ops.utils.index", "fla.utils")}
    mods["fla.modules"].FusedRMSNormGated = FusedRMSNormGated
    mods["fla.modules"].ShortConvolution = ShortConvolution
    mods["fla.ops.kda"].chunk_kda = kda_op
    mods["fla.ops.kda"].fused_recurrent_kda = kda_op
    mods["fla.ops.utils.index"].prepare_cu_seqlens_from_mask = None
    mods["fla.ops.utils.index"].prepare_lens_from_mask = None
    mods["fla.utils"].tensor_cache = lambda f: f
    sys.modules.update(mods)


def shim_transformers():
    """transformers 5 moved two names the 4.56-era file imports; identity stand-ins keep its forward unchanged."""
    import transformers.utils.generic as generic
    from transformers.utils.output_capturing import OutputRecorder
    generic.OutputRecorder = getattr(generic, "OutputRecorder", OutputRecorder)
    generic.check_model_inputs = getattr(generic, "check_model_inputs", lambda f: f)


def load_official(src: Path):
    shim_transformers()
    pkg = types.ModuleType("k3official")
    pkg.__path__ = [str(src)]
    sys.modules["k3official"] = pkg
    install_fla_stubs()
    return importlib.import_module("k3official.modeling_kimi_linear")


def small_config(cfg_mod_cls):
    """A five-layer K3 in miniature: KDA dense, KDA MoE, MLA MoE, KDA MoE, MLA MoE; blocks of 2."""
    c = ref.Config(hidden=128, layers=5, vocab=512, dense_inter=192, experts=8, topk=3, shared=2, moe_inter=64,
                   latent=96, kda_heads=4, kda_dim=32, mla_heads=4, q_lora=48, kv_lora=32, nope=16, rope=8,
                   v_dim=16, block=2, full_attn=(3, 5))
    off = cfg_mod_cls(
        vocab_size=c.vocab, hidden_size=c.hidden, intermediate_size=c.dense_inter, num_hidden_layers=c.layers,
        num_attention_heads=c.mla_heads, num_key_value_heads=c.mla_heads, hidden_act="situ", rms_norm_eps=c.eps,
        moe_intermediate_size=c.moe_inter, num_experts=c.experts, num_experts_per_token=c.topk,
        num_shared_experts=c.shared, first_k_dense_replace=1, q_lora_rank=c.q_lora, kv_lora_rank=c.kv_lora,
        qk_nope_head_dim=c.nope, qk_rope_head_dim=c.rope, v_head_dim=c.v_dim, mla_use_nope=True,
        mla_use_output_gate=True, attn_res_block_size=c.block, latent_moe_use_norm=True,
        activation_situ_beta=c.situ_beta, activation_situ_linear_beta=c.situ_linear, routed_expert_hidden_size=c.latent,
        linear_attn_config={"full_attn_layers": list(c.full_attn), "kda_layers": [1, 2, 4], "head_dim": c.kda_dim,
                            "num_heads": c.kda_heads, "short_conv_kernel_size": c.conv, "gate_lower_bound": -5.0,
                            "use_full_rank_gate": True}, pad_token_id=0, bos_token_id=1, eos_token_id=2)
    return c, off


def synth_weights(model, cfg):
    """Our synthetic values for every official parameter, and the same values under the checkpoint's names."""
    W = {}
    for name, prm in model.named_parameters():
        key = name.removeprefix("model.")
        shape = tuple(prm.shape)
        if ".experts." in key:
            K = shape[1]
            packed = synthetic(key + "_packed", (shape[0], K // 2))
            scale = synthetic(key + "_scale", (shape[0], K // 32))
            W[key.replace(".weight", ".weight_packed")], W[key.replace(".weight", ".weight_scale")] = packed, scale
            val = mxfp4_decode(packed, scale)
        else:
            val = synthetic(key, shape)
        W[key] = val
        with torch.no_grad():
            prm.copy_(torch.from_numpy(np.ascontiguousarray(val)).to(prm.dtype))
        if prm.dtype == torch.bfloat16 and ".experts." not in key:
            W[key] = bf16(val)
    return W


def compare(name, ours, theirs, limit):
    """Print one comparison; True when the largest difference is within `limit` of the largest value."""
    ours, theirs = np.asarray(ours, np.float64), np.asarray(theirs, np.float64)
    rel = np.abs(ours - theirs).max() / np.abs(theirs).max()
    print(f"{name:34s} rel {rel:.2e}  bf16 words equal {np.mean(ours == theirs) * 100:5.1f}%  (limit {limit:.0e})")
    return rel <= limit


def capture(model):
    """Each layer's attention and MLP inputs and outputs, recorded by forward hooks."""
    io = {}

    def hook(name):
        def f(_m, args, kwargs, out):
            x = args[0] if args else kwargs["hidden_states"]
            o = out[0] if isinstance(out, tuple) else out
            io[name] = (x.float().numpy().reshape(-1, x.shape[-1]), o.float().numpy().reshape(-1, o.shape[-1]))
        return f

    for i, lay in enumerate(model.model.layers):
        lay.self_attn.register_forward_hook(hook(f"attn{i}"), with_kwargs=True)
        mlp = lay.block_sparse_moe if hasattr(lay, "block_sparse_moe") else lay.mlp
        mlp.register_forward_hook(hook(f"mlp{i}"), with_kwargs=True)
    return io


def sublayers(io, model, W, cfg) -> bool:
    """Every sublayer of ours on the official layer's own inputs (one prefill call)."""
    ok = True
    for i in range(cfg.layers):
        p = f"layers.{i}."
        x, theirs = io[f"attn{i}"]
        if cfg.is_kda(i):
            ok &= compare(f"layer {i} KDA", ref.kda(x, W, p + "self_attn.", cfg, ref.kda_state(cfg)), theirs, 8e-3)
        else:
            ok &= compare(f"layer {i} MLA plain form", ref.mla_plain(x, W, p + "self_attn.", cfg, []), theirs, 8e-3)
            ok &= compare(f"layer {i} MLA absorbed (ours)", ref.mla(x, W, p + "self_attn.", cfg, []), theirs, 1.6e-2)
        x, theirs = io[f"mlp{i}"]
        if i == 0:
            ok &= compare("layer 0 dense MLP", ref.mlp(x, *(W[f"{p}mlp.{n}_proj.weight"] for n in
                                                           ("gate", "up", "down")), cfg), theirs, 8e-3)
            continue
        ours, idx, _ = ref.moe(x, W, p + "block_sparse_moe.", cfg)
        with torch.no_grad():
            tidx, _ = model.model.layers[i].block_sparse_moe.gate(torch.from_numpy(x).to(torch.bfloat16)[None])
        same = all(set(a) == set(b) for a, b in zip(idx, tidx.numpy()))
        print(f"layer {i} router: expert sets {'equal' if same else 'DIFFER'}")
        ok &= same and compare(f"layer {i} latent MoE", ours, theirs, 8e-3)
    return ok


def main() -> int:
    global FLA_SRC
    ap = argparse.ArgumentParser()
    ap.add_argument("--official", required=True, help="folder with modeling_kimi_linear.py and configuration_kimi_k3.py")
    ap.add_argument("--fla", required=True, help="unpacked fla-core source (fla/ops/kda/naive.py)")
    a = ap.parse_args()
    FLA_SRC = a.fla
    mod = load_official(Path(a.official))
    conf = importlib.import_module("k3official.configuration_kimi_k3")
    cfg, off = small_config(conf.KimiLinearConfig)
    model = mod.KimiLinearForCausalLM(off).to(torch.bfloat16).eval()
    model.config._attn_implementation = "sdpa"
    W = synth_weights(model, cfg)
    io = capture(model)
    tokens = (np.arange(7) * 37 + 11) % cfg.vocab
    with torch.no_grad():
        cache = mod.KimiDynamicCache(off)
        logits = [model(input_ids=torch.tensor(tokens[None, :5]), past_key_values=cache, use_cache=True).logits[0]]
        ok = sublayers(io, model, W, cfg)
        for t in tokens[5:]:
            logits.append(model(input_ids=torch.tensor([[t]]), past_key_values=cache, use_cache=True).logits[0])
    theirs = torch.cat(logits).float().numpy()
    stream = ref.Stream()
    ours = np.concatenate([ref.forward(tokens[:5], W, cfg, stream)] + [ref.forward([t], W, cfg, stream)
                                                                      for t in tokens[5:]])
    agree = np.mean(ours.argmax(-1) == theirs.argmax(-1))
    compare("logits: 5-token prefill, 2 decodes", ours, theirs, 1.0)
    print(f"next-token argmax agrees on {agree * 100:.0f}% of rows")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
