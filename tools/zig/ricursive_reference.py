"""A Ripple (ricursive) checkpoint's teacher-forced logits on the CPU, a layer at a time from its safetensors."""
import importlib.util
import json
import sys
import time

import numpy as np
import torch
from safetensors import safe_open

USAGE = """usage: ricursive_reference.py MODEL_DIR tokens.json out.bin [--bf16-points | --release]
writes raw bf16 logits [positions, vocab]; an NVFP4 release's MLPs are its exact dequantized values
  (default)      float64: bf16 weights cast up exactly, the release's arithmetic, one rounding at the end
  --bf16-points  float64 rounded to bf16 at every point the release's bf16 forward rounds
  --release      the checkpoint's own modeling_ricursive.py layers in bf16 (bf16 releases only)"""


def bf16_once(x: np.ndarray) -> np.ndarray:
    """float64 to bf16 bits, rounded once to nearest (ties to even) at bf16's spacing."""
    spacing = np.ldexp(1.0, np.frexp(np.abs(x))[1] - 1 - 7)
    y = (np.round(x / spacing) * spacing).astype(np.float32)  # exact: y has at most 8 significant bits
    return (y.view(np.uint32) >> 16).astype(np.uint16)


def checked(cfg: dict) -> None:
    """Only the release's layer: gated (fp32 gate), value residuals, sliding and global layers, no scaled RoPE."""
    want = {"model_type": "ricursive", "use_gated_attention": True, "gate_full_precision": True, "use_valres": True,
            "is_hybrid_swa": True, "tie_word_embeddings": False, "hidden_act": "silu"}
    for key, value in want.items():
        if cfg.get(key) != value:
            raise SystemExit(f"config {key} is {cfg.get(key)!r}, not the release's {value!r}")
    q = cfg.get("quantization_config")
    if q and (q.get("quant_method") != "modelopt" or q.get("quant_algo") != "NVFP4"):
        raise SystemExit("only the publisher's ModelOpt NVFP4 MLPs are read")
    if cfg.get("rope_scaling"):
        raise SystemExit("scaled RoPE is not the release's")


def main(model_dir: str, tokens: str, out: str, mode: str = "") -> None:
    torch.set_num_threads(24)
    cfg = json.load(open(f"{model_dir}/config.json"))
    checked(cfg)
    index = json.load(open(f"{model_dir}/model.safetensors.index.json"))["weight_map"]
    ids = json.load(open(tokens))
    n = len(ids)

    def tensor(name):
        with safe_open(f"{model_dir}/{index[name]}", framework="pt") as f:
            return f.get_tensor(name)

    t0 = time.time()
    with torch.no_grad():
        h = release(model_dir, cfg, tensor, ids) if mode == "--release" else exact(cfg, tensor, ids, mode == "--bf16-points", index)
        head = tensor("lm_head.weight").to(h.dtype)
        with open(out, "wb") as f:
            for s in range(0, n, 64):
                rows = (h[s : s + 64] @ head.T).to(torch.float64).numpy()
                bf16_once(rows).tofile(f)
    print(f"{n} positions' logits ({mode or 'float64'}) in {time.time() - t0:.0f} s")


E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float64)


def nvfp4(packed, scale, global_):
    """ModelOpt NVFP4 [n, k] in float64: E2M1 nibbles (low first) x each block's E4M3 scale x the global, exact."""
    n = packed.shape[0]
    codes = torch.stack((packed & 15, packed >> 4), -1).reshape(n, -1).long()
    blocks = scale.to(torch.float64).repeat_interleave(16, 1)
    return E2M1[codes] * blocks * global_.to(torch.float64)


def exact(cfg, tensor, ids, points, index):
    """The release's forward in float64; with `points`, rounded to bf16 where its bf16 code rounds."""
    f64 = torch.float64
    layers, heads, kv, dim = cfg["num_hidden_layers"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    eps, theta, window, clamp = cfg["rms_norm_eps"], float(cfg["rope_theta"]), cfg["sliding_window"], cfg["ffn_clamp_threshold"]
    n = len(ids)

    def w(name):
        if name.endswith(".weight") and f"{name}_scale" in index:  # an NVFP4 projection, its exact values
            return nvfp4(tensor(name), tensor(f"{name}_scale"), tensor(f"{name}_scale_2"))
        return tensor(name).to(f64)

    def r(x):  # a rounding point: bf16 when `points`, else exact
        return x.to(torch.bfloat16).to(f64) if points else x

    def norm(x, g):
        return r(g * r(x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)))

    def rotate(x):
        return torch.cat((-x[..., dim // 2 :], x[..., : dim // 2]), -1)

    pos = torch.arange(n, dtype=f64)
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=f64) / dim))
    angle = torch.outer(pos, inv)
    cos, sin = torch.cat((angle, angle), -1).cos(), torch.cat((angle, angle), -1).sin()
    if points:  # the release's fp32 angle, its cos and sin rounded to bf16
        inv32 = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        a32 = torch.outer(torch.arange(n, dtype=torch.float32), inv32)
        cos, sin = r(torch.cat((a32, a32), -1).cos().to(f64)), r(torch.cat((a32, a32), -1).sin().to(f64))
    key, query = torch.arange(n)[None, :], torch.arange(n)[:, None]
    causal = torch.where(key <= query, 0.0, float("-inf")).to(f64)
    sliding = torch.where((key <= query) & (key >= query - (window - 1)), 0.0, float("-inf")).to(f64)
    scale = cfg["embed_scale"]
    x = r(w("model.embed_tokens.weight")[torch.tensor(ids)] * (np.float32(scale) if points else scale))
    v0 = x
    t0 = time.time()
    for i in range(layers):
        p, global_ = f"model.layers.{i}.", cfg["layer_types"][i] == "full_attention"
        a = p + "self_attn."
        hs = norm(x, w(p + "input_layernorm.weight"))
        blend = r(torch.sigmoid(w(a + "value_res_lambda")[0]))
        vin = r(torch.lerp(v0, hs, blend))
        q = r(hs @ w(a + "q_proj.weight").T).view(n, heads, dim)
        k = r(hs @ w(a + "k_proj.weight").T).view(n, kv, dim)
        v = r(vin @ w(a + "v_proj.weight").T).view(n, kv, dim)
        q, k = norm(q, w(a + "q_norm.weight")), norm(k, w(a + "k_norm.weight"))
        q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        if not global_:
            q, k = r(q * cos + rotate(q) * sin), r(k * cos + rotate(k) * sin)
        k, v = k.repeat_interleave(heads // kv, 0), v.repeat_interleave(heads // kv, 0)
        att = r(torch.softmax(q @ k.transpose(1, 2) / dim**0.5 + (causal if global_ else sliding), -1) @ v)
        att = att.transpose(0, 1).reshape(n, heads * dim)
        att = r(att * r(torch.sigmoid(r(hs @ w(a + "g_proj.weight").T))))  # the gate's sigmoid in fp32, then bf16
        x = r(x + norm(r(att @ w(a + "o_proj.weight").T), w(p + "post_attention_layernorm.weight")))
        hs = norm(x, w(p + "pre_feedforward_layernorm.weight"))
        gate = r(hs @ w(p + "mlp.gate_proj.weight").T).clamp(max=clamp)
        up = r(hs @ w(p + "mlp.up_proj.weight").T).clamp(-clamp, clamp)
        act = r(r(torch.nn.functional.silu(gate)) * up)
        x = r(x + norm(r(act @ w(p + "mlp.down_proj.weight").T), w(p + "post_feedforward_layernorm.weight")))
        print(f"layer {i + 1} of {layers}, {time.time() - t0:.0f} s", flush=True)
    return norm(x, w("model.norm.weight"))


def release(model_dir, cfg, tensor, ids):
    """The checkpoint's own decoder layers and norms in bf16, each layer's weights loaded alone."""
    sys.path.insert(0, model_dir)  # its modeling file imports configuration_ricursive as a package sibling
    pkg = importlib.util.spec_from_file_location("ricursive_release", f"{model_dir}/modeling_ricursive.py",
                                                 submodule_search_locations=[model_dir])
    spec_c = importlib.util.spec_from_file_location("ricursive_release.configuration_ricursive",
                                                    f"{model_dir}/configuration_ricursive.py")
    conf = importlib.util.module_from_spec(spec_c)
    sys.modules[spec_c.name] = conf
    spec_c.loader.exec_module(conf)
    spec_m = importlib.util.spec_from_file_location("ricursive_release.modeling_ricursive", f"{model_dir}/modeling_ricursive.py")
    mod = importlib.util.module_from_spec(spec_m)
    sys.modules["ricursive_release"] = importlib.util.module_from_spec(pkg)
    sys.modules[spec_m.name] = mod
    spec_m.loader.exec_module(mod)
    config = conf.RicursiveConfig.from_pretrained(model_dir)
    bf = torch.bfloat16
    n = len(ids)
    x = tensor("model.embed_tokens.weight")[torch.tensor(ids)].unsqueeze(0) * config.embed_scale
    v0 = x
    position_ids = torch.arange(n).unsqueeze(0)
    t0 = time.time()
    for i in range(config.num_hidden_layers):
        layer = mod.RicursiveDecoderLayer(config, i).to(bf)
        prefix = f"model.layers.{i}."
        state = {k[len(prefix):]: tensor(k) for k in index_names(model_dir, prefix)}
        layer.load_state_dict(state, strict=True)
        x = layer(x, position_ids=position_ids, value_residual=v0)[0]
        print(f"layer {i + 1} of {config.num_hidden_layers}, {time.time() - t0:.0f} s", flush=True)
    final = mod.RicursiveRMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(bf)
    final.load_state_dict({"weight": tensor("model.norm.weight")})
    return final(x)[0]


def index_names(model_dir, prefix):
    return [k for k in json.load(open(f"{model_dir}/model.safetensors.index.json"))["weight_map"] if k.startswith(prefix)]


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5) or sys.argv[4:] not in ([], ["--bf16-points"], ["--release"]):
        sys.exit(USAGE)
    main(*sys.argv[1:4], mode=sys.argv[4] if len(sys.argv) > 4 else "")
