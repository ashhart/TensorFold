#!/usr/bin/env python3
"""Ling-3.1-flash EXL3 layer parity: exllamav3's blocks on a real text against TensorFold's ling-exl3-layers."""

# ref PACK REF.bin LAYERS [T]: exllamav3's blocks 0..max(LAYERS) on T tokens, no cache; listed blocks' rows (TFLX3L01).

# REF.bin.floor.pt holds the reference's own noise: each block re-run on a prefix of its rows, router margins in fp32.

# compare PACK REF.bin OUT.bin: TensorFold's outputs (ling-exl3-layers) against the reference, beside that floor.

# Run with exllamav3 v1.6.6 installed and TORCH_ALLOW_TF32_CUBLAS_OVERRIDE unset.
import json
import os
import struct
import sys

import torch

# LING_POSITION: the rope's first position (exllamav3's params["position"]; keys are still only these T tokens)
POSITION = int(os.environ.get("LING_POSITION", "0"))


def params_():
    p = {"attn_mode": "flash_attn_nc"}
    if POSITION:
        p["position"] = POSITION
    return p


def rope_info(model, out):
    """Every rope object's inv_freq and attn_factor in the loaded modules (YaRN's table as exllamav3 built it)."""
    seen = {}
    for m in model.modules:
        for sub in [m] + list(getattr(m, "modules", []) or []):
            for name in dir(sub):
                o = getattr(sub, name, None)
                if o is not None and hasattr(o, "inv_freq") and hasattr(o, "attn_factor") and id(o) not in seen:
                    f = o.inv_freq
                    seen[id(o)] = {"where": f"{type(sub).__name__}.{name}", "attn_factor": float(o.attn_factor),
                                   "inv_freq": [float(x) for x in (f.flatten().tolist() if torch.is_tensor(f) else f)]}
    json.dump(list(seen.values()), open(out, "w"), indent=1)
    print("rope objects:", [(v["where"], v["attn_factor"], len(v["inv_freq"])) for v in seen.values()])

TEXT = (
    "def merge_intervals(intervals):\n"
    "    \"\"\"Merge overlapping [start, end] intervals and return them sorted.\"\"\"\n"
    "    out = []\n"
    "    for s, e in sorted(intervals):\n"
    "        if out and s <= out[-1][1]:\n"
    "            out[-1][1] = max(out[-1][1], e)\n"
    "        else:\n"
    "            out.append([s, e])\n"
    "    return out\n\n"
    "The function above sorts the intervals by their start, then walks them once. Each interval either extends "
    "the last merged one or starts a new one, so the whole pass costs O(n log n) for the sort and O(n) after it. "
    "Ling-3.1-flash mixes linear attention with a latent attention layer every eighth block."
)


def read_ref(path):
    b = open(path, "rb").read()
    assert b[:8] == b"TFLX3L01"
    T, D, n = struct.unpack_from("<3I", b, 8)
    at = 20
    ids = list(struct.unpack_from(f"<{T}i", b, at))
    at += 4 * T
    layers = list(struct.unpack_from(f"<{n}I", b, at))
    at += 4 * n
    v = torch.frombuffer(bytearray(b[at:]), dtype=torch.float32).view(n, 2, T, D)
    return ids, layers, v


def route(y, wg, bias, c):
    """Ling's grouped sigmoid top-k in fp32 (DeepSeek-V3 style): picks [T][k], the k-th vs next margin, group margin."""
    s = torch.sigmoid(y.float() @ wg.float().t())
    ch = s + bias.float()
    E, G, keep, k = s.shape[-1], c["n_group"], c["topk_group"], c["num_experts_per_tok"]
    gs = ch.view(-1, G, E // G).topk(2, dim=-1).values.sum(-1)
    gt = gs.topk(G, dim=-1).values
    kept = gs.topk(keep, dim=-1).indices
    mask = torch.zeros_like(gs, dtype=torch.bool).scatter_(1, kept, True)
    m = ch.masked_fill(~mask.repeat_interleave(E // G, dim=1), float("-inf"))
    top = m.topk(k + 1, dim=-1)
    return top.indices[:, :k], (top.values[:, k - 1] - top.values[:, k]), (gt[:, keep - 1] - gt[:, keep])


def ref(pack, out, layers, T):
    from exllamav3 import Config, Model
    from safetensors import safe_open
    from tokenizers import Tokenizer

    config = Config.from_directory(pack)
    model = Model.from_config(config)
    dev = torch.device("cuda:0")
    nb = max(layers) + 1
    mods = model.modules[: model.first_block_idx + nb]
    for m in mods:
        m.load(dev)
    tok = Tokenizer.from_file(os.path.join(pack, "tokenizer.json"))
    ids = tok.encode(TEXT).ids[:T]
    assert len(ids) == T, f"text gives {len(ids)} tokens"
    cfg = json.load(open(os.path.join(pack, "config.json")))
    index = json.load(open(os.path.join(pack, "model.safetensors.index.json")))["weight_map"]

    def tensor(name):
        with safe_open(os.path.join(pack, index[name]), framework="pt") as f:
            return f.get_tensor(name)

    def run_blocks(n_ids, x_in=None, only=None):
        """Embedding then blocks; or one block `only` on x_in. Every block's input and output kept (fp32)."""
        params = params_()
        x = model.prepare_inputs(torch.tensor([n_ids], dtype=torch.long), params)
        caps, mlp_in = {}, {}
        hooks = []
        for m in mods[model.first_block_idx:]:
            if getattr(m, "mlp", None) is not None:
                orig = m.mlp.forward

                def fwd(y, *a, i=m.layer_idx, orig=orig, **k):
                    mlp_in[i] = y.detach().float().cpu().clone()
                    return orig(y, *a, **k)
                m.mlp.forward = fwd
                hooks.append((m.mlp, orig))
        with torch.inference_mode():
            if only is None:
                for m in mods:
                    x = m.prepare_for_device(x, params)
                    xin = x.detach().float().cpu().clone() if m in mods[model.first_block_idx:] else None
                    x = m.forward(x, params)
                    if xin is not None:
                        caps[m.layer_idx] = (xin[0], x.detach().float().cpu().clone()[0])
            else:
                m = mods[model.first_block_idx + only]
                xx = x_in[None].to(dev, dtype=torch.float)
                y = m.forward(m.prepare_for_device(xx, params), params)
                caps[only] = (x_in, y.detach().float().cpu().clone()[0])
        for mlp, orig in hooks:
            mlp.forward = orig
        return caps, mlp_in

    caps, mlp_in = run_blocks(ids)
    D = caps[0][0].shape[-1]
    with open(out, "wb") as f:
        f.write(b"TFLX3L01" + struct.pack("<3I", T, D, len(layers)))
        f.write(struct.pack(f"<{T}i", *ids))
        f.write(struct.pack(f"<{len(layers)}I", *layers))
        for li in layers:
            for t in caps[li]:
                f.write(t.contiguous().numpy().tobytes())
    floor = {}
    for li in layers:
        xin, yout = caps[li]
        for P in (T // 2, T - 16):
            c2, _ = run_blocks(ids[:P], x_in=xin[:P].clone(), only=li)
            floor[f"{li}.prefix{P}"] = c2[li][1]
        # the same rows again in one more full run: a run-to-run difference would show here
        c3, _ = run_blocks(ids, x_in=xin.clone(), only=li)
        floor[f"{li}.again"] = c3[li][1]
        # the reference with a bf16 residual (as TensorFold keeps it): input rounded, output rounded
        c4, _ = run_blocks(ids, x_in=xin.bfloat16().float(), only=li)
        floor[f"{li}.bf16io"] = c4[li][1].bfloat16().float()
        if li in mlp_in and f"model.layers.{li}.mlp.gate.weight" in index:
            p, mk, mg = route(mlp_in[li][0], tensor(f"model.layers.{li}.mlp.gate.weight"),
                              tensor(f"model.layers.{li}.mlp.gate.expert_bias"), cfg)
            floor[f"{li}.picks"], floor[f"{li}.margin"], floor[f"{li}.gmargin"] = p, mk, mg
    torch.save(floor, out + ".floor.pt")
    rope_info(model, out + ".rope.json")
    print(f"position {POSITION}; saved {len(layers)} layers of {T} x {D}, floor runs {sorted(floor)}")


def stats(a, b, x):
    """rel L2 of a vs b, of (a - b) against the layer's change (b - x), and the worst row of the latter."""
    e = (a - b).double()
    d = (b - x).double()
    row = e.norm(dim=-1) / d.norm(dim=-1).clamp_min(1e-12)
    return float(e.norm() / b.double().norm()), float(e.norm() / d.norm()), float(row.max()), int(row.argmax())


def compare(pack, refp, outp):
    ids, layers, v = read_ref(refp)
    floor = torch.load(refp + ".floor.pt")
    b = open(outp, "rb").read()
    assert b[:8] == b"TFLX3O01"
    T, D, n, top = struct.unpack_from("<4I", b, 8)
    at = 24
    res = {"T": T, "layers": {}}
    for j, li in enumerate(layers):
        prompt = torch.frombuffer(bytearray(b[at:at + T * D * 4]), dtype=torch.float32).view(T, D)
        at += T * D * 4
        dec = torch.frombuffer(bytearray(b[at:at + T * D * 4]), dtype=torch.float32).view(T, D)
        at += T * D * 4
        pk = torch.frombuffer(bytearray(b[at:at + T * (top + 1) * 4]), dtype=torch.int32).view(T, top + 1)[:, :top]
        at += T * (top + 1) * 4
        dk = torch.frombuffer(bytearray(b[at:at + T * (top + 1) * 4]), dtype=torch.int32).view(T, top + 1)[:, :top]
        at += T * (top + 1) * 4
        x, y = v[j, 0], v[j, 1]
        r = {}
        bfx = x.bfloat16().float()
        r["bf16_io_floor"] = stats(y.bfloat16().float(), y, x)[1]  # our bf16 residual's rounding alone
        for P in (T // 2, T - 16):
            r[f"ref_prefix{P}"] = stats(floor[f"{li}.prefix{P}"], y[:P], x[:P])
            r[f"tf_prompt_rows{P}"] = stats(prompt[:P], y[:P], x[:P])
        r["ref_again"] = stats(floor[f"{li}.again"], y, x)
        if f"{li}.rerun" in floor:
            r["ref_rerun"] = stats(floor[f"{li}.rerun"], y, x)
        if f"{li}.bf16io" in floor:
            r["ref_bf16_residual"] = stats(floor[f"{li}.bf16io"], y, x)
            r["tf_prompt_vs_ref_bf16_residual"] = stats(prompt, floor[f"{li}.bf16io"], x)
        if f"{li}.emu" in floor:
            r["ref_bf16_act"] = stats(floor[f"{li}.emu"], y, x)
            r["tf_prompt_vs_ref_bf16_act"] = stats(prompt, floor[f"{li}.emu"], x)
        r["tf_prompt"] = stats(prompt, y, x)
        r["tf_decode"] = stats(dec, y, x)
        r["tf_prompt_vs_decode"] = stats(prompt, dec, x)
        if f"{li}.picks" in floor:
            rp = floor[f"{li}.picks"].sort(-1).values
            fl = [(pk[t].long().sort().values != rp[t]).any().item() for t in range(T)]
            fld = [(dk[t].long().sort().values != rp[t]).any().item() for t in range(T)]
            mk, mg = floor[f"{li}.margin"], floor[f"{li}.gmargin"]
            r["flips_prompt"] = [(t, float(mk[t]), float(mg[t])) for t in range(T) if fl[t]]
            r["flips_decode"] = [(t, float(mk[t]), float(mg[t])) for t in range(T) if fld[t]]
            keep = torch.tensor([not (fl[t] or fld[t]) for t in range(T)])
            r["tf_prompt_unflipped"] = stats(prompt[keep], y[keep], x[keep])
            r["min_margin"] = float(mk.min())
        res["layers"][li] = r
        fl_txt = ""
        if "flips_prompt" in r:
            keep = torch.tensor([t not in {f[0] for f in r["flips_prompt"] + r["flips_decode"]} for t in range(T)])
            if "ref_bf16_residual" in r:
                r["ref_bf16_residual_unflipped"] = stats(floor[f"{li}.bf16io"][keep], y[keep], x[keep])
            if "ref_bf16_act" in r:
                r["ref_bf16_act_unflipped"] = stats(floor[f"{li}.emu"][keep], y[keep], x[keep])
            fl_txt = (f"  | flips prompt {len(r['flips_prompt'])} decode {len(r['flips_decode'])} at "
                      f"{[(t, f'{m:.1e}') for t, m, _ in r['flips_prompt']]}; unflipped TF {r['tf_prompt_unflipped'][1]:.2e}"
                      + (f" ref-bf16 {r['ref_bf16_residual_unflipped'][1]:.2e}" if "ref_bf16_residual_unflipped" in r else "")
                      + (f" ref-bf16-act {r['ref_bf16_act_unflipped'][1]:.2e}" if "ref_bf16_act_unflipped" in r else "")
                      + f"; min margin {r['min_margin']:.2e}")
        print(f"layer {li:2d}: of the layer's change  ref prefix {r[f'ref_prefix{T // 2}'][1]:.2e}/"
              f"{r[f'ref_prefix{T - 16}'][1]:.2e}  ref again {r['ref_again'][1]:.2e}"
              + (f"  ref rerun {r['ref_rerun'][1]:.2e}" if "ref_rerun" in r else "")
              + f"  bf16 out {r['bf16_io_floor']:.2e}"
              + (f"  ref bf16-residual {r['ref_bf16_residual'][1]:.2e}" if "ref_bf16_residual" in r else "")
              + (f"  ref bf16-act {r['ref_bf16_act'][1]:.2e} (TF vs it {r['tf_prompt_vs_ref_bf16_act'][1]:.2e})" if "ref_bf16_act" in r else "")
              + f"  | TF prompt {r['tf_prompt'][1]:.2e} (worst row {r['tf_prompt'][2]:.2e} @{r['tf_prompt'][3]})"
              f"  decode {r['tf_decode'][1]:.2e}  prompt~decode {r['tf_prompt_vs_decode'][1]:.2e}" + fl_txt, flush=True)
    json.dump(res, open(outp + ".json", "w"), indent=1)


def rerun(pack, refp, outp):
    """Each saved layer again on its saved input rows, in this fresh process: the reference's run-to-run noise."""
    from exllamav3 import Config, Model

    ids, layers, v = read_ref(refp)
    config = Config.from_directory(pack)
    model = Model.from_config(config)
    dev = torch.device("cuda:0")
    mods = model.modules[: model.first_block_idx + max(layers) + 1]
    for li in layers:
        mods[model.first_block_idx + li].load(dev)
    model.modules[0].load(dev)
    out = {}
    with torch.inference_mode():
        for j, li in enumerate(layers):
            params = params_()
            model.prepare_inputs(torch.tensor([ids], dtype=torch.long), params)
            m = mods[model.first_block_idx + li]
            y = m.forward(m.prepare_for_device(v[j, 0][None].clone().to(dev), params), params)
            out[f"{li}.rerun"] = y.detach().float().cpu()[0]
    floor = torch.load(refp + ".floor.pt")
    floor.update(out)
    torch.save(floor, refp + ".floor.pt")
    for j, li in enumerate(layers):
        print(f"layer {li}: rerun vs saved {stats(out[f'{li}.rerun'], v[j, 1], v[j, 0])[1]:.2e} of the layer's change")


def emu(pack, refp, _):
    """Each saved layer with every module's input and output rounded to bf16: TensorFold's dtype, exllamav3's maths."""
    # rounded: RMSNorm, Linear, GatedMLP, BlockSparseMLP, GatedDeltaNet, MLAttention; saved as floor[f"{li}.emu"]
    from exllamav3 import Config, Model
    from exllamav3 import modules as M

    def rnd(t):
        return t.to(torch.bfloat16).to(t.dtype) if torch.is_tensor(t) and t.is_floating_point() else t

    for cls in (M.RMSNorm, M.Linear, M.GatedMLP, M.BlockSparseMLP, M.GatedDeltaNet, M.MLAttention):
        orig = cls.forward

        def fwd(self, x, *a, orig=orig, **k):
            return rnd(orig(self, rnd(x), *a, **k))
        cls.forward = fwd
    ids, layers, v = read_ref(refp)
    config = Config.from_directory(pack)
    model = Model.from_config(config)
    dev = torch.device("cuda:0")
    mods = model.modules[: model.first_block_idx + max(layers) + 1]
    for li in layers:
        mods[model.first_block_idx + li].load(dev)
    model.modules[0].load(dev)
    out = {}
    with torch.inference_mode():
        for j, li in enumerate(layers):
            params = params_()
            model.prepare_inputs(torch.tensor([ids], dtype=torch.long), params)
            m = mods[model.first_block_idx + li]
            x = v[j, 0].bfloat16().float()[None].clone().to(dev)
            y = m.forward(m.prepare_for_device(x, params), params)
            out[f"{li}.emu"] = y.detach().float().cpu()[0].bfloat16().float()
    floor = torch.load(refp + ".floor.pt")
    floor.update(out)
    torch.save(floor, refp + ".floor.pt")
    for j, li in enumerate(layers):
        print(f"layer {li}: bf16-activation emulation vs saved {stats(out[f'{li}.emu'], v[j, 1], v[j, 0])[1]:.2e} of the layer's change")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "ref":
        ref(sys.argv[2], sys.argv[3], [int(x) for x in sys.argv[4].split(",")], int(sys.argv[5]) if len(sys.argv) > 5 else 96)
    elif mode == "rerun":
        rerun(sys.argv[2], sys.argv[3], None)
    elif mode == "emu":
        emu(sys.argv[2], sys.argv[3], None)
    else:
        compare(sys.argv[2], sys.argv[3], sys.argv[4])
