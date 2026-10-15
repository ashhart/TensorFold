"""fp32 PyTorch reference of Bolt's image path (preprocess, C-RADIO tower, projector), spliced into the LM one."""

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bolt_reference as lm  # noqa: E402

FIXTURES = "/storage/models/bolt-fixtures/vision.jsonl"
VPRE = "vision_model.radio_model."


# ---------------------------------------------------------------- preprocessing


def target_patches(w, h, cfg):
    """(w_patches, h_patches) of the single dynamic-resolution tile (image path, one image)."""
    ps, mn, mx = cfg["patch_size"], cfg["min_num_patches"], cfg["max_num_patches"]
    r = int(round(1 / cfg["downsample_ratio"]))
    budget = (cfg["max_model_len"] - 4) * r * r  # vLLM subtracts the text length too; capped anyway
    avail = max(min(budget, mx), mn)
    ph = round(h / ps + 0.5)
    pw = round(w / ps + 0.5)
    f = min(math.sqrt(avail / (ph * pw)), 1.0)
    th, tw = math.floor(f * ph), math.floor(f * pw)
    if avail > mn and th * tw < mn:
        up = math.sqrt(mn / (th * tw))
        th, tw = math.ceil(up * th), math.ceil(up * tw)
    if th % r:
        th = th + (r - th % r) if (th + r - th % r) * tw <= avail else max(r, th - th % r)
    if tw % r:
        tw = tw + (r - tw % r) if th * (tw + r - tw % r) <= avail else max(r, tw - tw % r)
    return tw, th


def preprocess(path, cfg):
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    wp, hp = target_patches(img.width, img.height, cfg)
    ps = cfg["patch_size"]
    t = torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1)[None].float()
    if t.shape[-2:] != (hp * ps, wp * ps):
        t = F.interpolate(t, size=(hp * ps, wp * ps), mode="bicubic", align_corners=False, antialias=True)
    mean = torch.tensor(cfg["norm_mean"]).view(1, 3, 1, 1)
    std = torch.tensor(cfg["norm_std"]).view(1, 3, 1, 1)
    return ((t / 255.0 - mean) / std)[0], (img.width, img.height), (hp, wp)


# ---------------------------------------------------------------- C-RADIO tower


def pos_embed(table, hp, wp):
    """CPE: [1, R*R, C] table -> [hp*wp, C]; bilinear (align_corners=False) to max side, crop."""
    n = int(math.isqrt(table.shape[1]))
    pos = table.reshape(1, n, n, -1).permute(0, 3, 1, 2).float()
    m = max(hp, wp)
    pos = F.interpolate(pos, size=(m, m), mode="bilinear", align_corners=False)[..., :hp, :wp]
    return pos.flatten(2)[0].T


def radio(ck, vcfg, px, dump=None, dump_layers=False, verbose=True):
    """px [3, H, W] normalized -> (all tokens [10 + hp*wp, 1280], patch features [hp*wp, 1280])."""
    ps, C, nh = vcfg["patch_size"], vcfg["hidden_size"], vcfg["num_attention_heads"]
    eps = vcfg["layer_norm_eps"]
    _, H, W = px.shape
    hp, wp = H // ps, W // ps
    m = VPRE + "model."
    g = lambda n: ck.raw(n).float()  # noqa: E731
    patches = px.reshape(3, hp, ps, wp, ps).permute(1, 3, 0, 2, 4).reshape(hp * wp, 3 * ps * ps)
    x = patches @ g(m + "patch_generator.embedder.weight").T
    x = x + pos_embed(g(m + "patch_generator.pos_embed"), hp, wp)
    prefix = g(m + "patch_generator.cls_token.token")  # [3 cls + 7 registers, C]
    x = torch.cat([prefix, x], 0)
    if dump:
        np.save(dump / "tower_embed.npy", x.numpy())
    hd = C // nh
    T = x.shape[0]
    for i in range(vcfg["num_hidden_layers"]):
        t0 = time.time()
        b = f"{m}blocks.{i}."
        y = F.layer_norm(x, (C,), g(b + "norm1.weight"), g(b + "norm1.bias"), eps)
        qkv = (y @ g(b + "attn.qkv.weight").T + g(b + "attn.qkv.bias")).view(T, 3, nh, hd).permute(1, 2, 0, 3)
        o = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])  # [nh, T, hd], scale hd^-0.5
        o = o.transpose(0, 1).reshape(T, C)
        x = x + o @ g(b + "attn.proj.weight").T + g(b + "attn.proj.bias")
        y = F.layer_norm(x, (C,), g(b + "norm2.weight"), g(b + "norm2.bias"), eps)
        y = F.gelu(y @ g(b + "mlp.fc1.weight").T + g(b + "mlp.fc1.bias"))
        x = x + y @ g(b + "mlp.fc2.weight").T + g(b + "mlp.fc2.bias")
        if dump and dump_layers:
            np.save(dump / f"tower_layer{i:02d}.npy", x.numpy())
        if verbose:
            print(f"  radio block {i:2d} |x|={x[10:].norm(dim=-1).mean():8.3f}  {time.time() - t0:5.1f}s",
                  file=sys.stderr, flush=True)
    skip = vcfg["num_cls_tokens"] + vcfg["num_registers"]
    return x, x[skip:]


def pixel_shuffle(feat, hp, wp, r=2):
    """ps_version v2: [hp*wp, C] -> [(hp/r)*(wp/r), r*r*C], channel index (dy*r + dx)*C + c."""
    C = feat.shape[-1]
    x = feat.view(hp // r, r, wp // r, r, C).permute(0, 2, 1, 3, 4)
    return x.reshape((hp // r) * (wp // r), r * r * C)


def projector(ck, x):
    w0 = ck.raw("mlp1.0.weight").float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * w0
    x = F.relu(x @ ck.raw("mlp1.1.weight").float().T).square()
    return x @ ck.raw("mlp1.3.weight").float().T


# ---------------------------------------------------------------- prompt


def build_prompt(question, n_img, image_first=True):
    """The chat template (no system content, thinking off) with the processor's image expansion."""
    img = "<img>" + "<image>" * n_img + "</img>"
    content = img + question if image_first else question + img
    return ("<|im_start|>system\n<|im_end|>\n<|im_start|>user\n" + content
            + "<|im_end|>\n<|im_start|>assistant\n<think></think>")


# ---------------------------------------------------------------- scoring


def compare_step(tok, lp, top, cache):
    """lp [vocab] vs vLLM {token string: logprob}; returns (n mapped, max |d|, worst)."""
    if "rev" not in cache or cache["rev_lp"] is not lp:
        rev = {}
        for i in torch.topk(lp, 4000).indices.tolist():
            rev.setdefault(tok.decode([i]), i)
        cache["rev"], cache["rev_lp"] = rev, lp
    rev = cache["rev"]
    d, worst, n = 0.0, None, 0
    for s, ref in top.items():
        i = rev.get(s)
        if i is None:
            ids = tok.encode(s, add_special_tokens=False).ids
            if len(ids) != 1:
                continue
            i = ids[0]
        n += 1
        e = abs(lp[i].item() - ref)
        if e > d:
            d, worst = e, (s, ref, lp[i].item())
    return n, d, worst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", type=int, default=0, help="line index in vision.jsonl")
    ap.add_argument("--image", help="image file (overrides the fixture's; needs --question)")
    ap.add_argument("--question")
    ap.add_argument("--model", default=lm.MODEL)
    ap.add_argument("--fixtures", default=FIXTURES)
    ap.add_argument("--text-first", action="store_true", help="question before the image in the user turn")
    ap.add_argument("--vision-only", action="store_true", help="stop after the projector")
    ap.add_argument("--generate", type=int, default=8, help="free-running greedy tokens")
    ap.add_argument("--forced", type=int, default=16, help="teacher-forced steps scored against the fixture")
    ap.add_argument("--dump", help="directory for npy dumps (pixels, tower, shuffle, projector)")
    ap.add_argument("--dump-tower-layers", action="store_true", help="also every RADIO block output")
    ap.add_argument("--dump-lm", action="store_true", help="also the LM per-layer dumps (last 8 positions)")
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)

    root = json.loads((Path(a.model) / "config.json").read_text())
    pcfg = json.loads((Path(a.model) / "preprocessor_config.json").read_text())
    vcfg = root["vision_config"]
    assert root["ps_version"] == "v2" and not root["use_thumbnail"] and not root.get("vision_final_layernorm")
    assert not vcfg["use_swiglu_ffn"] and vcfg["hidden_act"] == "gelu"

    fx = None
    if a.image:
        img_path, question = a.image, a.question
    else:
        fx = json.loads(open(a.fixtures).readlines()[a.case])
        img_path, question = str(Path(a.fixtures).parent / fx["image"]), fx["question"]

    dump = None
    if a.dump:
        dump = Path(a.dump)
        dump.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    px, (w, h), (hp, wp) = preprocess(img_path, pcfg)
    n_img = hp * wp // 4
    print(f"image {Path(img_path).name} {w}x{h} -> {wp * 16}x{hp * 16} px, patch grid {hp}x{wp} = {hp * wp}, "
          f"{n_img} image tokens", flush=True)

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(a.model) / "tokenizer.json"))
    ids = tok.encode(build_prompt(question, n_img, not a.text_first), add_special_tokens=False).ids
    img_pos = torch.tensor([i for i, t in enumerate(ids) if t == root["img_context_token_id"]])
    assert len(img_pos) == n_img
    line = f"prompt tokens {len(ids)}"
    if fx:
        line += f" (vLLM usage.prompt_tokens {fx['usage']['prompt_tokens']}: " \
                f"{'MATCH' if len(ids) == fx['usage']['prompt_tokens'] else 'MISMATCH'})"
    print(line, flush=True)

    ck = lm.Checkpoint(a.model)
    allx, feat = radio(ck, vcfg, px, dump, a.dump_tower_layers)
    shuf = pixel_shuffle(feat, hp, wp, int(round(1 / root["downsample_ratio"])))
    proj = projector(ck, shuf)
    print(f"vision: tower {tuple(allx.shape)} -> features {tuple(feat.shape)} -> shuffle {tuple(shuf.shape)} "
          f"-> projector {tuple(proj.shape)} |row| mean {proj.norm(dim=-1).mean():.3f}  {time.time() - t0:.1f}s",
          flush=True)
    if dump:
        np.save(dump / "pixel_values.npy", px.numpy())
        np.save(dump / "image_rgb.npy", np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8))
        np.save(dump / "tower_out.npy", allx.numpy())
        np.save(dump / "tower_features.npy", feat.numpy())
        np.save(dump / "pixel_shuffle.npy", shuf.numpy())
        np.save(dump / "projector_out.npy", proj.numpy())
        (dump / "ids.json").write_text(json.dumps(ids))
        (dump / "meta.json").write_text(json.dumps({"image": str(img_path), "orig_wh": [w, h],
                                                    "grid_hw": [hp, wp], "n_image_tokens": n_img,
                                                    "image_positions": [int(img_pos[0]), int(img_pos[-1])]}))
    if a.vision_only:
        return

    t1 = time.time()
    logits, st, _ = lm.forward(ck, ids, splice=(img_pos, proj), dump=dump if a.dump_lm else None, logits_last=1)
    print(f"LM prompt {len(ids)} tokens in {time.time() - t1:.1f}s", flush=True)
    lp = torch.log_softmax(logits[-1], -1)
    st0 = copy.deepcopy(st)
    out = [int(lp.argmax())]
    steps = [lp]
    for _ in range(a.generate - 1):
        lg, st, _ = lm.forward(ck, [out[-1]], st, verbose=False)
        steps.append(torch.log_softmax(lg[-1], -1))
        out.append(int(lg[-1].argmax()))
    mine = [tok.decode([i]) for i in out]
    print("ours:", repr("".join(mine)), out)
    if fx is None:
        print(f"total {time.time() - t0:.1f}s")
        return
    ref = [g["t"] for g in fx["gen"]]
    print("vLLM:", repr("".join(ref[:len(out)])))
    match = 0
    for m_, r_ in zip(mine, ref):
        if m_ != r_:
            break
        match += 1
    print(f"free-running greedy prefix match vs vLLM: {match}/{len(out)}")
    top = torch.topk(lp, 5)
    print("  step 0 ours top-5:", [(tok.decode([i]), round(v, 3)) for v, i in zip(top.values.tolist(), top.indices.tolist())])
    rank = sorted(fx["gen"][0]["top"], key=fx["gen"][0]["top"].get, reverse=True)
    print("  step 0 vLLM top-5:", [(r, round(fx["gen"][0]["top"][r], 3)) for r in rank[:5]])

    # teacher-forced: feed vLLM's tokens, score every step (a vLLM near-tie can't derail the rest)
    n = min(max(a.forced, 1), len(fx["gen"]))
    st, s_lp, cache = st0, lp, {}
    strict = tie = 0
    diffs = []
    for k in range(n):
        g = fx["gen"][k]
        cnt, d, worst = compare_step(tok, s_lp, g["top"], cache)
        o = int(s_lp.argmax())
        os_ = tok.decode([o])
        best = max(g["top"].values())
        is_tie = os_ in g["top"] and g["top"][os_] >= best - 1e-4
        strict += os_ == g["t"]
        tie += is_tie
        diffs.append(d)
        rank = sorted(g["top"], key=g["top"].get, reverse=True)
        print(f"  forced step {k:2d}: ours {os_!r} {s_lp[o].item():7.3f} | vLLM {g['t']!r} {g['lp']:7.3f} "
              f"(2nd {rank[1]!r} {g['top'][rank[1]]:7.3f}) | mapped {cnt}/20 max|d| {d:.3f}"
              + ("" if os_ == g["t"] else f"  {'TIE' if is_tie else 'MISS'} worst {worst}"))
        if k + 1 < n:
            nxt = cache["rev"].get(g["t"])
            if nxt is None:
                nxt = tok.encode(g["t"], add_special_tokens=False).ids[0]
            lg, st, _ = lm.forward(ck, [nxt], st, verbose=False)
            s_lp = torch.log_softmax(lg[-1], -1)
    diffs = np.array(diffs)
    print(f"teacher-forced {n} steps: top-1 {strict}/{n} (counting vLLM ties {tie}/{n}) | max|dlogprob| over "
          f"vLLM top-20: mean {diffs.mean():.3f} median {np.median(diffs):.3f} max {diffs.max():.3f}")
    print(f"total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
