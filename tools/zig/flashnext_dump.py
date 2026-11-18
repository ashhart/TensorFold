"""Write what the Zig Flash Next family runs from the Python engine: kernels, launch sites, a prepared-weight pack,
reference tokens from one-row steps, and per-call fixtures of one step. Run with TF_FLASH_PLE_KERNELS=1."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metal_source import PREAMBLE, Arg, kernel_text  # noqa: E402

KERNELS: dict[str, dict] = {}
CALLS: list[dict] = []
ROLE: dict[int, str] = {}           # id of a persistent prepared array -> its role name
FIX = {"on": False, "blobs": [], "offset": 0, "rows": 1, "phase": ""}
_orig = mx.fast.metal_kernel


def _dtype(a: mx.array) -> str:
    return str(a.dtype).replace("mlx.core.", "")


def metal_kernel(name, input_names, output_names, source, header="", ensure_row_contiguous=True,
                 atomic_outputs=False):
    k = _orig(name=name, input_names=input_names, output_names=output_names, source=source, header=header,
              ensure_row_contiguous=ensure_row_contiguous, atomic_outputs=atomic_outputs)
    KERNELS[name] = {"inputs": list(input_names), "outputs": list(output_names), "source": source, "header": header}

    def call(*args, **kw):
        outs = k(*args, **kw)
        if FIX["on"]:
            ins = kw.get("inputs", args[0] if args else [])
            entry = {"kernel": name, "rows": FIX["rows"], "phase": FIX["phase"], "grid": list(kw["grid"]), "threadgroup": list(kw["threadgroup"]),
                     "template": [[t[0], t[1] if isinstance(t[1], (bool, int)) else str(t[1])]
                                  for t in kw.get("template", [])],
                     "inputs": [], "outputs": []}
            for a in ins:
                d = {"dtype": _dtype(a), "shape": list(a.shape), "size": int(a.size)}
                if id(a) in ROLE:
                    d["role"] = ROLE[id(a)]
                elif a.size <= 64:
                    d["value"] = a.tolist()
                entry["inputs"].append(d)
            mx.eval(*outs)
            for o in outs:
                d = {"dtype": _dtype(o), "shape": list(o.shape), "size": int(o.size)}
                if o.size <= 1 << 16:
                    raw = np.array(o.view(mx.uint8)).tobytes()
                    d["at"], d["bytes"] = FIX["offset"], len(raw)
                    FIX["blobs"].append(raw)
                    FIX["offset"] += len(raw)
                entry["outputs"].append(d)
            CALLS.append(entry)
        return outs

    return call


mx.fast.metal_kernel = metal_kernel



# Simdgroup-matrix layout: a lane's columns fn + {0, 1, 8, 9} a block, results at r * 8 + f * 4 + j; same sums.
SIMD_LANES = (
    ("  const short fn = ((qid & 2) | (lane & 1)) * 4;\n",
     "  const short fn = ((lane & 8) >> 1) | ((lane & 1) << 1);   // simdgroup matrices: columns fn + {0, 1, 8, 9}\n"),
    ("  const device uint4* sbv = (const device uint4*)SBt;\n  bool colok[NF];\n"
     "  for (int f = 0; f < NF; f++) colok[f] = n0 + f * 16 + fn < N;\n",
     "  const device uint2* sbv = (const device uint2*)SBt;\n  bool colok[NF][2];\n"
     "  for (int f = 0; f < NF; f++) for (int h = 0; h < 2; h++) colok[f][h] = n0 + f * 16 + fn + 8 * h < N;\n"),
    ("      const uint4 q = colok[f] ? sbv[(g * N + n0 + f * 16 + fn) / 4] : uint4(0);\n",
     "      const uint4 q = uint4(colok[f][0] ? sbv[(g * N + n0 + f * 16 + fn) / 2] : uint2(0),\n"
     "                            colok[f][1] ? sbv[(g * N + n0 + f * 16 + fn + 8) / 2] : uint2(0));\n"),
    ("C[t][i] = fma(s[f][j], P[t * NF * 8 + i], fma(bb[f][j], r ? xs1 : xs0, C[t][i]));",
     "C[t][i] = fma(s[f][j], P[t * NF * 8 + r * 8 + f * 4 + j], fma(bb[f][j], r ? xs1 : xs0, C[t][i]));"),
    ("          if (m < M && nn < N)\n"
     "            for (int j = 0; j < 4; j++) Y[m * N + nn + j] = static_cast<bfloat>(C[t][f * 8 + r * 4 + j]);\n",
     "          if (m < M)\n            for (int j = 0; j < 4; j++)\n"
     "              if (colok[f][j >> 1])\n"
     "                Y[m * N + nn + (j & 1) + 8 * (j >> 1)] = static_cast<bfloat>(C[t][f * 8 + r * 4 + j]);\n"),
)


def simd_lane_source(source: str) -> str:
    """A widening lane kernel reading the op's results in the simdgroup-matrix layout (GPUs before M5)."""

    for old, new in SIMD_LANES:
        assert source.count(old) == 1, f"lane kernel changed: {old[:60]!r}"
        source = source.replace(old, new)
    return source


def simd_lanes() -> None:
    """Before M5: the lane path the Zig engine replays, its widening kernels in the simdgroup-matrix layout."""

    from tensorfold.kernels.qwen.dense.v1 import lane_widen

    os.environ["TF_FLASH_DENSE"] = "lane"
    for name in ("NIBBLES", "BYTES", "NIBBLES_GROUPED", "BYTES_GROUPED"):
        setattr(lane_widen, name, simd_lane_source(getattr(lane_widen, name)))


def lane(decode, linear) -> tuple[mx.array, mx.array, int]:
    hit = decode._lane[id(linear)]
    return hit[1], hit[2], int(hit[3])


def build_pack(rt) -> tuple[dict[str, mx.array], dict[str, int]]:
    from tensorfold.families.qwen4_exp import decode

    fused, pack, tiles = rt.fused, {}, {}

    def put(name, a):
        pack[name] = a
        ROLE[id(a)] = name

    def put_lane(name, linear):
        wq, sbt, nt = lane(decode, linear)
        put(name + ".wq", wq)
        put(name + ".sbt", sbt)
        tiles[name] = nt

    def put_hc(name, hc):
        put(name + ".scale", hc.scale)
        for part, q in (("down", hc.down), ("up", hc.up)):
            put(f"{name}.{part}.w", q.weight)
            put(f"{name}.{part}.s", q.scales)
            put(f"{name}.{part}.b", q.biases)

    put("eps", fused.eps)
    for i, (layer, entry) in enumerate(zip(rt.model.layers, fused.layers)):
        put_hc(f"L{i}.ahc", entry["attn_hc"])
        put_hc(f"L{i}.mhc", entry["mlp_hc"])
        if layer.is_linear:
            proj, conv_w, g = entry["gdn"]
            put_lane(f"L{i}.gdn.in", proj)
            put_lane(f"L{i}.gdn.out", g.out_proj)
            put(f"L{i}.gdn.conv", conv_w)
            put(f"L{i}.gdn.alog", g.A_log)
            put(f"L{i}.gdn.dt", g.dt_bias)
            put(f"L{i}.gdn.norm", g.norm.weight)
        else:
            proj, qs, ks, iqs, pool, a = entry["attn"]
            put_lane(f"L{i}.att.proj", proj)
            put_lane(f"L{i}.att.o", a.o_proj)
            for nm, v in (("qn", qs), ("kn", ks), ("iqn", iqs), ("pool", pool)):
                put(f"L{i}.att.{nm}", v)
        moe, router = entry["moe"]
        put(f"L{i}.moe.router", router)
        sw, se = moe.switch_mlp, moe.shared_expert
        for nm, lin in (("gate", sw.gate_proj), ("up", sw.up_proj), ("down", sw.down_proj),
                        ("sgate", se.gate_proj), ("sup", se.up_proj), ("sdown", se.down_proj)):
            ROLE[id(lin.weight)], ROLE[id(lin.scales)], ROLE[id(lin.biases)] = (
                f"ck:L{i}.moe.{nm}.w", f"ck:L{i}.moe.{nm}.s", f"ck:L{i}.moe.{nm}.b")
    put_hc("mix", fused.mixer)
    put_lane("head", rt.model.lm_head)
    for i, layer in enumerate(rt.model.layers):
        if "ple" in layer:
            kv, ks, qs, cs, cw = fused.ple_parts[id(layer.ple)]
            put_lane("ple.kv", kv)
            put("ple.ks", ks)
            put("ple.qs", qs)
            put("ple.cs", cs)
            put("ple.conv", cw)
            put("ple.starts", fused.ple_tables.starts)
            for s, (w, sc, b) in enumerate(zip(fused.ple_tables.weights, fused.ple_tables.scales,
                                               fused.ple_tables.biases)):
                ROLE[id(w)], ROLE[id(sc)], ROLE[id(b)] = f"ck:ple.w{s}", f"ck:ple.s{s}", f"ck:ple.b{s}"
    if rt.mtp is not None:                      # the MTP head: its layer, mixer, input projections and cut head
        head, entry = rt.mtp, rt.mtp_fused.layers[0]
        put_hc("mtp.ahc", entry["attn_hc"])
        put_hc("mtp.mhc", entry["mlp_hc"])
        proj, qs, ks, iqs, pool, a = entry["attn"]
        put_lane("mtp.att.proj", proj)
        put_lane("mtp.att.o", a.o_proj)
        for nm, v in (("qn", qs), ("kn", ks), ("iqn", iqs), ("pool", pool)):
            put(f"mtp.att.{nm}", v)
        moe, router = entry["moe"]
        put("mtp.moe.router", router)
        sw, se = moe.switch_mlp, moe.shared_expert
        for nm, lin in (("gate", sw.gate_proj), ("up", sw.up_proj), ("down", sw.down_proj),
                        ("sgate", se.gate_proj), ("sup", se.up_proj), ("sdown", se.down_proj)):
            ROLE[id(lin.weight)], ROLE[id(lin.scales)], ROLE[id(lin.biases)] = (
                f"ck:mtp.moe.{nm}.w", f"ck:mtp.moe.{nm}.s", f"ck:mtp.moe.{nm}.b")
        put_hc("mtp.mix", rt.mtp_fused.mixer)
        put_lane("mtp.fce", head.fc_embedding)
        put_lane("mtp.fch", head.fc_hidden)
        put("mtp.enorm.scale", rt._mtp_scales[0])
        put("mtp.hnorm.scale", rt._mtp_scales[1])
        put_lane("mtp.draft", rt._draft_head)
        put("mtp.draft_ids", rt._draft_ids)
    emb = rt.model.model.embed_tokens
    ROLE[id(emb.weight)], ROLE[id(emb.scales)], ROLE[id(emb.biases)] = "ck:embed.w", "ck:embed.s", "ck:embed.b"
    return {k: v for k, v in pack.items() if not k.startswith("ck:")}, tiles


def cut_draft_head(rt, choice: str) -> None:
    """The MTP head cut to another list's ids: "cjk" (shipped beside the default list) or a file of ids."""

    from tensorfold.families.qwen4_exp.draft_head import VOCAB_FILE, cut_head, draft_ids

    ids = draft_ids(VOCAB_FILE.with_name("draft_vocab_cjk.txt") if choice == "cjk" else Path(choice))
    assert int(ids[-1]) < rt.model.args.vocab_size, f"{choice}: an id past the vocabulary"
    rt._draft_ids, rt._draft_head = mx.array(ids), cut_head(rt.model.lm_head, ids)


def site_of(entry: dict) -> str:
    """A launch site: the kernel family and the role stem of its first weight (layer dropped), else its input shape."""

    base = entry["kernel"].rsplit("_", 1)[0]
    for d in entry["inputs"]:
        if "role" in d:
            stem = d["role"].split(":", 1)[-1].rsplit(".", 1)[0]
            stem = ".".join(p for p in stem.split(".") if not (p.startswith("L") and p[1:].isdigit()))
            return f"{base}@{stem}"
    return f"{base}#{entry['inputs'][0]['shape']}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--prompt", default="Write a story about a lighthouse keeper.")
    ap.add_argument("--prompt-file", type=Path, help="the prompt's text from a file (long contexts)")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--fixture-step", type=int, default=3)
    ap.add_argument("--windows", default="2,3,4,6,8")
    ap.add_argument("--absorb", type=int, default=8, help="the MTP head's absorb windows: 1 .. this many rows")
    ap.add_argument("--draft-vocab", default="default",
                    help="the MTP head's draft ids: default (the shipped list), cjk (it and every CJK id) or a file")
    args = ap.parse_args()
    assert os.environ.get("TF_FLASH_PLE_KERNELS") == "1", "run with TF_FLASH_PLE_KERNELS=1"
    from tensorfold.kernels.device import tensor_units
    if not tensor_units():
        simd_lanes()
    out = args.out
    (out / "kernels").mkdir(parents=True, exist_ok=True)
    from tensorfold.families.qwen4_exp.runtime import load
    from tensorfold.server.text import render_prompt_ids

    rt, tok = load(args.model, drafts=3)
    if args.draft_vocab != "default":               # before the first draft draw tiles the head
        cut_draft_head(rt, args.draft_vocab)
    from tensorfold.families.qwen4_exp.mtp_cache import MTPCache
    text = args.prompt_file.read_text() if args.prompt_file else args.prompt
    prompt = [int(t) for t in render_prompt_ids(tok, [{"role": "user", "content": text}],
                                                enable_thinking=False)]
    cache = rt.model.make_cache()

    def step(t: int) -> int:
        logits = rt.head(rt.model.hidden(np.array([[t]], dtype=np.int64), cache))
        return int(mx.argmax(logits[0, -1]).item())

    nxt = 0
    for t in prompt:
        nxt = step(t)
    warm = MTPCache()                        # one MTP step and draw, so the head's lane weights are tiled too
    m0, _ = rt._mtp_step([nxt], rt.fused.last_streams[-1:], warm)
    mx.eval(rt._draft_draw(m0, None, [len(prompt)]))
    pack, tiles = build_pack(rt)            # after a forward: every lane weight is tiled
    made, fixture_input = [nxt], -1
    for g in range(args.tokens - 1):
        FIX["on"] = g == args.fixture_step
        if FIX["on"]:
            fixture_input = made[-1]
        made.append(step(made[-1]))
        FIX["on"] = False
    rounds, sizes, seen = [], [int(x) for x in args.windows.split(",")], set()
    cache = rt.model.make_cache()
    for t in prompt:
        pick = step(t)
    got, r = [pick], 0
    while len(got) < len(made):
        rows = min(sizes[r % len(sizes)], len(made) - len(got) + 1)
        drafts = list(made[len(got):len(got) + rows - 1])
        bad = r % rows
        if bad:
            drafts[bad - 1] = (drafts[bad - 1] + 1) % rt.model.args.vocab_size
        window = [got[-1]] + drafts
        FIX["on"], FIX["rows"] = rows not in seen, rows
        logits = rt.head(rt.model.hidden(np.array([window], dtype=np.int64), cache))
        picks = [int(x) for x in mx.argmax(logits[0], axis=-1).tolist()]
        FIX["on"], FIX["rows"] = False, 1
        seen.add(rows)
        keep = 1
        while keep < rows and drafts[keep - 1] == picks[keep - 1]:
            keep += 1
        got.extend(picks[:keep])
        rt.keep_rows(cache, rows, keep)
        rounds.append({"rows": rows, "window": window, "picks": picks, "keep": keep})
        r += 1
    assert got[:len(made)] == made, "drafted windows differ from one-row steps"
    # the MTP head: absorb windows of 1..--absorb rows (prompt rows, then generated ones) and a chain of drafts
    cache, mtp_cache, streams = rt.model.make_cache(), MTPCache(), []
    seq = prompt + made[:16]
    for t in seq:
        step(t)
        streams.append(rt.fused.last_streams[-1:])
    mtp_ref, at = {"absorb": [], "chain": []}, 0
    FIX["phase"] = "mtp:"
    for rows in range(1, args.absorb + 1):
        nexts = seq[at + 1:at + 1 + rows]
        FIX["on"], FIX["rows"] = True, rows
        mixed, out_streams = rt._absorb(mx.concatenate(streams[at:at + rows]), nexts, mtp_cache)
        d = int(rt._draft_draw(mixed, None, [at + rows + 1]).item())
        FIX["on"], FIX["rows"] = False, 1
        mtp_ref["absorb"].append({"start": at, "rows": rows, "next": nexts, "draft": d})
        at += rows
    for j in range(6):
        FIX["on"] = j == 0
        mixed, out_streams = rt._mtp_step([d], out_streams, mtp_cache)
        mtp_cache.drafted += 1
        d = int(rt._draft_draw(mixed, None, [at + 2 + j]).item())
        FIX["on"] = False
        mtp_ref["chain"].append(d)
    FIX["phase"] = ""
    mx.save_safetensors(str(out / "pack.safetensors"), pack)
    (out / "fixtures.bin").write_bytes(b"".join(FIX["blobs"]))
    variants, sites = {}, {}
    for c in CALLS:
        k = KERNELS[c["kernel"]]
        ins = [Arg(n, d["dtype"], d["size"], len(d["shape"])) for n, d in zip(k["inputs"], c["inputs"])]
        outs = [Arg(n, d["dtype"], d["size"], len(d["shape"])) for n, d in zip(k["outputs"], c["outputs"])]
        fname, text = kernel_text(c["kernel"], ins, outs, k["source"], k["header"], c["template"])
        c["function"] = fname
        if fname not in variants:
            (out / "kernels" / f"{c['kernel']}.metal").write_text(PREAMBLE + text)
            variants[fname] = {"kernel": c["kernel"], "file": f"kernels/{c['kernel']}.metal", "inputs": k["inputs"], "outputs": k["outputs"],
                               "meta": sorted({m for n in k["inputs"] for m in (n + "_shape", n + "_strides")
                                               if m in k["source"]})}
        c["site"] = site = c["phase"] + site_of(c) + (f"|{c['rows']}" if c["rows"] > 1 else "")
        found = sites.setdefault(site, {"function": fname, "grid": c["grid"], "threadgroup": c["threadgroup"]})
        assert found == {"function": fname, "grid": c["grid"], "threadgroup": c["threadgroup"]}, f"two launches share {site}"
    cfg = rt.model.args
    ple = next(l.ple for l in rt.model.layers if "ple" in l)
    e = ple.ple_embedding
    ref = {"prompt": prompt, "tokens": made, "rounds": rounds, "mtp": mtp_ref, "fixture_step": args.fixture_step, "fixture_input": fixture_input,
           "lane_tiles": tiles, "ple": {"n": int(e.n), "context": int(e.context), "per": int(e.per_ngram),
                                        "heads": int(e.heads), "eos": int(e.eos), "sizes": e.head_sizes.tolist(),
                                        "offsets": e.head_offsets.tolist(),
                                        "multipliers": [int(m) for m in e.multipliers],
                                        "starts": list(map(int, e.shard_starts)), "tail": int(ple.tail),
                                        "dilation": int(ple.dilation), "layer": next(
                                            i for i, l in enumerate(rt.model.layers) if "ple" in l)},
           "attention_scale": float(next(l.self_attn.scale for l in rt.model.layers if not l.is_linear)),
           "eps": float(cfg.rms_norm_eps)}
    (out / "ref.json").write_text(json.dumps(ref, indent=1))
    (out / "plan.json").write_text(json.dumps({"variants": variants, "sites": sites, "calls": CALLS}, indent=1))
    print(f"prompt {len(prompt)} tokens, {len(made)} made, {len(CALLS)} calls, {len(variants)} variants, "
          f"{len(sites)} sites, pack {len(pack)} tensors, fixtures {FIX['offset']} bytes")


if __name__ == "__main__":
    main()
