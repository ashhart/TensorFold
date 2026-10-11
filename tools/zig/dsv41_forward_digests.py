#!/usr/bin/env python3
"""Single-GPU reference digests: backbone layers 0-24 + head over a prompt, then decode windows of 1, 3 and 16 rows."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WINDOWS = (1, 3, 16)


def digest(t) -> str:
    import torch

    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", type=int, default=1536)
    ap.add_argument("--limit", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=4101)
    ap.add_argument("--layers", default="0-8", help="backbone layers a-b (consecutive from 0: their sources load)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(HERE))
    import struct

    import torch
    import triton_aot_manifest as aot
    from dsv41_triton_capture import IndexRows, ZeroPeer, pack_aot, specialization
    from tensorfold.families.deepseek_v41.cuda import rope as RO
    from tensorfold.families.deepseek_v41.cuda import weights as W
    from tensorfold.families.deepseek_v41.cuda.config import Config
    from tensorfold.families.deepseek_v41.cuda.csa2.backend import Triton
    from tensorfold.families.deepseek_v41.cuda.engram_host import NgramHasher, token_map
    from tensorfold.families.deepseek_v41.cuda.forward import Forward, Seg

    t0 = time.time()
    dev = torch.device("cuda:0")
    pack = Path(a.pack)
    cfg = Config.from_file(pack / "config.json")
    tmap, _ = token_map(pack / "tokenizer.json")
    src = W.Shards(pack)
    loader = W.Loader(cfg, src, 0, 2, dev)
    vh = cfg.vocab_size // 2
    embed = loader._dev(src.rows("embed.weight", 0, vh).to(torch.bfloat16))
    head, norm = loader.x3("head", "col"), loader.native("norm.weight")
    lo, hi = (int(v) for v in a.layers.split("-"))
    layers = []
    for L in range(lo, hi + 1):
        layers.append(loader.layer(L))
        print(f"loaded layer {L} at {time.time() - t0:.0f} s", flush=True)
    tree = W.RankW(0, 2, 0, embed, layers, norm, head, None, {})
    load_s = time.time() - t0
    rows = {L: IndexRows(cfg.engram_head_dim) for L in cfg.engram_layer_ids}
    fw = Forward(cfg, tree, csa2=Triton(128, cfg.num_attention_heads // 2, dev), comm=ZeroPeer(), device=dev,
                 limit=a.limit, chunk=128, engram_rows=rows, hasher=NgramHasher(cfg, tmap), slots=1,
                 contiguous=True, pchunk=2048)
    g = torch.Generator().manual_seed(a.seed)
    ids = torch.randint(3, vh, (a.prompt + sum(WINDOWS),), generator=g).tolist()
    fw.prompt(ids[:a.prompt])
    torch.cuda.synchronize()
    print(f"prompt of {a.prompt} at {time.time() - t0:.0f} s", flush=True)
    pos = a.prompt
    # the slot after the prompt, by our forward's role names (block.zig: s.L<i>.swa / comp .v .s, .ik, .carry)
    torch.cuda.synchronize()
    sd = out / "state"
    sd.mkdir(exist_ok=True)
    roles = {}
    s0 = fw.slot

    def put(role, t):
        f = sd / f"{role}.bin"
        f.write_bytes(t.contiguous().view(torch.uint8).cpu().numpy().tobytes())
        roles[role] = f.name

    for L in range(lo, hi + 1):
        v, sc = s0.swa[L].kernel()
        put(f"s.L{L}.swa.v", v)
        put(f"s.L{L}.swa.s", sc)
        if L in s0.comp:
            v, sc = s0.comp[L].kernel()
            put(f"s.L{L}.comp.v", v)
            put(f"s.L{L}.comp.s", sc)
            put(f"s.L{L}.ik", s0.ikeys[L].kernel())
        if L in s0.carry:
            put(f"s.L{L}.carry", s0.carry[L])
    (sd / "state.json").write_text(json.dumps({"pos": s0.pos, "tail": list(s0.tail), "roles": roles}, indent=1) + "\n")
    rec = aot.Recorder().install()
    meta = {"pack": str(pack), "prompt": a.prompt, "limit": a.limit, "seed": a.seed, "load_s": round(load_s, 1),
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
            "env": {k: v for k, v in os.environ.items() if k.startswith("TF_DSV41")}, "peer": "zeros",
            "engram_rows": "index function (dsv41_triton_capture.IndexRows)", "layers": list(range(lo, hi + 1)),
            "ids": ids, "prompt_windows": 16, "windows": []}
    for n in WINDOWS:
        fw.trace = {}
        start = fw.slot.pos
        logits = fw.run([Seg(0, start, tuple(ids[pos:pos + n]))])
        torch.cuda.synchronize()
        lb = out / f"logits-w{n}.bin"
        lb.write_bytes(logits.contiguous().view(torch.uint8).cpu().numpy().tobytes())
        meta["windows"].append({"n": n, "start": start, "ids": ids[pos:pos + n],
                                "layers": {str(L): digest(t) for L, t in sorted(fw.trace.items())},
                                "logits": {"file": lb.name, "shape": list(logits.shape), "sha256": digest(logits)}})
        fw.trace = None
        fw.keep(0, n - 1)
        pos += n
        print(f"window {n} at {time.time() - t0:.0f} s", flush=True)
    meta["run_s"] = round(time.time() - t0 - load_s, 1)
    # the rope tables of this host (torch's CPU cos / sin), and the Triton variants the run launched
    tabs = RO.tables(cfg, a.limit + 2048)
    (out / "rope.bin").write_bytes(b"DSV41RP1" + struct.pack("<II", *tabs[0].cs.shape) + tabs[0].cs.contiguous().numpy().tobytes()
                                   + tabs[1].cs.contiguous().numpy().tobytes())
    rec.dump(out / "launches.json")
    cache = Path(os.environ.get("TRITON_CACHE_DIR", Path.home() / ".triton" / "cache"))
    aot.build(cache, out / "launches.json", out / "manifest.json", None)
    meta["aot"] = pack_aot(out / "manifest.json", cache, out / "aot", specialization())
    (out / "digests.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps({"windows": len(meta["windows"]), "layers": len(meta["windows"][0]["layers"]),
                      "load_s": meta["load_s"], "run_s": meta["run_s"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
