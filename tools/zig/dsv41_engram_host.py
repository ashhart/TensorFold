#!/usr/bin/env python3
"""Engram host tables for the Zig hasher (token map + per-layer hash multipliers), one small binary per pack."""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--py-src", required=True, help="the reference engine's src/")
    ap.add_argument("--config", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fixture")
    a = ap.parse_args()
    sys.path.insert(0, a.py_src)
    import torch
    from tensorfold.families.deepseek_v41.cuda import engram_host as EH

    raw = json.loads(Path(a.config).read_text())
    t = raw.get("text_config", raw)
    cfg = SimpleNamespace(engram_layer_ids=t["engram_layer_ids"], engram_max_ngram_size=t["engram_max_ngram_size"],
                          engram_n_heads=t["engram_n_heads"], engram_vocab_size=t["engram_vocab_size"],
                          engram_compressed_vocab_size=t["engram_compressed_vocab_size"],
                          engram_pad_token_id=t["engram_pad_token_id"])
    tmap, compressed = EH.token_map(a.tokenizer)
    if compressed != cfg.engram_compressed_vocab_size:
        raise SystemExit(f"token map: {compressed} compressed ids, config says {cfg.engram_compressed_vocab_size}")
    mult = EH.hash_multipliers(cfg.engram_layer_ids, cfg.engram_max_ngram_size, cfg.engram_compressed_vocab_size)
    L, M = len(cfg.engram_layer_ids), cfg.engram_max_ngram_size

    def header(vocab: int) -> bytes:
        return b"DSV41EH1" + struct.pack("<4I", L, M, vocab, compressed) + struct.pack(f"<{L}I", *cfg.engram_layer_ids) \
            + struct.pack(f"<{L * M}q", *mult.reshape(-1).tolist())

    Path(a.out).write_bytes(header(len(tmap)) + struct.pack(f"<{len(tmap)}I", *tmap))
    print(f"{a.out}: {len(tmap)} ids -> {compressed}, multipliers {mult.tolist()}")
    if a.fixture:
        # ids: ordinary tokens, the pad id, repeats; a lookback of 2 (fewer than max_ngram - 1: the sequence starts)
        g = torch.Generator().manual_seed(1401)
        ids = torch.randint(3, len(tmap), (37,), generator=g).tolist()
        ids[5] = cfg.engram_pad_token_id
        ids[9] = ids[8]
        lookback = [ids[0], ids[1]]
        window = ids[2:]
        hasher = EH.NgramHasher(cfg, tmap)
        cases = [(window, lookback), (window[:7], []), (window[7:], window[4:7])]
        used = sorted(set(ids) | {cfg.engram_pad_token_id})
        out = header(len(tmap)) + struct.pack("<I", len(used)) + b"".join(struct.pack("<II", i, tmap[i]) for i in used)
        out += struct.pack("<I", len(cases))
        for w, lb in cases:
            h = hasher(torch.tensor(w), lookback=torch.tensor(lb, dtype=torch.long))
            out += struct.pack("<II", len(w), len(lb)) + struct.pack(f"<{len(w)}I", *w) + struct.pack(f"<{len(lb)}I", *lb)
            out += struct.pack(f"<{h.numel()}q", *h.reshape(-1).tolist())
        Path(a.fixture).write_bytes(out)
        print(f"{a.fixture}: {len(cases)} cases, {len(used)} map entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
