#!/usr/bin/env python3
"""The 27B's verify windows as the Zig engine's oracle: chains of 1-16 rows and shared forwards, each row hashed."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

TOPICS = ["lighthouses", "volcanoes", "bread", "rivers", "chess", "glaciers", "owls", "trains",
          "deserts", "bees", "clocks", "ferns", "kites", "salt", "comets", "violins"]
REPLY = 24                # serial tokens each stream decodes first: the chain windows' tokens


def chat(user: str) -> str:
    return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def sha(t) -> str:
    import torch

    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def rows(logits) -> list[dict]:
    return [{"sha": sha(logits[r]), "argmax": int(logits[r].argmax())} for r in range(logits.shape[0])]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tools", default="")
    ap.add_argument("--wide", action="store_true", help="also rounds past 16 rows")
    ap.add_argument("--sweep", action="store_true", help="also every alignment of a shared round's packed inputs")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = None
    if a.tools:
        sys.path.insert(0, a.tools)
        import triton_aot_manifest as aot

        rec = aot.Recorder().install()
    import torch
    from tokenizers import Tokenizer

    from tensorfold.families.qwen3_5.cuda.decode import clone_state, prefill, serial_decode
    from tensorfold.families.qwen3_5.cuda.engine import Qwen27Engine
    from tensorfold.families.qwen3_5.cuda.forward import commit_streams, multi_tree_forward, tree_forward

    model = Path(a.model)
    tok = Tokenizer.from_file(str(model / "tokenizer.json"))
    prompts = [tok.encode(chat(f"Write four sentences about {t}."), add_special_tokens=False).ids for t in TOPICS]
    eng = Qwen27Engine(model, None, streams=1, context=4096, context_explicit=True)
    if rec is not None:
        from tensorfold.cuda.kernels import gdn, prefill_attention, qmm

        rec.wrap(qmm._ext(), ("qmm", "qmm_group", "qmm_prefill"), "qmm")
        rec.wrap(gdn._ext(), ("tree", "replay", "prefill"), "gdn")
        rec.wrap(prefill_attention._ext(), ("prefill_attention",), "prefill_attention")
    w = eng.w
    states, replies = [], []
    for p in prompts:
        st, pending = prefill(w, p, None, limit=4096)
        states.append(st)
        replies.append(serial_decode(w, st, pending, REPLY, None, stop_eos=False).tokens)
    dev = w.norm.device
    cases = []

    def tokens(ids):
        return torch.tensor(ids, dtype=torch.int32, device=dev)

    # chains of one stream: the serial reply's first W tokens, each row's draw the next serial token
    for width in range(1, 17):
        st = clone_state(states[0])
        logits, _ = tree_forward(w, tokens(replies[0][:width]), list(range(-1, width - 1)), st)
        cases.append({"kind": "chain", "streams": [{"stream": 0, "tokens": replies[0][:width]}], "rows": rows(logits)})
    # shared forwards: each stream its own chain from its own state, rows in stream order
    layouts = [[1, 1], [1] * 4, [1] * 8, [1] * 16, [4, 4, 4, 4], [2, 5, 1, 8], [3] * 5, [16]]
    if a.wide:                          # rounds past 16 rows: the wider matmul tiles (to 32, 64, more)
        layouts += [[16, 16], [5, 16, 3], [16] * 4, [9, 16, 2, 16, 7, 16, 1, 5], [12] * 8, [16] * 16]
    if a.sweep:                         # the packed copy's pointer alignments: rows and streams mod 4, 16-row multiples
        for n in range(2, 6):
            for total in (16 * n, 16 * n - 12, 16 * n - 11, 16 * n - 10, 16 * n - 9):
                layouts.append([total // n + (i < total % n) for i in range(n)])
    for layout in layouts:
        sts = [clone_state(states[s]) for s in range(len(layout))]
        wins = [(replies[s][:n], list(range(-1, n - 1)), sts[s]) for s, n in enumerate(layout)]
        logits, record, _, starts = multi_tree_forward(w, wins)
        case = {"kind": "shared", "streams": [{"stream": s, "tokens": replies[s][:n]} for s, n in enumerate(layout)],
                "rows": rows(logits)}
        # each stream keeps a prefix of its rows (as multi._commit names them: window rows), then one more row each
        keep = [list(range(max(1, n - 1))) for n in layout]
        commit_streams(sts, record, [[starts[s] + r for r in k] for s, k in enumerate(keep)])
        nxt = [([replies[s][len(k)]], [-1], sts[s]) for s, k in enumerate(keep)]
        after, _, _, _ = multi_tree_forward(w, nxt)
        case["keep"] = [len(k) for k in keep]
        case["after"] = rows(after)
        cases.append(case)
    # trees: each stream's serial reply as the main chain, with other tokens hung beside it
    def tree(stream: int, shape: str):
        r = replies[stream]
        alt = lambda t: (t + 7919) % w.config.vocab  # noqa: E731
        if shape == "fork":            # 0 r0; 1 r1 (0); 2 alt (0); 3 r2 (1); 4 alt (1); 5 r3 (3); 6 alt (2)
            return [r[0], r[1], alt(r[1]), r[2], alt(r[2]), r[3], alt(r[3])], [-1, 0, 0, 1, 1, 3, 2]
        if shape == "fan":             # the root's six children (r1 first), then r2 under r1
            return [r[0], r[1]] + [alt(r[1] + j) for j in range(5)] + [r[2]], [-1, 0, 0, 0, 0, 0, 0, 1]
        raise ValueError(shape)

    tree_cases = [([(0, "fork")], [[0, 1, 3, 5]]), ([(1, "fan")], [[0, 1, 7]]),
                  ([(2, "fork"), (3, "fork")], [[0, 1, 3, 5], [0, 2, 6]])]
    if a.wide:                          # every stream a tree in one round, 120 rows
        tree_cases.append(([(s, "fork" if s % 2 else "fan") for s in range(16)],
                           [[0, 1, 3, 5] if s % 2 else [0, 1, 7] for s in range(16)]))
    for spec, paths in tree_cases:
        sts = [clone_state(states[s]) for s, _ in spec]
        wins = [(*tree(s, shape), st) for (s, shape), st in zip(spec, sts)]
        if len(spec) == 1:
            logits, record = tree_forward(w, tokens(wins[0][0]), wins[0][1], sts[0])
            starts = [0, len(wins[0][0])]
        else:
            logits, record, _, starts = multi_tree_forward(w, [(t, p, st) for t, p, st in wins])
        case = {"kind": "tree", "streams": [{"stream": s, "tokens": t, "parents": p} for (s, _), (t, p, _) in
                                            zip(spec, wins)], "rows": rows(logits), "paths": paths}
        nxt = [int(logits[starts[k] + path[-1]].argmax()) for k, path in enumerate(paths)]
        commit_streams(sts, record, [[starts[k] + r for r in path] for k, path in enumerate(paths)])
        after, _, _, _ = multi_tree_forward(w, [([t], [-1], st) for t, st in zip(nxt, sts)])
        case["next"] = nxt
        case["after"] = rows(after)
        cases.append(case)
    torch.cuda.synchronize()
    (out / "windows.json").write_text(json.dumps({"prompts": prompts, "replies": replies, "cases": cases}) + "\n")
    if rec is not None:
        rec.dump(out / "launches.json")
    print(f"{len(cases)} cases", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
