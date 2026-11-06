"""Ripple's engine logits against ricursive_reference.py's: KL, top-1, worst rows, identical dumps, a greedy reply."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

USAGE = """
Logits files are bf16 bits [rows][vocab]. The reference writes them raw, with no header. The engine's `tensorfold kl
--dump` writes u32 rows, u32 vocab, then a record a row: the token fed next (u32) and the row's logits. Either is read
(the format is told from the file size); row r is the distribution of token r + 1 given tokens 0..r.

  ricursive_gate_compare.py finite FILE... [--ids ids.json]    (non-finite logits, distinct top-1 tokens, hits on the next token)
  ricursive_gate_compare.py kl REF ENGINE --ids ids.json [--max-kl X --min-top1 Y] [--out report.json]
                                   (a REF with more rows than ids, as one run over prompt + reply gives, is read for its first rows)
  ricursive_gate_compare.py same FILE FILE... [--vocab N]     (every row's logits bit for bit equal to the first file's)
  ricursive_gate_compare.py reply REF REPLY.json --prompt-len P   (REF over prompt + reply: argmax of row P-1+i is reply[i])
  ricursive_gate_compare.py join IDS.json REPLY.json OUT.json     (the prompt's ids followed by the reply's)
  ricursive_gate_compare.py split IDS.json PROMPT.json --prompt N (the first N ids to PROMPT.json; the rest on stdout, comma
                                   separated, for `tensorfold kl --gold`)
"""

HEADER = 8  # a kl dump's u32 rows and u32 vocab


def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def read_ids(path: str) -> list[int]:
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        data = data.get("ids", data.get("tokens"))
    return [int(t) for t in data]


def open_logits(path: str, vocab: int) -> tuple[np.ndarray, np.ndarray | None]:
    """The file's logits as a [rows, vocab] uint16 view, and the fed tokens when it is a kl dump."""
    size = os.path.getsize(path)
    rec = 4 + 2 * vocab
    raw = size > 0 and size % (2 * vocab) == 0
    kl = size > HEADER and (size - HEADER) % rec == 0
    if raw == kl:
        raise SystemExit(f"{path}: {size} bytes is neither [rows, {vocab}] raw bf16 nor a kl dump of that vocabulary")
    if raw:
        rows = size // (2 * vocab)
        return np.memmap(path, dtype="<u2", mode="r", shape=(rows, vocab)), None
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    rows, v = (int(x) for x in np.frombuffer(mm[:HEADER], dtype="<u4"))
    if v != vocab or HEADER + rows * rec != size:
        raise SystemExit(f"{path}: kl dump header says {rows} rows of vocab {v}, not {vocab}")
    logits = np.ndarray((rows, vocab), dtype="<u2", buffer=mm, offset=HEADER + 4, strides=(rec, 2))
    fed = np.ndarray((rows,), dtype="<u4", buffer=mm, offset=HEADER, strides=(rec,))
    return logits, fed


def compare(ref: np.ndarray, eng: np.ndarray, ids: list[int], top: int, chunk: int = 64) -> dict:
    """Per row KL(ref || engine) over the whole vocabulary in float64, both greedy picks, and the worst rows."""
    rows = ref.shape[0]
    kl = np.zeros(rows)
    agree = np.zeros(rows, dtype=bool)
    ref_top = np.zeros((rows, 3), dtype=np.int64)
    eng_top = np.zeros((rows, 3), dtype=np.int64)
    ref_lp = np.zeros((rows, 3))
    eng_lp = np.zeros((rows, 3))
    finite = np.ones(rows, dtype=bool)
    for lo in range(0, rows, chunk):
        hi = min(rows, lo + chunk)
        r, e = bf16_to_f32(np.asarray(ref[lo:hi])), bf16_to_f32(np.asarray(eng[lo:hi]))
        finite[lo:hi] = np.isfinite(e).all(-1)
        lp, lq = log_softmax(r), log_softmax(e)
        kl[lo:hi] = (np.exp(lp) * (lp - lq)).sum(-1)
        ref_top[lo:hi] = np.argsort(-lp, axis=-1, kind="stable")[:, :3]
        eng_top[lo:hi] = np.argsort(-lq, axis=-1, kind="stable")[:, :3]
        ref_lp[lo:hi] = np.take_along_axis(lp, ref_top[lo:hi], -1)
        eng_lp[lo:hi] = np.take_along_axis(lq, eng_top[lo:hi], -1)
        agree[lo:hi] = ref_top[lo:hi, 0] == eng_top[lo:hi, 0]
    worst = []
    for r in np.argsort(-np.nan_to_num(kl, nan=np.inf), kind="stable")[:top]:
        worst.append({
            "row": int(r),
            "next_token": ids[r + 1] if r + 1 < len(ids) else None,
            "kl": float(kl[r]),
            "same_top1": bool(agree[r]),
            "reference": [[int(t), round(float(p), 4)] for t, p in zip(ref_top[r], ref_lp[r])],
            "engine": [[int(t), round(float(p), 4)] for t, p in zip(eng_top[r], eng_lp[r])],
            "reference_margin": round(float(ref_lp[r, 0] - ref_lp[r, 1]), 4),
        })
    return {
        "rows": rows,
        "kl_mean": float(kl.mean()),
        "kl_median": float(np.median(kl)),
        "kl_p99": float(np.percentile(kl, 99)),
        "kl_max": float(kl.max()),
        "top1_agreement": float(agree.mean()),
        "top1_misses": int((~agree).sum()),
        "nonfinite_rows": int((~finite).sum()),
        "worst": worst,
    }


def run_kl(a: argparse.Namespace) -> int:
    ids = read_ids(a.ids)
    ref, ref_fed = open_logits(a.ref, a.vocab)
    eng, fed = open_logits(a.engine, a.vocab)
    if ref.shape[0] > len(ids):
        ref = ref[: len(ids)]  # a causal model: the first rows of a longer run are the shorter run's
    if fed is None and eng.shape[0] > len(ids):
        eng = eng[: len(ids)]  # the same for a raw file, so two reference passes compare over the chat's rows
    if ref.shape != eng.shape or ref.shape[0] != len(ids):
        raise SystemExit(f"shapes differ: reference {ref.shape}, engine {eng.shape}, {len(ids)} ids")
    if fed is not None:  # a kl dump says which token it fed after each row: the chat's next one, but for the last row
        wrong = np.flatnonzero(np.asarray(fed[:-1]) != np.asarray(ids[1:], dtype=np.uint32))
        if wrong.size:
            raise SystemExit(f"the engine's file fed other tokens than the ids at {wrong.size} rows, first row {wrong[0]}")
    rep = compare(ref, eng, ids, a.top)
    print(f"{rep['rows']} rows: mean KL {rep['kl_mean']:.5f}, median {rep['kl_median']:.5f}, p99 {rep['kl_p99']:.4f}, "
          f"max {rep['kl_max']:.4f}; top-1 {100 * rep['top1_agreement']:.2f}% ({rep['top1_misses']} misses); "
          f"{rep['nonfinite_rows']} rows with a non-finite engine logit")
    print(f"the {len(rep['worst'])} worst rows (row, next token, KL, reference and engine top 3 as token:logprob, margin):")
    for w in rep["worst"]:
        ref3 = " ".join(f"{t}:{p}" for t, p in w["reference"])
        eng3 = " ".join(f"{t}:{p}" for t, p in w["engine"])
        print(f"  row {w['row']}, next {w['next_token']}, KL {w['kl']:.4f}, reference [{ref3}], engine [{eng3}], "
              f"margin {w['reference_margin']}{'' if w['same_top1'] else ', TOP-1 DIFFERS'}")
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1))
    bad = rep["nonfinite_rows"] > 0 or not rep["kl_mean"] == rep["kl_mean"]
    if a.max_kl is not None and not rep["kl_mean"] <= a.max_kl:
        print(f"FAIL: mean KL {rep['kl_mean']:.5f} is above {a.max_kl}")
        bad = True
    if a.min_top1 is not None and not rep["top1_agreement"] >= a.min_top1:
        print(f"FAIL: top-1 {100 * rep['top1_agreement']:.2f}% is below {100 * a.min_top1:.2f}%")
        bad = True
    return 1 if bad else 0


def run_finite(a: argparse.Namespace) -> int:
    """A first look at a dump before any reference exists: a broken engine shows as NaNs, one token, or no hits."""
    ids = read_ids(a.ids) if a.ids else None
    status = 0
    for path in a.files:
        logits, _ = open_logits(path, a.vocab)
        bad, top = 0, []
        for lo in range(0, logits.shape[0], 64):
            x = bf16_to_f32(np.asarray(logits[lo:lo + 64]))
            bad += int((~np.isfinite(x).all(-1)).sum())
            top.append(x.argmax(-1))
        top = np.concatenate(top)
        text = f"{path}: {logits.shape[0]} rows, {bad} with a non-finite logit, {len(set(top.tolist()))} distinct top-1 tokens"
        if ids is not None and len(ids) == len(top):
            text += f", top-1 is the next token at {100 * np.mean(top[:-1] == np.array(ids[1:])):.1f}% of rows"
        print(text)
        status |= int(bad > 0)
    return status


def run_same(a: argparse.Namespace) -> int:
    base, _ = open_logits(a.files[0], a.vocab)
    status = 0
    for path in a.files[1:]:
        other, _ = open_logits(path, a.vocab)
        if other.shape != base.shape:
            print(f"{path}: {other.shape[0]} rows, not {base.shape[0]}")
            status = 1
            continue
        differ = [r for r in range(base.shape[0]) if not np.array_equal(base[r], other[r])]
        if not differ:
            print(f"{path}: all {base.shape[0]} rows bit-identical to {a.files[0]}")
            continue
        status = 1
        first = differ[0]
        print(f"{path}: {len(differ)} of {base.shape[0]} rows differ from {a.files[0]}; first row {first} "
              f"({int((base[first] != other[first]).sum())} of {a.vocab} logits)")
    return status


def read_reply(path: str) -> list[int]:
    return read_ids(path)  # a plain list, or the run command's --report (its "tokens")


def run_reply(a: argparse.Namespace) -> int:
    reply = read_reply(a.reply)
    ref, _ = open_logits(a.ref, a.vocab)
    need = a.prompt_len - 1 + len(reply)
    if ref.shape[0] < need:
        raise SystemExit(f"the reference has {ref.shape[0]} rows, the prompt and reply need {need}")
    lp = log_softmax(bf16_to_f32(np.asarray(ref[a.prompt_len - 1:need])))
    pick = lp.argmax(-1)
    miss = [i for i, t in enumerate(reply) if int(pick[i]) != t]
    print(f"greedy reply of {len(reply)} tokens: {len(reply) - len(miss)} equal the float64 pass's argmax")
    for i in miss[:5]:
        print(f"  reply token {i}: engine {reply[i]} (reference logprob {lp[i, reply[i]]:.4f}), "
              f"reference {int(pick[i])} ({lp[i, pick[i]]:.4f}), margin {lp[i, pick[i]] - lp[i, reply[i]]:.4f}")
    return 1 if miss else 0


def run_join(a: argparse.Namespace) -> int:
    ids = read_ids(a.ids) + read_reply(a.reply)
    Path(a.out).write_text(json.dumps(ids))
    print(f"{len(ids)} ids to {a.out}")
    return 0


def run_split(a: argparse.Namespace) -> int:
    ids = read_ids(a.ids)
    Path(a.prompt_out).write_text(json.dumps(ids[: a.prompt]))
    print(",".join(str(t) for t in ids[a.prompt:]))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, epilog=USAGE, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    kl = sub.add_parser("kl")
    kl.add_argument("ref")
    kl.add_argument("engine")
    kl.add_argument("--ids", required=True)
    kl.add_argument("--top", type=int, default=10)
    kl.add_argument("--max-kl", type=float, help="exit 1 if the mean KL is above this")
    kl.add_argument("--min-top1", type=float, help="exit 1 if the top-1 agreement (0 to 1) is below this")
    kl.add_argument("--out", default="")
    fin = sub.add_parser("finite")
    fin.add_argument("files", nargs="+")
    fin.add_argument("--ids", default="")
    same = sub.add_parser("same")
    same.add_argument("files", nargs="+")
    reply = sub.add_parser("reply")
    reply.add_argument("ref")
    reply.add_argument("reply")
    reply.add_argument("--prompt-len", type=int, required=True)
    join = sub.add_parser("join")
    join.add_argument("ids")
    join.add_argument("reply")
    join.add_argument("out")
    split = sub.add_parser("split")
    split.add_argument("ids")
    split.add_argument("prompt_out")
    split.add_argument("--prompt", type=int, required=True)
    for p in (kl, fin, same, reply):
        p.add_argument("--vocab", type=int, default=100352)
    a = ap.parse_args(argv)
    return {"kl": run_kl, "finite": run_finite, "same": run_same, "reply": run_reply, "join": run_join, "split": run_split}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
