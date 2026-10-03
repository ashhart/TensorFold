#!/usr/bin/env python3
# Adapted from jayleaton/deepseek-v41-tensorfold-spark bench/stress.py, Apache License 2.0, Copyright 2026 Jay Leaton
# (https://x.com/jayleaton). Modified for tensorfold dsv41-cuda: the API key from TF_API_KEY, our port, model and
# context defaults (120K / 32K under CONTEXT=131072), no min_tokens (ignore_eos already decodes to max_tokens), ids
# in the checkpoint's text range from --vocab-lo / --vocab-hi.
"""Memory stress / admission client (stdlib only) for a running tensorfold server, prompts as token ids.

  dsv41_stress.py stress --base URL --long 120000 --dctx 32768 --decode 2048 --out stress.json
      one stream prefills ``--long`` tokens while three streams prefill ``--dctx`` tokens and then decode
      ``--decode`` tokens (``ignore_eos``: the decode really runs instead of stopping at EOS); the streams start
      ``--stagger`` s apart. Token ids are random text-range ids (no specials), a different seed a stream (no prefix
      sharing). Reports each stream's first-token / total seconds, completion tokens (the server's usage), its decode
      tok/s, and its decode tok/s WHILE the long stream was still prefilling (chunk times before the long stream's
      first token), and errors. Exit 1 on an error or (``--require-full``) a decode stream short of ``--decode``.
  dsv41_stress.py admit --base URL --tokens 512 --max-wait 10
      one request; the seconds to its first token (admission + prefill); exit 1 if over ``--max-wait``.

Admission refuses (HTTP 400 / 503) what does not fit the server's context or memory: that is an error here, so size
``--long`` / ``--dctx`` to the server's ``--context``. The key is read from TF_API_KEY only, never a flag.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
import urllib.request


def headers() -> dict[str, str]:
    key = os.environ.get("TF_API_KEY", "")
    return {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}


def _model(base: str) -> str:
    req = urllib.request.Request(base + "/v1/models", headers=headers())
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["data"][0]["id"]


def _ids(n: int, seed: int, lo: int, hi: int) -> list[int]:
    rng = random.Random(seed)
    return [0] + [rng.randrange(lo, hi) for _ in range(n - 1)]


def _stream(base: str, model: str, ids: list[int], max_tokens: int, out: dict, timeout: float,
            t_ref: float | None = None) -> None:
    body = {"model": model, "prompt": ids, "max_tokens": max_tokens, "temperature": 0, "ignore_eos": True,
            "stream": True, "stream_options": {"include_usage": True}}
    t_ref = time.time() if t_ref is None else t_ref
    out["start_s"] = round(time.time() - t_ref, 2)
    times: list[float] = []
    req = urllib.request.Request(base + "/v1/completions", data=json.dumps(body).encode(), headers=headers())
    t0 = time.time()
    out.update(prompt=len(ids), max_tokens=max_tokens, first_token_s=None, tokens=0)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                try:
                    chunk = json.loads(line[5:])
                except ValueError:
                    chunk = {}
                if chunk.get("error"):
                    out["error"] = json.dumps(chunk["error"])[:300]
                if chunk.get("usage"):
                    out["usage_completion"] = chunk["usage"].get("completion_tokens")
                if not chunk.get("choices") or not chunk["choices"][0].get("text"):
                    continue
                if out["first_token_s"] is None:
                    out["first_token_s"] = round(time.time() - t0, 2)
                out["tokens"] += 1
                times.append(round(time.time() - t_ref, 3))
        out["total_s"] = round(time.time() - t0, 2)
        out["chunks"] = out["tokens"]
        if out.get("usage_completion"):
            out["tokens"] = int(out["usage_completion"])
        out["chunk_times"] = times
        if not out["tokens"] and "error" not in out:
            out["error"] = "no streamed tokens"
        ft = out["first_token_s"] or out["total_s"]
        out["decode_tok_s"] = round(out["tokens"] / max(out["total_s"] - ft, 1e-6), 2) if out["tokens"] > 1 else None
    except Exception as e:                                   # noqa: BLE001 - reported
        detail = e.read()[:200] if hasattr(e, "read") else b""
        out["error"] = f"{type(e).__name__}: {e} {detail!r}"[:300]
        out["total_s"] = round(time.time() - t0, 2)


def _during(r: dict, end_s: float | None) -> dict:
    """Decode tok/s of a stream while the long stream prefilled: its chunks after its own first one, before
    ``end_s`` (the long stream's first token, seconds from the shared start), scaled from chunks to tokens."""

    times = r.get("chunk_times") or []
    if end_s is None or len(times) < 2:
        return {}
    inside = [t for t in times[1:] if t <= end_s]
    span = min(end_s, times[-1]) - times[0]
    if not inside or span <= 0:
        return {"during_prefill_tokens": 0}
    scale = r["tokens"] / max(r.get("chunks") or len(times), 1)
    return {"during_prefill_tokens": round(len(inside) * scale), "during_prefill_s": round(span, 2),
            "during_prefill_tok_s": round(len(inside) * scale / span, 2)}


def stress(a) -> int:
    model = a.model or _model(a.base)
    plan = [(_ids(a.long, 1, a.vocab_lo, a.vocab_hi), a.long_decode)] + \
        [(_ids(a.dctx, 2 + i, a.vocab_lo, a.vocab_hi), a.decode) for i in range(3)]
    res = [dict(role="long" if i == 0 else "decode") for i in range(4)]
    t0 = time.time()
    th = [threading.Thread(target=_stream, args=(a.base, model, ids, mt, res[i], a.timeout, t0), daemon=True)
          for i, (ids, mt) in enumerate(plan)]
    for t in th:
        t.start()
        time.sleep(a.stagger)
    for t in th:
        t.join()
    long = res[0]
    long_ft = None if long.get("first_token_s") is None else long["start_s"] + long["first_token_s"]
    for r in res[1:]:
        r.update(_during(r, long_ft))
    full = all(r.get("tokens", 0) >= a.decode for r in res[1:])
    rep = {"model": model, "long": a.long, "dctx": a.dctx, "decode": a.decode, "wall_s": round(time.time() - t0, 1),
           "long_first_token_s": long_ft, "decode_full": full, "streams": res,
           "ok": all("error" not in r for r in res) and (full or not a.require_full)}
    brief = {k: v for k, v in rep.items() if k != "streams"}
    brief["streams"] = [{k: v for k, v in r.items() if k != "chunk_times"} for r in res]
    print(json.dumps(brief), flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rep, f, indent=1)
    return 0 if rep["ok"] else 1


def admit(a) -> int:
    res: dict = {}
    _stream(a.base, a.model or _model(a.base), _ids(a.tokens, 9, a.vocab_lo, a.vocab_hi), 8, res, a.timeout)
    first = res.get("first_token_s")
    res["pass"] = "error" not in res and first is not None and first <= a.max_wait
    print(json.dumps({k: v for k, v in res.items() if k != "chunk_times"}), flush=True)
    return 0 if res["pass"] else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--base", default="http://127.0.0.1:8888")
    common.add_argument("--model", default="", help="default: the server's first /v1/models id")
    common.add_argument("--vocab-lo", type=int, default=1000, help="random prompt ids from here ...")
    common.add_argument("--vocab-hi", type=int, default=120_000, help="... to here (V4.1: text ids below 128000)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stress", parents=[common])
    s.add_argument("--long", type=int, default=120_000)
    s.add_argument("--dctx", type=int, default=32_768)
    s.add_argument("--decode", type=int, default=2048)
    s.add_argument("--long-decode", type=int, default=16, help="max_tokens of the long stream")
    s.add_argument("--stagger", type=float, default=1.0, help="seconds between the four submits")
    s.add_argument("--require-full", action="store_true", help="exit 1 when a decode stream is short of --decode")
    s.add_argument("--timeout", type=float, default=7200)
    s.add_argument("--out", default="")
    m = sub.add_parser("admit", parents=[common])
    m.add_argument("--tokens", type=int, default=512)
    m.add_argument("--max-wait", type=float, default=10.0)
    m.add_argument("--timeout", type=float, default=900)
    a = ap.parse_args(argv)
    a.base = a.base.rstrip("/")
    return {"stress": stress, "admit": admit}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
