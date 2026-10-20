"""On rank 0's host, against `tensorfold serve` of GLM-5.3-Flash at TP=2: Z0's three decode-check prompts (dc.json) as
ids through /v1/completions, 512 tokens greedy with ignore_eos, each with drafts and with "draft": false. A reply's ids
are named by the server's tensorfold.token_sha256 (sha256 of the decimal ids joined by commas).
  - drafts and "draft": false give the same ids (drafted = plain): a gate
  - --golden FILE: the ids equal an earlier run's file (F2 against upstream 1.1.0, case B against case A): a gate
  - --z0 TOKENIZER_DIR: where the text leaves Z0's (2.x's ids, decoded), recorded only: the Zig engine leaves them at
    three close starts of thinking (<redacted-plan>'s E5, the same on two TP=2 host pairs)
The ids go to OUT. One line a request, then "exit 0|1".

    python3 golden_http.py PORT DC_JSON OUT [--golden FILE] [--z0 TOKENIZER_DIR]
"""

import json
import sys
import time
import urllib.request


def post(port, body, timeout=900):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    return d, time.perf_counter() - t0


def main(port, dc_path, out, golden, z0_dir):
    kinds = json.loads(open(dc_path).read())["kinds"]
    want = json.loads(open(golden).read()) if golden else None
    tok = None
    if z0_dir:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(f"{z0_dir}/tokenizer.json")
    got = {}
    failed = 0
    for k in kinds:
        body = {"model": "local-model", "max_tokens": len(k["want"]), "ignore_eos": True, "temperature": 0,
                "prompt": k["prompt"]}
        row = {}
        for name, extra in (("drafts", {}), ("plain", {"draft": False})):
            d, secs = post(port, {**body, **extra})
            spec = d.get("speculative") or {}
            row[name] = {"n": d.get("usage", {}).get("completion_tokens"),
                         "token_sha256": d.get("tensorfold", {}).get("token_sha256"), "seconds": round(secs, 2),
                         "text": d["choices"][0]["text"], "per_round": spec.get("tokens_per_round"),
                         "accepted": spec.get("acceptance_rate")}
            print(f"{k['kind']} {name}: {row[name]['n']} tokens, {row[name]['token_sha256']}, {secs:.1f} s, "
                  f"{row[name]['per_round']} a round", flush=True)
        a, b = row["drafts"], row["plain"]
        ok = a["token_sha256"] == b["token_sha256"] and a["text"] == b["text"] and a["n"] == len(k["want"])
        print(f"{'PASS' if ok else 'FAIL'} {k['kind']}: drafts and plain {'equal' if ok else 'differ'}", flush=True)
        failed += not ok
        if want is not None:
            ok = b["token_sha256"] == want[k["kind"]]["token_sha256"] and b["text"] == want[k["kind"]]["text"]
            print(f"{'PASS' if ok else 'FAIL'} {k['kind']}: {'equal to' if ok else 'not'} the golden's ids", flush=True)
            failed += not ok
        if tok is not None:
            z = tok.decode(k["want"], skip_special_tokens=False)
            at = next((i for i, (x, y) in enumerate(zip(b["text"], z)) if x != y), None)
            if at is None and len(b["text"]) == len(z):
                print(f"INFO {k['kind']}: equal to Z0's text", flush=True)
            else:
                at = min(len(b["text"]), len(z)) if at is None else at
                print(f"INFO {k['kind']}: leaves Z0's text at char {at}: {b['text'][at:at + 24]!r} vs {z[at:at + 24]!r}",
                      flush=True)
        got[k["kind"]] = {"n": b["n"], "token_sha256": b["token_sha256"], "text": b["text"],
                          "seconds": {"drafts": a["seconds"], "plain": b["seconds"]},
                          "drafts": {"per_round": a["per_round"], "accepted": a["accepted"]}}
    with open(out, "w") as f:
        json.dump(got, f, ensure_ascii=False, indent=1)
    print("exit", 1 if failed else 0, flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    args = sys.argv[1:]

    def opt(name):
        return args[args.index(name) + 1] if name in args else None

    sys.exit(main(int(args[0]), args[1], args[2], opt("--golden"), opt("--z0")))
