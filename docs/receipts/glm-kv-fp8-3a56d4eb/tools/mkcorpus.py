"""The FP8 cache qualification's teacher-forced sequences: two public text domains of TensorFold itself at one commit
(prose: docs/**/*.md outside docs/receipts; code: zig/src/**/*.zig), each file through a running server's /tokenize
without special tokens, in path order, until the domain holds its plan's tokens. Each sequence is the checkpoint's own
prefix ("" with add_special_tokens) and then the next run of the domain's ids, so no two sequences share a token.
Plan a domain: 1 x 65,536, 2 x 32,768, 4 x 8,192 and 8 x 2,048 tokens. Standard library only.

    python3 mkcorpus.py PORT TEXTS_JSON OUT_IDS_JSON OUT_PLAN_JSON   (TEXTS_JSON: {"prose": [[path, text]...], "code": ...})
"""
import hashlib
import json
import sys
import urllib.request

PLAN = [65536] + [32768] * 2 + [8192] * 4 + [2048] * 8


def tok(port, text, special):
    body = {"model": "local-model", "prompt": text, "add_special_tokens": special}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/tokenize", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["tokens"]


def main(port, texts_path, out_ids, out_plan):
    texts = json.load(open(texts_path))
    prefix = tok(port, "", True)
    need = sum(PLAN) - len(PLAN) * len(prefix)
    seqs, plan = [], []
    for domain in ("prose", "code"):
        ids, used = [], []
        for path, text in texts[domain]:
            if len(ids) >= need:
                break
            ids.extend(tok(port, text, False))
            used.append(path)
        if len(ids) < need:
            sys.exit(f"{domain}: {len(ids)} tokens, {need} needed")
        at = 0
        for n in PLAN:
            body = ids[at:at + n - len(prefix)]
            at += len(body)
            seqs.append(prefix + body)
            plan.append({"seq": len(seqs) - 1, "domain": domain, "n": n})
        print(f"{domain}: {len(used)} files, {len(ids)} tokens, {at} used, first {used[0]}, last {used[-1]}")
    digest = hashlib.sha256(json.dumps(seqs).encode()).hexdigest()
    json.dump(seqs, open(out_ids, "w"))
    json.dump({"prefix": prefix, "ids_sha256": digest, "sequences": plan}, open(out_plan, "w"), indent=1)
    print(f"{len(seqs)} sequences, {sum(len(s) for s in seqs)} tokens, prefix {prefix}, ids sha256 {digest}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
