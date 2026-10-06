"""GLM-5.3-Flash references from the Python family for tf-glm-run (dev-time oracle): row-exact prompts, greedy replies, sublayer captures."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

PROMPTS = (
    ("story", "Write a short story about a lighthouse keeper who finds a message in a bottle."),
    ("code", "Write a Python function that checks whether a string is a palindrome, with a few tests."),
    ("facts", "Explain how a refrigerator moves heat out of its inside, in three short paragraphs."),
)
WINDOW = 16


def long_text(tokenizer, n: int, source: str) -> list[int]:
    """A prompt of about n tokens: a public text (our README) repeated, then a question about it."""

    words = Path(source).read_text(errors="replace")
    body = (words * (1 + 4 * n // max(len(words) // 4, 1)))
    ids = tokenizer.apply_chat_template([{"role": "user", "content": body[: 6 * n] + "\n\nSummarize the text above."}],
                                        add_generation_prompt=True, tokenize=True)
    ids = list(ids["input_ids"] if isinstance(ids, dict) else ids)
    if len(ids) > n:                       # keep the template's head and tail around a cut body
        ids = ids[: n // 2] + ids[-(n - n // 2):]
    return ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--max", type=int, default=64)
    ap.add_argument("--long", type=int, nargs="*", default=[])
    ap.add_argument("--capture", default="")
    ap.add_argument("--text", default=str(Path(__file__).resolve().parents[2] / "README.md"))
    ap.add_argument("--layers", type=int, default=0, help="load and run only the first N layers (0: all)")
    ap.add_argument("--trace", nargs=3, metavar=("NAME", "STEPS", "PATH"), default=None)
    args = ap.parse_args()
    import mlx.core as mx

    from tensorfold.families.glm5_next import weights as glm
    from tensorfold.families.glm5_next.prompts import GlmTokenizer
    from tensorfold.families.tokenizer import load_tokenizer

    info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    mx.set_wired_limit(int(info["max_recommended_working_set_size"]))
    t0 = time.time()
    model = glm.load_backbone(Path(args.model), layers=args.layers or None)
    tokenizer = GlmTokenizer(load_tokenizer(Path(args.model), eos_token_ids=model.args.eos_token_id))
    print(f"loaded {len(model.layers)} layers in {time.time() - t0:.1f} s", flush=True)
    eos = set(model.args.eos_token_id)
    prompts = []
    for name, text in PROMPTS:
        ids = tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                            tokenize=True)
        prompts.append((name, list(ids["input_ids"] if isinstance(ids, dict) else ids)))
    for n in args.long:
        prompts.append((f"long{n}", long_text(tokenizer, n, args.text)))
    out = []
    for name, ids in prompts:
        cache = model.make_cache()
        t1 = time.time()
        for at in range(0, len(ids), WINDOW):
            hidden = model.hidden(mx.array([ids[at:at + WINDOW]], dtype=mx.uint32), cache)
            mx.eval(hidden)
        logits = model.head(hidden[:, -1:])
        tok = int(mx.argmax(logits.reshape(-1)).item())
        first = logits.reshape(-1).astype(mx.float32)
        t2 = time.time()
        reply = [tok]
        while len(reply) < args.max and tok not in eos:
            logits = model.head(model.hidden(mx.array([[tok]], dtype=mx.uint32), cache))
            tok = int(mx.argmax(logits.reshape(-1)).item())
            reply.append(tok)
        t3 = time.time()
        top = mx.argsort(-first)[:5].tolist()
        print(f"{name}: {len(ids)} prompt tokens in {t2 - t1:.1f} s, {len(reply)} tokens at "
              f"{(len(reply) - 1) / max(t3 - t2, 1e-9):.1f} tok/s; first top5 {top}", flush=True)
        print("  " + repr(tokenizer.decode(reply)[:300]), flush=True)
        out.append({"name": name, "ids": ids, "expect": reply, "first_top5": top,
                    "first_logits_top5": [float(first[i].item()) for i in top]})
    Path(args.out).write_text(json.dumps({"layers": len(model.layers), "prompts": out}))
    if args.capture:
        capture(model, prompts[0][1][:WINDOW], args.capture)
    if args.trace:
        name, steps, path = args.trace[0], int(args.trace[1]), args.trace[2]
        row = next(r for r in out if r["name"] == name)
        trace(model, row["ids"], row["expect"], steps, path)
    return 0


def trace(model, ids: list[int], reply: list[int], steps: int, path: str) -> None:
    """Every call of a reply's forward, sublayer by sublayer (keys c{call}.{array}): the prompt's windows, then `steps` one-row steps."""

    import mlx.core as mx

    cache = model.make_cache()
    calls = [ids[at:at + WINDOW] for at in range(0, len(ids), WINDOW)] + [[t] for t in reply[:steps]]
    found = {}
    for c, tokens in enumerate(calls):
        for key, value in sublayers(model, cache, tokens).items():
            found[f"c{c}.{key}"] = value
        mx.eval(list(found.values()))
    mx.save_safetensors(path, {k: mx.contiguous(v) for k, v in found.items()})
    print(f"traced {len(calls)} calls ({len(ids)} prompt rows, {min(steps, len(reply))} steps) in {path}", flush=True)


def capture(model, tokens: list[int], path: str) -> None:
    """The first window's sublayer inputs (normed) and outputs, layer by layer, and its final rows: bf16."""

    import mlx.core as mx

    found = sublayers(model, model.make_cache(), tokens)
    mx.save_safetensors(path, {k: mx.contiguous(v) for k, v in found.items()})
    print(f"captured {len(found)} arrays for {len(tokens)} rows in {path}", flush=True)


def sublayers(model, cache, tokens: list[int]) -> dict:
    """One call's rows through the layers with `cache` (advanced): embed, each sublayer's input and output, final, logits."""

    import mlx.core as mx

    ids = mx.array(tokens, dtype=mx.uint32)
    rows = int(ids.shape[0])
    h = model.embed_tokens(ids)
    x = mx.contiguous(mx.broadcast_to(h[:, None, :], (rows, model.args.hc_mult, h.shape[-1])))
    pending = None
    found = {"embed": h}
    for i, layer in enumerate(model.layers):
        lc = [c[i] for c in [cache]]
        x, normed, post, comb = model.boundary(x, pending, layer.attn_hc, layer.in_norm, True)
        att = layer.attn(normed, lc, (rows,), True)
        found[f"l{i}.attn_in"], found[f"l{i}.attn_out"] = normed, att
        pending = (att, post, comb)
        x, normed, post, comb = model.boundary(x, pending, layer.ffn_hc, layer.post_norm, True)
        mlp = layer.mlp(normed, True)
        found[f"l{i}.mlp_in"], found[f"l{i}.mlp_out"] = normed, mlp
        pending = (mlp, post, comb)
        mx.eval(x, *pending)
    found["final"] = model.final_norm(model.boundary(x, pending, None, None, True)[0])
    found["logits"] = model.head(found["final"])
    return found


if __name__ == "__main__":
    raise SystemExit(main())
