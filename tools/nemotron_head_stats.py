"""Measure Nemotron's MTP head against greedy and sampled prose replies: acceptance by depth, the true token's rank, and candidate sources.

  python tools/nemotron_head_stats.py MODEL_DIR --tokens 512 --out head.json
  python tools/nemotron_head_stats.py --analyze head.json [--costs profile.json]

The measurement decodes each prompt's reply one row at a time (greedy, then sampled at T=1, top_p 0.95, top_k 20),
then walks the reply with the MTP head fed the true tokens: at every position the rank of the true token in the
head's draft logits at depths 1..6, the head's keyed draft (the engine's) and whether it lands, the head's own
chain of 4 (its guesses fed back), the copy proposer's offers at the engine's 8-token match and at 3, and an n-gram
pool of the head's past chains (the lookahead idea's pool, filled from drafts already paid for). ``--analyze``
needs no MLX: chains and the best trees by node count with tokens a round and tok/s on a price list.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

PROMPTS = (
    "Write a 700-word short story about a lighthouse keeper who receives a letter forty years late. Continuous "
    "prose, no headings or lists.",
    "Explain to a curious teenager how vaccines train the immune system, in about 600 words of plain prose with "
    "no lists or headings.",
    "Write an essay of about 600 words on why cities should plant more street trees, in flowing paragraphs.",
    "Describe a day in the life of a medieval blacksmith as a narrative of about 600 words.",
    "Write a letter of about 600 words from a grandmother to her grandchild about what the sea taught her.",
    "Give a thoughtful account, in about 600 words of continuous prose, of how the printing press changed Europe.",
)
DEPTH = 6                 # teacher-forced depths measured
CHAIN = 4                 # the head's own chain, its guesses fed back
POOL_KEEP = 8             # continuations kept a key in the n-gram pool
POOL_TAKE = 4             # candidates verified a position (lookahead's G)
RANKS = 8                 # ranks a tree may branch over at a depth
# the lead's M5 Max price list (ms): verify windows and shared forwards, 2 Oct 2026
M5_COSTS = {1: 5.9, 2: 7.3, 3: 8.5, 4: 9.7, 8: 14.3, 16: 20.4, 17: 21.3, 32: 37.1, 48: 57.0, 64: 70.8, 96: 108.6,
            128: 141.0}
M5_MTP_MS = 0.6


# -- measurement -----------------------------------------------------------------------------------------------------
def chat_ids(tokenizer: Any, prompt: str) -> list[int]:
    """The chat template's token ids for one user message, thinking off (a flat list under transformers 4 or 5)."""

    messages = [{"role": "user", "content": prompt}]
    kwargs = {"add_generation_prompt": True, "tokenize": True, "enable_thinking": False}
    try:
        out = tokenizer.apply_chat_template(messages, return_dict=False, **kwargs)
    except TypeError:
        out = tokenizer.apply_chat_template(messages, **kwargs)
    if isinstance(out, dict):
        out = out["input_ids"]
    if out and isinstance(out[0], (list, tuple)):
        out = out[0]
    return [int(t) for t in out]


def common_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class HeadWalk:
    """The MTP head along one reply's truth, one position at a time."""

    def __init__(self, model: Any, sampling: Any, prompt_ids: list[int], prompt_hidden: Any) -> None:
        import mlx.core as mx

        self.mx = mx
        self.model = model
        self.sampling = sampling
        self.embed = model.model.backbone.embeddings
        self.ids = model._draft_ids
        self.index = {int(t): i for i, t in enumerate(self.ids.tolist())} if self.ids is not None else None
        self.mcache = model.mtp.make_cache()
        self.start = len(prompt_ids)
        # the prompt's (hidden_i, token_i+1) pairs enter the head's attention cache without drafting
        model._head_step(prompt_hidden[:, :-1], self.embed(mx.array([prompt_ids[1:]], dtype=mx.uint32)),
                         self.mcache, 0)

    def _token(self, value: int) -> Any:
        return self.embed(self.mx.array([[value]], dtype=self.mx.uint32))

    def _draw(self, logits: Any, position: int) -> Any:
        from tensorfold.engine.gpu_sampling import sample

        return sample(logits.reshape(1, -1), self.sampling, [position], ids=self.ids)

    def _rank(self, logits: Any, truth: int) -> Any:
        """Draft-vocabulary tokens scoring above the truth (-1 when the truth is not draftable)."""

        if self.index is not None:
            at = self.index.get(truth)
            if at is None:
                return self.mx.array(-1, dtype=self.mx.int32)
        else:
            at = truth
        flat = logits.reshape(-1)
        return self.mx.sum(flat > flat[at]).astype(self.mx.int32)

    def position(self, j: int, hidden: Any, tokens: list[int]) -> dict[str, Any]:
        """Absorb (hidden, tokens[j]) and measure the head on the truth tokens[j + 1:], then its own chain."""

        mx, model, mcache = self.mx, self.model, self.mcache
        base = model._head_step(hidden, self._token(tokens[j]), mcache, 1)
        state, ranks, keyed = base, [], []
        for d in range(1, DEPTH + 1):
            if j + d >= len(tokens):
                break
            logits = model._draft_logits(state)
            truth = tokens[j + d]
            rank, draw = self._rank(logits, truth), self._draw(logits, self.start + j + d)
            mx.eval(rank, draw)
            ranks.append(int(rank.item()))
            keyed.append(int(draw.item()))
            if d < DEPTH and j + d + 1 < len(tokens):
                state = model._head_step(state, self._token(truth), mcache, 1)
                mcache.drafted += 1
        model._trim_chained(mcache)
        chain, state, off = [keyed[0]] if keyed else [], base, False
        for d in range(2, CHAIN + 1):
            if j + d >= len(tokens):
                break
            off = off or chain[-1] != tokens[j + d - 1]
            state = model._head_step(state, self._token(chain[-1]), mcache, 1)
            mcache.drafted += 1
            if off:                                               # its own guess fed back: a fresh draw
                draw = self._draw(model._draft_logits(state), self.start + j + d)
                mx.eval(draw)
                chain.append(int(draw.item()))
            else:                                                 # on the truth so far: the teacher-forced draw
                chain.append(keyed[d - 1])
        model._trim_chained(mcache)
        return {"ranks": ranks, "keyed": [int(k == tokens[j + d + 1]) for d, k in enumerate(keyed)],
                "keyed_tokens": keyed, "chain": chain}


def copy_offers(proposer: Any, context: list[int], truth: list[int], need: int, width: int = 15) -> dict | None:
    """The copy proposer's offer at this position under the engine's rule, and how much of it the truth keeps."""

    copied = [int(t) for t in proposer.propose(context, width)]
    if len(copied) < 2 or int(getattr(proposer, "last_match", 0) or 0) < need:
        return None
    accepted = common_prefix(copied, truth)
    proposer.observe(len(copied), accepted)
    return {"offered": len(copied), "accepted": accepted}


def pool_lookup(pool: dict[int, list[tuple[int, ...]]], current: int, truth: list[int]) -> dict | None:
    """Verify up to POOL_TAKE continuations the pool holds for ``current``: the best match and the rows they cost."""

    found = pool.get(current)
    if not found:
        return None
    take = found[:POOL_TAKE]
    return {"candidates": len(take), "rows": sum(len(c) for c in take),
            "best": max(common_prefix(list(c), truth) for c in take)}


def pool_add(pool: dict[int, list[tuple[int, ...]]], sequence: list[int]) -> None:
    """Every suffix of a chain enters the pool under its first token, most recent first."""

    for i in range(len(sequence) - 1):
        key, cont = sequence[i], tuple(sequence[i + 1:i + 4])
        held = pool.setdefault(key, [])
        if cont in held:
            held.remove(cont)
        held.insert(0, cont)
        del held[POOL_KEEP:]


def measure_reply(model: Any, tokenizer: Any, prompt: str, sampling: Any, max_tokens: int) -> dict[str, Any]:
    import time

    import mlx.core as mx

    from tensorfold.engine.family_common import cache_arrays
    from tensorfold.engine.gpu_sampling import sample
    from tensorfold.engine.lane_engine import SuffixLookupProposer

    ids = chat_ids(tokenizer, prompt)
    cache = model.make_cache()
    prompt_hidden = model.hidden(mx.array([ids], dtype=mx.uint32), cache)
    mx.eval(prompt_hidden, *cache_arrays(cache))
    eos = set(getattr(tokenizer, "eos_token_ids", None) or [tokenizer.eos_token_id])
    hiddens = [prompt_hidden[:, -1:]]
    position = len(ids)
    token = int(sample(model.head(hiddens[-1]).reshape(1, -1), sampling, [position]).item())
    tokens, step_ms = [token], []
    while len(tokens) < max_tokens and token not in eos:
        started = time.perf_counter()
        h = model.hidden(mx.array([[token]], dtype=mx.uint32), cache)
        position += 1
        token = int(sample(model.head(h).reshape(1, -1), sampling, [position]).item())
        step_ms.append((time.perf_counter() - started) * 1e3)
        hiddens.append(h)
        tokens.append(token)
    walk = HeadWalk(model, sampling, ids, prompt_hidden)
    proposers = {"copy8": (SuffixLookupProposer(), 8), "copy3": (SuffixLookupProposer(ngram=3, min_match=3), 3)}
    pool: dict[int, list[tuple[int, ...]]] = {}
    positions = []
    for j in range(len(tokens) - 1):
        found = walk.position(j, hiddens[j], tokens)
        context, truth = ids + tokens[:j + 1], tokens[j + 1:]
        for name, (proposer, need) in proposers.items():
            found[name] = copy_offers(proposer, context, truth, need)
        found["pool"] = pool_lookup(pool, tokens[j], truth)
        pool_add(pool, [tokens[j], *found["chain"]])
        positions.append(found)
    return {"prompt": prompt, "mode": "greedy" if sampling is None else "sampled", "prompt_tokens": len(ids),
            "tokens": tokens, "text": tokenizer.decode(tokens), "serial_step_ms": statistics.median(step_ms),
            "positions": positions}


# -- analysis ----------------------------------------------------------------------------------------------------------
def interpolate(costs: dict[int, float], rows: int) -> float:
    """A window's ms from the price list: measured, interpolated between neighbours, or scaled past the widest."""

    known = sorted(costs)
    if rows in costs:
        return costs[rows]
    if rows > known[-1]:
        return costs[known[-1]] * rows / known[-1]
    above = next(w for w in known if w > rows)
    below = max(w for w in known if w < rows)
    return costs[below] + (costs[above] - costs[below]) * (rows - below) / (above - below)


def best_trees(positions: list[dict], sizes: tuple[int, ...]) -> dict[int, dict[str, Any]]:
    """For each node count, the prefix-closed set of rank paths with the most expected landed tokens, from the data."""

    counts: Counter = Counter()
    n = 0
    for p in positions:
        ranks = p["ranks"]
        if not ranks:
            continue
        n += 1
        prefix: list[int] = []
        for r in ranks:
            if r < 0 or r >= RANKS:
                break
            prefix.append(r)
            counts[tuple(prefix)] += 1
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], len(kv[0]), kv[0]))
    out: dict[int, dict[str, Any]] = {}
    for size in sizes:
        chosen = ordered[:size]
        by_depth = Counter(len(path) for path, _ in chosen)
        out[size] = {"tokens": 1 + sum(c for _, c in chosen) / n, "depth": max((len(p) for p, _ in chosen), default=0),
                     "shape": [by_depth[d] for d in range(1, max(by_depth, default=0) + 1)],
                     "nodes": [list(p) for p, _ in chosen[:8]]}
    return out


def chain_landed(p: dict, depth: int = CHAIN) -> int:
    """Drafts a chain of ``depth`` lands at this position: its leading keyed hits."""

    landed = 0
    for k in p["keyed"][:depth]:
        if not k:
            break
        landed += 1
    return landed


def chain_tokens(positions: list[dict], depth: int) -> float:
    """Expected tokens a round of a chain of ``depth`` keyed drafts: 1 + sum of P(the first j all land)."""

    here = [p for p in positions if p["keyed"]]
    return 1 + statistics.mean(chain_landed(p, depth) for p in here) if here else 1.0


def analyze(runs: list[dict], costs: dict[int, float], mtp_ms: float, label: str) -> str:
    lines = []
    sizes = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
    for mode in ("greedy", "sampled"):
        positions = [p for run in runs if run["mode"] == mode for p in run["positions"]]
        if not positions:
            continue
        lines.append(f"== {mode}: {len(positions)} positions from {sum(1 for r in runs if r['mode'] == mode)} replies "
                     f"(serial step {statistics.median([r['serial_step_ms'] for r in runs if r['mode'] == mode]):.2f} ms"
                     f" here) ==")
        lines.append("teacher-forced depth: keyed draft lands | truth in top-1 / top-2 / top-4 / top-8 | not draftable")
        for d in range(1, DEPTH + 1):
            here = [p for p in positions if len(p["ranks"]) >= d]
            if not here:
                break
            ranks = [p["ranks"][d - 1] for p in here]
            keyed = statistics.mean(p["keyed"][d - 1] for p in here)
            tops = [statistics.mean(0 <= r < k for r in ranks) for k in (1, 2, 4, 8)]
            miss = statistics.mean(r < 0 for r in ranks)
            lines.append(f"  depth {d}: {keyed:.1%} | " + " / ".join(f"{t:.1%}" for t in tops) + f" | {miss:.1%}"
                         f"  (n {len(here)})")
        lines.append("chains (keyed drafts fed back): depth -> tokens a round, rows, ms, tok/s")
        for d in range(1, CHAIN + 1):
            tokens = chain_tokens(positions, d)
            ms = interpolate(costs, d + 1) + d * mtp_ms
            lines.append(f"  depth {d}: {tokens:.3f} tokens, {d + 1} rows, {ms:.1f} ms, {1e3 * tokens / ms:.0f} tok/s")
        lines.append(f"best trees over the head's top-{RANKS} by depth ({label} price list, one head step a level):")
        lines.append(f"  {'nodes':>5} {'rows':>4} {'tokens':>7} {'depth':>5} {'ms':>6} {'tok/s':>6}  shape (nodes a depth)")
        for size, tree in best_trees(positions, sizes).items():
            rows = size + 1
            ms = interpolate(costs, rows) + tree["depth"] * mtp_ms
            lines.append(f"  {size:>5} {rows:>4} {tree['tokens']:>7.3f} {tree['depth']:>5} {ms:>6.1f} "
                         f"{1e3 * tree['tokens'] / ms:>6.0f}  {tree['shape']}")
        for name in ("copy8", "copy3"):
            offers = [p[name] for p in positions if p.get(name)]
            offered = sum(o["offered"] for o in offers)
            accepted = sum(o["accepted"] for o in offers)
            lines.append(f"{name}: fired at {len(offers)}/{len(positions)} positions, {offered} rows offered, "
                         f"{accepted} tokens landed ({accepted / offered if offered else 0:.3f} a row, "
                         f"{accepted / len(positions):.3f} a position)")
        hits = [p["pool"] for p in positions if p.get("pool")]
        rows = sum(h["rows"] for h in hits)
        best = sum(h["best"] for h in hits)
        marginal = sum(max(0, p["pool"]["best"] - chain_landed(p)) for p in positions if p.get("pool"))
        lines.append(f"head-chain pool (lookahead proxy): candidates at {len(hits)}/{len(positions)} positions, "
                     f"{rows} rows verified, {best} tokens landed ({best / rows if rows else 0:.3f} a row), "
                     f"{marginal} beyond the chain's own ({marginal / rows if rows else 0:.3f} a row)")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model", nargs="?")
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--prompts", type=int, default=len(PROMPTS))
    parser.add_argument("--modes", default="greedy,sampled")
    parser.add_argument("--full-vocab", action="store_true", help="draft from the whole LM head, not draft_ids.txt")
    parser.add_argument("--out", default="")
    parser.add_argument("--analyze", default="", help="a saved run to summarize (no MLX needed)")
    parser.add_argument("--costs", default="", help="a nemotron_step_profile.py JSON whose window and shared costs price rows")
    args = parser.parse_args()
    if args.analyze:
        runs = json.loads(Path(args.analyze).read_text())
        costs, mtp_ms, label = dict(M5_COSTS), M5_MTP_MS, "M5 Max"
        if args.costs:
            profile = json.loads(Path(args.costs).read_text())
            costs = {int(k): float(v) for table in (profile["window_costs"], profile["shared_costs"])
                     for k, v in table.items()}
            mtp_ms, label = float(profile.get("mtp_step_ms") or mtp_ms), profile.get("device", "profile")
        print(analyze(runs, costs, mtp_ms, label))
        return 0
    if not args.model:
        parser.error("MODEL_DIR required")
    import os

    for key, value in (("MLX_MAX_OPS_PER_BUFFER", "200"), ("MLX_MAX_MB_PER_BUFFER", "100000")):
        os.environ.setdefault(key, value)
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families import nemotron_h

    model, tokenizer = nemotron_h.load(Path(args.model))
    if model.mtp is None:
        raise SystemExit("the MTP head did not load (mtp-4bit.safetensors beside the weights?)")
    if args.full_vocab:
        model._draft_ids, model._draft_head = None, None
        print(f"[head] drafting over the whole vocabulary: a head step {model._time_mtp_step():.2f} ms "
              f"(draft vocabulary: {model.mtp_step_ms:.2f})", flush=True)
    runs = []
    vocab = "full" if args.full_vocab else "draft"
    for i, prompt in enumerate(PROMPTS[:args.prompts]):
        for mode in args.modes.split(","):
            sampling = None if mode == "greedy" else Sampling(seed=1000 + i, temperature=1.0, top_k=20, top_p=0.95)
            run = measure_reply(model, tokenizer, prompt, sampling, args.tokens)
            run["vocabulary"] = vocab
            runs.append(run)
            landed = statistics.mean(p["keyed"][0] for p in run["positions"] if p["keyed"])
            print(f"[head] prompt {i} {mode}: {len(run['tokens'])} tokens, serial step {run['serial_step_ms']:.2f} ms, "
                  f"depth-1 draft lands {landed:.1%}", flush=True)
            if args.out:
                Path(args.out).write_text(json.dumps(runs))
    print(analyze(runs, M5_COSTS, M5_MTP_MS, "M5 Max"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
