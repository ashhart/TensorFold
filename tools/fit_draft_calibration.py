"""Fit a draft head's node calibration (``tensorfold.drafters.calibration``) on public prompts.

    collect: send CORPUS to a server started with TF_DRAFT_LOG=<log> (sampled and greedy, fixed seeds)
    fit:     label every logged tree node by whether its path matches what the stream committed, fit a table a
             sampling regime, check it on held-out streams, and write the file with its source

The corpus, the collect command and its seed go into the file's ``source``; the logs never ship.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import urllib.request
from pathlib import Path

CORPUS = [
    ("completion", "Write a Python class for a bounded LRU cache with get and put methods, then add unit tests."),
    ("completion", "Implement binary search over a sorted list in JavaScript and explain the loop invariant."),
    ("completion", "Write a SQL query that returns each customer's three most recent orders, with a short note."),
    ("completion", "Write a bash script that renames every .jpeg file in a folder to .jpg and reports the count."),
    ("completion", "Write a Rust function that parses a comma-separated line of integers and returns their sum."),
    ("completion", "Refactor this Python loop into a list comprehension and explain the change:\n\n"
                   "result = []\nfor x in range(20):\n    if x % 3 == 0:\n        result.append(x * x)\n"),
    ("chat", "Write a short story about a lighthouse keeper who finds a message in a bottle."),
    ("chat", "Summarise the causes of the French Revolution in five bullet points."),
    ("chat", "Explain photosynthesis to a ten-year-old, then list three common misconceptions."),
    ("chat", "Draft a polite email asking a colleague to review a pull request by Friday."),
    ("chat", "Compare TCP and UDP in a table, then say when to choose each."),
    ("chat", "Solve step by step: a train travels 180 km in 2.5 hours; what is its average speed in m/s?"),
    ("chat", "Give a JSON object describing three fictional books with title, author, year and genre fields."),
    ("chat", "Translate into French and Spanish: 'The library opens at nine and closes at six on weekdays.'"),
    ("chat", "Write a haiku sequence of four haiku about the seasons in a city."),
    ("chat", "Explain what a hash table is, how collisions are handled, and the cost of each operation."),
]


def _send(base: str, model: str, kind: str, prompt: str, tokens: int, temperature: float, seed: int) -> None:
    body = {"model": model, "max_tokens": tokens, "temperature": temperature, "seed": seed, "ignore_eos": True}
    if temperature > 0:
        body.update(top_k=20, top_p=0.95)
    if kind == "chat":
        url = base + "/v1/chat/completions"
        body["messages"] = [{"role": "user", "content": prompt}]
        body["chat_template_kwargs"] = {"enable_thinking": False}
    else:
        url = base + "/v1/completions"
        body["prompt"] = prompt
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as response:
        response.read()


def collect(args: argparse.Namespace) -> None:
    jobs = [(kind, prompt, t, args.seed + i) for t in (1.0, 0.0) for i, (kind, prompt) in enumerate(CORPUS)]
    for at in range(0, len(jobs), args.streams):
        threads = [threading.Thread(target=_send, args=(args.base, args.model, kind, prompt, args.tokens, t, seed))
                   for kind, prompt, t, seed in jobs[at:at + args.streams]]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        print(f"sent {min(at + args.streams, len(jobs))} of {len(jobs)}", flush=True)


def samples(paths: list[str]) -> dict[str, list[tuple[int, float, bool, int]]]:
    """(depth, score, landed, stream) for every logged node whose positions the stream committed, by regime."""

    records: dict[tuple[int, int], list[dict]] = {}
    for f, path in enumerate(paths):
        for line in Path(path).read_text().splitlines():
            record = json.loads(line)
            records.setdefault((f, record["stream"]), []).append(record)
    out: dict[str, list[tuple[int, float, bool, int]]] = {"greedy": [], "sampled": []}
    for number, (key, trees) in enumerate(records.items()):
        committed: dict[int, int] = {}
        for tree in trees:
            first = tree["position"] - len(tree["kept"])
            committed.update({first + j: t for j, t in enumerate(tree["kept"])})
        for tree in trees:
            depths: list[int] = []
            hits: list[bool | None] = []
            for token, q, score in zip(tree["tokens"], tree["parents"], tree["scores"]):
                depth = 0 if q < 0 else depths[q] + 1
                truth = committed.get(tree["position"] + depth)
                above = True if q < 0 else hits[q]
                hit = None if truth is None or above is None else (above and truth == token)
                depths.append(depth)
                hits.append(hit)
                if hit is not None:
                    out["greedy" if tree["greedy"] else "sampled"].append((depth, score, hit, number))
    return out


def fit(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from tensorfold.drafters import calibration

    tables = {}
    for regime, rows in samples(args.logs).items():
        if not rows:
            continue
        tables[regime] = calibration.fit((d, s, hit) for d, s, hit, _ in rows)
        train = calibration.fit((d, s, hit) for d, s, hit, n in rows if n % 2 == 0)
        held = [(d, s, hit) for d, s, hit, n in rows if n % 2 == 1]
        print(f"{regime}: {len(rows)} nodes; held-out streams, predicted vs landed by predicted band:")
        for name, predict in (("table", lambda d, s: train.probability(d, s)), ("e^score", lambda d, s: math.exp(s))):
            bands: dict[int, list[float]] = {}
            for d, s, hit in held:
                p = predict(d, s)
                band = bands.setdefault(min(9, int(p * 10)), [0.0, 0.0, 0.0])
                band[0] += p
                band[1] += hit
                band[2] += 1
            cells = [f"{b[0] / b[2]:.2f}/{b[1] / b[2]:.2f} ({int(b[2])})" for _, b in sorted(bands.items())]
            print(f"  {name:8s} " + "  ".join(cells))
    source = json.loads(Path(args.source).read_text()) if args.source else {}
    source.setdefault("prompts", [{"kind": kind, "prompt": prompt} for kind, prompt in CORPUS])
    calibration.save(args.output, tables, source)
    print(f"wrote {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("base")
    c.add_argument("model")
    c.add_argument("--tokens", type=int, default=512)
    c.add_argument("--streams", type=int, default=8)
    c.add_argument("--seed", type=int, default=7)
    f = sub.add_parser("fit")
    f.add_argument("logs", nargs="+")
    f.add_argument("--output", required=True)
    f.add_argument("--source", help="a JSON file naming the model, drafter, collect command and seed")
    args = parser.parse_args()
    collect(args) if args.command == "collect" else fit(args)


if __name__ == "__main__":
    main()
