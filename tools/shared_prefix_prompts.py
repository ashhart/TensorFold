"""Build a shared-prefix labelling workload from the Python standard library: one long system prompt, N user items."""

import argparse
import ast
import json
import random
import sysconfig
from pathlib import Path

SKIP = ("test", "tests", "idlelib", "lib2to3", "site-packages", "dist-packages", "ensurepip", "turtledemo",
        "__pycache__", "tkinter", "config-")

RULES = """You label Python source files for a code-search index. Every request holds one file excerpt. Reply in
exactly two parts and nothing else.

Part one is a single line of JSON with the keys "primary", "secondary" and "summary". "primary" is the one label
from the catalogue below that best describes what the excerpt does. "secondary" is a list of up to three further
catalogue labels, most relevant first, and may be empty. "summary" is one plain sentence of at most thirty words.
Use label names exactly as the catalogue spells them. Never invent a label.

Part two starts with the line "Walkthrough" and explains the excerpt block by block, in order, in 250 to 350
words. Name each class and function you describe. Say what it takes, what it returns and what it changes. When a
block only imports names or sets constants, say so in one sentence. Do not quote more than one line of code at a
time. Do not speculate about code that is not in the excerpt.

How to choose labels:
1. Prefer the label whose catalogue line names the excerpt's main job, not a helper it calls.
2. When the excerpt is a thin wrapper around another module, label what the wrapper adds.
3. When the excerpt defines data only, label the domain the data serves.
4. If two labels fit equally, choose the one listed first in the catalogue.
5. The catalogue lists each label with a one-line description taken from that module's own documentation.

Label catalogue:
"""


def stdlib_modules() -> list[Path]:
    root = Path(sysconfig.get_paths()["stdlib"])
    files = [p for p in sorted(root.rglob("*.py")) if not any(part in SKIP or part.startswith(SKIP)
                                                                  for part in p.relative_to(root).parts)]
    return [p for p in files if p.stat().st_size > 6000]


def first_line(path: Path) -> str:
    try:
        doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")))
    except (SyntaxError, UnicodeDecodeError, ValueError):
        return ""
    return (doc or "").strip().splitlines()[0].strip() if doc else ""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model_dir")
    p.add_argument("--items", type=int, default=48)
    p.add_argument("--system-tokens", type=int, default=2900)
    p.add_argument("--min-user", type=int, default=100)
    p.add_argument("--max-user", type=int, default=2200)
    p.add_argument("--seed", type=int, default=38)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    from tokenizers import Tokenizer

    from tensorfold.cuda.server import ChatTemplate

    model_dir = Path(args.model_dir)
    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    template = ChatTemplate(model_dir)

    def rendered(messages: list[dict]) -> int:
        return len(tok.encode(template.render(messages, tools=None, enable_thinking=False),
                              add_special_tokens=False).ids)

    stub = [{"role": "user", "content": "x"}]

    def system_size(text: str) -> int:
        """Rendered tokens the system message adds (the template needs a user message to render)."""

        return rendered([{"role": "system", "content": text}] + stub) - rendered(stub)

    root = Path(sysconfig.get_paths()["stdlib"])
    files = stdlib_modules()
    catalogue: list[str] = []
    system = RULES
    for path in files:
        line = first_line(path)
        if not line:
            continue
        name = ".".join(path.relative_to(root).with_suffix("").parts)
        trial = system + f"- {name}: {line}\n"
        if system_size(trial) > args.system_tokens:
            break
        system, catalogue = trial, catalogue + [name]
    system_tokens = system_size(system)
    rng = random.Random(args.seed)
    sources = [f for f in files if f.stat().st_size > 12000]
    rng.shuffle(sources)
    n = args.items
    targets = [args.min_user + (args.max_user - args.min_user) * i // max(1, n - 1) for i in range(n)]
    rng.shuffle(targets)
    items = []
    for i, (path, target) in enumerate(zip(sources, targets)):
        lines = path.read_text(encoding="utf-8").splitlines()
        rel = path.relative_to(root).as_posix()

        def user(k: int) -> str:
            return f"Label this file.\n\nFile: {rel}\n```python\n" + "\n".join(lines[:k]) + "\n```"

        lo, hi = 1, len(lines)
        while lo < hi:                      # the most lines whose message stays within the target
            mid = (lo + hi + 1) // 2
            if len(tok.encode(user(mid), add_special_tokens=False).ids) <= target:
                lo = mid
            else:
                hi = mid - 1
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user(lo)}]
        items.append({"id": i, "file": rel, "target_user_tokens": target, "messages": messages,
                      "prompt_tokens": rendered(messages)})
    lengths = [it["prompt_tokens"] for it in items]
    meta = {"system_tokens": system_tokens, "catalogue_labels": len(catalogue), "items": n,
            "prompt_tokens_min": min(lengths), "prompt_tokens_max": max(lengths),
            "prompt_tokens_mean": round(sum(lengths) / n, 1), "prompt_tokens_total": sum(lengths),
            "source": f"CPython {sysconfig.get_python_version()} standard library, seed {args.seed}"}
    Path(args.output).write_text(json.dumps({"meta": meta, "items": items}, indent=1))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
