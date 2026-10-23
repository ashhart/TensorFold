"""Tool calls of a recipe against the BF16 reference's (free-running greedy replies): same name, same arguments."""
# Reads OUT/<name>.json ("reply") of two directories written by gemma4_e4b_reference.py without --force.

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

CALL = re.compile(r"<\|tool_call>call:(\w+)\{(.*?)\}<tool_call\|>", re.S)


def calls(reply: str) -> list[tuple[str, str]]:
    return [(m.group(1), ",".join(sorted(m.group(2).split(",")))) for m in CALL.finditer(reply)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ref", type=Path)
    p.add_argument("test", type=Path)
    p.add_argument("-v", action="store_true")
    args = p.parse_args()
    n = same_name = same_all = same_text = 0
    for f in sorted(args.ref.glob("*.json")):
        a = json.loads(f.read_text())
        if not isinstance(a, dict) or "reply" not in a or not (args.test / f.name).exists():
            continue
        b = json.loads((args.test / f.name).read_text())
        ca, cb = calls(a["reply"]), calls(b["reply"])
        n += 1
        same_name += [c[0] for c in ca] == [c[0] for c in cb]
        same_all += ca == cb
        same_text += a["reply"] == b["reply"]
        if args.v and ca != cb:
            print(f"  {a['name']}: ref {a['reply'][:110]!r}\n  {'':{len(a['name'])}}  test {b['reply'][:110]!r}")
    print(f"{n} prompts: same tool names {same_name}, same name and arguments {same_all}, identical replies {same_text}")


if __name__ == "__main__":
    main()
