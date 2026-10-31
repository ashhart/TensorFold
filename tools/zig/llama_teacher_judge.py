"""Judge a Llama checkpoint's teacher-forced logits against a reference's at every position."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from qwen27_margins import verdict


def top2(row) -> dict:
    """A row's best two ids and scores, ties to the lower id (the engine's argmax and numpy's)."""
    import numpy as np

    first = int(np.argmax(row))
    rest = row.copy()
    rest[first] = -np.inf
    second = int(np.argmax(rest))
    return {"ids": [first, second], "logits": [float(row[first]), float(row[second])]}


def judge(native: list[dict], reference: list[dict]) -> dict:
    if len(native) != len(reference) or not native:
        raise ValueError("expected the same positive number of positions on both sides")
    exceptions = []
    for position, (a, b) in enumerate(zip(native, reference)):
        actual, wanted = a["ids"][0], b["ids"][0]
        if actual != wanted:
            exceptions.append({"position": position, "expected": wanted, "actual": actual, **verdict(a, b, wanted, actual)})
    return {"positions": len(native), "strict_equal": len(native) - len(exceptions), "exceptions": exceptions,
            "policy_pass": all(item["verdict"] == "near-tie" for item in exceptions)}


def load(path: Path, vocab: int):
    import numpy as np

    words = np.fromfile(path, dtype=np.uint16)
    if words.size % vocab:
        raise ValueError(f"{path}: not whole rows of {vocab} bf16 logits")
    return (words.astype(np.uint32) << 16).view(np.float32).reshape(-1, vocab)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--vocab", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ref, nat = load(args.reference, args.vocab), load(args.native, args.vocab)
    result = judge([top2(row) for row in nat], [top2(row) for row in ref])
    result["sha256"] = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in (("reference", args.reference), ("native", args.native))}
    with args.output.open("x") as out:
        out.write(json.dumps(result, indent=2) + "\n")
    for item in result["exceptions"]:
        print(f"position {item['position']}: {item['verdict']}: reference {item['python_top2']}, native {item['native_top2']}")
    print(f"{result['strict_equal']} of {result['positions']} positions equal; {len(result['exceptions'])} differ; "
          f"policy {'pass' if result['policy_pass'] else 'FAIL'}")
    if not result["policy_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
