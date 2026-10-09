"""Judge complete teacher-forced receipts on the same fixed reference history."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from qwen27_margins import verdict


def judge(native, reference, margins):
    if native.get("teacher_forced") is not True or native.get("drafts") is not False:
        raise ValueError("expected teacher-forced plain generation")
    if native.get("prompt_state_byte_equal") is not True:
        raise ValueError("native prompt chunk contract missing")
    if len(reference) != 256 or len(margins) != 256:
        raise ValueError("expected 256 reference positions and margins")
    tokens, top2 = native["tokens"], native["top2"]
    if len(tokens) != 256 or len(top2) != 256:
        raise ValueError("expected 256 native positions and margins")
    exceptions = []
    strict = 0
    for position, (actual, wanted, a, b) in enumerate(zip(tokens, reference, top2, margins)):
        if len(a["ids"]) != 2 or len(b["ids"]) != 2 or len(a["logits"]) != 2 or len(b["logits"]) != 2:
            raise ValueError("expected exactly two ordered scores")
        for item in (a, b):
            if any(type(token) is not int or not 0 <= token < 2**32 for token in item["ids"]):
                raise ValueError("invalid token ID")
            if item["ids"][0] == item["ids"][1]:
                raise ValueError("duplicate top-two token ID")
            if not all(math.isfinite(score) for score in item["logits"]):
                raise ValueError("nonfinite top-two score")
            if item["logits"][0] < item["logits"][1]:
                raise ValueError("top-two scores are not ordered")
        if a["ids"][0] != actual or b["ids"][0] != wanted:
            raise ValueError("argmax and recorded top-two disagree")
        if actual == wanted:
            strict += 1
            continue
        exceptions.append({"position": position, "expected": wanted, "actual": actual,
                           **verdict(a, b, wanted, actual)})
    return {"positions": 256, "strict_equal": strict, "exceptions": exceptions,
            "teacher_forced": True, "native_prompt_byte_equal": True,
            "policy_pass": all(item["verdict"] == "near-tie" for item in exceptions),
            "prompt_last_reference_phase": "official LaneEngine oracle",
            "alternate_prompt_last_phase_checked": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = {}
    for name in ("copy", "arithmetic", "story", "code"):
        paths = {"reference": args.oracle / f"{name}.reference.json",
                 "margins": args.oracle / f"{name}.top2.json",
                 "native": args.native / f"{name}-teacher-native.log"}
        values = {key: json.loads(path.read_text()) for key, path in paths.items()}
        report = judge(values["native"], values["reference"], values["margins"])
        report["sha256"] = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()}
        reports[name] = report
    result = {"prompts": reports, "positions": 1024,
              "strict_equal": sum(item["strict_equal"] for item in reports.values()),
              "policy_pass": all(item["policy_pass"] for item in reports.values())}
    with args.output.open("x") as out:
        out.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    if not result["policy_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
