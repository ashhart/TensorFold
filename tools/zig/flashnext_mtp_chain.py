#!/usr/bin/env python3
"""Compare serial and GPU-chained MTP drafts with alternating HTTP blocks and exact token receipts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics

import flashnext_batch_serve as gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("fixtures", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=4)
    args = parser.parse_args()
    if args.tokens < 1 or args.rounds < 3:
        parser.error("tokens must be positive; rounds must include one warmup and at least two measured rounds")
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = {"exact": False, "tokens": args.tokens, "rounds": args.rounds,
               "warmup_rounds": 1, "blocks": []}

    def save():
        (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")

    save()
    receipt["binary_sha256"] = hashlib.sha256(args.binary.read_bytes()).hexdigest()
    cases = [json.loads((args.fixtures / f"{name}.json").read_text()) for name in
             ("short", "thinking", "sparse", "sparse-thinking")]
    reference = None
    previous = os.environ.get("FZ_BATCH_MTP_CHAIN")
    try:
        # The one-slot legacy engine is a control; four-slot blocks reverse order to limit drift.
        for block, (slots, chain) in enumerate(((1, 0), (1, 1), (4, 0), (4, 1), (4, 1), (4, 0))):
            os.environ["FZ_BATCH_MTP_CHAIN"] = str(chain)
            print(f"BLOCK {block}: parallel={slots} chain={chain}", flush=True)
            batches = gate.run(args, slots, cases, block)
            receipt["blocks"].append({"parallel": slots, "chain": chain, "batches": batches})
            save()
            if slots > 1 and not any(batch["peak_running"] > 1 for batch in batches):
                raise RuntimeError(f"shared sessions were not observed in block {block}")
            for batch in batches:
                replies = [{key: result[key] for key in ("reply", "tokens", "prompt_tokens", "token_sha")}
                           for result in batch["results"]]
                if reference is None:
                    reference = replies
                if replies != reference:
                    raise RuntimeError(f"token or reply mismatch in block {block}, round {batch['iteration']}")
        receipt["exact"] = True
        medians = {}
        for slots in (1, 4):
            seconds = {chain: statistics.median(
                batch["seconds"] for block in receipt["blocks"]
                if block["parallel"] == slots and block["chain"] == chain
                for batch in block["batches"][1:]) for chain in (0, 1)}
            medians[str(slots)] = {"serial_seconds": seconds[0], "chain_seconds": seconds[1],
                                  "throughput_gain_percent": (seconds[0] / seconds[1] - 1) * 100}
        receipt["warm_medians"] = medians
        save()
        print(json.dumps(medians, indent=2), flush=True)
    finally:
        if previous is None:
            os.environ.pop("FZ_BATCH_MTP_CHAIN", None)
        else:
            os.environ["FZ_BATCH_MTP_CHAIN"] = previous


if __name__ == "__main__":
    main()
