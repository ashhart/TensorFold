from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .controller import Controller, LearningError
from .evaluator import evaluate
from .ledger import Ledger, LedgerError, exclusive_file
from .schema import ManifestError, load_manifest
from .transport import Client, TransportError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m living_weights_v2", description="External guarded learning controller for native TensorFold Living Weights")
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="validate and summarize a manifest without writing weights")
    plan.add_argument("manifest", type=Path)
    evaluation = sub.add_parser("evaluate", help="read-only lexical recall probes over any OpenAI-compatible chat server")
    evaluation.add_argument("manifest", type=Path)
    evaluation.add_argument("--base-url", default="http://127.0.0.1:8080")
    evaluation.add_argument("--api-key-file", type=Path, required=True)
    evaluation.add_argument("--allow-remote", action="store_true")
    teach = sub.add_parser("teach", help="explicitly teach user-approved facts through the native API")
    teach.add_argument("manifest", type=Path)
    teach.add_argument("--base-url", default="http://127.0.0.1:8080")
    teach.add_argument("--api-key-file", type=Path, required=True)
    teach.add_argument("--ledger", type=Path, default=Path.home() / ".local/state/tensorfold/living-weights-v2.sqlite3")
    teach.add_argument("--allow-remote", action="store_true", help="explicitly allow an authenticated remote server")
    teach.add_argument("--timeout", type=int, default=3600)
    status = sub.add_parser("status", help="show durable audit history and pending recovery")
    status.add_argument("--ledger", type=Path, default=Path.home() / ".local/state/tensorfold/living-weights-v2.sqlite3")
    status.add_argument("--model")
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            manifest = load_manifest(args.manifest)
            output = {"schema": 1, "model": manifest.model, "manifest_digest": manifest.digest, "source": manifest.source, "facts": [
                {"subject": f.subject, "relation": f.relation, "fact_digest": f.digest, "positive_count": len(f.questions), "near_miss_count": len(f.near_misses)}
                for f in manifest.facts
            ], "anchor_count": len(manifest.anchors), "mutates_weights": False}
            print(json.dumps(output, indent=2))
            return 0
        if args.command == "status":
            ledger = Ledger(args.ledger)
            try:
                print(json.dumps(ledger.history(args.model), indent=2, ensure_ascii=False))
            finally:
                ledger.close()
            return 0
        if args.command == "evaluate":
            manifest = load_manifest(args.manifest)
            client = Client(args.base_url, args.api_key_file, args.allow_remote)
            if manifest.model not in client.models():
                raise LearningError("model not listed by chat server")
            print(json.dumps(evaluate(manifest, client), indent=2, ensure_ascii=False))
            return 0
        if args.timeout <= 0:
            raise ValueError("timeout must be positive")
        manifest = load_manifest(args.manifest)
        client = Client(args.base_url, args.api_key_file, args.allow_remote, args.timeout)
        with exclusive_file(args.ledger.with_suffix(args.ledger.suffix + ".lock")):
            ledger = Ledger(args.ledger)
            try:
                result = Controller(client, ledger).teach(manifest)
                print(json.dumps(result, indent=2, ensure_ascii=False))
                return 3 if any(r["state"] == "needs_recovery" for r in result) else (2 if any(r["state"] in {"rejected", "rolled_back"} for r in result) else 0)
            finally:
                ledger.close()
    except (ManifestError, TransportError, LearningError, LedgerError, ValueError) as exc:
        print(f"living_weights_v2: {exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
