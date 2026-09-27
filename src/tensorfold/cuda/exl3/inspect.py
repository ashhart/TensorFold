"""``python -m tensorfold.cuda.exl3.inspect MODEL_DIR [--json]``: an EXL3 checkpoint's bits and codebooks per tensor category, from headers alone."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from . import format as fmt

_NUMBER = re.compile(r"\.\d+(?=\.|$)")


def category(name: str) -> str:
    return _NUMBER.sub(".*", name)


def summarize(model_dir: str | Path) -> dict:
    """The checkpoint's EXL3 contents as a dict (what ``main`` prints)."""

    ck = fmt.scan(model_dir)
    q = ck.config.get("quantization_config") or ck.config.get("text_config", {}).get("quantization_config") or {}
    cats: dict[str, Counter] = defaultdict(Counter)
    codebooks: Counter = Counter()
    scales: Counter = Counter()
    bits_all: Counter = Counter()
    weights = 0
    trellis_bytes = 0
    for g in ck.groups.values():
        cats[category(g.prefix)][g.bits] += 1
        codebooks[g.codebook] += 1
        scales[f"{g.in_scales}/{g.out_scales}" + ("+bias" if g.bias else "")] += 1
        bits_all[g.bits] += g.k * g.n
        weights += g.k * g.n
        trellis_bytes += g.trellis_bytes
    bad_markers = {name: hex(v) for name, v in ck.markers.items()
                   if v != fmt.MARKERS[name.rsplit(".", 1)[1]]}
    plain: dict[str, Counter] = defaultdict(Counter)
    for name, (dtype, shape) in ck.plain.items():
        plain[category(name)][f"{dtype} {list(shape)}" if len(shape) != 1 else dtype] += 1
    return {
        "model_dir": str(model_dir),
        "model_type": ck.config.get("model_type"),
        "config": {k: q.get(k) for k in ("quant_method", "version", "bits", "head_bits", "codebook", "out_scales")
                   if k in q},
        "exl3_tensors": len(ck.groups),
        "codebooks": dict(codebooks),
        "scales": dict(scales),
        "bits_by_weights": {f"{b:g}": round(c / max(weights, 1), 4) for b, c in sorted(bits_all.items())},
        "average_bits": round(8 * trellis_bytes / max(weights, 1), 4),
        "trellis_gb": round(trellis_bytes / 1e9, 3),
        "categories": {c: {f"{b:g}": n for b, n in sorted(v.items())} for c, v in sorted(cats.items())},
        "plain": {c: dict(v) for c, v in sorted(plain.items())},
        "unsupported": dict(sorted(ck.bad.items())),
        "bad_markers": bad_markers,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tensorfold.cuda.exl3.inspect", description=__doc__.split("\n")[0])
    parser.add_argument("model_dir")
    parser.add_argument("--json", action="store_true", help="print one JSON object")
    args = parser.parse_args(argv)
    s = summarize(args.model_dir)
    if args.json:
        print(json.dumps(s, indent=1))
        return 1 if s["unsupported"] or s["bad_markers"] else 0
    print(f"{s['model_dir']}  model_type {s['model_type']}  config {s['config']}")
    print(f"EXL3 tensors {s['exl3_tensors']}  codebooks {s['codebooks']}  scales {s['scales']}")
    print(f"average {s['average_bits']} bits over {s['trellis_gb']} GB of trellis; share of weights by bits: "
          + ", ".join(f"{b}: {v:.1%}" for b, v in s["bits_by_weights"].items()))
    print("\nquantized categories (tensors per bits):")
    width = max((len(c) for c in s["categories"]), default=10)
    for c, hist in s["categories"].items():
        print(f"  {c:<{width}}  " + "  ".join(f"{b}b x{n}" for b, n in hist.items()))
    print("\nplain tensors:")
    for c, kinds in s["plain"].items():
        print(f"  {c:<{width}}  " + "  ".join(f"{k} x{n}" for k, n in kinds.items()))
    if s["unsupported"] or s["bad_markers"]:
        print("\nUNSUPPORTED:")
        by_reason: dict[str, list[str]] = defaultdict(list)
        for prefix, why in s["unsupported"].items():
            by_reason[why].append(prefix)
        for why, prefixes in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
            examples = ", ".join(category(p) if len(prefixes) > 3 else p for p in sorted(prefixes)[:3])
            print(f"  {len(prefixes)} group(s): {why}")
            print(f"      e.g. {examples}")
        for name, v in s["bad_markers"].items():
            print(f"  {name}: unexpected marker value {v}")
        return 1
    print("\nevery EXL3 tensor is readable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
