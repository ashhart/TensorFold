"""Release gate step for the native engine: passed cells as manifest entries, zig/gate.json from them, and a check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from tensorfold import families, native
from tensorfold.native import contract, switch


def _installed() -> tuple[Path, dict]:
    binary = native.binary()
    if binary is None:
        raise SystemExit(f"no native engine in this install ({native.ROOT / native.BINARY})")
    return binary, contract.capabilities(binary)


def cell(run: str, model: Path, backend: str = "auto") -> dict:
    """The manifest entry for ``model`` on this machine with this install's bundle, checked against the schema."""

    _, found = _installed()
    if found["chip"] is None:
        raise SystemExit("the native engine reports no usable GPU here")
    entry = {"family": families.model_type(model), "format": contract.weight_format(families.read_config(model)),
             "backend": switch.backend(backend), "chip": found["chip"], "run": run, "bundle": native.bundle_digest()}
    contract.check(entry, contract.schema("gate")["properties"]["entries"]["items"])
    return entry


def write(out: Path, cells: list[Path], bundles: list[str] = ()) -> dict:
    """zig/gate.json from cell files (an entry or a whole manifest each), sorted, deduplicated, checked."""

    entries = []
    for path in cells:
        document = json.loads(path.read_text())
        entries += document["entries"] if "entries" in document else [document]
    keep = sorted({json.dumps(entry, sort_keys=True) for entry in entries if not bundles or entry["bundle"] in bundles})
    manifest = {"schema": 1, "entries": [json.loads(entry) for entry in keep]}
    contract.check(manifest, contract.schema("gate"))
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify() -> int:
    """Print this install's native engine, chip, bundle and the gate entries that name its bundle."""

    _, found = _installed()
    digest = native.bundle_digest()
    entries = contract.manifest()["entries"]
    mine = [entry for entry in entries if entry["bundle"] == digest]
    print(json.dumps({"version": found["version"], "chip": found["chip"], "bundle": digest,
                      "entries": len(entries), "this_bundle": mine}, indent=1))
    return 0 if mine else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("cell", help="print the entry for a cell whose native arm passed, run in that install")
    one.add_argument("--run", required=True, help="the release gate label, e.g. rel066a (public: no machine names)")
    one.add_argument("--model", required=True, type=Path, help="the checkpoint directory the arm served")
    one.add_argument("--backend", choices=("auto", "mlx", "cuda"), default="auto")
    out = commands.add_parser("write", help="write zig/gate.json from cell files or manifests")
    out.add_argument("--out", required=True, type=Path)
    out.add_argument("--bundle", action="append", default=[], help="keep only entries for this bundle sha256")
    out.add_argument("cells", nargs="*", type=Path)
    commands.add_parser("verify", help="exit 0 only when this install's bundle has gate entries")
    args = parser.parse_args(argv)
    if args.command == "cell":
        print(json.dumps(cell(args.run, args.model, args.backend), indent=1))
        return 0
    if args.command == "write":
        manifest = write(args.out, args.cells, args.bundle)
        print(f"{args.out}: {len(manifest['entries'])} entries", file=sys.stderr)
        return 0
    return verify()


if __name__ == "__main__":
    raise SystemExit(main())
