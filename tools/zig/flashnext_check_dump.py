"""Compare checked-in Flash Next kernels and the role table with a dump. Exit 1 on any difference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dump", type=Path)
    args = ap.parse_args()

    table = (ROOT / "zig" / "src" / "families" / "flashnext" / "roles_gen.zig").read_text()
    checked = ROOT / "zig" / "kernels" / "metal" / "flashnext"

    bad = 0
    dump_files = sorted((args.dump / "kernels").glob("*.metal"))
    for src in dump_files:
        mine = checked / src.name
        if not mine.exists():
            print(f"MISSING  {src.name}: no checked-in file")
            bad += 1
        elif mine.read_bytes() != src.read_bytes():
            print(f"DIFFERS  {src.name}: the checked-in text is not the recorded one")
            bad += 1
    for mine in sorted(checked.glob("*.metal")):
        if mine.name == "sources_gen.zig":
            continue
        if not (args.dump / "kernels" / mine.name).exists():
            # the -lanes twins: the no-tensor-unit recorder writes them; a tensor-unit dump does not carry them
            if "-lanes." in mine.name:
                continue
            print(f"EXTRA    {mine.name}: no recorded file in the dump")
            bad += 1
    print(f"{len(dump_files)} recorded kernel files compared, byte equal where checked in")

    # every recorded site must be in the generated table with the same function, grid and threadgroup
    import json

    plan = json.loads((args.dump / "plan.json").read_text())
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from flashnext_checked import role_key

    entries = {}
    for line in table.splitlines():
        line = line.strip()
        if not line.startswith('.{ .site = '):
            continue
        parts = line.split('.site = "', 1)[1].split('"', 1)
        site = parts[0]
        fields = {}
        for key in ("function", "file"):
            fields[key] = line.split(f'.{key} = "', 1)[1].split('"', 1)[0]
        fields["grid"] = line.split(".grid = .{ ", 1)[1].split(" }", 1)[0]
        fields["tg"] = line.split(".tg = .{ ", 1)[1].split(" }", 1)[0]
        entries[site] = fields
    checked_sites, wrong = 0, 0
    for site, v in plan["sites"].items():
        key = role_key(site)
        mine = entries.get(key)
        checked_sites += 1
        if mine is None:
            print(f"NO ROLE  {key}: the dump recorded it, the table has no entry")
            bad += 1
            wrong += 1
            continue
        want_fn = v["function"]
        want_grid = ", ".join(str(x) for x in v["grid"])
        want_tg = ", ".join(str(x) for x in v["threadgroup"])
        if mine["function"] != want_fn or mine["grid"] != want_grid or mine["tg"] != want_tg:
            print(f"WRONG    {key}: function {mine['function']} vs {want_fn}, grid {mine['grid']} vs {want_grid}, tg {mine['tg']} vs {want_tg}")
            bad += 1
            wrong += 1
    print(f"{checked_sites} recorded sites checked against the table, {wrong} wrong")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
