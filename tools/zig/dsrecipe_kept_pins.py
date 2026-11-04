"""Rewrite the pins in deepseek_recipe_kept.zig from a TF_DSRECIPE_REPORT run, after reviewing what changed."""
import argparse
import hashlib
import json
import re
from pathlib import Path

KEPT = Path("zig/src/server/deepseek_recipe_kept.zig")
PIN = re.compile(r"\.\{ (\d+), 0x([0-9a-f]{16}) \}")


def pin(entry):
    """The test's pinOf: SHA-256 of "text" or "refused", a zero byte and the outcome; its first 8 bytes."""
    kind = "text" if entry["text"] is not None else "refused"
    digest = hashlib.sha256(f"{kind}\0{entry[kind]}".encode("utf-8")).digest()
    return digest[:8].hex()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="the TF_DSRECIPE_REPORT directory, holding server.jsonl")
    parser.add_argument("--kept", type=Path, default=KEPT)
    args = parser.parse_args()
    lines = (args.report / "server.jsonl").read_text(encoding="utf-8").splitlines()
    pins = {}
    for line in lines:
        entry = json.loads(line)
        if f"0x{pin(entry)}" != entry["pin"]:
            raise SystemExit(f"case {entry['id']}: the report's pin {entry['pin']} is not this script's 0x{pin(entry)}")
        pins[entry["id"]] = pin(entry)
    changed = []

    def repin(m):
        case = int(m.group(1))
        if pins[case] != m.group(2):
            changed.append(case)
        return f".{{ {case}, 0x{pins[case]} }}"

    source = PIN.sub(repin, args.kept.read_text(encoding="utf-8"))
    args.kept.write_text(source, encoding="utf-8")
    print(f"{len(PIN.findall(source))} kept cases, pins changed: {changed}")


if __name__ == "__main__":
    main()
