"""Rewrite fixtures/serve/template.jsonl's `expected` with the DeepSeek-V4.1 encoder's text, V4's in `expected_v4`."""
import argparse
import json
from pathlib import Path

GOLDEN = Path(__file__).resolve().parents[2] / "zig/src/families/deepseek_v41/fixtures/serve/template.jsonl"


def updated(case, text):
    """The golden line with `text` as its V4.1 prompt; V4's text moves to `expected_v4` the first time they part."""
    v4 = case.get("expected_v4", case["expected"])
    if text is None or v4 is None:
        if text != case["expected"]:
            raise SystemExit(f"a refusal changed: {case['messages']!r}")
        return case
    out = {}
    for key, value in case.items():
        if key == "expected_v4":
            continue
        out[key] = text if key == "expected" else value
        if key == "expected" and text != v4:
            out["expected_v4"] = v4
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rendered", type=Path, help="the file TF_DSV41_TEMPLATE_OUT=FILE zig build test-dsv41-serve-host wrote")
    args = parser.parse_args()
    cases = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line]
    rendered = [json.loads(line) for line in args.rendered.read_text(encoding="utf-8").splitlines() if line]
    if len(cases) != len(rendered):
        raise SystemExit(f"{len(cases)} golden lines but {len(rendered)} renderings")
    out = [updated(c, r["text"]) for c, r in zip(cases, rendered)]
    GOLDEN.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in out), encoding="utf-8")
    changed = sum(1 for c, o in zip(cases, out) if c["expected"] != o["expected"])
    print(f"{len(out)} lines, {changed} V4.1 prompts changed, {sum('expected_v4' in o for o in out)} differ from V4's")


if __name__ == "__main__":
    main()
