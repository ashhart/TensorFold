"""Pack deepseek-recipe's DeepSeek-V4.1 prompts with their request cases into the committed gzip fixture."""
import argparse
import gzip
import json
from pathlib import Path

INPUTS = ("messages", "tools", "thinking", "budget", "drop_thinking", "generation", "response_format", "image_text")
OPTIONAL = ("effort_field", "default_thinking")


def case(index, given, official):
    """One fixture line: the request case, DeepSeek's prompt text (null where it refuses), and its protocol verdicts."""
    if official["id"] != index:
        raise SystemExit(f"official case {official['id']} is on line {index}")
    for key in INPUTS:
        if given[key] != official.get(key, given[key]):
            raise SystemExit(f"case {index}: {key} differs between the two files")
    out = {"id": index}
    out.update({key: given[key] for key in INPUTS})
    out.update({key: given[key] for key in OPTIONAL if key in given})
    out["prompt"] = official.get("official_prompt")
    cc = official.get("official_cc")
    out["chat_ok"] = bool(cc.get("ok")) if isinstance(cc, dict) else None
    effort = official.get("official_effort")
    if isinstance(effort, dict):
        out["effort_ok"] = bool(effort.get("ok"))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True, help="template.jsonl: the request cases")
    parser.add_argument("--official", type=Path, required=True, help="official_template.jsonl: deepseek-recipe's output")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    given = [json.loads(line) for line in args.cases.read_text(encoding="utf-8").splitlines() if line]
    official = [json.loads(line) for line in args.official.read_text(encoding="utf-8").splitlines() if line]
    if len(given) != len(official):
        raise SystemExit(f"{len(given)} cases but {len(official)} official lines")
    lines = [json.dumps(case(i, g, o), ensure_ascii=False) for i, (g, o) in enumerate(zip(given, official))]
    data = ("\n".join(lines) + "\n").encode("utf-8")
    args.out.write_bytes(gzip.compress(data, compresslevel=9, mtime=0))
    print(f"{len(lines)} cases, {len(data)} bytes, {args.out.stat().st_size} gzip bytes")


if __name__ == "__main__":
    main()
