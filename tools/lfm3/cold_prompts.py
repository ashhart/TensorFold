"""Regenerate the synthetic cold-prompt request bodies and rendered token arrays, and check them against their hashes."""
import argparse
import hashlib
import json
import pathlib

SHA256 = pathlib.Path(__file__).resolve().parents[2] / "docs/receipts/lfm3/cold-prompt-sha256.json"
LENGTHS = (2048, 8192, 32768, 65536)
PREFIX = "A public benchmark fixture. Explain this sequence.\n"
# The chat template's header and the prefix render to 13 tokens, the trailer to 7, each repeated word to token 23018.
HEAD = [124894, 124899, 5922, 207, 41, 1251, 40524, 51639, 22, 42847, 532, 8055, 318]
TAIL = [124900, 207, 124899, 63514, 207, 124901, 124902]
WORD = 23018
FIXED = {
    "max_tokens": 16, "temperature": 0, "top_p": 1, "top_k": 0, "repetition_penalty": 1, "stream": True,
    "stream_options": {"include_usage": True}, "enable_thinking": False,
    "chat_template_kwargs": {"enable_thinking": False},
}


def body(tokens: int) -> bytes:
    """The request body whose rendered prompt is exactly `tokens` long."""
    words = tokens - len(HEAD) - len(TAIL)
    return (json.dumps({"messages": [{"role": "user", "content": PREFIX + " alpha" * words}]} | FIXED) + "\n").encode()


def ids(tokens: int) -> bytes:
    return (json.dumps(HEAD + [WORD] * (tokens - len(HEAD) - len(TAIL)) + TAIL) + "\n").encode()


def files() -> dict[str, bytes]:
    return {f"{kind}-{n}.{part}.json": make(n) for kind in ("text", "vl") for n in LENGTHS
            for part, make in (("body", body), ("ids", ids))}


def mismatches(produced: dict[str, bytes], wanted: dict[str, str]) -> list[str]:
    """Files whose hash differs from the record, and recorded files nothing produced."""
    bad = [name for name, data in produced.items() if hashlib.sha256(data).hexdigest() != wanted.get(name)]
    return bad + [name for name in wanted if name not in produced]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", type=pathlib.Path, help="directory to write the 16 files into")
    args = parser.parse_args()
    wanted = json.loads(SHA256.read_text())
    produced = files()
    bad = mismatches(produced, wanted)
    for name in bad:
        print(f"MISMATCH {name}")
    print(f"{len(produced) - len(bad)} of {len(wanted)} files match their recorded SHA-256")
    if args.write:
        args.write.mkdir(parents=True, exist_ok=True)
        for name, data in produced.items():
            (args.write / name).write_bytes(data)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
