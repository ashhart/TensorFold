"""Generate the public short and synthetic long EXL3 qualification prompts."""
import json
from pathlib import Path
import sys

SHORT = [
    {
        "name": "code-quicksort",
        "kind": "completion",
        "prompt": "def quicksort(xs):\n    \"\"\"Sort a list with quicksort.\"\"\"\n"
    },
    {
        "name": "rust-reverse",
        "kind": "chat",
        "prompt": "Write a Rust function that reverses the words of a sentence, with a unit test."
    },
    {
        "name": "network-protocols",
        "kind": "chat",
        "prompt": "Compare TCP and UDP: reliability, ordering, congestion control, and typical uses."
    },
    {
        "name": "database-join",
        "kind": "chat",
        "prompt": "Explain what a SQL LEFT JOIN returns compared with an INNER JOIN, with a tiny example table."
    }
]
PARAGRAPH = 'A cache stores recently used values so a later request can avoid repeating expensive work. Each entry has a key, a value and a version. Readers check the version before reusing an entry. A writer publishes the new value before making its version visible. Independent requests keep their own state.\n'


def main():
    directory = Path(sys.argv[1])
    directory.mkdir(parents=True, exist_ok=True)
    long = [{"name": f"long-cache-{i}", "kind": "chat",
             "prompt": f"Document {i}. Read the following engineering notes.\n"
             + PARAGRAPH * 250
             + f"\nFor document {i}, explain why checking a cached entry version matters. Give a concrete example."}
            for i in range(4)]
    for name, prompts in (("short", SHORT), ("long", long)):
        (directory / f"{name}-prompts.json").write_text(json.dumps(prompts, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
