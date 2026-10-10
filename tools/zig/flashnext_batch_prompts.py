#!/usr/bin/env python3
"""Write public, deterministic token fixtures for tf-flashnext-batch using a local checkpoint tokenizer."""
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    args.output.mkdir(parents=True, exist_ok=True)
    cases = [
        ("short", "List the first twenty prime numbers, with no introduction.", False),
        ("thinking", "Explain why a square root of two cannot be a rational number.", True),
        ("sparse", "\n".join(f"Item {i}: value {i % 97}." for i in range(240))
         + "\nSummarize the list and explain how the values repeat.", False),
        ("sparse-thinking", "\n".join(f"Row {i}: the box contains {i % 31} blue counters." for i in range(720))
         + "\nDescribe an algorithm to calculate the total number of blue counters.", True),
    ]
    for name, text, thinking in cases:
        tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=True,
            add_generation_prompt=True, enable_thinking=thinking, return_dict=False,
        )
        path = args.output / f"{name}.json"
        path.write_text(json.dumps({"prompt": tokens, "text": text, "thinking": thinking}) + "\n")
        print(f"{path.name}: {len(tokens)} prompt tokens, thinking={thinking}")


if __name__ == "__main__":
    main()
