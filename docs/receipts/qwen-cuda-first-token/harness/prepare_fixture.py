"""Pin literal PR836 prompts to the selected server's rendered token IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import run_panel


def sha(path: Path) -> str:
    """Hash a literal source without relying on its older token IDs."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def body(case: dict[str, Any]) -> dict[str, Any]:
    """Construct the same tokenize request used by the panel preflight."""
    if case["kind"] == "chat":
        return {"messages": case["messages"], "add_generation_prompt": True,
                "chat_template_kwargs": {"enable_thinking": False}}
    return {"prompt": case["prompt"], "add_special_tokens": True}


def prepare(corpus_source: Path, long_source: Path,
            tokenize: Callable[[dict[str, Any]], list[int]]) -> tuple[dict, dict]:
    """Regenerate and check prompt IDs before any timed request."""
    literal = json.loads(corpus_source.read_text())
    anchors = literal["anchors"]
    cases = literal["cases"]
    if len(anchors) != 2 or len(cases) != 42:
        raise ValueError("literal source corpus changed shape")
    for case in anchors + cases:
        ids = tokenize(body(case))
        if not ids or any(type(token) is not int or token < 0 for token in ids):
            raise ValueError(f"{case['case_id']}: invalid rendered token IDs")
        case["input_ids"] = ids
        case["prompt_tokens"] = len(ids)
    corpus = {"schema": "pr836_ttft_literal_corpus_v1",
              "source_sha256": sha(corpus_source),
              "anchors": anchors, "cases": cases}
    changes = [
        {"case_id": "code-changed", "name": "code_changed",
         "kind": "completion", "prompt": anchors[0]["prompt"]
         + "\nNow handle negative inputs too."},
        {"case_id": "chat-changed", "name": "chat_changed", "kind": "chat",
         "messages": anchors[1]["messages"] + [
             {"role": "assistant", "content": "Acknowledged."},
             {"role": "user", "content": "What changes if the matrices are sparse?"},
         ]},
    ]
    for case in changes:
        ids = tokenize(body(case))
        if not ids:
            raise ValueError(f"{case['case_id']}: empty rendered prompt")
        case["input_ids"] = ids
        case["prompt_tokens"] = len(ids)
    corpus["changes"] = changes

    old_long = json.loads(long_source.read_text())
    messages = old_long["messages"]
    rendered = {}
    for name in ("initial", "followup"):
        ids = tokenize({"messages": messages[name],
                        "add_generation_prompt": True,
                        "chat_template_kwargs": {"enable_thinking": False}})
        if not ids:
            raise ValueError(f"long/{name}: empty rendered prompt")
        rendered[name] = {"generation": ids}
    first = rendered["initial"]["generation"]
    second = rendered["followup"]["generation"]
    common = 0
    for a, b in zip(first, second):
        if a != b:
            break
        common += 1
    if common < 4096:
        raise ValueError("changing chat history has fewer than 4096 shared tokens")
    long = {"schema": "pr836_chat_history_fixture_v1",
            "source_sha256": sha(long_source), "messages": messages,
            "rendered": rendered, "common_full_prefix_tokens": common}
    return corpus, long


def main() -> None:
    """Build two immutable input files against the selected running baseline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--corpus-source", type=Path, required=True)
    parser.add_argument("--long-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    corpus, long = prepare(
        args.corpus_source, args.long_source,
        lambda request: run_panel.tokenize(args.base, request),
    )
    run_panel.write(args.output_dir / "corpus.json", corpus)
    run_panel.write(args.output_dir / "long-manifest.json", long)


if __name__ == "__main__":
    main()
