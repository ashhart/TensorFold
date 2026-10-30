"""Compare the model fixtures' checkpoint-template output against Jinja2."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from jinja2.sandbox import ImmutableSandboxedEnvironment


def reject(message: str) -> None:
    """Implement the checkpoint template's explicit error function."""
    raise ValueError(message)


def main() -> int:
    """Compile pinned renderers and compare both complete outputs with Jinja2."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--zig", default="zig")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    repo = here.parents[3]
    zig = shutil.which(args.zig)
    if zig is None or subprocess.check_output([zig, "version"], text=True).strip() != "0.17.0":
        raise ValueError("Zig 0.17.0 is required")
    pinned = json.loads((here / "environment.json").read_text())
    spec = importlib.util.spec_from_file_location("model_client", here / "client.py")
    client = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client)
    source = args.template.read_text()
    expected_hash = pinned["checkpoint_assets_sha256"]["chat_template.jinja"]
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    if source_hash != expected_hash:
        raise ValueError("checkpoint template does not match the pinned hash")
    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"],
    )
    env.globals["raise_exception"] = reject
    cases, conversations = [], []
    for name, messages in client.fixtures().items():
        index = len(conversations)
        conversations.append(messages)
        for generation in (False, True):
            expected = env.from_string(source).render(
                messages=messages, tools=None, add_generation_prompt=generation, enable_thinking=False,
            )
            cases.append({
                "id": name + ("-generation" if generation else "-history"), "t": "qwen", "m": index,
                "tl": None, "c": 0, "g": generation, "ok": True, "raised": False, "text": expected,
            })
    args.out.mkdir(parents=True, exist_ok=False)
    output = args.out.resolve()
    corpus = output / "corpus.json"
    corpus.write_text(json.dumps({
        "templates": {"qwen": source}, "conversations": conversations, "tools": [],
        "contexts": [{"enable_thinking": False}], "cases": cases,
    }, ensure_ascii=False))
    summary = {"template_sha256": source_hash, "cases": len(cases), "revisions": {}}
    with tempfile.TemporaryDirectory(prefix="template-model-parity-") as temporary:
        for side in ("base", "candidate"):
            snapshot = Path(temporary) / side
            snapshot.mkdir()
            archive = subprocess.check_output([
                "git", "-C", str(repo), "archive", pinned[side],
                "zig/src/core/template", "zig/src/platform",
            ])
            subprocess.run(["tar", "-xf", "-", "-C", str(snapshot)], input=archive, check=True)
            binary = output / (side + "-parity")
            subprocess.run([
                zig, "build-exe", "-O", "ReleaseFast", "--dep", "sys",
                "-Mroot=zig/src/core/template/parity.zig", "-Msys=zig/src/platform/sys.zig",
                "-lc", "-femit-bin=" + str(binary),
            ], cwd=snapshot, check=True)
            result_path = output / (side + "-results.json")
            subprocess.run([str(binary), str(corpus), str(result_path)], check=True)
            results = json.loads(result_path.read_text())
            if len(results) != len(cases):
                raise ValueError("unexpected result count")
            records = []
            for want, got in zip(cases, results):
                actual = got.get("text", "")
                actual = actual.encode() if isinstance(actual, str) else bytes(actual)
                expected = want["text"].encode()
                records.append({
                    "id": want["id"], "equal": got["ok"] and actual == expected,
                    "bytes": len(expected), "sha256": hashlib.sha256(expected).hexdigest(),
                })
            summary["revisions"][side] = records
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    passed = all(r["equal"] for records in summary["revisions"].values() for r in records)
    print(json.dumps({"cases_per_revision": len(cases), "all_equal": passed}))
    return int(not passed)


if __name__ == "__main__":
    raise SystemExit(main())
