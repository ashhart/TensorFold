"""Check synthetic accumulation output against Jinja2 without a model or GPU."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

TEMPLATES = {
    'two-live-chains': (
        '{% set ns=namespace(a=prefix,b=prefix) %}{% for m in messages %}'
        "{% set ns.a=ns.a+m.content %}{% set ns.b=ns.b+('B' ~ m.content) %}{% endfor %}{{ ns.a }}"
        '<SPLIT>{{ ns.b }}'
    ),
    'long-grouped-rhs': (
        "{% set ns=namespace(out='') %}{% for m in messages %}"
        '{% set ns.out=ns.out+(prefix ~ m.content ~ suffix) %}{% endfor %}{{ ns.out }}'
    ),
    'saved-growing-snapshots': (
        '{% set ns=namespace(out=prefix,snaps=[]) %}{% for m in messages %}'
        '{% if loop.index0 % 8 == 0 %}{% set ns.snaps=ns.snaps+[ns.out] %}{% endif %}'
        '{% set noise=prefix ~ m.role %}{% set ns.out=ns.out+m.content %}{% endfor %}'
        '{% for s in ns.snaps %}[{{ s }}]{% endfor %}|{{ ns.out }}'
    ),
    'snapshot-branches': (
        '{% set ns=namespace(out=prefix) %}{% for m in messages %}'
        '{% set ns.out=ns.out+m.content %}{% endfor %}{% set saved=ns.out %}'
        "{% set ns.left=saved+suffix %}{% set ns.right=saved+'RIGHT' %}{% for m in messages %}"
        "{% set ns.left=ns.left+m.content %}{% set ns.right=ns.right+('R' ~ m.content) %}"
        '{% endfor %}{{ saved }}|{{ ns.left }}|{{ ns.right }}|{{ saved }}'
    ),
    'self-concat': (
        '{% set ns=namespace(out=prefix) %}{% for i in range(rounds) %}{% set snapshot=ns.out %}'
        "{% set ns.out=ns.out+ns.out %}{% set ns.out=ns.out ~ '' %}{% endfor %}{{ ns.out }}"
    ),
    'safe-escaping': (
        '{% set ns=namespace(out=prefix|escape,raw=prefix) %}{% for m in messages %}'
        '{% set ns.out=ns.out+(m.content|escape) %}{% set ns.raw=ns.raw+m.content %}{% endfor %}'
        '{{ ns.out }}|{{ ns.out|escape }}|{{ ns.raw|escape }}|{{ ns.raw|safe|escape }}'
    ),
    'macro-interleave': (
        '{% macro wrap(m) %}<{{ m.role }}>{{ m.content }}</{{ m.role }}>{% endmacro %}'
        "{% set ns=namespace(out=prefix,other='') %}{% for m in messages %}"
        "{% set ns.out=ns.out+wrap(m) %}{% set ns.other=ns.other+('X' ~ m.content) %}{% endfor %}"
        '{{ ns.out }}|{{ ns.other }}'
    ),
    'empty-aliases': (
        "{% set ns=namespace(out=prefix,empty='') %}{% set saved=ns.out %}{% for m in messages %}"
        '{% set ns.out=ns.out+ns.empty %}{% set ns.out=ns.empty+ns.out %}'
        '{% set saved_empty=ns.empty %}{% set ns.out=ns.out+m.content %}{% endfor %}{{ saved }}'
        "|{{ ns.out }}|{{ ns.empty }}|{{ saved_empty|default('') }}"
    ),
    'macro-mutates-held-lhs': (
        '{% set ns=namespace(out=prefix) %}{% macro mutate(m) %}'
        '{% set ns.out=ns.out+prefix+m.content %}[{{ m.role }}]{% endmacro %}'
        '{% for m in messages %}{% set ns.out=ns.out+mutate(m) %}{% endfor %}{{ ns.out }}'
    ),
}


def sha256(path: Path) -> str:
    """Hash one source or result artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_corpus(path: Path) -> dict[str, Any]:
    """Build expected bytes from Jinja2 and synthetic Unicode, HTML and NUL data."""
    environment = ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=["jinja2.ext.loopcontrols"],
    )
    conversations: list[list[dict[str, str]]] = []
    contexts: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    for kind, source in TEMPLATES.items():
        for size in (1, 8, 32, 96):
            messages = [
                {
                    "role": "user" if i % 2 == 0 else "assistant",
                    "content": f"item{i:03d}:<&>\"'é雪\x00" + chr(65 + i % 26) * 240,
                }
                for i in range(size)
            ]
            context = {
                "prefix": 'P<&>"é' * 780,
                "suffix": "TAIL<&>" * 80,
                "rounds": min(5, 1 + size // 8),
            }
            expected = environment.from_string(source).render(
                messages=messages, tools=None, add_generation_prompt=False, **context
            )
            index = len(conversations)
            conversations.append(messages)
            contexts.append(context)
            cases.append({
                "id": f"{kind}-{size}", "t": kind, "m": index, "tl": None,
                "c": index, "g": False, "ok": True, "raised": False,
                "text": expected,
            })
    corpus = {
        "templates": TEMPLATES,
        "conversations": conversations,
        "tools": [],
        "contexts": contexts,
        "cases": cases,
    }
    path.write_text(json.dumps(corpus, ensure_ascii=False), encoding="utf-8")
    return corpus


def compare(corpus: dict[str, Any], results_path: Path) -> list[dict[str, Any]]:
    """Compare actual bytes independently because parity.zig only prints mismatch counts."""
    actual = json.loads(results_path.read_text(encoding="utf-8"))
    if len(actual) != len(corpus["cases"]):
        raise ValueError("result count differs from corpus")
    mismatches = []
    for want, got in zip(corpus["cases"], actual):
        expected = want["text"].encode("utf-8")
        text_value = got.get("text", "")
        value = text_value.encode("utf-8") if isinstance(text_value, str) else bytes(text_value)
        if not got.get("ok") or expected != value:
            first = next(
                (i for i, pair in enumerate(zip(expected, value)) if pair[0] != pair[1]),
                min(len(expected), len(value)),
            )
            mismatches.append({
                "id": want["id"], "ok": got.get("ok"),
                "expected_bytes": len(expected), "actual_bytes": len(value),
                "first_difference": first,
                "expected_sha256": hashlib.sha256(expected).hexdigest(),
                "actual_sha256": hashlib.sha256(value).hexdigest(),
                "error": got.get("error", ""),
            })
    return mismatches


def main() -> int:
    """Compile the checked-out runner and verify 36 independently rendered cases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zig", default="zig", help="Zig executable, matching .zig-version")
    parser.add_argument("--out", type=Path, help="artifact directory; default is a fresh temporary directory")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    output = args.out or Path(tempfile.mkdtemp(prefix="tf-template-parity-"))
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    executable = shutil.which(args.zig)
    if executable is None:
        raise FileNotFoundError(f"Zig executable unavailable: {args.zig}")
    zig_version = subprocess.check_output([executable, "version"], text=True).strip()
    required = (repo / ".zig-version").read_text().strip()
    if zig_version != required:
        raise ValueError(f"Zig {required} required; got {zig_version}")
    source_names = [
        "zig/src/core/template/eval.zig",
        "zig/src/core/template/provenance.zig",
        "zig/src/core/template/template.zig",
        "zig/src/core/template/parity.zig",
        "zig/src/platform/sys.zig",
    ]
    source_hashes = {name: sha256(repo / name) for name in source_names}
    binary = output / "parity"
    compile_command = [
        executable, "build-exe", "-lc", "--dep", "sys",
        "-Mroot=zig/src/core/template/parity.zig",
        "-Msys=zig/src/platform/sys.zig", "-O", "Debug",
        f"-femit-bin={binary}", "--cache-dir", str(output / "cache"),
        "--global-cache-dir", str(output / "global-cache"),
    ]
    with (output / "compile.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            compile_command, cwd=repo, stdout=log, stderr=subprocess.STDOUT,
            check=True, timeout=180,
        )
    corpus_path = output / "corpus.json"
    corpus = write_corpus(corpus_path)
    results_path = output / "results.json"
    begin = time.monotonic()
    process = subprocess.run(
        [str(binary), str(corpus_path), str(results_path)],
        capture_output=True, text=True, timeout=120, check=True,
    )
    elapsed = time.monotonic() - begin
    for name, digest in source_hashes.items():
        if sha256(repo / name) != digest:
            raise ValueError(f"source changed during verification: {name}")
    mismatches = compare(corpus, results_path)
    summary = {
        "jinja2_version": jinja2.__version__, "zig_version": zig_version,
        "source_sha256": source_hashes,
        "cases": len(corpus["cases"]), "passed": len(corpus["cases"]) - len(mismatches),
        "mismatches": mismatches, "elapsed_s": elapsed,
        "runner_stderr": process.stderr.strip(), "binary_sha256": sha256(binary),
        "corpus_sha256": sha256(corpus_path),
        "expected_output_bytes": sum(len(case["text"].encode("utf-8")) for case in corpus["cases"]),
    }
    encoded = json.dumps(summary, indent=2) + "\n"
    (output / "summary.json").write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    print(f"Artifacts: {output}")
    return int(bool(mismatches))


if __name__ == "__main__":
    raise SystemExit(main())
