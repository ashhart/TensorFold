#!/usr/bin/env python3
"""quality_suite.py URL OUT: TensorFold Web's quality suite against one served model, scored as its Bench scores it."""
# Each suite item runs once, greedy, with thinking off when the server advertises that switch.
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

WEB = pathlib.Path(__file__).resolve().parent.parent / "web"
sys.path.insert(0, str(WEB))
from tensorfold_web import quality  # noqa: E402

OFF = {"chat_template_kwargs": {"enable_thinking": False}}


def call(url: str, body: dict | None = None, timeout: float = 30) -> dict:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ask(base: str, model: str, it: dict, off: dict, timeout: float) -> dict:
    """One item: the reply's verdict, or an unscored error as the Bench keeps a failed request apart."""
    body = {"model": model, "messages": it["messages"], "max_tokens": it["max_tokens"], "temperature": 0.0,
            "top_p": 1.0, "stream": False, **off}
    try:
        choice = call(base + "/v1/chat/completions", body, timeout)["choices"][0]
    except (urllib.error.URLError, OSError, KeyError, IndexError, ValueError) as e:
        return {"category": it["category"], "id": it["id"], "passed": None, "read": f"error: {e}"[:200], "text": ""}
    text = choice.get("message", {}).get("content") or ""
    passed, read = quality.score(it["category"], it["item"], text)
    if not passed and choice.get("finish_reason") == "length":
        read = f"cut at {it['max_tokens']} tokens ({read})"
    return {"category": it["category"], "id": it["id"], "passed": passed, "read": read, "text": text[:2000]}


def summary(data: dict, results: list[dict]) -> dict:
    """The Bench's quality summary: graded items and passes, overall and per category; errors counted apart."""
    graded = [r for r in results if r["passed"] is not None]
    cats = {c: [r for r in graded if r["category"] == c] for c in data["categories"]}
    return {"overall": {"items": len(graded), "passed": sum(r["passed"] for r in graded),
                        "unscored": len(results) - len(graded)},
            "categories": {c: {"items": len(v), "passed": sum(r["passed"] for r in v)} for c, v in cats.items()}}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("url", help="the server, for example http://127.0.0.1:8080")
    p.add_argument("out", help="the receipt to write (JSON)")
    p.add_argument("--checkpoint", default="", help="REPO@REVISION served, kept in the receipt")
    p.add_argument("--engine", default="", help="the engine commit served, kept in the receipt")
    p.add_argument("--machine", default="", help="the chip it ran on, kept in the receipt")
    p.add_argument("--timeout", type=float, default=900, help="seconds a request may take (900)")
    a = p.parse_args(argv)
    base = a.url.rstrip("/")
    entry = call(base + "/v1/models")["data"][0]
    off = OFF if (entry.get("capabilities") or {}).get("thinking_switch") else {}
    data = quality.load()
    water = json.loads((WEB / "tensorfold_web" / "data" / "suites.json").read_text(encoding="utf-8"))["passages"]["water"]
    work = quality.items(data, water + "\n\n")
    started, results = time.time(), []
    for i, it in enumerate(work, 1):
        results.append(ask(base, entry["id"], it, off, a.timeout))
        print(f"{i}/{len(work)} {it['category']} {it['id']}: {results[-1]['read']}", file=sys.stderr, flush=True)
    q = summary(data, results)
    receipt = {"suite": {"quality": data["version"], "digest": data["digest"]},
               "protocol": {"temperature": 0.0, "top_p": 1.0, "concurrency": 1,
                            "thinking": "off: chat_template_kwargs.enable_thinking" if off else "model default"},
               "model": entry["id"], "checkpoint": a.checkpoint, "engine": a.engine, "machine": a.machine,
               "seconds": round(time.time() - started, 1), "quality": q, "items": results}
    pathlib.Path(a.out).write_text(json.dumps(receipt, indent=1) + "\n", encoding="utf-8")
    o = q["overall"]
    cats = ", ".join(f"{c} {v['passed']}/{v['items']}" for c, v in q["categories"].items())
    print(f"quality {o['passed']}/{o['items']} ({100 * o['passed'] / max(1, o['items']):.1f}%); {cats}; "
          f"unscored {o['unscored']}")
    return 0 if o["unscored"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
