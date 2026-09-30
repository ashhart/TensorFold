"""Measure served Flash Next probabilities and compare drafted, serial and repeated replies."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import json
from pathlib import Path
import resource
import threading
import time

import torch

from tensorfold.cuda.server import App, Server, make_handler
from tensorfold.families.qwen4_exp import cuda_engine

PROMPTS = [
    "Answer A or B only: is Paris in France? A yes B no",
    "Write a Python function that reverses a linked list and explain it.",
    "Explain why the sky is blue in simple language.",
    "Write a Python function that merges two sorted lists.",
    "Describe a calm morning beside a lake.",
    "Write a Python binary search with boundary checks.",
    "Explain photosynthesis to a child.",
    "Write a small Python CSV parser example.",
    "Describe how to prepare a vegetable soup.",
]


def request(port, body):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
    start = time.perf_counter()
    try:
        connection.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        response = connection.getresponse()
        result = json.loads(response.read())
        elapsed = time.perf_counter() - start
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {result}")
        stats = result.get("tensorfold", {})
        ids = stats.get("token_ids", [])
        if "token_ids" not in stats or len(ids) != result["usage"]["completion_tokens"]:
            raise RuntimeError("qualification requires every emitted token id")
        result["measurement"] = {"elapsed_s": elapsed, "stats": stats,
                                 "sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                                 "tokens": len(ids), "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                 "rss_peak_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        return result
    finally:
        connection.close()


def body(prompt, seed=13, temperature=0, count=64, **extra):
    return {"messages": [{"role": "user", "content": prompt}], "max_tokens": count,
            "temperature": temperature, "top_k": 20, "top_p": 0.95, "min_p": 0, "seed": seed,
            "return_token_ids": True, "chat_template_kwargs": {"enable_thinking": False}, **extra}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--supported", action="store_true")
    parser.add_argument("--pp", action="store_true")
    args = parser.parse_args()
    engine = cuda_engine(args.model, parallel=5, context=69632, context_explicit=True, mtp_confidence=0.7)
    app = App(engine, args.model, "probability-check", context_window=69632)
    server = Server(("127.0.0.1", 0), make_handler(app))
    worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    worker.start()
    port, rows = server.server_port, []

    def record(name, payload):
        torch.cuda.reset_peak_memory_stats()
        result = request(port, payload)
        row = {"cell": name, "response": result}
        rows.append(row)
        print(json.dumps({"cell": name, **result["measurement"]}), flush=True)
        args.output.write_text(json.dumps({"label": args.label, "rows": rows}, indent=2) + "\n")
        return result

    try:
        decision = record("decision-reproduction", body(PROMPTS[0], count=1, logprobs=True, top_logprobs=5))
        present = "logprobs" in decision["choices"][0]
        assert present == args.supported, ("feature capability", present)
        exact = []
        for index, prompt in enumerate(PROMPTS):
            payload = body(prompt, seed=13 + index, temperature=1 if index in (3, 4, 7, 8) else 0)
            request(port, payload)
            plain = record(f"cell-{index}-off", payload)
            if not args.supported:
                continue
            wanted = {**payload, "logprobs": True, "top_logprobs": 20}
            drafted = record(f"cell-{index}-drafted", wanted)
            serial = record(f"cell-{index}-serial", {**wanted, "draft": False})
            repeat = record(f"cell-{index}-repeat", wanted)
            hashes = [r["measurement"]["sha256"] for r in (plain, drafted, serial, repeat)]
            assert len(set(hashes)) == 1, ("token mismatch", index, hashes)
            probabilities = [r["choices"][0]["logprobs"] for r in (drafted, serial, repeat)]
            assert probabilities[0] == probabilities[1] == probabilities[2], ("probability mismatch", index)
            exact.append(index)
        if args.supported:
            from tensorfold.engine.exact_sampling import Sampling
            from tensorfold.engine.probabilities import Probabilities

            prompt = app.prepare(body(PROMPTS[2]), True).prompt

            def direct(tokens, drafted):
                probabilities = Probabilities(20, len(tokens), 32)
                out = []
                stats = engine.generate(tokens, 32, Sampling(seed=47, top_k=20, top_p=0.95),
                                        lambda new: out.extend(new) or False, draft=drafted,
                                        probabilities=probabilities)
                return {"tokens": out, "probabilities": probabilities.emitted(out), "stats": stats}

            first = direct(prompt, True)
            continuation = prompt + first["tokens"] + app.tok.encode(" Continue.").ids
            resumed, fresh = direct(continuation, True), direct(continuation, False)
            assert resumed["stats"].get("cached", 0) > 0, "resume check did not reuse a prefix"
            assert resumed["tokens"] == fresh["tokens"] and resumed["probabilities"] == fresh["probabilities"]
            rows.append({"cell": "resumed-vs-fresh", "resumed": resumed, "fresh": fresh})
            from tensorfold.cuda.logprobs import capture

            # Replay a real target row at the largest supported tree width, outside timed requests.
            width = max(16, engine.multi.buf.rows)
            logits = engine.multi.buf.logits[:1].expand(width, -1).contiguous()
            chosen = logits.argmax(-1).cpu().tolist()
            collector = Probabilities(20, 0, width)
            torch.cuda.synchronize()
            live = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            capture(logits, chosen, list(range(width)), collector, rows=list(range(width)))
            peak = torch.cuda.max_memory_allocated()
            rows.append({"cell": "top20-maximum-tree-memory", "rows": width, "vocab": logits.shape[1],
                         "live_bytes": live, "peak_allocated_bytes": peak, "transient_bytes": peak - live})
            assert peak - live < 256 * 1024**2
            del logits, collector
            payloads = [body(prompt, count=32, logprobs=True, top_logprobs=5) for prompt in PROMPTS[1:6]]
            alone = [record(f"alone-{i}", p) for i, p in enumerate(payloads)]
            with ThreadPoolExecutor(max_workers=5) as pool:
                together = list(pool.map(lambda p: request(port, p), payloads))
            for i, (a, b) in enumerate(zip(alone, together)):
                assert a["measurement"]["sha256"] == b["measurement"]["sha256"], ("concurrent tokens", i)
                assert a["choices"][0]["logprobs"] == b["choices"][0]["logprobs"], ("concurrent probabilities", i)
                rows.append({"cell": f"together-{i}", "response": b})
        if args.pp:
            for size in (2048, 8192, 16384, 32768, 65536):
                prompt = f"Prompt length {size}. " + "data " * max(1, size - 40)
                payload = body(prompt, count=1)
                count = len(app.prepare(payload, True).prompt)
                prompt += "data " * max(0, size - count)
                result = record(f"cold-pp-{size}", body(prompt, count=1))
                assert result["measurement"]["stats"].get("cached", 0) == 0
        args.output.write_text(json.dumps({"label": args.label, "exact_cells": exact, "rows": rows}, indent=2) + "\n")
    finally:
        server.shutdown()
        server.server_close()
        worker.join(5)


if __name__ == "__main__":
    main()
