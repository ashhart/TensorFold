"""Queue every workload item at t=0 and serve it with N in flight; output tok/s, per-stream decode, TTFT, memory.

Output tok/s = completion tokens / (last token of any request - first request sent). Per stream: (tokens - 1) /
(last - first token). ``--mem`` samples nvidia-smi's compute processes (all but ``--ignore``) and the host's
MemTotal - MemAvailable every 0.5 s. Standard library only.
"""

import argparse
import hashlib
import json
import queue
import statistics
import subprocess
import threading
import time
import urllib.request


def one(base: str, model: str, item: dict, max_tokens: int, temperature: float, seed: int | None,
        draft: bool) -> dict:
    body = {"model": model, "messages": item["messages"], "max_tokens": max_tokens, "temperature": temperature,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    if seed is not None:
        body["seed"] = seed
    if temperature > 0:
        body.update(top_k=20, top_p=0.95)
    if not draft:
        body["draft"] = False
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    out = {"id": item["id"], "sent": time.perf_counter(), "first": None, "last": None, "tokens": 0, "pieces": 0,
           "text_sha": None, "token_sha": None, "finish": None, "prompt_tokens": None, "error": None, "arrivals": []}
    text = []
    try:
        with urllib.request.urlopen(req, timeout=3600) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[5:])
                if chunk.get("error"):
                    out["error"] = str(chunk["error"])[:300]
                usage = chunk.get("usage")
                if usage:
                    out["tokens"] = int(usage.get("completion_tokens", 0))
                    out["prompt_tokens"] = usage.get("prompt_tokens")
                stats = chunk.get("tensorfold")
                if stats:
                    out["token_sha"] = stats.get("token_sha")
                    out["stats"] = stats
                for choice in chunk.get("choices", []):
                    piece = (choice.get("delta") or {}).get("content") or ""
                    if piece:
                        now = time.perf_counter()
                        out["first"] = out["first"] or now
                        out["last"] = now
                        out["pieces"] += 1
                        out["arrivals"].append((round(now, 4), len(piece)))
                        text.append(piece)
                    if choice.get("finish_reason"):
                        out["finish"] = choice["finish_reason"]
    except Exception as exc:  # noqa: BLE001 - a failed request is reported
        out["error"] = f"{type(exc).__name__}: {exc}"[:300]
    out["text_sha"] = hashlib.sha256("".join(text).encode()).hexdigest()[:16]
    return out


class Memory(threading.Thread):
    def __init__(self, ignore: tuple[str, ...]) -> None:
        super().__init__(daemon=True)
        self.ignore, self.stop, self.samples = ignore, threading.Event(), []

    @staticmethod
    def host_used() -> float:
        info = {}
        for row in open("/proc/meminfo"):
            key, value = row.split(":")
            info[key] = int(value.split()[0])
        return (info["MemTotal"] - info["MemAvailable"]) / 2**20

    def gpu(self) -> float:
        rows = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        total = 0.0
        for row in rows.strip().splitlines():
            parts = [p.strip() for p in row.split(",")]
            if len(parts) == 3 and not any(name in parts[1] for name in self.ignore):
                try:
                    total += float(parts[2]) / 1024
                except ValueError:
                    pass
        return total

    def snapshot(self) -> tuple[float, float]:
        return self.gpu(), self.host_used()

    def run(self) -> None:
        while not self.stop.is_set():
            self.samples.append((time.perf_counter(), *self.snapshot()))
            self.stop.wait(0.5)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("base")
    p.add_argument("model")
    p.add_argument("prompts")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--first", type=int, default=0, help="only the first N items")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=None, help="sampled: item i gets seed + i")
    p.add_argument("--no-draft", action="store_true", help='send "draft": false (the serial reference)')
    p.add_argument("--mem", action="store_true")
    p.add_argument("--ignore", default="", help="comma list of process names nvidia-smi should not count")
    p.add_argument("--label", default="")
    p.add_argument("--output")
    args = p.parse_args()
    items = json.load(open(args.prompts))["items"]
    items = items[:args.first] if args.first else items
    memory = Memory(tuple(s for s in args.ignore.split(",") if s)) if args.mem else None
    idle = memory.snapshot() if memory else None
    todo: queue.Queue = queue.Queue()
    for item in items:
        todo.put(item)
    results: list[dict] = []
    lock = threading.Lock()

    def worker() -> None:
        while True:
            try:
                item = todo.get_nowait()
            except queue.Empty:
                return
            seed = None if args.seed is None else args.seed + item["id"]
            r = one(args.base, args.model, item, args.max_tokens, args.temperature, seed, not args.no_draft)
            with lock:
                results.append(r)

    if memory:
        memory.start()
    start = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(args.concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if memory:
        memory.stop.set()
        memory.join()
    ok = [r for r in results if not r["error"] and r["first"] is not None]
    end = max((r["last"] for r in ok), default=start)
    tokens = sum(r["tokens"] for r in ok)
    per = [(r["tokens"] - 1) / (r["last"] - r["first"]) for r in ok if r["tokens"] > 1 and r["last"] > r["first"]]
    ttft = sorted(r["first"] - r["sent"] for r in ok)
    summary = {"label": args.label, "concurrency": args.concurrency, "requests": len(items), "ok": len(ok),
               "failed": len(results) - len(ok), "output_tokens": tokens, "wall_s": round(end - start, 2),
               "output_tps": round(tokens / (end - start), 1) if end > start else 0.0,
               "per_stream_tps_median": round(statistics.median(per), 1) if per else None,
               "ttft_s_median": round(statistics.median(ttft), 2) if ttft else None,
               "ttft_s_p90": round(ttft[int(0.9 * (len(ttft) - 1))], 2) if ttft else None,
               "ttft_s_max": round(max(ttft), 2) if ttft else None,
               "start": start,
               "prompt_tokens_total": sum(r["prompt_tokens"] or 0 for r in ok),
               "stopped_early": sum(1 for r in ok if r["finish"] == "stop")}
    if memory and memory.samples:
        summary.update(idle_gpu_gib=round(idle[0], 2), idle_used_gib=round(idle[1], 2),
                       peak_gpu_gib=round(max(s[1] for s in memory.samples), 2),
                       peak_used_gib=round(max(s[2] for s in memory.samples), 2))
    print(json.dumps(summary), flush=True)
    if args.output:
        results.sort(key=lambda r: r["id"])
        json.dump({"summary": summary, "results": results,
                   "memory": [list(s) for s in memory.samples] if memory else []}, open(args.output, "w"), indent=1)


if __name__ == "__main__":
    main()
