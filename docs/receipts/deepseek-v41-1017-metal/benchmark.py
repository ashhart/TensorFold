"""Measure cold prompt processing and concurrent prose on the installed server."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import statistics
import subprocess
import threading
import time
import urllib.request

MODEL = "deepseek-v4.1-flash-1017"
PP_LENGTHS = (2048, 8192, 16384, 32760)
DECODE_TOKENS = 256
PARAGRAPHS = (
    "The harbor clerks met before sunrise to compare the records that had arrived the night before. "
    "A freight ship waited outside the breakwater, and two of the lamps above the pier needed replacing. "
    "In the long silence between ferry crossings, the clerks wrote out each change in a ruled notebook.",
    "The surveyors set out along the old canal, stopping where the embankment had shifted in the spring. "
    "They compared a map with the broken pavement and marked each place where someone could cross on foot. "
    "At noon they sat beneath an ash tree and described the work still to be done.",
    "The library opened an hour early on Thursday. A driver left a wooden case on the steps. "
    "Inside lay a set of field notebooks, their labels faded but still readable. "
    "The librarian began a fresh inventory, noting the weather and the year beside each account.",
    "An electrician carried the plans upstairs to a room with several long tables. "
    "The room had once held a switchboard. The windows were high enough that nobody could see the street. "
    "Small pieces of tape marked the positions where new lights had to hang.",
)


def response(base, path, request=None, timeout=240):
    data = None if request is None else json.dumps(request).encode()
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def get_json(base, path, request=None):
    with response(base, path, request) as resp:
        return json.load(resp)


def before(base, want_batch=None):
    health = get_json(base, "/health")
    assert health["status"] == "ok" and health["model"] == MODEL, health
    if want_batch is not None:
        assert health["max_batch_size"] == want_batch, health
    for _ in range(40):
        if health["live"]["connections"] == health["live"]["waiting"] == 0:
            break
        time.sleep(0.25)
        health = get_json(base, "/health")
    else:
        raise RuntimeError(f"Benchmark endpoint stayed busy: {health['live']}")
    return {
        "device": health["device"],
        "model": health["model"],
        "version": health["version"],
        "server_lanes": health["max_batch_size"],
    }


class Memory:
    def __init__(self, pid):
        self.pid = pid
        self.done = threading.Event()
        self.peak_kib = 0

    def monitor(self):
        while not self.done.is_set():
            raw = subprocess.check_output(["/bin/ps", "-p", str(self.pid), "-o", "rss="], text=True)
            self.peak_kib = max(self.peak_kib, int(raw.strip()))
            self.done.wait(0.4)

    def __enter__(self):
        self.thread = threading.Thread(target=self.monitor, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.done.set()
        self.thread.join()

    @property
    def peak_gib(self):
        return self.peak_kib / (1024 ** 2)


def chat(prompt, output_tokens):
    return {
        "model": MODEL, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": output_tokens, "temperature": 0, "seed": 1234, "ignore_eos": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def pp_prompt(base, size, index):
    labels = ("Bluejay", "Tungsten", "Daffodil", "Marigold", "Juniper", "Halibut",
              "Quicksilver", "Sundial", "Zeppelin", "Woodlark", "Starling", "Argon")
    label = labels[3 * PP_LENGTHS.index(size) + index]
    lead = f"{label} is the cover title. Read this fictional journal before responding.\n"
    text = "\n".join(f"{i+1}. {PARAGRAPHS[i % len(PARAGRAPHS)]}" for i in range((size + 49) // 50 + 100))

    def build(length):
        content = lead + text[:length] + "\nReply in one word."
        messages = chat(content, 1)["messages"]
        count = get_json(base, "/v1/tokenize", {
            "messages": messages, "chat_template_kwargs": {"enable_thinking": False}
        })["count"]
        return count, content

    low, high = 0, len(text)
    assert build(high)[0] >= size
    selected = None
    while low <= high:
        mid = (low + high) // 2
        count, content = build(mid)
        if count <= size and (selected is None or selected[0] < count):
            selected = count, content
        if count == size:
            break
        if count < size:
            low = mid + 1
        else:
            high = mid - 1
    assert selected is not None and size - selected[0] <= 4, (size, selected[0] if selected else None)
    return selected


def prompt_processing(base, pid, out):
    before(base, 1)
    results = []
    for target in PP_LENGTHS:
        for rep in range(3):
            n, prompt = pp_prompt(base, target, rep)
            before(base, 1)
            with Memory(pid) as memory:
                start = time.perf_counter()
                reply = get_json(base, "/v1/chat/completions", chat(prompt, 1))
                wall = time.perf_counter() - start
            usage, runtime = reply["usage"], reply["tensorfold"]
            cached = usage["prompt_tokens_details"]["cached_tokens"]
            assert usage["prompt_tokens"] == n and 0 <= cached <= 2 and usage["completion_tokens"] == 1, (n, usage)
            seconds = runtime["prefill_seconds"]
            assert seconds and seconds > 0
            record = dict(target=target, prompt_tokens=n, cached_tokens=cached,
                          prefill_seconds=seconds, prefill_tps=(n - cached) / seconds,
                          ttft_seconds=runtime["time_to_first_token"], wall_seconds=wall,
                          peak_process_rss_gib=memory.peak_gib, rep=rep,
                          prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
            print(json.dumps(record), flush=True)
            results.append(record)
            out.write_text(json.dumps(results, indent=2) + "\n")


def streaming(base, prompt, gate=None):
    body = dict(chat(prompt, DECODE_TOKENS), stream=True, stream_options={"include_usage": True}, draft=True)
    if gate is not None:
        gate.wait(timeout=20)
    start = time.perf_counter()
    first = last = runtime = usage = None
    complete = False
    pieces = 0
    with response(base, "/v1/chat/completions", body) as resp:
        for raw in resp:
            if not raw.startswith(b"data:"):
                continue
            payload = raw[5:].strip()
            if payload == b"[DONE]":
                complete = True
                break
            event = json.loads(payload)
            if "error" in event:
                raise RuntimeError(event["error"])
            runtime = event.get("tensorfold") or runtime
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                delta = choice.get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content"):
                    last = time.perf_counter()
                    first = first or last
                    pieces += 1
    end = time.perf_counter()
    assert complete and runtime and usage and first and last > first and pieces >= 4, (bool(runtime), usage, pieces)
    n = usage["completion_tokens"]
    assert n == DECODE_TOKENS and usage["prompt_tokens_details"]["cached_tokens"] == 0, usage
    return dict(first=first, last=last, start=start, end=end, tokens=n,
                ttft_seconds=first-start, client_decode_tps=(n-1)/(last-first),
                server_decode_tps=runtime["tokens_per_second"], output_ids_sha256=runtime["token_sha256"])


def collect_burst(base, variants):
    gate = threading.Barrier(len(variants))
    with ThreadPoolExecutor(max_workers=len(variants)) as pool:
        futures = [pool.submit(streaming, base, prompt, gate) for _, prompt in variants]
        return [dict(name=name, **f.result()) for (name, _), f in zip(variants, futures)]


def summarize_burst(rows, level, rep, reference, rss):
    start, first, last = min(r["start"] for r in rows), min(r["first"] for r in rows), max(r["last"] for r in rows)
    chats = sorted(r["ttft_seconds"] for r in rows)
    first_wait = sorted(r["first"] for r in rows)
    return dict(level=level, rep=rep, requests=len(rows), completion_tokens=sum(r["tokens"] for r in rows),
                aggregate_decode_tps=sum(r["tokens"]-1 for r in rows)/(last-first),
                aggregate_end_to_end_tps=sum(r["tokens"] for r in rows)/(last-start),
                median_server_per_request_tps=statistics.median(r["server_decode_tps"] for r in rows),
                median_client_per_request_tps=statistics.median(r["client_decode_tps"] for r in rows),
                ttft_p50_seconds=statistics.median(chats), ttft_p95_seconds=chats[math.ceil(0.95*len(chats))-1],
                first_response_offset_seconds=[t-start for t in first_wait],
                exact_count=sum(r["output_ids_sha256"] == reference[r["name"]]["output_ids_sha256"] for r in rows),
                peak_process_rss_gib=rss, streams=rows)
