"""Soak of tensorfold-native serve on the Intel GPU: mixed requests for hours, then SIGINT; run under the GPU guard."""
import csv
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request

PORT = 18420
BASE = f"http://127.0.0.1:{PORT}"
CHAT = "Write one sentence about the sea."


def post(path, body, timeout=3600):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def metric(name):
    text = urllib.request.urlopen(BASE + "/metrics", timeout=30).read().decode()
    m = re.search(rf"^{re.escape(name)} (\S+)$", text, re.M)
    return float(m.group(1)) if m else None


def chat(tokens, temperature=0.0, seed=None, think=False):
    body = {"model": "x", "messages": [{"role": "user", "content": CHAT}], "max_tokens": tokens, "temperature": temperature,
            "ignore_eos": True, "chat_template_kwargs": {"enable_thinking": think}}
    if seed is not None:
        body["seed"] = seed
    return post("/v1/chat/completions", body)


def decode_job(tokens=128, **kw):
    r = chat(tokens, **kw)
    return r["tensorfold"]["tokens_per_second"], r["tensorfold"]["token_sha"]


def prefill_job(arrays, length):
    body = json.load(open(os.path.join(arrays, f"cold_{length}.json")))
    r = post("/v1/completions", body)
    return length / r["tensorfold"]["prefill_seconds"], None


def concurrent_job():
    import threading
    out = []
    ts = [threading.Thread(target=lambda: out.append(decode_job(64))) for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    shas = {s for _, s in out}
    if len(shas) != 1:
        raise RuntimeError(f"four simultaneous greedy replies gave {len(shas)} different digests")
    return sum(t for t, _ in out) / len(out), None


def disconnect_job():
    body = {"model": "x", "messages": [{"role": "user", "content": CHAT}], "max_tokens": 3000, "stream": True,
            "ignore_eos": True, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"})
    r = urllib.request.urlopen(req, timeout=60)
    r.read(2000)
    r.close()
    time.sleep(1)
    return decode_job(64)[0], None


# usage: xpu_soak_server.py BIN MODEL ARRAYS_DIR OUT_DIR [--hours H | --cycles N] [--quick]
def main():
    a = sys.argv[1:]
    binp, model, arrays, out = a[:4]
    hours = float(a[a.index("--hours") + 1]) if "--hours" in a else 0
    cycles = int(a[a.index("--cycles") + 1]) if "--cycles" in a else (0 if hours else 1)
    quick = "--quick" in a
    os.makedirs(out, exist_ok=True)
    # soak.csv has the columns xpu_soak_report.py reads; ARRAYS_DIR holds cold_<L>.json from xpu_token_arrays.py
    ctx = 9000 if quick else 70000
    lengths = [2048, 8192] if quick else [2048, 8192, 32768, 65536]
    log = open(os.path.join(out, "server.log"), "w")
    server = subprocess.Popen([binp, "serve", model, "--port", str(PORT), "--context", str(ctx), "--prompt-cache-gib", "0",
                               "--no-update-check"], stdout=log, stderr=subprocess.STDOUT)
    for _ in range(300):
        try:
            urllib.request.urlopen(BASE + "/v1/models", timeout=5)
            break
        except OSError:
            time.sleep(1)
    csv_file = open(os.path.join(out, "soak.csv"), "w", newline="")
    w = csv.writer(csv_file)
    w.writerow(["cycle", "job", "toks", "peak_gb", "rc", "seconds"])
    solo = None
    start, cycle, failed = time.time(), 0, 0
    while True:
        cycle += 1
        jobs = [("decode_greedy", lambda: decode_job(128)), ("decode_sampled", lambda: decode_job(128, temperature=0.8, seed=cycle)),
                ("concurrent4", concurrent_job), ("disconnect", disconnect_job)]
        jobs += [(f"prefill_{n}", (lambda n=n: prefill_job(arrays, n))) for n in lengths]
        for name, fn in jobs:
            t0, rc, toks = time.time(), 0, "-"
            try:
                tps, sha = fn()
                toks = f"{tps:.2f}"
                if name == "decode_greedy":
                    solo = solo or sha
                    if sha != solo:
                        raise RuntimeError("greedy digest changed")
            except Exception as e:
                rc = 1
                print(f"cycle {cycle} {name}: {e}", flush=True)
            peak = metric("tensorfold:device_memory_peak_bytes")
            failed += rc
            w.writerow([cycle, name, toks, f"{peak / 1e9:.3f}" if peak else "-", rc, f"{time.time() - t0:.0f}"])
            csv_file.flush()
            if os.path.exists("/tmp/arc_stop"):
                cycles = cycle
        if (cycles and cycle >= cycles) or (hours and time.time() - start >= hours * 3600):
            break
    server.send_signal(signal.SIGINT)
    code = server.wait(timeout=60)
    print(f"server exit {code}; failed jobs {failed}", flush=True)
    sys.exit(0 if code == 0 and failed == 0 else 1)


main()
