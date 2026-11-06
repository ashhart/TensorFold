"""One drafted arm: the TTFT table's prompts (2k/16k/32k x 3, temp 0) with drafts on. usage: abd.py LABEL BIN MODEL OUTDIR  (extra serve flags in PT_EXTRA)"""
import hashlib, json, os, random, re, signal, subprocess, sys, time, urllib.request
LABEL, BIN, MODEL, OUT = sys.argv[1:5]
PORT = int(os.environ.get("PT_PORT", "18409"))
VOCAB = ["river", "stone", "lamp", "copper", "garden", "signal", "harbor", "winter", "ledger", "orbit", "candle", "meadow"]
os.makedirs(OUT, exist_ok=True)

def text(words, seed):
    r = random.Random(seed)
    return f"Document {seed}.\n\n" + " ".join(f"{r.choice(VOCAB)}{r.randint(0, 999)}" for _ in range(words)) + "\n\nSummarize:"

def post(msg, **kw):
    body = {"model": "m", "messages": [{"role": "user", "content": msg}], "temperature": 0, **kw}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())

def drop_caches():
    cmd = os.environ.get("PT_DROP_CACHES", "sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'")
    subprocess.run(cmd, shell=True, check=True, timeout=300)

def server(extra):
    drop_caches()
    log = f"{OUT}/{LABEL}.server.log"
    p = subprocess.Popen([BIN, "serve", MODEL, "--port", str(PORT), "--name", "m", "--no-update-check", "--parallel", "1", "--context", "40960", "--prompt-cache-gib", "0"] + os.environ.get("PT_EXTRA", "").split() + extra, stdout=open(log, "w"), stderr=subprocess.STDOUT, start_new_session=True)
    t0 = time.monotonic()
    while True:
        if p.poll() is not None: raise SystemExit(f"server exited {p.returncode}")
        if time.monotonic() - t0 > 600: p.kill(); raise SystemExit("not ready")
        try: urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=3).read(); break
        except Exception: time.sleep(1)
    return p, log

def stop(p):
    p.send_signal(signal.SIGTERM)
    try: p.wait(120)
    except subprocess.TimeoutExpired: p.kill(); p.wait()

def done(log):
    return re.findall(r"prompt=(\d+) cached=(\d+).*?tokens=(\d+) sha=(\w+).*?accepted=(\d+)/(\d+) ttft=([\d.]+) decode=([\d.]+)", open(log).read())

R = {"label": LABEL, "md5": hashlib.md5(open(BIN, "rb").read()).hexdigest(), "cold": {}}
p, log = server([])
try:
    R["serving"] = [l for l in open(log).read().splitlines() if " serving m at " in l]
    post(text(40, 1), max_tokens=4); time.sleep(2)
    for name, w in (("2k", 520), ("16k", 4100), ("32k", 8200)):
        R["cold"][name] = []
        for rep in range(3):
            d = post(text(w, 1000 * (rep + 1) + w), max_tokens=64)
            l = done(log)[-1]
            R["cold"][name].append({"prompt": int(l[0]), "cached": int(l[1]), "tokens": int(l[2]), "sha": l[3], "accepted": int(l[4]), "drafted": int(l[5]), "ttft": float(l[6]), "decode": float(l[7]), "msg": d["choices"][0]["message"]})
            print(LABEL, name, rep, l[0], "accepted", f"{l[4]}/{l[5]}", "sha", l[3], flush=True)
finally:
    stop(p)
json.dump(R, open(f"{OUT}/{LABEL}.json", "w"), indent=1)
print("ARM-DONE", LABEL, flush=True)
