"""One arm of base/new: cold TTFT, content, decode. usage: ab.py LABEL BIN MODEL OUTDIR  (page cache dropped before each server)"""
import hashlib, json, os, random, re, signal, subprocess, sys, time, urllib.request
LABEL, BIN, MODEL, OUT = sys.argv[1:5]
PORT = 18408
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

def server(tag, extra):
    subprocess.run(os.environ.get("PT_DROP_CACHES", "sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'"), shell=True, check=True, timeout=300)
    log = f"{OUT}/{LABEL}-{tag}.server.log"
    p = subprocess.Popen([BIN, "serve", MODEL, "--port", str(PORT), "--name", "m", "--no-update-check", "--parallel", "1", "--context", "40960", "--no-drafts"] + os.environ.get("PT_EXTRA", "").split() + extra, stdout=open(log, "w"), stderr=subprocess.STDOUT, start_new_session=True)
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
    return re.findall(r"prompt=(\d+) cached=(\d+).*?tokens=(\d+) sha=(\w+).*?ttft=([\d.]+) decode=([\d.]+)", open(log).read())

def content(d):
    c = d["choices"][0]
    return {"msg": c["message"], "lp": c.get("logprobs"), "plp": d.get("prompt_logprobs") or c.get("prompt_logprobs")}

R = {"label": LABEL, "md5": hashlib.md5(open(BIN, "rb").read()).hexdigest(), "cold": {}}
p, log = server("cold", ["--prompt-cache-gib", "0"])
try:
    post(text(40, 1), max_tokens=4); time.sleep(2)
    for name, w in [x for x in (("2k", 520), ("16k", 4100), ("32k", 8200)) if x[0] in os.environ.get("PT_LEN", "2k,16k,32k").split(",")]:
        R["cold"][name] = []
        for rep in range(3):
            d = post(text(w, 1000 * (rep + 1) + w), max_tokens=16, logprobs=True, top_logprobs=20)
            l = done(log)[-1]
            R["cold"][name].append({"prompt": int(l[0]), "cached": int(l[1]), "sha": l[3], "ttft": float(l[4]), **content(d)})
            print(LABEL, name, rep, l[0], l[4], flush=True)
    d = post(text(520, 77), max_tokens=128, ignore_eos=True); l = done(log)[-1]
    R["decode"] = {"tokens": int(l[2]), "s": float(l[5]), "sha": l[3], **content(d)}
    d = post(text(4100, 78), max_tokens=16, temperature=0.8, seed=7); R["seeded"] = {"sha": done(log)[-1][3], **content(d)}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions", json.dumps({"model": "m", "prompt": text(2200, 79), "max_tokens": 4, "temperature": 0, "prompt_logprobs": 5}).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r: d = json.loads(r.read())
    c = d["choices"][0]; R["prompt_lp"] = {"msg": c.get("text"), "lp": c.get("logprobs"), "plp": c.get("prompt_logprobs")}
finally:
    stop(p)
p, log = server("kept", ["--prompt-cache-gib", "4"])
try:
    a = text(4100, 80); post(a, max_tokens=8); d = post(a + " And then in one line.", max_tokens=16, logprobs=True, top_logprobs=20)
    d2 = post(a + " And then in one line.", max_tokens=16, logprobs=True, top_logprobs=20); l = done(log)
    R["kept"] = {"cached": [int(x[1]) for x in l], "shas": [x[3] for x in l], "a": content(d), "b": content(d2)}
finally:
    stop(p)
json.dump(R, open(f"{OUT}/{LABEL}.json", "w"))
print("ARM-DONE", LABEL, flush=True)
