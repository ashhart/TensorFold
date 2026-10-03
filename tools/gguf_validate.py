"""GGUF-path validation for the PR: every check writes its raw replies to RECEIPTS.

    python3 tools/gguf_validate.py PORT MODEL RECEIPTS_DIR [--parallel]

Without --parallel (a server with one stream): a long prompt drafted == serial, the same prompt again from the
prompt-end cache == the first reply, and JSON-schema output drafted == serial and valid. With --parallel (a server
started with --parallel 3): three concurrent requests each equal their solo run.
"""
import concurrent.futures as cf, hashlib, json, os, sys, time, urllib.request

port, model, receipts = sys.argv[1], sys.argv[2], sys.argv[3]
parallel = "--parallel" in sys.argv
os.makedirs(receipts, exist_ok=True)
URL = f"http://127.0.0.1:{port}/v1/chat/completions"


def call(body, timeout=3600):
    r = urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}),
                               timeout=timeout)
    return json.load(r)


def chat(content, draft=True, **extra):
    body = {"model": model, "messages": [{"role": "user", "content": content}], "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}, "max_tokens": 200, **extra}
    if not draft:
        body["draft"] = False
    d = call(body)
    return d["choices"][0]["message"]["content"], d


def save(name, **data):
    json.dump(data, open(os.path.join(receipts, name + ".json"), "w"), indent=1)


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()[:12]


results = []


def record(name, ok, detail):
    results.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)


if not parallel:
    # 1. long prompt (about 6k tokens, several prefill chunks): drafted == serial
    long = " ".join(f"Section {i}: in a distributed system, nodes exchange messages over unreliable networks and must "
                    f"agree on shared state despite failures; section {i} adds detail number {i * 7 % 13}." for i in range(200))
    long += " Summarise the sections above in five bullet points."
    a, da = chat(long)
    s, ds = chat(long, draft=False)
    save("long_prompt", drafted=a, serial=s, usage=da.get("usage"), tensorfold=da.get("tensorfold"))
    record("long prompt drafted == serial", a == s, f"prompt {da['usage']['prompt_tokens']} tokens, reply {sha(a)} vs {sha(s)}")
    # 2. the same long prompt again (prompt-end cache): same reply as the first, fresh prefill
    b, db = chat(long)
    save("long_prompt_cached", reply=b, usage=db.get("usage"), tensorfold=db.get("tensorfold"))
    record("long prompt repeated (cache) == first", b == a, f"{sha(b)} vs {sha(a)}")
    # 3. structured output: JSON schema, drafted == serial, and the reply parses
    schema = {"type": "object", "properties": {"city": {"type": "string"}, "population": {"type": "integer"},
              "landmarks": {"type": "array", "items": {"type": "string"}}}, "required": ["city", "population", "landmarks"]}
    rf = {"type": "json_schema", "json_schema": {"name": "city", "schema": schema}}
    q = "Describe Helsinki as JSON with its population and three landmarks."
    a, _ = chat(q, response_format=rf)
    s, _ = chat(q, draft=False, response_format=rf)
    try:
        parsed = json.loads(a); valid = all(k in parsed for k in schema["required"])
    except Exception:
        valid = False
    save("structured", drafted=a, serial=s)
    record("structured output drafted == serial, valid JSON", a == s and valid, f"valid={valid} {sha(a)} vs {sha(s)}")
else:
    # 4. concurrent requests (server --parallel 3): each reply equals its solo run
    prompts = ["Explain in two paragraphs what RDMA is and why it lowers latency compared with TCP.",
               "Write a short story of about 200 words about a lighthouse keeper who finds a message in a bottle.",
               "Write a Python function that merges two sorted lists into one sorted list, with a docstring and three doctests."]
    solo = [chat(p)[0] for p in prompts]
    with cf.ThreadPoolExecutor(3) as ex:
        together = list(ex.map(lambda p: chat(p)[0], prompts))
    save("concurrent", solo=solo, together=together)
    for i, (x, y) in enumerate(zip(solo, together)):
        record(f"concurrent request {i} == solo", x == y, f"{sha(y)} vs {sha(x)}")
print(f"\n{sum(ok for _, ok, _ in results)}/{len(results)} passed; receipts in {receipts}")
