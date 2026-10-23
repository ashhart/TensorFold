"""X9's live checks (<plan>): structured output from GLM's CUDA lanes on a running TP server.
Copied from <records> (<session>) for <plan>'s S9 on 1.1.1 (FP4-Local and NVFP4, in <container>):
PLAIN_SHA from the environment, else this run's first plain reply.

Run in the <container> container on rank 0's host: python3 x9_live.py OUT.jsonl [--concurrent] [--mixed4] [--cold-first].
One JSON line a check in OUT, a summary on stdout; exit 1 when a check fails. The last line (check "_greedy") holds
every greedy reply's text, token_sha and calls, for `compare`.
- schema: a json_schema reply parses and has the schema's keys and types (thinking off, then on)
- same-as-plain: a schema the plain greedy reply already meets gives the same text under the grammar
- draft-false: the schema request with "draft": false gives the drafted reply's text
- sampled: the schema request at temperature 0.7 with a seed, twice: the same valid text
- tool-required: tool_choice "required" gives a call with arguments that parse and name the city
- plain: a request without a grammar keeps PLAIN_SHA (unset: this run's first plain reply, the later ones against it)
- --concurrent (PARALLEL 2): the plain request beside a schema request keeps its sha, the schema reply its solo text
<session>'s specs of 10-18 (<records>/reply-1017.md, reply-1018.md):
- nested (#1017): an array of objects with an enum parses and meets the schema
- resume (#1017): (a) a new schema request twice: the same text, the second from the cache; (b) a new text's plain
  chat, then its schema request, which resumes from the plain one's cache (`compare` holds it to the text the same
  request gives cold: --cold-first sends that request first, and skips (b), in another launch, as --no-drafts)
- tool-required-3, tool-named, tool-parallel, tool-single, tool-nested, tool-empty (#1018), each again with
  "draft": false (tool-*-draft-false: the same calls)
- --concurrent adds (#1017) passes-a (a 5,000-token plain chat, then 0.5 s later the short schema request) and passes-b
  (a schema request behind a 5,000-token text, then 0.5 s later the short plain one); --mixed4 (PARALLEL 4) adds mixed4
  (two schema and two plain requests 0.2 s apart). A plain prompt resumes from the cache: each 5,000-token prompt is
  new to the server when it is sent beside another, and its alone reference comes after it (`cached` in the line).
python3 x9_live.py compare A.jsonl B.jsonl: every greedy reply the two runs share has the same text, token_sha and
calls (the --no-drafts check, #1017), and a resumed (b) equals the other run's cold one; exit 1 on a difference.
"""

import concurrent.futures
import json
import os
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8095"
MODEL = os.environ.get("MODEL", "local-model")  # the served name: S5's runs used local-model, <fork>'s serve script's glm-tf
TOK = "/hub/models--nvidia--GLM-5.3-Flash-NVFP4/snapshots/423acf37583782c51c142d145aef733d72943d93/tokenizer.json"
PLAIN_SHA = os.environ.get("PLAIN_SHA")  # <fork>'s NVFP4 gave 09a15f0582e7 (<image>, TP=2 and TP=3)
CITY = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "population": {"type": "integer"}},
    "required": ["name", "population"],
    "additionalProperties": False,
}
CITIES = {
    "type": "object",
    "properties": {
        "cities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "country": {"type": "string"},
                               "size": {"type": "string", "enum": ["large", "medium", "small"]}},
                "required": ["name", "country", "size"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["cities"],
    "additionalProperties": False,
}
LEDGER = {
    "type": "object",
    "properties": {"entry": {"type": "integer"}, "boats": {"type": "integer"}},
    "required": ["entry", "boats"],
    "additionalProperties": False,
}


def tool(name, description, properties, required):
    return {"type": "function", "function": {"name": name, "description": description,
                                             "parameters": {"type": "object", "properties": properties, "required": required}}}


WEATHER = tool("get_weather", "Current weather for a city", {"city": {"type": "string"}}, ["city"])
TIME = tool("get_time", "Current local time in a city", {"city": {"type": "string"}}, ["city"])
TRIP = tool("plan_trip", "Plan a trip", {
    "travellers": {"type": "array", "items": {"type": "object", "required": ["name", "age"],
                                              "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}},
    "stops": {"type": "array", "items": {"type": "object", "required": ["city", "nights"],
                                         "properties": {"city": {"type": "string"}, "nights": {"type": "integer"}}}},
}, ["travellers", "stops"])
SERVER_TIME = {"type": "function", "function": {"name": "get_server_time", "description": "The server's clock",
                                                "parameters": {"type": "object", "properties": {}}}}


def post(path, body, timeout=1800):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def chat(text, thinking=False, schema=None, **more):
    body = {"model": MODEL, "messages": [{"role": "user", "content": text}], "max_tokens": 400 if not thinking else 3000,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": thinking}, **more}
    if schema is not None:
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema}}
    t = time.time()
    reply = post("/v1/chat/completions", body)
    reply["_wall"] = round(time.time() - t, 3)
    return reply


def content(reply):
    return reply["choices"][0]["message"].get("content") or ""


def sha(reply):
    return reply["tensorfold"]["token_sha"]


def cached(reply):
    return (reply.get("usage", {}).get("prompt_tokens_details") or {}).get("cached_tokens") or 0


def calls(reply):
    """A reply's tool calls as (name, arguments text) pairs."""
    return [(c["function"]["name"], c["function"]["arguments"]) for c in reply["choices"][0]["message"].get("tool_calls") or []]


def fits(schema, v):
    """The parts of JSON Schema these checks use: object (properties, required, additionalProperties false), array
    (items), string, integer, boolean and enum."""
    if "enum" in schema and v not in schema["enum"]:
        return False
    t = schema.get("type")
    if t == "object":
        props = schema.get("properties", {})
        return (isinstance(v, dict) and all(k in v for k in schema.get("required", []))
                and not (schema.get("additionalProperties") is False and set(v) - set(props))
                and all(fits(props[k], x) for k, x in v.items() if k in props))
    if t == "array":
        return isinstance(v, list) and all(fits(schema.get("items", {}), x) for x in v)
    if t == "string":
        return isinstance(v, str)
    if t == "integer":
        return isinstance(v, int) and not isinstance(v, bool)
    if t == "boolean":
        return isinstance(v, bool)
    return True


def parsed(text):
    try:
        return json.loads(text)
    except ValueError:
        return None


def city_ok(text):
    return fits(CITY, parsed(text))


def main(argv):
    if argv[1] == "compare":
        return compare(argv[2], argv[3])
    out = open(argv[1], "w", encoding="utf-8")  # noqa: SIM115 (one file for the run)
    failed = []
    seen = {}  # every greedy reply: text, token_sha, calls

    def record(name, ok, **facts):
        out.write(json.dumps({"check": name, "ok": ok, **facts}, ensure_ascii=False) + "\n")
        out.flush()
        print(f"{'PASS' if ok else 'FAIL'} {name}: {json.dumps(facts, ensure_ascii=False)[:400]}")
        if not ok:
            failed.append(name)

    def keep(name, reply):
        seen[name] = {"text": content(reply) if "message" in reply["choices"][0] else reply["choices"][0]["text"],
                      "sha": sha(reply), "calls": calls(reply) if "message" in reply["choices"][0] else []}
        return reply

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(TOK)

    def ledger(word, n):
        line = "Entry {}: the " + word + " carried {} boats past the old mill, and the keeper wrote the count in a ledger. "
        return tok.decode(tok.encode("".join(line.format(i, (i * 37) % 101) for i in range(260))).ids[:n])

    count_q = "\n\nHow many boats did entry 17 record? Answer as JSON with entry and boats."
    later = ledger("ferry", 3000) + count_q   # resume (b): new to the server until its plain chat or --cold-first
    if "--cold-first" in argv:
        cold = keep("resume-b-cold", chat(later, schema=LEDGER))
        record("resume-b-cold", fits(LEDGER, parsed(content(cold))), text=content(cold), cached=cached(cold), wall=cold["_wall"])

    ask = "What is the largest city in Japan and its population? Answer as JSON with name and population."
    a = keep("schema", chat(ask, schema=CITY))
    record("schema", city_ok(content(a)), text=content(a), sha=sha(a), wall=a["_wall"])
    t = keep("schema-thinking", chat(ask, thinking=True, schema=CITY))
    record("schema-thinking", city_ok(content(t)), text=content(t), reasoning=len(t["choices"][0]["message"].get("reasoning_content") or ""), wall=t["_wall"])

    say = 'Reply with exactly {"ok": true} and nothing else.'
    plain = keep("same-as-plain-plain", chat(say))
    shaped = keep("same-as-plain-shaped", chat(say, schema={"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}))
    record("same-as-plain", content(plain) == content(shaped), plain=content(plain), shaped=content(shaped),
           shas=[sha(plain), sha(shaped)])

    d = keep("draft-false", chat(ask, schema=CITY, draft=False))
    record("draft-false", content(d) == content(a), text=content(d), wall=d["_wall"])

    s1 = chat(ask, schema=CITY, temperature=0.7, seed=7)
    s2 = chat(ask, schema=CITY, temperature=0.7, seed=7)
    record("sampled", content(s1) == content(s2) and city_ok(content(s1)), texts=[content(s1), content(s2)])

    w = keep("tool-required", chat("What is the weather in Osaka?", tools=[WEATHER], tool_choice="required"))
    wc = calls(w)
    args = parsed(wc[0][1]) if wc else None
    record("tool-required", bool(wc) and wc[0][0] == "get_weather" and isinstance(args, dict) and "osaka" in str(args.get("city", "")).lower(),
           calls=wc, finish=w["choices"][0]["finish_reason"])

    body = {"model": MODEL, "prompt": "The capital of France is", "max_tokens": 8, "temperature": 0}
    p = keep("plain", post("/v1/completions", body))
    plain_sha = PLAIN_SHA or sha(p)
    record("plain", sha(p) == plain_sha, text=p["choices"][0]["text"], sha=sha(p), against=PLAIN_SHA or "this run's")

    many = "Name two or three large cities in Japan, each with its country and size. Answer as JSON."
    n = keep("nested", chat(many, schema=CITIES))
    v = parsed(content(n))
    record("nested", fits(CITIES, v), text=content(n), cities=len(v["cities"]) if fits(CITIES, v) else None, wall=n["_wall"])

    first_q = ledger("lake", 3000) + count_q
    m1 = keep("resume-a-1", chat(first_q, schema=LEDGER))
    m2 = keep("resume-a-2", chat(first_q, schema=LEDGER))
    prompt_tokens = m2["usage"]["prompt_tokens"]
    record("resume-a", content(m1) == content(m2) and fits(LEDGER, parsed(content(m1))) and cached(m2) * 2 > prompt_tokens,
           text=content(m1), cached=[cached(m1), cached(m2)], prompt_tokens=prompt_tokens, walls=[m1["_wall"], m2["_wall"]])
    if "--cold-first" not in argv:
        warm = chat(later, max_tokens=16)
        res = keep("resume-b-resumed", chat(later, schema=LEDGER))
        record("resume-b", fits(LEDGER, parsed(content(res))) and cached(res) * 2 > res["usage"]["prompt_tokens"],
               text=content(res), cached={"plain": cached(warm), "schema": cached(res)},
               prompt_tokens=res["usage"]["prompt_tokens"], against="compare: the cold one of a --cold-first run")

    three = [WEATHER, TIME, TRIP]
    named_choice = {"type": "function", "function": {"name": "get_time"}}
    trip_q = "Plan a trip for Aiko (34) and Ken (36): two nights in Kyoto, then one in Nara."
    both_q = "What is the weather in Osaka and in Sapporo?"
    cases = {
        "tool-required-3": ("What is the weather in Osaka?", three, "required", {}),
        "tool-named": ("What is the weather in Osaka?", three, named_choice, {}),
        "tool-parallel": (both_q, [WEATHER], "required", {}),
        "tool-single": (both_q, [WEATHER], "required", {"parallel_tool_calls": False}),
        "tool-nested": (trip_q, [TRIP], "required", {}),
        "tool-empty": ("What time is it on the server?", [SERVER_TIME], "required", {}),
    }
    params = {x["function"]["name"]: x["function"]["parameters"] for x in three + [SERVER_TIME]}

    def judge(name, cs):
        args = [parsed(x) for _, x in cs]
        valid = bool(cs) and all(c in params and fits(params[c], x) for (c, _), x in zip(cs, args))
        if name == "tool-required-3":
            return valid
        if name == "tool-named":
            return valid and [c for c, _ in cs] == ["get_time"] and "osaka" in str(args[0].get("city", "")).lower()
        if name == "tool-parallel":
            return (valid and len(cs) == 2 and all(c == "get_weather" for c, _ in cs)
                    and sorted(str(x.get("city", "")).lower() for x in args) == ["osaka", "sapporo"])
        if name == "tool-single":
            return valid and len(cs) == 1
        if name == "tool-nested":
            return valid and len(cs) == 1
        return valid and len(cs) == 1 and args[0] == {}

    for name, (text, tools, choice, more) in cases.items():
        r = keep(name, chat(text, tools=tools, tool_choice=choice, **more))
        cs = calls(r)
        record(name, judge(name, cs) and r["choices"][0]["finish_reason"] == "tool_calls", calls=cs,
               finish=r["choices"][0]["finish_reason"], text=content(r)[:200])
        rd = keep(name + "-draft-false", chat(text, tools=tools, tool_choice=choice, draft=False, **more))
        record(name + "-draft-false", calls(rd) == cs, calls=calls(rd))

    if "--concurrent" in argv:
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            shaped_f = pool.submit(chat, ask, False, CITY)
            time.sleep(0.5)
            plain_f = pool.submit(post, "/v1/completions", body)
            shaped_c, plain_c = shaped_f.result(), plain_f.result()
        record("concurrent", sha(plain_c) == plain_sha and content(shaped_c) == content(a),
               plain_sha=sha(plain_c), text=content(shaped_c))

        creek = ledger("creek", 5000)
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            long_f = pool.submit(chat, creek, max_tokens=8)
            time.sleep(0.5)
            short_f = pool.submit(chat, ask, False, CITY)
            long_c, short_c = long_f.result(), short_f.result()
        alone = chat(creek, max_tokens=8)
        record("passes-a", content(short_c) == content(a) and sha(long_c) == sha(alone), text=content(short_c),
               plain_sha={"beside": sha(long_c), "alone": sha(alone)}, cached={"beside": cached(long_c), "alone": cached(alone)})

        brook = ledger("brook", 5000) + "\n\n" + ask
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            long_f = pool.submit(chat, brook, False, CITY)
            time.sleep(0.5)
            short_f = pool.submit(post, "/v1/completions", body)
            long_c, short_c = long_f.result(), short_f.result()
        alone = chat(brook, schema=CITY)
        record("passes-b", sha(short_c) == plain_sha and content(long_c) == content(alone) and city_ok(content(long_c)),
               plain_sha=sha(short_c), texts={"beside": content(long_c), "alone": content(alone)},
               cached={"beside": cached(long_c), "alone": cached(alone)})

    if "--mixed4" in argv:
        delta = ledger("delta", 5000)
        sends = [lambda: chat(ask, schema=CITY), lambda: chat(many, schema=CITIES),
                 lambda: post("/v1/completions", body), lambda: chat(delta, max_tokens=8)]
        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            futures = []
            for send in sends:
                futures.append(pool.submit(send))
                time.sleep(0.2)
            got = [f.result() for f in futures]
        alone = chat(delta, max_tokens=8)
        same = [content(got[0]) == content(a), content(got[1]) == content(n), sha(got[2]) == plain_sha, sha(got[3]) == sha(alone)]
        record("mixed4", all(same), same=same, texts=[content(got[0]), content(got[1])], shas=[sha(got[2]), sha(got[3]), sha(alone)],
               cached={"beside": cached(got[3]), "alone": cached(alone)})

    out.write(json.dumps({"check": "_greedy", "replies": seen}, ensure_ascii=False) + "\n")
    print("failed: " + ", ".join(failed) if failed else "all passed")
    return 1 if failed else 0


def compare(path_a, path_b):
    def greedy(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row.get("check") == "_greedy":
                    return row["replies"]
        return {}

    a, b = greedy(path_a), greedy(path_b)
    differ = []
    for name in sorted(set(a) & set(b)):
        same = a[name] == b[name]
        print(f"{'SAME' if same else 'DIFF'} {name}: {json.dumps(a[name], ensure_ascii=False)[:160]}"
              + ("" if same else f" | {json.dumps(b[name], ensure_ascii=False)[:160]}"))
        if not same:
            differ.append(name)
    for x, y in ((a, b), (b, a)):
        if "resume-b-resumed" in x and "resume-b-cold" in y:
            same = x["resume-b-resumed"]["text"] == y["resume-b-cold"]["text"]
            print(f"{'SAME' if same else 'DIFF'} resume-b (resumed against cold): {x['resume-b-resumed']['text']} | {y['resume-b-cold']['text']}")
            if not same:
                differ.append("resume-b")
    print(f"{len(set(a) & set(b))} shared replies; " + ("differ: " + ", ".join(differ) if differ else "all the same"))
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
