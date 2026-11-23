"""Cold against resumed first token on a continued chat: a first turn, a short second turn, a long third turn, and the third
turn's whole prompt again cold (a different first line, so no kept state matches)."""
import json, os, sys, time, urllib.request
base, name = sys.argv[1], sys.argv[2]
DOC = open(os.environ["NOTES"]).read() * 2
def ask(messages, tokens=8):
    body = {"model": "m", "messages": messages, "max_tokens": tokens, "temperature": 0}
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    raw = json.loads(urllib.request.urlopen(req, timeout=3600).read())
    tf = raw.get("tensorfold", {})
    rec = {"prompt": raw["usage"]["prompt_tokens"], "cached": (raw["usage"].get("prompt_tokens_details") or {}).get("cached_tokens"), "ttft": tf.get("time_to_first_token"), "wall": round(time.time() - t0, 3)}
    return rec, raw["choices"][0]["message"]["content"]
for rep in range(3):
    off = 2000 * rep + 500
    first = "Read these notes and answer in one sentence what they cover.\n\n" + DOC[off:off + 8800]
    m = [{"role": "user", "content": first}]
    a, text = ask(m)
    m += [{"role": "assistant", "content": text}, {"role": "user", "content": "Which kernels or tests do the notes name? List three."}]
    b, text = ask(m)
    m += [{"role": "assistant", "content": text}, {"role": "user", "content": "Now read this further passage and say how it relates.\n\n" + DOC[off + 8800:off + 12600]}]
    c, _ = ask(m)
    z = [dict(x) for x in m]
    z[0]["content"] = "Z%d. " % rep + z[0]["content"]
    d, _ = ask(z)
    print(json.dumps({"server": name, "rep": rep, "first_cold": a, "second_resumed": b, "third_resumed": c, "third_cold": d}))
