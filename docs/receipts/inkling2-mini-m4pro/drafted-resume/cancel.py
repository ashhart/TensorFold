"""A prompt pass cancelled mid-way keeps its finished prefix; the same request then resumes it: same tokens as the plain server's."""
import hashlib, os, http.client, json, sys, threading, time, urllib.parse
base, name = sys.argv[1], sys.argv[2]
DOC = open(os.environ["NOTES"]).read() * 2
host = urllib.parse.urlparse(base)
body = json.dumps({"model": "m", "messages": [{"role": "user", "content": "Read and summarize.\n" + DOC[3000:19000]}], "max_tokens": 24, "temperature": 0, "return_token_ids": True}).encode()
def conn():
    return http.client.HTTPConnection(host.hostname, host.port, timeout=3600)
c = conn()
c.request("POST", host.path + "/chat/completions", body, {"Content-Type": "application/json"})
time.sleep(float(sys.argv[3]) if len(sys.argv) > 3 else 4.0)
c.close()  # the client leaves during the prompt pass
time.sleep(15)
out = []
for i in range(2):
    c = conn()
    c.request("POST", host.path + "/chat/completions", body, {"Content-Type": "application/json"})
    raw = json.loads(c.getresponse().read())
    tf = raw.get("tensorfold", {})
    out.append({"server": name, "run": i, "prompt": raw["usage"]["prompt_tokens"], "cached": (raw["usage"].get("prompt_tokens_details") or {}).get("cached_tokens"), "ids_sha": hashlib.sha256(json.dumps(tf.get("token_ids")).encode()).hexdigest()[:12], "text": raw["choices"][0]["message"]["content"][:60], "ttft": tf.get("time_to_first_token"), "spec": raw.get("speculative")})
    print(json.dumps(out[-1]))
