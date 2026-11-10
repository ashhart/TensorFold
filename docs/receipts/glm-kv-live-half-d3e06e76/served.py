"""Shared by the served runners (lazy_receipt.py, recall_ab.py) and receipt.py: the Zig server's HTTP routes as
docs/api.md describes them, and the server log's KVCODEC, prompt cache and done lines. Standard library only.
"""
import hashlib, json, re, time, urllib.error, urllib.request


def post(base, path, body, timeout=1800):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def get(base, path, timeout=60):
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        return json.load(r)


def model_id(base):
    return get(base, "/v1/models")["data"][0]["id"]


class Server:
    """The routes a runner uses: tokenize / detokenize (vLLM's, docs/api.md) and chat completions."""

    def __init__(self, base):
        self.base = base.rstrip("/")
        self.model = None

    def ids(self, text):
        return post(self.base, "/tokenize", {"prompt": text, "add_special_tokens": False})["tokens"]

    def text(self, ids):
        return post(self.base, "/detokenize", {"tokens": [int(i) for i in ids]})["prompt"]

    def render(self, messages):
        return post(self.base, "/tokenize", {"messages": messages, "add_generation_prompt": True})["tokens"]

    def name(self):
        if self.model is None:
            self.model = model_id(self.base)
        return self.model

    def chat(self, body, stream):
        body = dict(body, model=self.name())
        return stream_chat(self.base, body) if stream else whole_chat(self.base, body)


def chat_body(messages, max_tokens):
    """A greedy chat body that returns the reply's token ids (tensorfold.token_ids, token_sha256)."""
    return {"messages": messages, "max_tokens": max_tokens, "temperature": 0, "return_token_ids": True}


def reply_fields(out, obj):
    """The fields a runner records, from a whole reply or a stream's finish event (`obj`)."""
    tf = obj.get("tensorfold") or {}
    spec = obj.get("speculative") or {}
    use = obj.get("usage") or {}
    if tf:
        out["token_sha256"] = tf.get("token_sha256")
        out["token_ids"] = tf.get("token_ids")
        out["server_ttft_s"] = tf.get("time_to_first_token")
        out["decode_tps"] = tf.get("tokens_per_second")
        out["decode_s"] = tf.get("seconds")
        out["prefill_s"] = tf.get("prefill_seconds")
    src = spec or tf  # the draft counters: `speculative` beside the block (openai.zig), or inside it (docs/api.md)
    for k in ("rounds", "drafted", "accepted"):
        if k in src:
            out[k] = src[k]
    if use:
        out["prompt_tokens"] = use.get("prompt_tokens")
        out["completion_tokens"] = use.get("completion_tokens")
        out["cached_tokens"] = (use.get("prompt_tokens_details") or {}).get("cached_tokens")


def whole_chat(base, body):
    t0 = time.time()
    r = post(base, "/v1/chat/completions", dict(body, stream=False))
    msg = r["choices"][0]["message"]
    out = {"total_s": time.time() - t0, "content": msg.get("content") or "", "reasoning": msg.get("reasoning_content") or ""}
    reply_fields(out, r)
    return finish(out)


def stream_chat(base, body):
    """Streamed: the first chunk carrying text (reasoning or content) stamps the time to first token."""
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(dict(body, stream=True)).encode(),
                                 headers={"content-type": "application/json"})
    t0 = time.time()
    out = {"ttft_s": None, "content": "", "reasoning": ""}
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            for ch in obj.get("choices") or []:
                d = ch.get("delta") or {}
                piece = (d.get("content") or "") + (d.get("reasoning_content") or "")
                if piece and out["ttft_s"] is None:
                    out["ttft_s"] = time.time() - t0
                out["content"] += d.get("content") or ""
                out["reasoning"] += d.get("reasoning_content") or ""
            reply_fields(out, obj)
    out["total_s"] = time.time() - t0
    return finish(out)


def finish(out):
    if not out.get("token_sha256") and out.get("token_ids") is not None:
        out["token_sha256"] = token_sha256(out["token_ids"])
    out["text_sha256"] = hashlib.sha256(out["content"].encode()).hexdigest()
    return out


def token_sha256(ids):
    """zig/src/server/tokens.zig tokenSha256: the decimal ids joined by commas (a runner records the server's)."""
    return hashlib.sha256(",".join(str(int(i)) for i in ids).encode()).hexdigest()


# ---- the server log ------------------------------------------------------------------------------------------------
KV = re.compile(r"(\w+)=([-\w.]+)")
TOTALS = re.compile(r"KVCODEC rank=(\d+) mode=(\w+) keeps=")
COMPACTED = re.compile(r"KVCODEC rank=(\d+) compacted keep=(\d+) raw=(\d+) stored=(\d+) pages (\d+) -> (\d+) after_ms=([\d.]+)")
TRIMMED = re.compile(r"KVCODEC rank=(\d+) trimmed keep=(\d+) pages (\d+) -> (\d+) free=(\d+)")
START = re.compile(r"KVCODEC rank=(\d+) mode=(\w+) delay_ms=(\d+) threads=(\d+)(.*)")
CACHE = re.compile(r"prompt cache: (\d+) tokens, resumed at (\d+), kept (\d+); (\d+) states, (\d+) MiB and (\d+) MiB spare of (\d+) MiB \(hits (\d+), misses (\d+), evicted (\d+), refused (\d+), failed (\d+)\)")
DONE = re.compile(r"\] done (\S+) (.*)")


def num(v):
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


def totals(line):
    """A KVCODEC totals line as a dict (every key=value, `pages a -> b` as raw_pages and kept_pages)."""
    d = {k: num(v) for k, v in KV.findall(line[line.index("KVCODEC"):])}
    m = re.search(r"pages (\d+) -> (\d+)", line)
    if m:
        d["raw_pages"], d["kept_pages"] = int(m.group(1)), int(m.group(2))
    return d


def parse_log(lines):
    """The lines a receipt reads, in log order (each with its line number `at`)."""
    out = {"start": [], "totals": [], "compacted": [], "trimmed": [], "cache": [], "done": [], "refused": []}
    for at, line in enumerate(lines):
        if "KVCODEC refused" in line or "BadSnapshotControl" in line:
            out["refused"].append(line.strip())
        if (m := COMPACTED.search(line)):
            g = m.groups()
            out["compacted"].append({"at": at, "rank": int(g[0]), "keep": int(g[1]), "raw": int(g[2]), "stored": int(g[3]),
                                     "pages": [int(g[4]), int(g[5])], "after_ms": float(g[6])})
        elif (m := TRIMMED.search(line)):
            g = [int(x) for x in m.groups()]
            out["trimmed"].append({"at": at, "rank": g[0], "keep": g[1], "pages": [g[2], g[3]], "free": g[4]})
        elif TOTALS.search(line):
            out["totals"].append(dict(totals(line), at=at))
        elif (m := START.search(line)):
            out["start"].append({"at": at, "line": line[line.index("KVCODEC"):].strip()})
        elif (m := CACHE.search(line)):
            g = [int(x) for x in m.groups()]
            keys = ("prompt", "resumed_at", "kept", "states", "held_mib", "spare_mib", "budget_mib", "hits", "misses", "evicted", "refused", "failed")
            out["cache"].append(dict(zip(keys, g), at=at))
        elif (m := DONE.search(line)):
            d = {k: num(v) for k, v in KV.findall(m.group(2))}
            acc = re.search(r"accepted=(\d+)/(\d+)", m.group(2))
            if acc:
                d["accepted"], d["drafted"] = int(acc.group(1)), int(acc.group(2))
            out["done"].append(dict(d, id=m.group(1), at=at))
    return out


def read_lines(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.readlines()


class LogTail:
    """New lines of a growing log since the last call (None path: nothing)."""

    def __init__(self, path):
        self.path, self.pos = path, 0
        if path:
            try:
                with open(path, "rb") as f:
                    f.seek(0, 2)
                    self.pos = f.tell()
            except OSError:
                self.pos = 0

    def new(self):
        if not self.path:
            return []
        try:
            with open(self.path, "rb") as f:
                f.seek(self.pos)
                data = f.read()
        except OSError:
            return []
        cut = data.rfind(b"\n") + 1
        self.pos += cut
        return data[:cut].decode("utf-8", "replace").splitlines(True)
