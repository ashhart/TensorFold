"""A request cancelled in its prompt pass frees the server at once; usage: serve_cancel.py URL MODEL [PROMPT_WORDS]."""

import http.client
import json
import socket
import sys
import time
import urllib.parse

URL, MODEL = sys.argv[1], sys.argv[2]
WORDS = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
HOST = urllib.parse.urlparse(URL)


def body(text, max_tokens):
    return json.dumps({"model": MODEL, "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens,
                       "temperature": 0.0, "stream": False, "chat_template_kwargs": {"enable_thinking": False}})


def short():
    conn = http.client.HTTPConnection(HOST.hostname, HOST.port, timeout=600)
    began = time.perf_counter()
    conn.request("POST", "/v1/chat/completions", body("Count from one to ten in words.", 32), {"Content-Type": "application/json"})
    text = json.load(conn.getresponse())["choices"][0]["message"]["content"]
    conn.close()
    return text, time.perf_counter() - began


def long_then_drop(nonce):
    text = f"{nonce} " + " ".join(f"item{i % 97}x{(i * 7919) % 1013}" for i in range(WORDS))
    sock = socket.create_connection((HOST.hostname, HOST.port))
    payload = body(text, 8).encode()
    sock.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: "
                 + str(len(payload)).encode() + b"\r\n\r\n" + payload)
    time.sleep(0.4)
    sock.close()


before, _ = short()
alone = min(short()[1] for _ in range(2))
failed = 0
for i in range(3):
    long_then_drop(f"cancel{i}-{time.time_ns()}")
    time.sleep(0.05)
    after, waited = short()
    ok = after == before and waited < alone + 3.0
    failed += not ok
    print(("ok   " if ok else "FAIL ") + f"cancel {i}: the next short request took {waited:.2f} s (alone {alone:.2f} s)", flush=True)
print("failed", failed)
sys.exit(1 if failed else 0)
