"""Image URLs served end to end on a running ``tensorfold-native --vision --vision-urls`` (#565's URL checks).

    python tools/zig/vision_urls_served.py --url http://127.0.0.1:8000

Checks, against public hosts: an HTTPS image URL answers exactly as the same bytes sent as a data URL (the prompt's
tokens and the greedy reply), and so does a URL that redirects to it; a URL and a data URL in one request; and the
refusals images_http.py gave, before any token (HTTP status, content type, a host that resolves to a private
address, a name that is never public, an address literal that isn't public, TLS that doesn't verify, a scheme, port,
credentials or fragment the URL rules refuse, an unresolvable host). The fetch's own rules (redirect limits,
lengths, chunking, encodings, deadlines) are the unit tests in zig/src/server/media_fetch.zig.
"""

from __future__ import annotations

import argparse
import base64
import json
import ssl
import sys
import urllib.error
import urllib.request

IMAGE = "https://raw.githubusercontent.com/ashhart/TensorFold/main/assets/tensorfold-hero.png"
REDIRECT = "https://github.com/ashhart/TensorFold/raw/main/assets/tensorfold-hero.png"  # 302 to IMAGE's host
OTHER = "https://raw.githubusercontent.com/ashhart/TensorFold/main/assets/tensorfold-cuda-banner.png"


def post(url: str, body: dict) -> tuple[int, dict]:
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--tokens", type=int, default=48)
    args = ap.parse_args()
    with urllib.request.urlopen(args.url + "/v1/models") as r:
        model = json.loads(r.read())["data"][0]["id"]
    try:  # python.org's macOS builds ship no CA store until "Install Certificates" runs
        import certifi
        tls = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        tls = ssl.create_default_context()
    with urllib.request.urlopen(IMAGE, timeout=30, context=tls) as r:
        inline = "data:image/png;base64," + base64.b64encode(r.read()).decode()
    failures = 0

    def ask(*parts) -> tuple[int, dict]:
        return post(args.url, {"model": model, "messages": [{"role": "user", "content": list(parts)}], "max_tokens": args.tokens,
                               "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})

    def image(url: str) -> dict:
        return {"type": "image_url", "image_url": {"url": url}}

    def text(t: str) -> dict:
        return {"type": "text", "text": t}

    def check(label: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        failures += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {label}{': ' + detail if detail else ''}")

    def same(label: str, a: tuple[int, dict], b: tuple[int, dict]) -> None:
        ok = a[0] == 200 and b[0] == 200 and a[1]["usage"]["prompt_tokens"] == b[1]["usage"]["prompt_tokens"] and \
            a[1]["choices"][0]["message"]["content"] == b[1]["choices"][0]["message"]["content"]
        detail = f"HTTP {a[0]}/{b[0]}" if a[0] != 200 or b[0] != 200 else \
            f"{a[1]['usage']['prompt_tokens']} prompt tokens, {a[1]['choices'][0]['message']['content'][:60]!r}"
        if a[0] != 200 or b[0] != 200:
            detail += " " + json.dumps(a[1].get("error") or b[1].get("error"))[:120]
        check(label, ok, detail)

    q = text("Describe this image in one sentence.")
    by_data = ask(image(inline), q)
    same("an HTTPS image URL answers as its bytes in a data URL", ask(image(IMAGE), q), by_data)
    same("a URL that redirects to the image answers the same", ask(image(REDIRECT), q), by_data)
    two = ask(image(OTHER), image(inline), text("How do these two images differ?"))
    check("a URL and a data URL in one request answer", two[0] == 200, f"HTTP {two[0]} {json.dumps(two[1].get('error', ''))[:100]}")

    refusals = [
        ("an HTTP error", "https://raw.githubusercontent.com/ashhart/TensorFold/main/assets/no-such-image.png", "image download returned HTTP 404"),
        ("a page that isn't an image", "https://example.com/", "image URL content type must be JPEG, PNG or WebP"),
        ("a host that resolves to a private address", "https://localtest.me/x.png", "image URLs must resolve only to public internet addresses"),
        ("a private address literal", "https://169.254.169.254/latest/meta-data", "image URLs must resolve only to public internet addresses"),
        ("localhost", "https://localhost/x.png", "image URLs must use public internet hosts"),
        ("cloud metadata by name", "https://metadata.google.internal/x.png", "image URLs must use public internet hosts"),
        ("an expired certificate", "https://expired.badssl.com/", "image download failed or timed out"),
        ("a certificate for another host", "https://wrong.host.badssl.com/", "image download failed or timed out"),
        ("a self-signed certificate", "https://self-signed.badssl.com/", "image download failed or timed out"),
        ("plain HTTP", "http://example.com/x.png", "images require data URLs or public HTTPS URLs"),
        ("another port", "https://example.com:8443/x.png", "image URL must be HTTPS on port 443, without credentials or a fragment"),
        ("credentials", "https://user:pw@example.com/x.png", "image URL must be HTTPS on port 443, without credentials or a fragment"),
        ("a fragment", "https://example.com/x.png#a", "image URL must be HTTPS on port 443, without credentials or a fragment"),
        ("a host that never resolves", "https://no-such-host.invalid/x.png", "image host could not be resolved"),
    ]
    for label, url, words in refusals:
        code, r = ask(image(url), text("hi"))
        message = r.get("error", {}).get("message", json.dumps(r))
        check(f"refused {label}", code == 400 and message == words, f"HTTP {code} {message[:100]!r}")
    print("PASS" if failures == 0 else f"FAIL: {failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
