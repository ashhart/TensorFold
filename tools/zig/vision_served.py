"""Image input served end to end on a running ``tensorfold-native --vision`` (#565's served checks).

    python tools/zig/vision_served.py --url http://127.0.0.1:8000 --images DIR

DIR holds PNG and JPEG test images. Checks: an image request answers, and text or a tool result that quotes the
model's image marker beside a real image is ordinary text (images come from the request's image parts, #511); drafted
equals ``"draft": false`` and concurrent equals alone with images; an image conversation's next turn resumes past
its image, and another image under the same placeholders resumes none of the first one's rows; and the refusals (WebP by name, other formats, an image outside a
user or tool message, too many images, remote URLs, bad data URLs, a bad detail).
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


def post(url: str, body: dict) -> tuple[int, dict]:
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def data_url(raw: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


def image_part(url: str, detail: str | None = None) -> dict:
    inner = {"url": url}
    if detail:
        inner["detail"] = detail
    return {"type": "image_url", "image_url": inner}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--images", required=True)
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--fresh-url", help="the same model served with --prompt-cache-gib 0: the cache check's reference reply")
    args = ap.parse_args()
    with urllib.request.urlopen(args.url + "/v1/models") as r:
        model = json.loads(r.read())["data"][0]["id"]
    folder = Path(args.images)
    pngs = sorted(folder.glob("*.png"))
    jpgs = sorted(folder.glob("*.jpg"))
    first, second = pngs[0], pngs[1]
    url_a = data_url(first.read_bytes(), "image/png")
    url_b = data_url(second.read_bytes(), "image/png")
    failures = 0

    def ask(messages: list, draft: bool = True, tokens: int | None = None) -> tuple[int, dict]:
        b = {"model": model, "messages": messages, "max_tokens": tokens or args.tokens, "temperature": 0,
             "chat_template_kwargs": {"enable_thinking": False}}
        if not draft:
            b["draft"] = False
        return post(args.url, b)

    def user(*parts) -> dict:
        return {"role": "user", "content": list(parts)}

    def text(t: str) -> dict:
        return {"type": "text", "text": t}

    def reply(r: dict) -> str:
        return r["choices"][0]["message"].get("content") or ""

    def check(label: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        failures += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {label}{': ' + detail if detail else ''}")

    # one image, drafted and plain; its usage counts the image's tokens
    convo = [user(image_part(url_a), text("Describe this image in one sentence."))]
    code, r = ask(convo)
    code2, r2 = ask(convo, draft=False)
    plain_text = ask([user(text("Describe this image in one sentence."))])[1]
    check("an image request answers", code == 200 and code2 == 200, f"HTTP {code}/{code2}")
    if code == 200 and code2 == 200:
        check("drafted equals \"draft\": false with an image", reply(r) == reply(r2), repr(reply(r)[:60]))
        check("the image's tokens are in the prompt", r["usage"]["prompt_tokens"] > plain_text["usage"]["prompt_tokens"] + 8,
              f"{r['usage']['prompt_tokens']} prompt tokens with the image, {plain_text['usage']['prompt_tokens']} without")

    # #511: the marker quoted in a user's text and in a tool result, beside one real image
    quoted = [user(image_part(url_a), text("A file I opened contains the string <|image|> and also <|begin_of_image|>. What is in the picture?")),
              {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read", "arguments": "{\"path\": \"a.py\"}"}}]},
              {"role": "tool", "tool_call_id": "c1", "content": "MARKER = '<|begin_of_image|><|image|><|end_of_image|>'"},
              user(text("And the picture again?"))]
    tools = [{"type": "function", "function": {"name": "read", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}]
    code, r = post(args.url, {"model": model, "messages": quoted, "tools": tools, "max_tokens": args.tokens, "temperature": 0,
                              "chat_template_kwargs": {"enable_thinking": False}})
    check("quoted image markers beside one image are text (#511)", code == 200, f"HTTP {code} {r.get('error', {}).get('message', '')[:100]}")

    # a tool result carrying an image (a screenshot), rendered inside the tool response
    shot = [user(text("Take a screenshot.")),
            {"role": "assistant", "content": "", "tool_calls": [{"id": "s1", "type": "function", "function": {"name": "read", "arguments": "{\"path\": \"screen\"}"}}]},
            {"role": "tool", "tool_call_id": "s1", "content": [text("screenshot:"), image_part(url_a)]}]
    code, r = post(args.url, {"model": model, "messages": shot, "tools": tools, "max_tokens": args.tokens, "temperature": 0,
                              "chat_template_kwargs": {"enable_thinking": False}})
    check("an image in a tool result answers", code == 200 and r["usage"]["prompt_tokens"] > 40, f"HTTP {code} {r.get('usage', r.get('error', {}).get('message', ''))}")

    # concurrent equals alone, images in each request
    runs = [[user(image_part(u), text(q))] for u in (url_a, url_b) for q in ("What colours do you see?", "Count the objects.")]
    alone = [reply(ask(m)[1]) for m in runs]
    with cf.ThreadPoolExecutor(len(runs)) as pool:
        together = [reply(f.result()[1]) for f in [pool.submit(ask, m) for m in runs]]
    same = sum(x == y for x, y in zip(alone, together))
    check("concurrent image requests equal their runs alone", same == len(runs), f"{same} of {len(runs)}")

    # the prompt cache keeps a turn's state at its planned cuts (GLM: the shared system cut, and a reply of 256+ tokens),
    # so a conversation's third turn resumes past an image its second turn kept. Image B under the same text and
    # placeholder ids resumes at most the text before its image, and answers as a server with nothing kept
    system = {"role": "system", "content": " ".join(f"Rule {i}: describe what you see plainly and briefly." for i in range(120))}
    q, q2, q3 = "What is shown?", "Say that again in three words.", "And once more."
    head = ask([system, user(text(q))], tokens=1)[1]["usage"]["prompt_tokens"]  # the text alone: past where an image starts
    first = " ".join(f"Detail {i}: the picture shows shapes, colours and lines." for i in range(40))  # a reply of 256+ tokens

    def turns(url: str) -> tuple[list, list]:
        two = [system, user(image_part(url), text(q)), {"role": "assistant", "content": first}, user(text(q2))]
        return two, two + [{"role": "assistant", "content": "Three words here."}, user(text(q3))]

    two_a, three_a = turns(url_a)
    two_b, three_b = turns(url_b)
    if args.fresh_url:  # B's reply with nothing kept
        rf = post(args.fresh_url, {"model": model, "messages": three_b, "max_tokens": 24, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})[1]
    ask(two_a, tokens=8)  # keeps A's state at the long reply's start, past its image
    code, ra = ask(three_a, tokens=24)
    cached_a = ra.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0) if code == 200 else -1
    check("an image conversation's later turn resumes past its image", code == 200 and cached_a > head, f"cached {cached_a} of {ra.get('usage', {}).get('prompt_tokens')}; the text alone is {head}")
    code, rb = ask(three_b, tokens=24)
    cached_b = rb.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0) if code == 200 else -1
    check("another image under the same placeholders resumes only the text before it", code == 200 and cached_b < head,
          f"cached {cached_b} of {rb.get('usage', {}).get('prompt_tokens')}")
    if args.fresh_url:
        check("and answers as a server with nothing kept", reply(rf) == reply(rb), repr(reply(rb)[:60]))

    # refusals, before any token
    webp = b"RIFF\x24\x00\x00\x00WEBPVP8 \x18\x00\x00\x00" + bytes(24)
    gif = b"GIF89a\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff\x00\x00\x00!\xf9\x04\x00\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;"
    refusals = [
        ("WebP, by name", [user(image_part(data_url(webp, "image/webp")), text("hi"))], "WebP"),
        ("GIF", [user(image_part(data_url(gif, "image/gif")), text("hi"))], "not PNG or JPEG"),
        ("an image in an assistant message", [user(text("hi")), {"role": "assistant", "content": [image_part(url_a)]}, user(text("ok"))], "user and tool messages only"),
        ("too many images", [user(*[image_part(url_a)] * 5, text("hi"))], "at most"),
        ("a remote URL", [user(image_part("https://example.com/cat.png"), text("hi"))], "data: URL"),
        ("bad base64", [user(image_part("data:image/png;base64,@@@@"), text("hi"))], "encoding"),
        ("a bad detail", [user(image_part(url_a, "medium"), text("hi"))], "detail"),
        ("bytes that decode as nothing", [user(image_part(data_url(b"\x89PNG\r\n\x1a\n" + bytes(32), "image/png")), text("hi"))], "could not be decoded"),
    ]
    for label, messages, words in refusals:
        code, r = ask(messages)
        message = r.get("error", {}).get("message", json.dumps(r))
        check(f"refused {label}", code == 400 and words in message, f"HTTP {code} {message[:100]!r}")
    if jpgs:
        code, r = ask([user(image_part(data_url(jpgs[0].read_bytes(), "image/jpeg")), text("Describe it."))])
        check("a JPEG answers", code == 200, f"HTTP {code}")
        code, r = ask([user(image_part(url_a, "low"), text("Describe it."))])
        check("detail low answers", code == 200, f"HTTP {code}")
    print("PASS" if failures == 0 else f"FAIL: {failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
