"""tools/zig/video_served.py's checks on a running ``tensorfold-native --vision``, with the processor's prompt ids
read from a file (video_ref.py writes it on a machine with transformers 5.19 and PyAV 19.0.1; <redacted-container> has neither).

usage: video_http.py URL CLIPS REF.json [IMAGE.png] OUT.json
Each check prints "ok  " or "FAIL" as video_served.py does; the replies go into OUT.json; the last line is PASS or
FAIL: N, then "exit 0|1".
"""

from __future__ import annotations

import base64
import concurrent.futures as cf
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


def post(url: str, path: str, body: dict) -> tuple[int, dict]:
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def data_url(raw: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


MIME = {".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm", ".mkv": "video/x-matroska"}
TOKENS = 32


def main() -> int:
    url, clips, ref_path = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    image = sys.argv[4] if len(sys.argv) > 5 else None
    out_path = Path(sys.argv[-1])
    refs = json.loads(ref_path.read_text())
    with urllib.request.urlopen(url + "/v1/models") as r:
        model = json.loads(r.read())["data"][0]["id"]
    failures = 0
    log: dict = {"ref": ref_path.name, "checks": [], "replies": {}}

    def check(label: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        failures += not ok
        log["checks"].append({"label": label, "ok": ok, "detail": detail})
        print(f"{'ok  ' if ok else 'FAIL'} {label}{': ' + detail if detail else ''}", flush=True)

    def video(p: Path) -> dict:
        return {"type": "video_url", "video_url": {"url": data_url(p.read_bytes(), MIME[p.suffix])}}

    def text(t: str) -> dict:
        return {"type": "text", "text": t}

    def user(*parts) -> dict:
        return {"role": "user", "content": list(parts)}

    off = {"enable_thinking": False}

    def ask(messages: list, draft: bool = True, tokens: int | None = None) -> tuple[int, dict]:
        b = {"model": model, "messages": messages, "max_tokens": tokens or TOKENS, "temperature": 0, "chat_template_kwargs": off}
        if not draft:
            b["draft"] = False
        return post(url, "/v1/chat/completions", b)

    def reply(r: dict) -> str:
        return r["choices"][0]["message"].get("content") or ""

    # the prompt ids per clip against the processor's (thinking on: no template switch; off, the server adds </think>)
    q = refs["question"]
    same = 0
    clip_files = sorted(p for p in clips.iterdir() if p.suffix in MIME and p.name != "short.mp4")
    assert sorted(p.name for p in clip_files) == sorted(refs["ids"]), "the clips differ from the reference's"
    for p in clip_files:
        ref = refs["ids"][p.name]
        code, r = post(url, "/tokenize", {"messages": [user(video(p), text(q))]})
        ok = code == 200 and r.get("tokens") == ref
        same += ok
        if not ok:
            got = r.get("tokens", [])
            at = next((i for i, (x, y) in enumerate(zip(got, ref)) if x != y), min(len(got), len(ref)))
            print(f"     {p.name}: HTTP {code} {len(got)} ids against {len(ref)}, first difference at {at}: {got[at:at + 6]} vs {ref[at:at + 6]} {r.get('error', '')}")
        else:
            print(f"     {p.name}: {len(ref)} ids equal")
    check("a video request's prompt ids equal the processor's", same == len(clip_files), f"{same} of {len(clip_files)} clips")

    a, b = clips / "h264.mp4", clips / "hevc.mp4"
    # the cache, before any concurrent run (a slot another stream takes leaves GLM's resident state stale)
    system = {"role": "system", "content": " ".join(f"Rule {i}: describe what you see plainly and briefly." for i in range(120))}
    head = ask([system, user(text("What is shown?"))], tokens=1)[1]["usage"]["prompt_tokens"]
    long = " ".join(f"Detail {i}: the clip shows shapes, colours and motion." for i in range(40))

    def turns(c: Path) -> tuple[list, list]:
        two = [system, user(video(c), text("What is shown?")), {"role": "assistant", "content": long}, user(text("Again, in three words."))]
        return two, two + [{"role": "assistant", "content": "Three words here."}, user(text("Once more."))]

    two_a, three_a = turns(a)
    _, three_b = turns(clips / "h264-odd.mp4")
    ask(two_a, tokens=8)
    code, ra = ask(three_a, tokens=16)
    cached_a = ra.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0) if code == 200 else -1
    check("a video conversation's later turn resumes past its video", code == 200 and cached_a > head, f"cached {cached_a} of {ra.get('usage', {}).get('prompt_tokens')}; the text alone is {head}")
    code, rb = ask(three_b, tokens=16)
    cached_b = rb.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0) if code == 200 else -1
    check("another video under the same text resumes only the text before it", code == 200 and cached_b < head, f"cached {cached_b} of {rb.get('usage', {}).get('prompt_tokens')}")

    convo = [user(video(a), text("Describe the video in one sentence."))]
    code, r = ask(convo)
    code2, r2 = ask(convo, draft=False)
    check("a video request answers", code == 200 and code2 == 200, f"HTTP {code}/{code2} {r.get('error', '')}")
    if code == 200 and code2 == 200:
        log["replies"]["describe"] = reply(r)
        check("drafted equals \"draft\": false with a video", reply(r) == reply(r2), repr(reply(r)[:60]))
        code3, r3 = ask(convo)
        check("a video request repeated answers the same", code3 == 200 and reply(r3) == reply(r))
    runs = [[user(video(c), text(t))] for c in (a, b) for t in ("What colours appear?", "Does anything move?")]
    alone = [reply(ask(m)[1]) for m in runs]
    with cf.ThreadPoolExecutor(len(runs)) as pool:
        together = [reply(f.result()[1]) for f in [pool.submit(ask, m) for m in runs]]
    log["replies"]["alone"] = alone
    check("concurrent video requests equal their runs alone", alone == together, f"{sum(x == y for x, y in zip(alone, together))} of {len(runs)}")
    if image:
        img = {"type": "image_url", "image_url": {"url": data_url(Path(image).read_bytes(), "image/png")}}
        code, r = ask([user(img, video(a), text("Does the picture appear in the video?"))])
        check("a video and an image in one request answer", code == 200, f"HTTP {code} {r.get('error', '')}")

    refusals = [
        ("three videos", [user(video(a), video(a), video(a), text("hi"))], "at most 2 videos"),
        ("a video in an assistant message", [user(text("hi")), {"role": "assistant", "content": [video(a)]}, user(text("ok"))], "only in user messages"),
        ("a video in a tool result", [user(text("hi")), {"role": "assistant", "content": "", "tool_calls": [{"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
                                       {"role": "tool", "tool_call_id": "c", "content": [video(a)]}], "only in user messages"),
        ("a clip too short to sample", [user(video(clips / "short.mp4"), text("hi"))], "too short"),
        ("bytes that aren't a video", [user({"type": "video_url", "video_url": {"url": data_url(b"not a video at all" * 20, "video/mp4")}}, text("hi"))], "invalid or unsupported"),
        ("a URL without --vision-urls", [user({"type": "video_url", "video_url": {"url": "https://example.com/a.mp4"}}, text("hi"))], "--vision-urls"),
        ("another scheme", [user({"type": "video_url", "video_url": {"url": "ftp://example.com/a.mp4"}}, text("hi"))], "data URLs or public HTTPS URLs"),
        ("a video_url without a url", [user({"type": "video_url", "video_url": {}}, text("hi"))], "non-empty url"),
    ]
    for label, messages, words in refusals:
        code, r = ask(messages)
        message = r.get("error", {}).get("message", json.dumps(r))
        check(f"refused {label}", code == 400 and words in message, f"HTTP {code} {message[:110]!r}")
    out_path.write_text(json.dumps(log, indent=1))
    print("PASS" if failures == 0 else f"FAIL: {failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
