"""Video input served end to end on a running ``tensorfold-native --vision`` (#565's video checks).

    python -I tools/zig/video_served.py --url http://127.0.0.1:8000 --model DIR --clips DIR

CLIPS holds tools/zig/glm_video_fixtures.py's clip set (--make). Checks: a video request's prompt ids (/tokenize,
media expanded as the chat route expands them) equal transformers' Glm5NextProcessor's input_ids for the same
messages, on every clip, with the server's --vision-video-tokens as the processor's max_image_tokens; a video request
answers, drafted equals "draft": false and concurrent equals alone; a video and an image in one request; a video
conversation's later turn resumes past its video, and another video under the same text resumes only the text
before it; and the refusals (three videos, a video outside a user message, a clip too short to sample, bytes that
aren't a video, a URL without --vision-urls, a scheme other than data or https).
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from glm_video_fixtures import load  # noqa: E402


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--image", help="a PNG for the video-and-image request")
    ap.add_argument("--video-tokens", type=int, default=16384)
    ap.add_argument("--tokens", type=int, default=32)
    args = ap.parse_args()
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(args.model)
    vp = proc.video_processor
    with urllib.request.urlopen(args.url + "/v1/models") as r:
        model = json.loads(r.read())["data"][0]["id"]
    clips = Path(args.clips)
    failures = 0

    def check(label: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        failures += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {label}{': ' + detail if detail else ''}")

    def video(p: Path) -> dict:
        return {"type": "video_url", "video_url": {"url": data_url(p.read_bytes(), MIME[p.suffix])}}

    def text(t: str) -> dict:
        return {"type": "text", "text": t}

    def user(*parts) -> dict:
        return {"role": "user", "content": list(parts)}

    off = {"enable_thinking": False}

    def ask(messages: list, draft: bool = True, tokens: int | None = None) -> tuple[int, dict]:
        b = {"model": model, "messages": messages, "max_tokens": tokens or args.tokens, "temperature": 0, "chat_template_kwargs": off}
        if not draft:
            b["draft"] = False
        return post(args.url, "/v1/chat/completions", b)

    def reply(r: dict) -> str:
        return r["choices"][0]["message"].get("content") or ""

    # the prompt ids, clip by clip, against the processor's (thinking on: the checkpoint's template has no switch; with
    # thinking off the server appends </think> itself)
    q = "What moves?"
    same = 0
    clip_files = sorted(p for p in clips.iterdir() if p.suffix in MIME and p.name != "short.mp4")
    for p in clip_files:
        frames, meta = load(p, vp, 256)
        msgs = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": q}]}]
        prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)  # the default: thinking on
        ref = proc(text=[prompt], videos=[frames], video_metadata=[meta], do_sample_frames=False, max_image_tokens=args.video_tokens, return_tensors="np")["input_ids"][0].tolist()
        code, r = post(args.url, "/tokenize", {"messages": [user(video(p), text(q))]})
        ok = code == 200 and r.get("tokens") == ref
        same += ok
        if not ok:
            got = r.get("tokens", [])
            at = next((i for i, (x, y) in enumerate(zip(got, ref)) if x != y), min(len(got), len(ref)))
            print(f"     {p.name}: HTTP {code} {len(got)} ids against {len(ref)}, first difference at {at}: {got[at:at + 6]} vs {ref[at:at + 6]} {r.get('error', '')}")
    check("a video request's prompt ids equal the processor's", same == len(clip_files), f"{same} of {len(clip_files)} clips")

    a, b = clips / "h264.mp4", clips / "hevc.mp4"
    # the cache, before any concurrent run (GLM keeps a state resident in the slot that made it; a slot another stream
    # takes leaves it stale): a third turn resumes past the video its second turn kept; another video under the same
    # text, only the text before it
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
        check("drafted equals \"draft\": false with a video", reply(r) == reply(r2), repr(reply(r)[:60]))
    runs = [[user(video(c), text(t))] for c in (a, b) for t in ("What colours appear?", "Does anything move?")]
    alone = [reply(ask(m)[1]) for m in runs]
    with cf.ThreadPoolExecutor(len(runs)) as pool:
        together = [reply(f.result()[1]) for f in [pool.submit(ask, m) for m in runs]]
    check("concurrent video requests equal their runs alone", alone == together, f"{sum(x == y for x, y in zip(alone, together))} of {len(runs)}")
    if args.image:
        img = {"type": "image_url", "image_url": {"url": data_url(Path(args.image).read_bytes(), "image/png")}}
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
    print("PASS" if failures == 0 else f"FAIL: {failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
