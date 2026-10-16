#!/usr/bin/env python3
"""Collect curated public posts for local product research.

Profile timelines on X are brittle without an official API or logged-in browser state. This script
therefore treats X as a curated-source workflow: collect high-signal post URLs, fetch each post via
public oEmbed-style mirrors, and write raw JSON into `.research/` for local analysis.

Examples:
  python scripts/public_post_research.py --post https://x.com/account/status/1234567890123456789
  python scripts/public_post_research.py --handle account --post-list .research/post-urls.txt
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_OUT = Path(".research/x-posts")
STATUS_ID_RE = re.compile(r"(?<!\d)(\d{15,25})(?!\d)")


@dataclasses.dataclass
class PostRef:
    status_id: str
    handle: str
    source: str


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def fetch_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 TensorFold Studio research"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def extract_post_ref(value: str, default_handle: str) -> PostRef:
    value = value.strip()
    parsed = urllib.parse.urlparse(value)
    path_parts = [part for part in parsed.path.split("/") if part]

    handle = default_handle
    status_id = None
    if parsed.netloc:
        if len(path_parts) >= 3 and path_parts[1] in {"status", "statuses"}:
            handle = path_parts[0]
            status_id = path_parts[2]
            if handle == "i":
                handle = default_handle
        else:
            match = STATUS_ID_RE.search(value)
            if match:
                status_id = match.group(1)
    elif STATUS_ID_RE.fullmatch(value):
        status_id = value

    if not status_id or not STATUS_ID_RE.fullmatch(status_id):
        raise ValueError(f"Could not extract an X status id from: {value}")
    if not handle:
        raise ValueError(f"Post id {status_id} needs --handle because no handle was present in the URL.")

    return PostRef(status_id=status_id, handle=handle.lstrip("@"), source=value)


def read_refs(args: argparse.Namespace) -> list[PostRef]:
    values = list(args.post or [])
    if args.post_list:
        values.extend(
            line.strip()
            for line in args.post_list.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        )

    seen: set[str] = set()
    refs: list[PostRef] = []
    for value in values:
        ref = extract_post_ref(value, args.handle)
        if ref.status_id in seen:
            continue
        seen.add(ref.status_id)
        refs.append(ref)
    return refs


def fetch_post(ref: PostRef) -> dict[str, Any]:
    errors: list[str] = []
    urls = [
        f"https://api.fxtwitter.com/{ref.handle}/status/{ref.status_id}",
        f"https://api.vxtwitter.com/{ref.handle}/status/{ref.status_id}",
    ]
    for url in urls:
        try:
            payload = fetch_json(url)
            payload["_vontra_fetch_url"] = url
            return payload
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
    raise RuntimeError("; ".join(errors))


def article_text(article: dict[str, Any] | None) -> str | None:
    if not article:
        return None
    blocks = article.get("content", {}).get("blocks", [])
    lines = [block.get("text", "").strip() for block in blocks if block.get("text", "").strip()]
    return "\n\n".join(lines) or None


def tweet_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("tweet", payload)


def normalize(ref: PostRef, payload: dict[str, Any], raw_path: Path) -> dict[str, Any]:
    tweet = tweet_from_payload(payload)
    quote = tweet.get("quote") or tweet.get("qrt")
    article = tweet.get("article")
    author = tweet.get("author") or {}

    return {
        "id": str(tweet.get("id") or tweet.get("tweetID") or ref.status_id),
        "source": ref.source,
        "url": tweet.get("url") or tweet.get("tweetURL") or f"https://x.com/{ref.handle}/status/{ref.status_id}",
        "handle": author.get("screen_name") or tweet.get("user_screen_name") or ref.handle,
        "author_name": author.get("name") or tweet.get("user_name"),
        "created_at": tweet.get("created_at") or tweet.get("date"),
        "text": tweet.get("text") or "",
        "article_title": article.get("title") if article else None,
        "article_preview": article.get("preview_text") if article else None,
        "article_text": article_text(article),
        "quote_id": str(quote.get("id") or quote.get("tweetID")) if quote else None,
        "quote_text": quote.get("text") if quote else None,
        "metrics": {
            "likes": tweet.get("likes"),
            "replies": tweet.get("replies"),
            "retweets": tweet.get("retweets"),
            "bookmarks": tweet.get("bookmarks"),
            "quotes": tweet.get("quotes"),
            "views": tweet.get("views"),
        },
        "raw_path": str(raw_path),
        "fetch_url": payload.get("_vontra_fetch_url"),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handle", default="", help="Default handle for bare status ids.")
    parser.add_argument("--post", action="append", help="X post URL or status id. Can be repeated.")
    parser.add_argument("--post-list", type=Path, help="File containing one X post URL or status id per line.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    refs = read_refs(args)
    if not refs:
        parser.error("Provide at least one --post or --post-list entry.")

    args.out.mkdir(parents=True, exist_ok=True)
    index: dict[str, Any] = {"fetched_at": utc_now(), "posts": []}

    for ref in refs:
        raw_path = args.out / "raw" / f"{ref.status_id}.json"
        try:
            payload = fetch_post(ref)
            write_json(raw_path, payload)
            index["posts"].append(normalize(ref, payload, raw_path))
        except Exception as exc:
            index["posts"].append(
                {
                    "id": ref.status_id,
                    "source": ref.source,
                    "url": f"https://x.com/{ref.handle}/status/{ref.status_id}",
                    "handle": ref.handle,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    write_json(args.out / "post-index.json", index)
    print(f"Wrote {len(refs)} post records to {args.out / 'post-index.json'}")
    print("Raw post JSON is local research material. Distill findings before committing docs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
