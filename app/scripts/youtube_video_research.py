#!/usr/bin/env python3
"""Collect YouTube metadata and optional transcripts for local product research.

The default output folder is `.research/youtube-videos`, which is intentionally ignored by git.
Use the generated transcript JSON as private source material, then check in distilled
product notes rather than raw captions.

Examples:
  uv run --with youtube-transcript-api python scripts/youtube_video_research.py --channel-id UC... --limit 20
  uv run --with youtube-transcript-api python scripts/youtube_video_research.py --video VIDEO_ID
  uvx yt-dlp --flat-playlist --print id https://www.youtube.com/@channel/videos > .research/youtube-videos/all-video-ids.txt
  uvx yt-dlp --flat-playlist --print '%(id)s\t%(title)s\t%(duration_string)s' https://www.youtube.com/@channel/videos > .research/youtube-videos/all-videos.tsv
  uv run --with youtube-transcript-api python scripts/youtube_video_research.py --video-list .research/youtube-videos/all-video-ids.txt --sleep 1 --stop-on-ip-block
  python scripts/youtube_video_research.py --channel-id UC... --metadata-only --limit 20
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


CHANNEL_ID = ""
DEFAULT_OUT = Path(".research/youtube-videos")
NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "yt": "http://www.youtube.com/xml/schemas/2015",
    "media": "http://search.yahoo.com/mrss/",
}


@dataclasses.dataclass
class VideoEntry:
    video_id: str
    title: str
    link: str
    published: str | None = None
    updated: str | None = None
    description: str | None = None
    views: int | None = None
    is_short: bool = False
    duration: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def fetch_text(url: str) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "Accept-Language": "en-US,en;q=0.9",
            "User-Agent": "Mozilla/5.0 TensorFold Studio research crawler",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8")


def text_at(element: ET.Element, path: str) -> str | None:
    match = element.find(path, NS)
    return match.text.strip() if match is not None and match.text else None


def parse_feed(channel_id: str) -> list[VideoEntry]:
    feed_url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
    root = ET.fromstring(fetch_text(feed_url))
    entries: list[VideoEntry] = []

    for entry in root.findall("atom:entry", NS):
        video_id = text_at(entry, "yt:videoId")
        title = text_at(entry, "atom:title")
        link = None
        alternate = entry.find("atom:link[@rel='alternate']", NS)
        if alternate is not None:
            link = alternate.attrib.get("href")
        if not video_id or not title or not link:
            continue

        stats = entry.find("media:group/media:community/media:statistics", NS)
        views = int(stats.attrib["views"]) if stats is not None and "views" in stats.attrib else None
        entries.append(
            VideoEntry(
                video_id=video_id,
                title=title,
                link=link,
                published=text_at(entry, "atom:published"),
                updated=text_at(entry, "atom:updated"),
                description=text_at(entry, "media:group/media:description"),
                views=views,
                is_short="/shorts/" in link,
            )
        )

    return entries


def extract_video_id(value: str) -> str:
    value = value.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", value):
        return value

    parsed = urllib.parse.urlparse(value)
    if parsed.netloc.endswith("youtu.be"):
        return parsed.path.strip("/").split("/")[0]
    if "youtube.com" in parsed.netloc:
        query_id = urllib.parse.parse_qs(parsed.query).get("v", [None])[0]
        if query_id:
            return query_id
        parts = [part for part in parsed.path.split("/") if part]
        if "shorts" in parts:
            index = parts.index("shorts")
            if index + 1 < len(parts):
                return parts[index + 1]

    raise ValueError(f"Could not extract a YouTube video id from: {value}")


def slugify(value: str, max_length: int = 72) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return slug[:max_length].strip("-") or "untitled"


def transcript_filename(entry: VideoEntry) -> str:
    date_prefix = f"{entry.published[:10]}-" if entry.published else ""
    return f"{date_prefix}{slugify(entry.title)}-{entry.video_id}.json"


def fallback_metadata(video_id: str) -> VideoEntry:
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        oembed_url = "https://www.youtube.com/oembed?" + urllib.parse.urlencode(
            {"url": url, "format": "json"}
        )
        payload = json.loads(fetch_text(oembed_url))
        title = payload.get("title") or video_id
    except Exception:
        title = video_id
    return VideoEntry(video_id=video_id, title=title, link=url)


def entry_from_list_line(value: str, by_id: dict[str, VideoEntry]) -> VideoEntry:
    parts = value.split("\t") if "\t" in value else value.split("\\t")
    video_id = extract_video_id(parts[0])
    if video_id in by_id:
        entry = by_id[video_id]
    elif len(parts) > 1 and parts[1].strip():
        entry = VideoEntry(
            video_id=video_id,
            title=parts[1].strip(),
            link=f"https://www.youtube.com/watch?v={video_id}",
            duration=parts[2].strip() if len(parts) > 2 and parts[2].strip() else None,
        )
    else:
        entry = fallback_metadata(video_id)

    if len(parts) > 2 and parts[2].strip() and not entry.duration:
        entry.duration = parts[2].strip()
    return entry


def import_transcript_api() -> Any:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError as exc:
        raise RuntimeError(
            "youtube-transcript-api is not installed. Run with: "
            "uv run --with youtube-transcript-api python scripts/youtube_video_research.py"
        ) from exc
    return YouTubeTranscriptApi


def fetch_transcript(video_id: str, language: str) -> list[dict[str, Any]]:
    transcript_api = import_transcript_api()

    try:
        fetched = transcript_api().fetch(video_id, languages=[language])
        snippets = list(fetched)
    except TypeError:
        snippets = transcript_api.get_transcript(video_id, languages=[language])

    segments: list[dict[str, Any]] = []
    for snippet in snippets:
        if isinstance(snippet, dict):
            text = snippet.get("text", "")
            start = snippet.get("start", 0)
            duration = snippet.get("duration", 0)
        else:
            text = snippet.text
            start = snippet.start
            duration = snippet.duration
        segments.append({"text": text, "start": start, "duration": duration})
    return segments


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def select_entries(args: argparse.Namespace) -> list[VideoEntry]:
    feed_entries = parse_feed(args.channel_id) if args.channel_id else []
    by_id = {entry.video_id: entry for entry in feed_entries}

    requested_videos = list(args.video or [])
    if args.video_list:
        requested_videos.extend(
            line.strip()
            for line in args.video_list.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        )

    if requested_videos:
        selected: list[VideoEntry] = []
        seen: set[str] = set()
        for value in requested_videos:
            entry = entry_from_list_line(value, by_id)
            if entry.video_id in seen:
                continue
            seen.add(entry.video_id)
            selected.append(entry)
        return selected[: args.max_videos] if args.max_videos else selected

    entries = feed_entries if args.include_shorts else [entry for entry in feed_entries if not entry.is_short]
    entries = entries[: args.limit]
    return entries[: args.max_videos] if args.max_videos else entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel-id", default=CHANNEL_ID, help="YouTube channel id for recent-feed metadata.")
    parser.add_argument("--limit", type=int, default=10, help="Recent feed entries to collect.")
    parser.add_argument("--video", action="append", help="Specific video id or URL. Can be repeated.")
    parser.add_argument("--video-list", type=Path, help="File containing one video id or URL per line.")
    parser.add_argument("--max-videos", type=int, help="Cap selected videos after dedupe.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--language", default="en")
    parser.add_argument("--include-shorts", action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--refresh", action="store_true", help="Re-fetch transcripts even if the JSON file exists.")
    parser.add_argument("--checkpoint-every", type=int, default=25, help="Write video-index.json every N records.")
    parser.add_argument("--stop-on-ip-block", action="store_true", help="Stop after recording the first YouTube IP block/429 error.")
    parser.add_argument("--sleep", type=float, default=0, help="Seconds to pause between transcript fetches.")
    args = parser.parse_args()

    if not args.channel_id and not args.video and not args.video_list:
        parser.error("Provide --channel-id, --video, or --video-list.")

    args.out.mkdir(parents=True, exist_ok=True)
    entries = select_entries(args)
    index: dict[str, Any] = {
        "source": {
            "channel_id": args.channel_id or None,
            "feed": f"https://www.youtube.com/feeds/videos.xml?channel_id={args.channel_id}" if args.channel_id else None,
        },
        "fetched_at": utc_now(),
        "language": args.language,
        "videos": [],
    }
    index_path = args.out / "video-index.json"

    for index_number, entry in enumerate(entries, start=1):
        print(f"[{index_number}/{len(entries)}] {entry.video_id} {entry.title}", flush=True)
        record = entry.as_dict()
        if not args.metadata_only:
            transcript_path = args.out / "transcripts" / transcript_filename(entry)
            try:
                if transcript_path.exists() and not args.refresh:
                    cached = json.loads(transcript_path.read_text(encoding="utf-8"))
                    segments = cached.get("segments", [])
                    record["transcript_cached"] = True
                else:
                    segments = fetch_transcript(entry.video_id, args.language)
                    write_json(
                        transcript_path,
                        {
                            "video": entry.as_dict(),
                            "source_url": entry.link,
                            "fetched_at": utc_now(),
                            "language": args.language,
                            "segments": segments,
                        },
                    )
                    record["transcript_cached"] = False
                record["transcript_path"] = str(transcript_path)
                record["transcript_segments"] = len(segments)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                record["transcript_error"] = error
        index["videos"].append(record)

        if args.checkpoint_every and index_number % args.checkpoint_every == 0:
            write_json(index_path, index)

        if args.stop_on_ip_block and "transcript_error" in record:
            error = record["transcript_error"]
            if "IpBlocked" in error or "RequestBlocked" in error or "Too Many Requests" in error:
                print("Stopping early after YouTube transcript throttling/IP block.", flush=True)
                break

        if args.sleep and entry is not entries[-1]:
            time.sleep(args.sleep)

    write_json(index_path, index)
    print(f"Wrote {len(index['videos'])} video records to {index_path}")
    if not args.metadata_only:
        print("Raw transcripts are local research material. Distill findings before committing docs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
