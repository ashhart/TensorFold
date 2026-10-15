#!/usr/bin/env python3
"""Collect public GitHub repository metadata and README snapshots for local research.

Raw repository metadata belongs under `.research/`. Product docs should contain adapted findings
only, without source attribution.

Examples:
  python scripts/github_profile_research.py --user account
  GITHUB_TOKEN=... python scripts/github_profile_research.py --user account --max-readmes 20
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_OUT = Path(".research/github-profiles")
KEYWORDS = {
    "agent",
    "automation",
    "browser",
    "bridge",
    "cli",
    "codex",
    "connector",
    "cost",
    "export",
    "hook",
    "kanban",
    "local",
    "mcp",
    "model",
    "provider",
    "proxy",
    "queue",
    "redact",
    "remote",
    "session",
    "token",
    "usage",
    "webhook",
    "worktree",
    "workflow",
}


@dataclasses.dataclass
class RepoScore:
    repo: dict[str, Any]
    score: int
    reasons: list[str]


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-") or "repo"


def request(url: str, *, accept: str = "application/vnd.github+json") -> bytes:
    headers = {
        "Accept": accept,
        "User-Agent": "TensorFold Studio research helper",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as response:
        return response.read()


def fetch_json(url: str) -> Any:
    return json.loads(request(url).decode("utf-8"))


def fetch_repos(user: str, per_page: int, sort: str) -> list[dict[str, Any]]:
    repos: list[dict[str, Any]] = []
    page = 1
    while True:
        query = urllib.parse.urlencode({"per_page": per_page, "page": page, "sort": sort})
        batch = fetch_json(f"https://api.github.com/users/{urllib.parse.quote(user)}/repos?{query}")
        if not batch:
            break
        repos.extend(batch)
        if len(batch) < per_page:
            break
        page += 1
    return repos


def score_repo(repo: dict[str, Any]) -> RepoScore:
    text = " ".join(str(repo.get(key) or "") for key in ("name", "description", "language", "topics")).lower()
    stars = int(repo.get("stargazers_count") or 0)
    score = min(stars, 500)
    reasons: list[str] = []

    if stars:
        reasons.append(f"{stars} stars")
    for keyword in sorted(KEYWORDS):
        if keyword in text:
            score += 75
            reasons.append(keyword)
    if repo.get("fork"):
        score -= 100
        reasons.append("fork")
    if repo.get("archived"):
        score -= 100
        reasons.append("archived")

    return RepoScore(repo=repo, score=score, reasons=reasons)


def shortlist(repos: list[dict[str, Any]], include_forks: bool, limit: int) -> list[RepoScore]:
    candidates = repos if include_forks else [repo for repo in repos if not repo.get("fork")]
    scored = [score_repo(repo) for repo in candidates]
    scored.sort(key=lambda item: (item.score, item.repo.get("pushed_at") or ""), reverse=True)
    return scored[:limit]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def fetch_readme(full_name: str) -> str | None:
    url = f"https://api.github.com/repos/{urllib.parse.quote(full_name, safe='/')}/readme"
    try:
        return request(url, accept="application/vnd.github.raw").decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def clean_repo_record(repo: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": repo.get("name"),
        "full_name": repo.get("full_name"),
        "html_url": repo.get("html_url"),
        "description": repo.get("description"),
        "language": repo.get("language"),
        "topics": repo.get("topics") or [],
        "stars": repo.get("stargazers_count"),
        "forks": repo.get("forks_count"),
        "fork": repo.get("fork"),
        "archived": repo.get("archived"),
        "pushed_at": repo.get("pushed_at"),
        "updated_at": repo.get("updated_at"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", action="append", required=True, help="GitHub username or org. Can be repeated.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--per-page", type=int, default=100)
    parser.add_argument("--sort", default="updated", choices=["created", "updated", "pushed", "full_name"])
    parser.add_argument("--max-readmes", type=int, default=12)
    parser.add_argument("--include-forks", action="store_true")
    args = parser.parse_args()

    for user in args.user:
        user_dir = args.out / slugify(user)
        repos = fetch_repos(user, args.per_page, args.sort)
        scored = shortlist(repos, args.include_forks, args.max_readmes)

        write_json(
            user_dir / "repos.json",
            {
                "fetched_at": utc_now(),
                "user": user,
                "count": len(repos),
                "repos": [clean_repo_record(repo) for repo in repos],
            },
        )
        write_json(
            user_dir / "shortlist.json",
            {
                "fetched_at": utc_now(),
                "user": user,
                "shortlist": [
                    {
                        **clean_repo_record(item.repo),
                        "research_score": item.score,
                        "research_reasons": item.reasons,
                    }
                    for item in scored
                ],
            },
        )

        for item in scored:
            full_name = str(item.repo.get("full_name") or "")
            if not full_name:
                continue
            readme = fetch_readme(full_name)
            if readme is None:
                continue
            write_text(user_dir / f"{slugify(item.repo.get('name') or full_name)}.README.md", readme)

        print(f"Wrote {len(repos)} repos and {len(scored)} README candidates under {user_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
