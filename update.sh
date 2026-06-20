#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
TensorFold update helper

Usage:
  ./update.sh

Environment:
  TENSORFOLD_REMOTE=origin       Git remote to pull from.
  TENSORFOLD_BRANCH=<branch>     Branch to pull. Defaults to current branch.
  TENSORFOLD_ALLOW_DIRTY=1       Allow update with local changes present.

This script fetches the configured remote and runs a fast-forward-only pull.
It refuses to update a dirty working tree by default so local edits are not
overwritten or tangled with upstream changes.
EOF
}

log() {
  printf 'tensorfold update: %s\n' "$*"
}

die() {
  printf 'tensorfold update: error: %s\n' "$*" >&2
  exit 1
}

case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
  "")
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

command -v git >/dev/null 2>&1 || die "git is not installed or not on PATH"

script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "not inside a git checkout"

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

remote="${TENSORFOLD_REMOTE:-origin}"
branch="${TENSORFOLD_BRANCH:-$(git symbolic-ref --quiet --short HEAD || true)}"

[[ -n "$branch" ]] || die "detached HEAD; set TENSORFOLD_BRANCH=<branch> and retry"
git remote get-url "$remote" >/dev/null 2>&1 || die "git remote '$remote' does not exist"

if [[ "${TENSORFOLD_ALLOW_DIRTY:-0}" != "1" ]]; then
  if [[ -n "$(git status --porcelain)" ]]; then
    git status --short >&2
    die "working tree has local changes; commit/stash them, or run TENSORFOLD_ALLOW_DIRTY=1 ./update.sh"
  fi
fi

log "repo: $repo_root"
log "remote: $remote ($(git remote get-url "$remote"))"
log "branch: $branch"
log "fetching"
git fetch --prune "$remote"

upstream="$(git rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || true)"
if [[ -n "$upstream" ]]; then
  log "pulling upstream $upstream"
  git pull --ff-only
else
  git rev-parse --verify --quiet "refs/remotes/$remote/$branch" >/dev/null \
    || die "remote branch '$remote/$branch' does not exist"
  log "no upstream configured; pulling $remote/$branch"
  git pull --ff-only "$remote" "$branch"
fi

log "updated to $(git rev-parse --short HEAD)"

if [[ -f pyproject.toml ]]; then
  log "if your virtualenv is active, refresh the editable install with: python -m pip install -e ."
fi
