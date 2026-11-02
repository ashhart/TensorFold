#!/usr/bin/env bash
# Builds an offline installation bundle: binaries for every platform, optional image, deploy assets, docs and checksums.
set -euo pipefail

usage() {
  echo "usage: $0 [--version V] [--out DIR] [--with-image]" >&2
  exit 2
}

root="$(cd "$(dirname "$0")/../.." && pwd)"
version=""
out=""
with_image=0
while [ $# -gt 0 ]; do
  case "$1" in
    --version) version="${2:?}"; shift 2 ;;
    --out) out="${2:?}"; shift 2 ;;
    --with-image) with_image=1; shift ;;
    -h|--help) usage ;;
    *) usage ;;
  esac
done

cd "$root"
command -v go >/dev/null || { echo "go is required" >&2; exit 1; }
if [ -z "$version" ]; then
  version="$(git describe --tags --always 2>/dev/null || echo 0.0.0-dev)"
fi
commit="$(git rev-parse --short=12 HEAD 2>/dev/null || true)"
out="${out:-$root/dist/tensorfold-starlight-$version}"
if [ -e "$out" ]; then
  echo "$out already exists" >&2
  exit 1
fi
mkdir -p "$out"

pkg="github.com/ashhart/TensorFold/starlight/internal/buildinfo"
ldflags="-s -w -X $pkg.Version=$version -X $pkg.Commit=$commit"
host="$(go env GOOS)/$(go env GOARCH)"

build() {
  local target="$1" cmd="$2" cgo="$3" tags="$4"
  local os="${target%/*}" arch="${target#*/}" ext=""
  [ "$os" = windows ] && ext=".exe"
  local dir="$out/bin/$os-$arch"
  mkdir -p "$dir"
  echo "building $cmd for $target"
  CGO_ENABLED="$cgo" GOOS="$os" GOARCH="$arch" go build -trimpath -tags "$tags" -ldflags "$ldflags" -o "$dir/$cmd$ext" "./cmd/$cmd"
}

for target in darwin/arm64 linux/amd64 linux/arm64 windows/amd64; do
  build "$target" starlight 0 ""
  build "$target" starlight-node 0 ""
  case "$target" in
    darwin/*)
      if [ "$host" = "$target" ]; then
        build "$target" tensorfold-connect 1 ""
      else
        echo "note: darwin Connect built without the tray; build on a Mac for the menu-bar app and Keychain storage"
        build "$target" tensorfold-connect 0 notray
      fi
      ;;
    *) build "$target" tensorfold-connect 0 "" ;;
  esac
done

mkdir -p "$out/deploy" "$out/docs"
cp -R deploy/compose deploy/helm deploy/node deploy/mdm deploy/prometheus deploy/grafana deploy/systemd deploy/starlight "$out/deploy/"
rm -rf "$out/deploy/compose/secrets" "$out/deploy/compose/.env"
cp -R docs/. "$out/docs/"
cp README.md THIRD_PARTY_NOTICES.md "$out/"

if [ "$with_image" = 1 ]; then
  command -v docker >/dev/null || { echo "--with-image needs docker" >&2; exit 1; }
  image="tensorfold/starlight:$version"
  docker build -f deploy/starlight/Dockerfile --build-arg VERSION="$version" --build-arg COMMIT="$commit" -t "$image" .
  docker save "$image" | gzip -9 > "$out/starlight-image-$version.tar.gz"
fi

if command -v sha256sum >/dev/null; then
  sum() { sha256sum "$@"; }
else
  sum() { shasum -a 256 "$@"; }
fi

cd "$out"
find . -type f ! -name SHA256SUMS ! -name manifest.json ! -name .files | LC_ALL=C sort | sed 's|^\./||' > .files
while IFS= read -r f; do sum "$f"; done < .files > SHA256SUMS
{
  printf '{\n  "version": "%s",\n  "commit": "%s",\n  "go": "%s",\n  "built_at": "%s",\n  "files": [\n' \
    "$version" "$commit" "$(go env GOVERSION)" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  first=1
  while read -r hash file; do
    size="$(wc -c < "$file" | tr -d ' ')"
    [ "$first" = 1 ] || printf ',\n'
    printf '    {"path": "%s", "sha256": "%s", "bytes": %s}' "$file" "$hash" "$size"
    first=0
  done < SHA256SUMS
  printf '\n  ]\n}\n'
} > manifest.json
rm -f .files
echo "bundle written to $out"
