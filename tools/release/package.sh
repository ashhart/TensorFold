#!/bin/sh
# Called by zig build dist with explicit artifacts; no checkout is put in the archive.
set -eu
version=$1
platform=$2
binary=$3
license=$4
notice=$5
runtime=$6
license_dir=$7
third_party=$8
output=$9
shift 9
# then optionally a CUDA capture directory, and --overlay=DIR: files copied into the archive's tree (libtfgrammar)
capture=
overlay=
path_checker=
build_root=
for extra in "$@"; do
    case "$extra" in
        --overlay=*) overlay=${extra#--overlay=} ;;
        --path-checker=*) path_checker=${extra#--path-checker=} ;;
        --build-root=*) build_root=${extra#--build-root=} ;;
        *) capture=$extra ;;
    esac
done
[ -n "$path_checker" ] && [ -n "$build_root" ] || { echo 'release packaging requires its privacy checker and build root' >&2; exit 2; }
name="tensorfold-$version-$platform"
mkdir -p "$output"
# Each invocation owns a fresh stage, including reruns of the same version/platform.
staging=$(mktemp -d "$output/../tensorfold-stage.XXXXXX")
stage="$staging/$name"
mkdir -p "$stage/bin" "$stage/lib" "$stage/LICENSES"
cp "$binary" "$stage/bin/tensorfold-native"
chmod 755 "$stage/bin/tensorfold-native"
cp "$license" "$stage/LICENSE"
cp "$notice" "$stage/NOTICE"
cp -R "$license_dir/." "$stage/LICENSES/"
cp "$third_party" "$stage/THIRD_PARTY_NOTICES.md"
cp "$runtime" "$stage/RUNTIME.md"
printf '%s\n' "$version" > "$stage/VERSION"
if [ -n "$overlay" ]; then
    [ -z "$(find "$overlay" -type l -print)" ] || { echo 'overlay input must not contain symlinks' >&2; exit 1; }
    cp -R "$overlay/." "$stage/"
fi
if [ -n "$capture" ]; then
    [ -z "$(find "$capture" -type l -print)" ] || { echo 'CUDA capture input must not contain symlinks' >&2; exit 1; }
    found=0
    for sm_dir in "$capture"/sm[0-9]*; do
        [ -d "$sm_dir" ] || continue
        sm=$(basename "$sm_dir")
        case "${sm#sm}" in ''|*[!0-9]*) echo "invalid CUDA capture directory: $sm" >&2; exit 1 ;; esac
        [ -f "$sm_dir/aot.json" ] || { echo "missing $sm/aot.json" >&2; exit 1; }
        mkdir -p "$stage/share/tensorfold/cuda/$sm/cubins"
        cp "$sm_dir/aot.json" "$stage/share/tensorfold/cuda/$sm/aot.json"
        cubins=0
        for cubin in "$sm_dir"/cubins/*.cubin; do
            [ -f "$cubin" ] || continue
            cp "$cubin" "$stage/share/tensorfold/cuda/$sm/cubins/"
            cubins=$((cubins + 1))
        done
        [ "$cubins" -gt 0 ] || { echo "missing $sm/cubins/*.cubin" >&2; exit 1; }
        found=$((found + 1))
    done
    [ "$found" -gt 0 ] || { echo 'CUDA capture input requires sm<capability>/aot.json and cubins/*.cubin' >&2; exit 1; }
fi
case "$platform" in
    *-host-only) printf '%s\n' 'CPU verification only. This binary has no embedded CUDA kernels and cannot run inference.' > "$stage/HOST-ONLY" ;;
esac
sh "$(dirname "$0")/check_paths.sh" "$path_checker" "$build_root" "$stage"
sha() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$@"; else shasum -a 256 "$@"; fi
}
# Hash every shipped file; keep relative paths so checks survive relocation.
(cd "$stage"; find . -type f ! -name SHA256SUMS | LC_ALL=C sort | while IFS= read -r file; do sha "$file"; done) > "$stage/SHA256SUMS"
# no macOS metadata or builder identity in the entries (Linux tar warned once per entry)
if tar --version 2>/dev/null | grep -q bsdtar; then owner="--no-xattrs --no-mac-metadata --uid 0 --gid 0 --uname root --gname root"; else owner="--owner=0 --group=0 --numeric-owner"; fi
# shellcheck disable=SC2086
COPYFILE_DISABLE=1 tar $owner -czf "$staging/$name.tar.gz" -C "$staging" "$name"
python3 "$(dirname "$0")/check_archive.py" "$path_checker" "$build_root" "$staging/$name.tar.gz"
mv "$staging/$name.tar.gz" "$output/$name.tar.gz"
(cd "$output"; sha "$name.tar.gz") > "$output/$name.tar.gz.sha256"
# Install only the final archive and checksum, never the staging tree.
# Fresh stages stay in the build cache for inspection.
