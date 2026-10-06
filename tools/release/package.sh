#!/bin/sh
# Called by zig build dist with explicit artifacts; no checkout is put in the archive.
set -eu
version=$1
platform=$2
binary=$3
license=$4
notice=$5
runtime=$6
output=$7
name="tensorfold-$version-$platform"
stage="$output/../staging/$name"
mkdir -p "$stage/bin" "$stage/lib"
cp "$binary" "$stage/bin/tensorfold-native"
chmod 755 "$stage/bin/tensorfold-native"
cp "$license" "$stage/LICENSE"
cp "$notice" "$stage/NOTICE"
cp "$runtime" "$stage/RUNTIME.md"
printf '%s\n' "$version" > "$stage/VERSION"
if [ "$#" -eq 8 ]; then
    [ -z "$(find "$8" -type l -print)" ] || { echo 'CUDA capture input must not contain symlinks' >&2; exit 1; }
    found=0
    for sm_dir in "$8"/sm[0-9]*; do
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
sha() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$@"; else shasum -a 256 "$@"; fi
}
# Hash every shipped file; keep relative paths so checks survive relocation.
(cd "$stage"; find . -type f ! -name SHA256SUMS | LC_ALL=C sort | while IFS= read -r file; do sha "$file"; done) > "$stage/SHA256SUMS"
COPYFILE_DISABLE=1 tar -czf "$output/$name.tar.gz" -C "$output/../staging" "$name"
(cd "$output"; sha "$name.tar.gz") > "$output/$name.tar.gz.sha256"
# Install only the final archive and checksum, never the staging tree.
# Staging lives beside the declared archives output in the build cache.
