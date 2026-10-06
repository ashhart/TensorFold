#!/bin/sh
# Run on each archive's OS and CPU. No Python, model, GPU, network or checkout is needed.
set -eu
[ "$#" -eq 2 ] || { echo 'usage: smoke.sh ARCHIVE MAJOR.MINOR.PATCH' >&2; exit 2; }
archive=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
version=$2
scratch=$(mktemp -d "${TMPDIR:-/tmp}/tensorfold-smoke.XXXXXX")
# Keep the receipt directory for inspection instead of deleting it.
mkdir "$scratch/unpacked" "$scratch/empty" "$scratch/home"
if command -v sha256sum >/dev/null 2>&1; then checker='sha256sum'; else checker='shasum -a 256'; fi
(cd "$(dirname "$archive")"; $checker -c "$(basename "$archive").sha256")
tar -xzf "$archive" -C "$scratch/unpacked"
root="$scratch/unpacked/$(basename "$archive" .tar.gz)"
(cd "$root"; $checker -c SHA256SUMS)
[ -f "$root/LICENSE" ] && [ -f "$root/NOTICE" ]
[ -f "$root/LICENSES/MIT.txt" ] && [ -f "$root/THIRD_PARTY_NOTICES.md" ]
[ -f "$root/LICENSES/Apache-2.0.txt" ] && [ -f "$root/LICENSES/MiaAI-Lab-MIT.txt" ]
if [ "$(uname -s)" = Darwin ]; then
    nm -m "$root/bin/tensorfold-native" > "$scratch/memcpy-symbols.txt"
    grep -q 'external _memcpy (from libSystem)' "$scratch/memcpy-symbols.txt" || {
        echo 'tensorfold-native does not import memcpy from libSystem' >&2
        exit 1
    }
fi
cd "$scratch/empty"
actual=$(env -i PATH=/usr/bin:/bin HOME="$scratch/home" "$root/bin/tensorfold-native" --version)
[ "$actual" = "tensorfold-native $version" ]
env -i PATH=/usr/bin:/bin HOME="$scratch/home" "$root/bin/tensorfold-native" --help > "$scratch/help.txt"
env -i PATH=/usr/bin:/bin HOME="$scratch/home" "$root/bin/tensorfold-native" serve --help > "$scratch/serve-help.txt"
grep -q 'serve MODEL' "$scratch/help.txt"
grep -q 'usage: tensorfold serve' "$scratch/serve-help.txt"
printf 'PASS %s: version/help from empty cwd, clean environment; receipt %s\n' "$(basename "$archive")" "$scratch"
