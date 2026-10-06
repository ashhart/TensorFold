#!/usr/bin/env bash
# Builds fake_serve and an engine-less tensorfold-native into zig-out/server/ (ZIG overrides the compiler, OPT the mode).
set -euo pipefail
zig="${ZIG:-zig}"
root="$(cd "$(dirname "$0")/../../.." && pwd)"
out="$root/zig-out/server"
mkdir -p "$out"
# The direct CPU-only harness uses the manifest version, like zig build native.
version="$(sed -n 's/^[[:space:]]*\.version = "\([0-9]*\.[0-9]*\.[0-9]*\)",$/\1/p' "$root/build.zig.zon")"
[ -n "$version" ] || { echo 'missing manifest version' >&2; exit 1; }
options="$out/build_options.zig"
printf 'pub const version = "%s";\n' "$version" > "$options"
core=(--dep lanes "-Mengine_api=$root/zig/src/core/engine_api.zig" "-Mlanes=$root/zig/src/core/lanes/lanes.zig"
      "-Mtokenizer=$root/zig/src/core/tokenizer/tokenizer.zig" "-Mtemplate=$root/zig/src/core/template/template.zig")
"$zig" build-exe -lc -O "${OPT:-ReleaseSafe}" --dep server --dep engine_api "-Mroot=$root/zig/tests/server/fake_serve.zig" \
  --dep engine_api --dep tokenizer --dep template "-Mserver=$root/zig/src/server/root.zig" "${core[@]}" \
  --name fake_serve -femit-bin="$out/fake_serve"
"$zig" build-exe -lc -O "${OPT:-ReleaseSafe}" --dep engine_api --dep tokenizer --dep template --dep native_engines --dep build_options \
  "-Mroot=$root/zig/src/server/main.zig" --dep engine_api "-Mnative_engines=$root/zig/src/native/none.zig" "${core[@]}" \
  "-Mbuild_options=$options" --name tensorfold-native -femit-bin="$out/tensorfold-native"
