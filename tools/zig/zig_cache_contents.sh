#!/bin/bash
# Zig 0.17's build cache misses on any stat change to a declared input, even with the same contents, and restoring a cache
# always changes a test binary's file id, so every test reran. Compare contents, as it does for files a compile finds.
set -eu
file="$(dirname "$(command -v zig)")/lib/std/Build/Cache.zig"
sed -i \
  -e '/fn checkInputPath(/,/fn checkDiscoveredPath(/s/const disk_file = file_off.get(disk_contents);/&\n        const manifest_digest = disk_file.digest;/' \
  -e '/fn checkInputPath(/,/fn checkDiscoveredPath(/s/if (mem.eql(u8, &disk_file.digest, &input_file.digest)) return .hit;/if (mem.eql(u8, \&disk_file.digest, \&manifest_digest)) return .hit;/' \
  "$file"
# Tests still pass without this and only run again, so a Zig that no longer matches shouldn't fail the job.
[ "$(grep -c manifest_digest "$file")" = 4 ] || echo "::warning::$file didn't take the patch, so restored caches won't reuse test runs"
