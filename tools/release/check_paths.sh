#!/bin/sh
# Check materialized release inputs or an archive stage, including every overlay and captured image.
set -eu
[ "$#" -eq 3 ] || { echo 'usage: check_paths.sh CHECKER BUILD_ROOT DIRECTORY' >&2; exit 2; }
[ -d "$3" ] || { echo 'release privacy input directory is missing' >&2; exit 1; }
[ ! -L "$3" ] && [ -z "$(find "$3" -type l -print -quit)" ] || { echo 'release privacy: symlink refused' >&2; exit 1; }
find "$3" -type f -exec "$1" "$2" {} +
