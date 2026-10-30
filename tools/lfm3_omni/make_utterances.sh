#!/bin/sh
# The 16 synthetic spoken turns (macOS `say`, eight system voices in turn) as 16 kHz WAVs: make_utterances.sh OUT_DIR
set -eu
here=$(cd "$(dirname "$0")" && pwd)
out=$1
mkdir -p "$out"
: > "$out/refs.tsv"
set -- Samantha Daniel Karen Moira Rishi Tessa Fred Albert
i=0
while IFS= read -r line; do
  [ -z "$line" ] && continue
  v=$(eval echo "\${$((i % 8 + 1))}")
  f=$(printf "u%02d.wav" "$i")
  say -v "$v" -o "$out/$f" --data-format=LEI16@16000 "$line"
  printf "%s\t%s\t%s\n" "$f" "$v" "$line" >> "$out/refs.tsv"
  i=$((i + 1))
done < "$here/utterances.txt"
