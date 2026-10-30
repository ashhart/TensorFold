#!/bin/sh
# Stage parity over WAV_DIR/u*.wav: parity.sh BIN MODEL REF_DIR OUT_DIR SYSTEM_FILE WAV_DIR [DEV] [DTYPE] ($PYTHON)
set -u
here=$(cd "$(dirname "$0")" && pwd)
B=$1; M=$2; REF=$3; OUT=$4; SYS=$5; WAVS=$6; DEV=${7:-cuda}; DT=${8:-bf16}; P=${PYTHON:-python3}
busy() { [ -n "${LFM3_OMNI_BUSY:-}" ] && sh -c "$LFM3_OMNI_BUSY"; }
for w in "$WAVS"/u*.wav; do
  u=$(basename "$w" .wav); mkdir -p "$OUT/$u"
  while busy; do sleep 5; done
  "$B" --model "$M" --wav "$w" --system-file "$SYS" --audio-temperature 0 --identity-order 1 --dump "$OUT/$u" \
    --codec-ref "$REF/$u" > "$OUT/$u/run.log" 2>&1
  "$P" "$here/compare_stages.py" "$REF/$u" "$OUT/$u" > /dev/null
  "$B" --model "$M" --wav "$w" --system-file "$SYS" --audio-temperature 0 --identity-order 1 --force "$REF/$u" \
    > "$OUT/$u/forced.log" 2>&1
  [ -f "$REF/$u/ref_depth.npy" ] || "$P" "$here/reference_depth.py" "$M" "$REF/$u" prep "$DEV" "$DT" > /dev/null
  "$B" --model "$M" --depth-check "$REF/$u" > "$OUT/$u/depth.log" 2>&1
  "$P" "$here/reference_depth.py" "$M" "$REF/$u" score > "$OUT/$u/depth_parity.json"
  echo "$u $(grep FORCED "$OUT/$u/forced.log") $(cat "$OUT/$u/depth_parity.json")"
done
"$P" "$here/summarize_parity.py" "$REF" "$OUT"
