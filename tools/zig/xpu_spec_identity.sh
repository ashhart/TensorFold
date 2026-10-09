#!/bin/bash
# usage: tools/zig/xpu_spec_identity.sh CHECKPOINT [tokens=299] [ks=1,2,3,4] [drafter=mtp] [mtp-from DIR]
set -u
# runs --spec K against plain greedy on each tools/zig/prompts prompt (digests must match); exit 1 if not
cd "$(dirname "$0")/../.."
C=${1:?checkpoint}; N=${2:-299}; KS=${3:-1,2,3,4}; D=${4:-mtp}; MF=${5:-}
BIN=${TF_XPU_BIN:-./zig-out/bin/tensorfold-xpu}; BD=$(dirname "$BIN")
L="tools/zig/xpu_guard.sh 1800"; [ -n "${TF_GUARD_HELD:-}" ] && L=""  # TF_GUARD_HELD: the caller runs the whole script under the guard
bad=0
[ -n "${SKIP_ROWS:-}" ] || echo "== rows test (window rows vs single-row decode)"
out=""; [ -n "${SKIP_ROWS:-}" ] || out=$($L $BD/tf-xpu-qwen_rows-test "$C" 40 2>&1)
echo "$out" | grep -E "differ|rror" | head -20
echo "$out" | grep -E "differ" | grep -vq ", 0 differ" && bad=1
# env PROMPTS (glob) or IDS_FILE (one prompt file); PREFILL_ROWS and CONTEXT apply to both runs
PF="${PREFILL_ROWS:+--prefill $PREFILL_ROWS} ${CONTEXT:+--context $CONTEXT}"
for f in ${IDS_FILE:-tools/zig/prompts/${PROMPTS:-*}.ids}; do
  name=$(basename "$f" .ids)
  plain=$($L $BIN run "$C" --tokens-file "$f" --max-tokens "$N" --no-drafts --ignore-eos $PF 2>&1 | grep -E "^tokens" | sed 's/.* sha \([0-9a-f]*\).*/\1/')
  for k in ${KS//,/ }; do
    spec=$($L $BIN run "$C" --tokens-file "$f" --max-tokens "$N" --no-drafts --ignore-eos $PF --spec "$k" --drafter "$D" ${MF:+--mtp-from "$MF"} 2>&1 | grep -E "^tokens|^spec" | tr '\n' ' ')
    sha=$(echo "$spec" | sed 's/.*tokens [0-9]* sha \([0-9a-f]*\).*/\1/')
    if [ -n "$plain" ] && [ "$sha" = "$plain" ]; then r=identical; else r=MISMATCH; bad=1; fi
    echo "$name k=$k: $r  ($(echo "$spec" | sed 's/tokens.*//' | cut -c1-110))"
  done
done
[ $bad = 0 ] && echo "SPEC IDENTITY: PASS" || echo "SPEC IDENTITY: FAIL"
exit $bad
