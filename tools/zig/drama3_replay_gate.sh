#!/bin/sh
# drama3_replay_gate.sh LOG: passes only on replay.zig's PCM-only outcome (chunk 4 or 16 PCM, all exact checks equal).
log=$1
n='[0-9]\.[0-9]{3}e-?[0-9]+'
tail -n 1 "$log" | grep -qx 'error: ReplayPcmOnly' || exit 1
tail -n 2 "$log" | head -n 1 |
    grep -Eqx "Drama replay PCM only: codec chunk (4|16) against 1, max abs $n, RMS $n; every exact check passed"
