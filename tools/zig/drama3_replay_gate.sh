#!/bin/sh
# drama3_replay_gate.sh LOG: passes when --replay 1 ended PCM-only, a codec chunk 4 or 16 PCM difference after every exact check.
# replay.zig prints that outcome only once frames, graphs, poison/reset, chunk 1 PCM and both fork lanes matched
log=$1
tail -n 1 "$log" | grep -qx 'error: ReplayPcmOnly' || exit 1
tail -n 2 "$log" | head -n 1 | grep -Eq '^Drama replay PCM only: codec chunk (4|16) against 1, max abs .*; every exact check passed$'
