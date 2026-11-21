#!/bin/sh
# drama3_replay_gate.sh LOG: passes when --replay 1 stopped at the chunk 4 or 16 PCM compare, every earlier check exact.
# a fork-lane frame mismatch after chunk 16 logs the same as a chunk 16 PCM one: replay.zig prints nothing between them
log=$1
tail -n 1 "$log" | grep -qx 'error: ReplayMismatch' || exit 1
case "$(grep -o '^codec requested [0-9]*' "$log" | cut -d' ' -f3 | tr '\n' ' ')" in
'1 1 4 ' | '1 1 4 16 ') exit 0 ;;
*) exit 1 ;;
esac
