#!/bin/sh
# Accept replay.zig's bounded PCM-only outcome after checking every chunk line.
log=$1
tail -n 1 "$log" | grep -qx 'error: ReplayPcmOnly' || exit 1
awk -v step=3.052e-5 '
    /^Drama replay PCM only/ {
        n = "[0-9]\\.[0-9][0-9][0-9]e-?[0-9]+"
        if ($0 !~ "^Drama replay PCM only: codec chunk (4|16) against 1, max abs " n ", RMS " n "; every exact check passed$" || $12 + 0 > step) bad = 1
        lines++
    }
    END { exit bad || !lines }' "$log"
