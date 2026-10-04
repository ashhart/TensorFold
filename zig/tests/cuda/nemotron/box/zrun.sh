#!/bin/bash
# One engine command on the GPU with journal lines; the caller holds the GPU lock. Usage: TF_JOURNAL=<file> bash -u zrun.sh RUN CMD ARGS...
set -u
RUN="${1:?run id}"
shift
JOURNAL="${TF_JOURNAL:-/dev/null}"
WHO="${TF_WHO:-zig-nemotron}"
apps=$(nvidia-smi --query-compute-apps=process_name --format=csv,noheader | grep -v -x -F "${TF_RESIDENT:-none}" || true)
if [ -n "$(docker ps -q)" ] || [ -n "$apps" ]; then
  echo "PREFLIGHT-FAIL: containers or compute apps present"; docker ps; echo "$apps"; exit 3
fi
printf '{"utc": "%s", "event": "START", "who": "%s", "run": "nemo-%s", "cmd": "%s"}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$WHO" "$RUN" "$(basename "$1") ${2:-}" >> "$JOURNAL"
rc=0
"$@" || rc=$?
printf '{"utc": "%s", "event": "END", "who": "%s", "run": "nemo-%s", "rc": %d}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$WHO" "$RUN" "$rc" >> "$JOURNAL"
exit $rc
