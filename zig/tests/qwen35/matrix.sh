#!/bin/sh
# usage: matrix.sh [options] MODELS_DIR DEVICES... [-- CHECK_ARGS]: a markdown table of `check` per model and setting.

# DEVICES: an argument a group, a comma list of HIP ordinals ("6,7" is tensor parallel over two); exit 1 on any FAIL.

bin=zig-out/bin/tf-qwen35-test
log=
port=29551
limit=3000
speed=
ids=
models=
truths=
envs=
extra=

die() { echo "matrix.sh: $*" >&2; exit 2; }
value() { [ "$1" -ge 2 ] || die "$2 needs a value"; }

while [ $# -gt 0 ]; do
  case $1 in
    --model) value $# "$1"; models="$models $2"; shift 2 ;; # a directory under MODELS_DIR; default: all of them
    --env) value $# "$1"; envs="$envs $2"; shift 2 ;; # VAR=V1,V2; every combination is run
    --truth) value $# "$1"; truths="$truths $2"; shift 2 ;; # NAME=FILE, the fp64 truth .npy model NAME is scored against
    --ids) value $# "$1"; ids=$2; shift 2 ;; # the ids .npy the long prompts and the truth rows are cut from
    --speed) speed=--speed; shift ;;
    --bin) value $# "$1"; bin=$2; shift 2 ;;
    --log) value $# "$1"; log=$2; shift 2 ;;
    --port) value $# "$1"; port=$2; shift 2 ;;
    --timeout) value $# "$1"; limit=$2; shift 2 ;;
    -*) die "unknown option $1" ;;
    *) break ;;
  esac
done
[ $# -ge 2 ] || die "usage: matrix.sh [options] MODELS_DIR DEVICES... [-- CHECK_ARGS]"
dir=$1
shift
groups=
while [ $# -gt 0 ] && [ "$1" != -- ]; do groups="$groups $1"; shift; done
[ $# -gt 0 ] && shift && extra="$*" # after `--`, arguments for every `check` as they are
[ -n "$groups" ] || die "no DEVICES"
[ -d "$dir" ] || die "$dir is not a directory"
[ -x "$bin" ] || die "$bin is not built"
if [ -z "$models" ]; then
  for d in "$dir"/[!.]*/; do models="$models $(basename "$d")"; done
fi
[ -n "$log" ] || log=$(mktemp -d)
mkdir -p "$log"

# every combination of the --env settings, a line each ("VAR=a OTHER=x"); one empty line when there are none
combos() (
  prefix=$1
  shift
  if [ $# -eq 0 ]; then echo "$prefix"; return; fi
  spec=$1
  shift
  for v in $(echo "${spec#*=}" | tr ',' ' '); do combos "$prefix ${spec%%=*}=$v" "$@"; done
)

truth_of() {
  for t in $truths; do
    [ "${t%%=*}" = "$1" ] && { echo "${t#*=}"; return; }
  done
}

# the numbers of one run's lines, as the cells of its table row
summarize() {
  awk '
    function num(re,   s) { if (match($0, re)) { s = substr($0, RSTART, RLENGTH); sub(/^[^=:]*[=:] */, "", s); sub(/[% ].*$/, "", s); return s } return "-" }
    /^(PASS|FAIL) (rows|lanes) / { total++; if ($1 == "PASS") ok++ }
    /^(PASS|FAIL) accuracy prefill/ { klp = num("KL mean=[^ ]+"); top = num("top1=[^ ]+") }
    /^(PASS|FAIL) accuracy decode/ { kld = num("KL mean=[^ ]+") }
    /^PASS speed prefill 2048 / { pf2 = num("[0-9]+ tok/s") }
    /^PASS speed prefill 8192 / { pf8 = num("[0-9]+ tok/s") }
    /^PASS speed decode 1 stream, drafts off/ { d1 = num(": [0-9.]+ tok/s") }
    /^PASS speed decode 1 stream, drafts on/ { d1m = num(": [0-9.]+ tok/s") }
    /^PASS speed decode 4 streams, drafts off/ { d4 = num(": [0-9.]+ tok/s") }
    /^PASS speed decode 4 streams, drafts on/ { d4m = num(": [0-9.]+ tok/s") }
    END {
      if (!klp) klp = "-"; if (!kld) kld = "-"; if (!top) top = "-"
      if (!pf2) pf2 = "-"; if (!pf8) pf8 = "-"; if (!d1) d1 = "-"; if (!d1m) d1m = "-"; if (!d4) d4 = "-"; if (!d4m) d4m = "-"
      printf "%d/%d|%s|%s|%s|%s|%s|%s|%s|%s|%s", ok, total, klp, kld, top, pf2, pf8, d1, d1m, d4, d4m
    }' "$1"
}

rows=$log/rows.txt
: > "$rows"
bad=0
n=0
for model in $models; do
  [ -d "$dir/$model" ] || die "$dir/$model is not a model directory"
  truth=$(truth_of "$model")
  for devs in $groups; do
    tp=$(echo "$devs" | awk -F, '{print NF}')
    combos "" $envs > "$log/settings.txt"
    while IFS= read -r setting; do
      setting=$(echo $setting)
      n=$((n + 1))
      out=$log/run$n.log
      args="$dir/$model --tp $tp --port $((port + n - 1))"
      [ -n "$ids" ] && args="$args --ids $ids"
      [ -n "$truth" ] && args="$args --truth $truth"
      args="$args $speed $extra"
      r=1
      while [ "$r" -lt "$tp" ]; do
        # shellcheck disable=SC2086
        env -u ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=$devs $setting timeout -s KILL "$limit" "$bin" check $args --rank $r > "$log/run$n.rank$r.log" 2>&1 &
        r=$((r + 1))
      done
      # shellcheck disable=SC2086
      env -u ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=$devs $setting timeout -s KILL "$limit" "$bin" check $args --rank 0 > "$out" 2>&1
      code=$?
      wait
      status=PASS
      if [ "$code" -ne 0 ] || [ "$(grep -c '^FAIL ' "$out")" -ne 0 ] || ! grep -q '^check PASS' "$out"; then status=FAIL; bad=1; fi
      echo "$model|$devs|$tp|${setting:--}|$status|$(summarize "$out")" >> "$rows"
      echo "run $n: $model on $devs ${setting:+($setting) }$status, $out" >&2
    done < "$log/settings.txt"
  done
done

echo "| model | devices | tp | env | status | invariants | KL prefill | KL decode | top-1 % | prefill 2048 tok/s | prefill 8192 tok/s | decode 1 | decode 1 + drafts | decode 4 | decode 4 + drafts |"
echo "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
sed 's/|/ | /g; s/^/| /; s/$/ |/' "$rows"
exit $bad
