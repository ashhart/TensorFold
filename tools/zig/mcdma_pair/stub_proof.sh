#!/usr/bin/env bash
# tf-mcdma-pair proven offline on this host through the hardware window's own scripts; its cases: README.md
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
# the exchange meets on the loopback: lo0's link-local address on a Mac, lo's IPv4 address on Linux (which has none)
os=linux
lo=lo/127.0.0.1
if [ "$(uname -s)" = Darwin ]; then os=macos; lo=lo0; fi
run="$here/run_$os.sh"
if [ ! -f "$here/BUILD-INFO" ] && [ -f "$here/../../../build.zig" ]; then base="$(cd "$here/../../.." && pwd)/build/mcdma-pair"; else base="$here/proof"; fi
proof="${1:-$base/proof-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$proof"
lib="$("$here/build_mcdma.sh" --stub "${TF_MCDMA_SRC:-$here/mcdma}" "$proof/lib")"
echo "stub library: $lib" >&2
loop="$("$here/build_mcdma.sh" --loop "${TF_MCDMA_SRC:-$here/mcdma}" "$proof/lib")"
echo "loop library: $loop" >&2
# loop_fabric's two ends of this proof know each other by this
LOOP_FABRIC_TOKEN="stub-proof-$$-$(date +%s)"
export LOOP_FABRIC_TOKEN

# A base port P with P .. P+N-1 free on the loopback (nothing listening or bound), so no run reaches another's socket.
free_ports() {
  python3 - "$1" "$$" <<'EOF'
import socket, sys
n, pid = int(sys.argv[1]), int(sys.argv[2])
def free(p):
    for fam, addr in ((socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")):
        c = socket.socket(fam, socket.SOCK_STREAM)
        c.settimeout(0.5)
        try:
            if c.connect_ex((addr, p)) == 0:
                return False
        except OSError:
            pass
        finally:
            c.close()
    for fam, addr in ((socket.AF_INET, "0.0.0.0"), (socket.AF_INET6, "::")):
        for kind in (socket.SOCK_STREAM, socket.SOCK_DGRAM):
            s = socket.socket(fam, kind)
            try:
                s.bind((addr, p))
            except OSError:
                return False
            finally:
                s.close()
    return True
for k in range(300):
    base = 30000 + ((pid + 37 * k) % 290) * 100
    if all(free(p) for p in range(base, base + n)):
        print(base)
        sys.exit()
sys.exit("stub_proof.sh: no block of %d free loopback ports between 30000 and 59000" % n)
EOF
}
port="$(free_ports 500)"
echo "ports $port .. $((port + 499)), checked free" >&2

# rank settings for a device pair: NAME DEV0 DEV1 KIND MTU PORT
make_pair() {
  python3 - "$proof/$1" "$lib" "$lo" "$2" "$3" "$4" "$5" "$6" <<'EOF'
import json, os, sys
out, lib, lo, d0, d1, kind, mtu, port = sys.argv[1:]
os.makedirs(out, exist_ok=True)
port = int(port)
for rank, dev in ((0, d0), (1, d1)):
    link = {"peer": 1 - rank, "device": dev, "via": lo, "port": port + 4 * rank, "peer_port": port + 4 * (1 - rank),
            "name": "stubpair", "gid": 1, "mtu": int(mtu)}
    s = {"rank": rank, "library": lib, "label": os.path.basename(out), "links": [link], "rounds": 2}
    if kind != "any":
        link["kind"] = kind
        s["expect_kind"] = kind
    json.dump(s, open(os.path.join(out, "r%d.json" % rank), "w"), indent=1)
EOF
}

make_pair cx5-1024 roce0 roce1 cx5 1024 $port
make_pair cx5-4096 roce0 roce1 any 4096 $((port + 100))
make_pair tb5 tb0 tb1 tb5 4096 $((port + 200))

results=()
OFFLINE=(--same-host --stub --allow-warn)
# COMPARE-ARGS...: compare.py (or $COMPARE) passes the offline proof, and says it is not the gate (exit 5)
offline_pass() {
  local got=0
  python3 "${COMPARE:-$here/compare.py}" "$@" || got=$?
  [ "$got" = 5 ]
}
# OUT: the verdict line an offline pass prints, and never GATE PASS
offline_said() {
  grep -qx 'OFFLINE PASS (not the gate)' "$1" && ! grep -q '^GATE PASS' "$1"
}
# CASE NAME COMPARE-ARGS -- RUN-ARGS: one run (label stub-NAME) through the run script, then compare.py on its directory
case_run() {
  local name="$1"; shift
  local cmp=()
  while [ "$1" != -- ]; do cmp+=("$1"); shift; done
  shift
  local dir="$proof/run-$name"
  mkdir -p "$dir"
  echo "--- $name" >&2
  TF_MCDMA_PAIR_OUT="$dir" "$run" "$@" --label "stub-$name" >/dev/null 2>"$dir/run.stderr" || true
  if offline_pass "$dir" "${OFFLINE[@]}" ${cmp[@]+"${cmp[@]}"} >"$dir/compare.txt" 2>&1 && offline_said "$dir/compare.txt"; then
    results+=("PASS  $name")
  else
    results+=("FAIL  $name (see $dir/compare.txt)")
  fi
  sed 's/^/    /' "$dir/compare.txt" >&2
}

# NAME 'PATTERN;;PATTERN..' COMPARE-ARGS...: compare.py (or $COMPARE) must say GATE FAIL (exit 1), for each reason
must_fail() {
  local name="$1" pats="$2"; shift 2
  local out="$proof/$name.compare.txt" missing="" got=0
  python3 "${COMPARE:-$here/compare.py}" "$@" >"$out" 2>&1 || got=$?
  if [ "$got" != 1 ]; then
    results+=("FAIL  $name: compare.py did not fail it, exit $got (see $out)")
  else
    local rest="$pats" pat
    while [ -n "$rest" ]; do
      pat="${rest%%;;*}"
      if [ "$pat" = "$rest" ]; then rest=""; else rest="${rest#*;;}"; fi
      grep -qE -- "$pat" "$out" || missing="$missing [$pat]"
    done
    if [ -z "$missing" ]; then results+=("PASS  $name: compare.py refuses it"); else results+=("FAIL  $name: refused, but not for$missing (see $out)"); fi
  fi
  echo "--- $name (must fail)" >&2
  sed 's/^/    /' "$out" >&2
}

# NAME CODE COMPARE-ARGS...: compare.py must exit CODE (3 PARTIAL, 4 N/A)
verdict_is() {
  local name="$1" want="$2"; shift 2
  local out="$proof/$name.compare.txt" got=0
  python3 "$here/compare.py" "$@" >"$out" 2>&1 || got=$?
  if [ "$got" = "$want" ]; then results+=("PASS  $name: compare.py exits $got"); else results+=("FAIL  $name: compare.py exits $got, not $want (see $out)"); fi
  echo "--- $name (exit $want)" >&2
  sed 's/^/    /' "$out" >&2
}

P="$proof"
case_run cx5-1024 --kind cx5 -- --pair "$P/cx5-1024/r0.json" "$P/cx5-1024/r1.json"
case_run cx5-4096 --kind cx5 --same-as "$P/run-cx5-1024" -- --pair "$P/cx5-4096/r0.json" "$P/cx5-4096/r1.json"
case_run tb5 --kind tb5 --same-as "$P/run-cx5-1024" -- --pair "$P/tb5/r0.json" "$P/tb5/r1.json"
# a 500 ms pause before the victim acts: the stub aborts on a RoCE write to a registration already gone
case_run cx5-exit-rank1 -- --pair "$P/cx5-1024/r0.json" "$P/cx5-1024/r1.json" --arm exit --arm-rank 1 --arm-test allreduce_bf16 --arm-step 3 --arm-delay-ms 500
case_run cx5-exit-rank0 -- --pair "$P/cx5-1024/r0.json" "$P/cx5-1024/r1.json" --arm exit --arm-rank 0 --arm-test bulk --arm-step 1 --arm-delay-ms 500
case_run cx5-stop -- --pair "$P/cx5-1024/r0.json" "$P/cx5-1024/r1.json" --arm stop --arm-rank 0 --arm-test pipes --arm-step 40
case_run cx5-hang -- --pair "$P/cx5-1024/r0.json" "$P/cx5-1024/r1.json" --arm hang --arm-rank 1 --arm-test bulk --arm-step 1 --watchdog-s 13
case_run tb5-exit -- --pair "$P/tb5/r0.json" "$P/tb5/r1.json" --arm exit --arm-rank 1 --arm-test allreduce_bf16 --arm-step 3 --arm-delay-ms 500
case_run tb5-stop -- --pair "$P/tb5/r0.json" "$P/tb5/r1.json" --arm stop --arm-rank 0 --arm-test pipes --arm-step 40
case_run tb5-hang -- --pair "$P/tb5/r0.json" "$P/tb5/r1.json" --arm hang --arm-rank 1 --arm-test bulk --arm-step 1 --watchdog-s 13
# both ends must agree on the run: rank 1 asks for another round count, and both refuse by name at the handshake
python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); d["rounds"]=3; json.dump(d,open(sys.argv[2],"w"))' "$P/cx5-1024/r1.json" "$P/cx5-1024/r1-rounds3.json"
case_run settings-mismatch -- --pair "$P/cx5-1024/r0.json" "$P/cx5-1024/r1-rounds3.json" --expect SettingsMismatch
# named startup failures, one rank alone (as G5 runs on hardware)
python3 - "$P/cx5-1024/r0.json" "$P" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
d["links"][0]["gid"] = -1
json.dump(d, open(sys.argv[2] + "/gid-scan-cx5.json", "w"))
d["links"][0].pop("kind"); d.pop("expect_kind")
json.dump(d, open(sys.argv[2] + "/gid-scan-any.json", "w"))
d = json.load(open(sys.argv[1]))
d["links"][0]["device"] = "roce1"
d["links"][0]["mtu"] = 4096
json.dump(d, open(sys.argv[2] + "/mtu-4096-alone.json", "w"))
EOF
export STUB_PORT_DOWN=roce0
case_run port-down -- "$P/cx5-1024/r0.json" --connect-timeout-s 2 --expect OpenFailed --expect-log "MCDMA status 4 DEVICE"
unset STUB_PORT_DOWN
case_run gid-scan-cx5 -- "$P/gid-scan-cx5.json" --connect-timeout-s 2 --expect NeedGidIndex
# G5c's offline twin: MCDMA's own "<device>: gid -1", and never "port not active"
G5C=(--connect-timeout-s 2 --expect OpenFailed --expect-log "MCDMA status 4 DEVICE" --expect-log "mcdma-fabric: roce0: gid -1" --refuse-log "port not active")
case_run gid-scan-any -- "$P/gid-scan-any.json" "${G5C[@]}"
case_run no-peer -- "$P/cx5-1024/r0.json" --connect-timeout-s 2 --expect ConnectFailed --expect-log "MCDMA status 6 TIMEOUT"
case_run mtu-1500 -- "$P/cx5-1024/r0.json" --mtu 1500 --expect InvalidConfig
if [ "$os" = macos ]; then
  # the Mac's kext takes 64 registrations per process: 300 MiB plus staging is 78 pieces, refused by name first
  case_run mac-regions -- "$P/cx5-1024/r0.json" --window-mib 300 --connect-timeout-s 2 --expect TooManyRegions
fi

# compare.py must refuse these: a run of one test is no gate (and --partial's verdict is PARTIAL, exit 3, never 0)
mkdir -p "$proof/run-neg-partial"
TF_MCDMA_PAIR_OUT="$proof/run-neg-partial" "$run" --pair "$P/cx5-1024/r0.json" "$P/cx5-1024/r1.json" --label stub-neg-partial --tests barrier --rounds 1 >/dev/null 2>&1 || true
must_fail neg-partial 'the whole suite' "$proof/run-neg-partial" --kind cx5 "${OFFLINE[@]}"
verdict_is neg-partial-exit 3 "$proof/run-neg-partial" --kind cx5 --partial "${OFFLINE[@]}"
# G5a counts only on an ACTIVE port whose active MTU is below 4096: the stub's is 4096 (N/A), and a down port fails it
G5A=(--connect-timeout-s 2 --expect OpenFailed --expect-log "MCDMA status 4 DEVICE" --expect-log "path MTU above active MTU")
mkdir -p "$proof/run-g5a-na" "$proof/run-neg-g5a-port-down" "$proof/run-neg-g5c-port-down"
TF_MCDMA_PAIR_OUT="$proof/run-g5a-na" "$run" "$P/mtu-4096-alone.json" "${G5A[@]}" --label stub-g5a >/dev/null 2>&1 || true
verdict_is g5a-na 4 "$proof/run-g5a-na" "${OFFLINE[@]}"
STUB_PORT_DOWN=roce1 TF_MCDMA_PAIR_OUT="$proof/run-neg-g5a-port-down" "$run" "$P/mtu-4096-alone.json" "${G5A[@]}" --label stub-g5a >/dev/null 2>&1 || true
must_fail neg-g5a-port-down 'FAIL .*shows the port ACTIVE \(port_state DOWN' "$proof/run-neg-g5a-port-down" "${OFFLINE[@]}"
STUB_PORT_DOWN=roce0 TF_MCDMA_PAIR_OUT="$proof/run-neg-g5c-port-down" "$run" "$P/gid-scan-any.json" "${G5C[@]}" --label stub-g5c >/dev/null 2>&1 || true
must_fail neg-g5c-port-down 'FAIL .*never says "port not active";;FAIL .*says "mcdma-fabric: roce0: gid -1"' "$proof/run-neg-g5c-port-down" "${OFFLINE[@]}"
# --label over two trees judges what it was given: one tree's failing run is never dropped silently (repro s-g5two)
TF_MCDMA_PAIR_RESULTS="$proof/trees-b" "$run" "$P/gid-scan-any.json" "${G5C[@]}" --label stub-g5c-trees --expect ConnectFailed >/dev/null 2>&1 || true
sleep 1
TF_MCDMA_PAIR_RESULTS="$proof/trees-a" "$run" "$P/gid-scan-any.json" "${G5C[@]}" --label stub-g5c-trees >/dev/null 2>&1 || true
must_fail neg-label-two-trees 'FAIL .*left out 1 log\(s\) of that label from another host or tree' --label stub-g5c-trees "$proof/trees-a" "$proof/trees-b" "${OFFLINE[@]}"

# the hardware flow in two processes over loop_fabric (not MCDMA): settings.py's runs as PLAN.txt runs them
two="$proof/two"
python3 - "$two/hosts.json" "$loop" "$lo" "$os" $((port + 300)) <<'EOF'
import json, os, sys
out, lib, lo, osname, port = sys.argv[1:]
os.makedirs(os.path.dirname(out), exist_ok=True)
# restore_rdma and native_check: only PLAN's prep lines name them (a Mac's are required), and nothing here runs those
host = lambda dev, n: {"os": osname, "library": lib, "device": dev, "gid": 0, "via": lo, "netdev": "lo",
                       "mac": "02:00:00:00:00:0%d" % n, "gid_addr": "fe80::ff:fe00:%d" % n,
                       "restore_rdma": "/opt/mcdma/tools/restore-rdma.py", "native_check": "/opt/mcdma/build/cx5-native-check"}
hosts = {"a": host("roce0", 1), "b": host("roce1", 2)}
hosts["b"]["free"] = True  # a link drop (G4) is loop_fabric shutting its own socket
json.dump({"pair": "loop", "label_prefix": "loop-", "mtu": 1024, "kind": "cx5", "window_mib": 96, "rounds": 2,
           "port": int(port), "name": "loop1594", "hosts": hosts}, open(out, "w"), indent=1)
EOF
python3 "$here/settings.py" "$two/hosts.json" "$two/settings" >/dev/null
plan="$two/settings/loop"
# TREE RUN 'ENV_A' 'ENV_B' 'ARGS_A' 'ARGS_B': both hosts' lines at once into TREE-a and TREE-b, each with its env
two_run() {
  local tree="$1" name="$2" env_a="$3" env_b="$4" args_a="$5" args_b="$6"
  echo "--- two processes: $name into $tree-a, $tree-b${env_b:+ (b: $env_b)}" >&2
  # shellcheck disable=SC2086
  env $env_a TF_MCDMA_PAIR_RESULTS="$two/$tree-a" "$run" "$plan/loop-$name/a.json" $args_a >/dev/null 2>>"$two/$tree.$name.a.stderr" &
  local pa=$!
  # shellcheck disable=SC2086
  env $env_b TF_MCDMA_PAIR_RESULTS="$two/$tree-b" "$run" "$plan/loop-$name/b.json" $args_b >/dev/null 2>>"$two/$tree.$name.b.stderr" || true
  wait "$pa" || true
}
LOOPCMP=(--kind cx5 --settings "$plan" "${OFFLINE[@]}")
mkdir -p "$two/results-b"
for name in G1 G2 G3a G3b G3d G4 G3c; do
  case "$name" in
    # the operator's drop, played by loop_fabric: b's socket shuts 4 s in, its time where PLAN's date line puts it
    G4) two_run results "$name" "" "LOOP_FAULT_FLAP_MS=4000 LOOP_FAULT_DROPPED_AT=$two/results-b/loop-G4.dropped_at" "" "" ;;
    *) two_run results "$name" "" "" "" "" ;;
  esac
  # PLAN.txt's compare line, offline
  if offline_pass --label "loop-$name" "${LOOPCMP[@]}" "$two/results-a" "$two/results-b" >"$two/$name.compare.txt" 2>&1 &&
     offline_said "$two/$name.compare.txt"; then
    results+=("PASS  two processes loop-$name"); else results+=("FAIL  two processes loop-$name (see $two/$name.compare.txt)"); fi
  sed 's/^/    /' "$two/$name.compare.txt" >&2
done
# PLAN's Thunderbolt line as written, by a compare.py beside the plan's settings as in a bundle (review repro s-tbline)
cp "$here/compare.py" "$here/compare_log.py" "$two/"
# G1's suite as PLAN says, at G1's window and rounds (compare.py holds the two runs' SETTINGS to each other)
g1_window="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["window_mib"])' "$plan/loop-G1/a.json")"
TF_MCDMA_PAIR_OUT="$two/tb-results/tb" "$run" --pair "$P/tb5/r0.json" "$P/tb5/r1.json" --label loop-tb-G1 --window-mib "$g1_window" >/dev/null 2>"$two/tb.stderr" || true
# TREE [TB]: PLAN's line, with "results results-b" as $two/TREE-a and TREE-b, and TB-RESULTS as $two/TB (tb-results)
tb_line() {
  local w words=()
  read -ra words <<<"$(grep -E '^  \./compare\.py --label loop-tb-G1 ' "$plan/PLAN.txt" || true)"
  tbargs=()
  for w in ${words[@]+"${words[@]}"}; do
    case "$w" in
      ./compare.py) ;;
      TB-RESULTS...) tbargs+=("$two/${2:-tb-results}") ;;
      results) tbargs+=("$two/$1-a") ;;
      results-b) tbargs+=("$two/$1-b") ;;
      "results/"*) tbargs+=("$two/receipts-$1-${2:-tb-results}/${w#results/}") ;; # its --receipt, one directory per variant
      *) tbargs+=("$w") ;;
    esac
  done
}
tb_line results
echo "--- two processes: PLAN's Thunderbolt line: ${tbargs[*]-none}" >&2
if COMPARE="$two/compare.py" offline_pass ${tbargs[@]+"${tbargs[@]}"} "${OFFLINE[@]}" >"$two/tbline.compare.txt" 2>&1 &&
   offline_said "$two/tbline.compare.txt" &&
   [ "$(grep -cE "^ok    rank [01] .*the plan's settings \(.*loop-G1/[ab]\.json\)" "$two/tbline.compare.txt")" = 2 ] &&
   grep -q '^verdict: OFFLINE PASS (not the gate' "$two/receipts-results-tb-results/loop-tb-G1.receipt.md"; then
  results+=("PASS  two processes: PLAN's Thunderbolt line (loop-tb-G1 against loop-G1, held to its plan, its receipt written)")
else
  results+=("FAIL  two processes: PLAN's Thunderbolt line (see $two/tbline.compare.txt)")
fi
sed 's/^/    /' "$two/tbline.compare.txt" >&2
# offline logs are never the gate: PLAN's compare line, without the offline flags, refuses them (repro s-stale)
must_fail neg-offline-as-gate 'FAIL .*both ranks report host;;FAIL .*not MCDMA.s libmcdma-fabric' --label loop-G1 --kind cx5 "$two/results-a" "$two/results-b"
# G1 again: host a's first G1 log with host b's second must not pair (their RUN ids differ)
first_a="$(ls -d "$two/results-a/"*-loop-G1-rank0-* | head -1)"
two_run results G1 "" "" "" ""
must_fail neg-stale-pair 'FAIL  one run: RUN id' "$first_a" "$(ls -d "$two/results-b/"*-loop-G1-rank1-* | tail -1)" "${LOOPCMP[@]}"
# G4 judged against a drop time a minute off: the failures are not the operator's drop
dropped="$(cat "$two/results-b/loop-G4.dropped_at" 2>/dev/null || echo 0)"
must_fail neg-g4-wrong-time 'FAIL .*after the drop' --label loop-G4 "${LOOPCMP[@]}" --dropped-at $((dropped - 60)) "$two/results-a" "$two/results-b"
# s-g3b-silent, the first repro that once printed GATE PASS (each in its own trees): the exit arm's BYE never arrives
two_run g3b-silent G3b "LOOP_FAULT_SILENT_EOF=1" "LOOP_FAULT_NO_BYE=1" "" ""
must_fail neg-g3b-silent "FAIL .*the exit arm's reason" --label loop-G3b "${LOOPCMP[@]}" "$two/g3b-silent-a" "$two/g3b-silent-b"
# s-g3d-flap: the stopped rank's link drops during its hold: the survivor's PeerTimeout is not with the link up
two_run g3d-flap G3d "" "LOOP_FAULT_FLAP_MS=1000" "" ""
must_fail neg-g3d-flap "FAIL .*the stop arm's reason" --label loop-G3d "${LOOPCMP[@]}" "$two/g3d-flap-a" "$two/g3d-flap-b"
# s-g4-flap: the link drops on its own 3 s in, with no operator and no recorded drop time
two_run g4-flap G4 "" "LOOP_FAULT_FLAP_MS=3000" "" ""
must_fail neg-g4-flap "FAIL .*the operator's drop time: none" --label loop-G4 "${LOOPCMP[@]}" "$two/g4-flap-a" "$two/g4-flap-b"
# s-sig: one intermediate signal of rank 1's storm corrupted (and left in place 200 ms); the last one lands right
two_run sig G1 "" "LOOP_FAULT_SIGNAL_NTH=3000 LOOP_FAULT_SIGNAL_HOLD_MS=200" "" ""
must_fail neg-sig 'MISMATCH test=signals;;FAIL .*rank 1 .*each equal to the host reference' --label loop-G1 "${LOOPCMP[@]}" "$two/sig-a" "$two/sig-b"
# s-expect: one rank expects a failure the other does not, and the other's data is wrong: refused at the handshake
two_run expect G1 "" "LOOP_FAULT_CORRUPT_NTH=1 LOOP_FAULT_CORRUPT_MINLEN=1000000" "--expect PeerTimeout" ""
must_fail neg-expect-one-rank 'FAIL .*both ranks expect the same failure' --label loop-G1 "${LOOPCMP[@]}" "$two/expect-a" "$two/expect-b"
# s-small: G1 with smaller settings (rounds, window, bulk) than the plan's
two_run small G1 "" "" "--rounds 1 --bulk-mib 1 --window-mib 0" "--rounds 1 --bulk-mib 1 --window-mib 0"
must_fail neg-small 'FAIL .*the plan.s settings .*differ: .*rounds=1' --label loop-G1 "${LOOPCMP[@]}" "$two/small-a" "$two/small-b"
# PLAN's Thunderbolt line against that small G1: loop-G1's logs are held to their plan, so it is refused for that too
tb_line small
COMPARE="$two/compare.py" must_fail neg-tb-small 'FAIL .*the plan.s settings .*loop-G1/[ab]\.json.*differ: .*rounds=1' ${tbargs[@]+"${tbargs[@]}"} "${OFFLINE[@]}"
# a-tbline: a Thunderbolt run of one round at the suite's own window has G1's hashes, but it is not G1's suite
TF_MCDMA_PAIR_OUT="$two/tb-short/tb" "$run" --pair "$P/tb5/r0.json" "$P/tb5/r1.json" --label loop-tb-G1 --rounds 1 >/dev/null 2>"$two/tb-short.stderr" || true
tb_line results tb-short
COMPARE="$two/compare.py" must_fail neg-tb-short 'FAIL  --same-as: .*only in the other run;;FAIL  --same-as: both runs. SETTINGS .*rounds=2 \(1\), window=' ${tbargs[@]+"${tbargs[@]}"} "${OFFLINE[@]}"
# a-g3d-lost39: b's pipes message of step 39 lost: a survivor timing out before b's stop at step 40 is not the arm's
two_run g3d-lost G3d "" "LOOP_FAULT_DROP_LEN=$(( (39 * 977 + 13) % 65536 ))" "" ""
must_fail neg-g3d-lost 'FAIL .*failed at pipes step 39, at or after the arm.s step 40' --label loop-G3d "${LOOPCMP[@]}" "$two/g3d-lost-a" "$two/g3d-lost-b"
# a-g5c-nolog: G5c without the plan's expect_log and refuse_log, on a port that is down: the plan's check refuses it
python3 - "$plan/loop-G5c/a.json" "$lib" "$two" <<'EOF2'
import json, sys
d = json.load(open(sys.argv[1]))
d["library"] = sys.argv[2]  # MCDMA's stub, whose "<device>: gid -1" the plan expects (loop_fabric logs none)
json.dump(d, open(sys.argv[3] + "/g5c-plan.json", "w"), indent=1)
d.pop("expect_log"), d.pop("refuse_log")
json.dump(d, open(sys.argv[3] + "/g5c-nolog.json", "w"), indent=1)
EOF2
TF_MCDMA_PAIR_RESULTS="$two/g5c-plan" "$run" "$two/g5c-plan.json" >/dev/null 2>&1 || true
if offline_pass --label loop-G5c "${LOOPCMP[@]}" "$two/g5c-plan" >"$two/g5c-plan.compare.txt" 2>&1 && offline_said "$two/g5c-plan.compare.txt"; then
  results+=("PASS  loop-G5c from its plan file, over MCDMA's stub"); else results+=("FAIL  loop-G5c from its plan file (see $two/g5c-plan.compare.txt)"); fi
sed 's/^/    /' "$two/g5c-plan.compare.txt" >&2
STUB_PORT_DOWN=roce0 TF_MCDMA_PAIR_RESULTS="$two/g5c-nolog" "$run" "$two/g5c-nolog.json" >/dev/null 2>&1 || true
must_fail neg-g5c-nolog 'FAIL .*the plan.s settings .*differ: expect_log=\[\] .*refuse_log=\[\]' --label loop-G5c "${LOOPCMP[@]}" "$two/g5c-nolog"
# the pre-flight WARNs no gate passes with, in a copy of cx5-1024's meta.txt: each fails it, unless --allow-warn
cp -R "$proof/run-cx5-1024" "$proof/run-neg-warn"
{
  echo "fact WARN the loaded MCDMA kext is 00000000-0000-0000-0000-000000000001, not the settings' expected_kext_uuid 00000000-0000-0000-0000-000000000002"
  echo "fact WARN relaxed ordering is on for mcrdma0: a signal written after its data may land before it"
  echo "fact WARN the clock is 1.5 s off time.apple.com: compare.py's cross-host windows (G3, G4) allow only 2 s"
  echo "fact WARN the clock is not NTP-synchronized (timedatectl): compare.py's cross-host windows (G3, G4) allow only 2 s"
} >>"$proof/run-neg-warn/meta.txt"
must_fail neg-warn 'FAIL .*pre-flight WARN \(not the kext the settings expect;;FAIL .*pre-flight WARN \(relaxed ordering;;FAIL .*pre-flight WARN \(the clock: .*1\.5 s off;;FAIL .*pre-flight WARN \(the clock: .*not NTP-synchronized' "$proof/run-neg-warn" --kind cx5 --same-host --stub
verdict_is warn-allowed 5 "$proof/run-neg-warn" --kind cx5 "${OFFLINE[@]}"

echo
echo "tf-mcdma-pair over MCDMA's stub verbs, and two processes over loop_fabric ($os): $proof"
bad=0
for r in "${results[@]}"; do
  echo "  $r"
  case "$r" in FAIL*) bad=$((bad + 1)) ;; esac
done
[ "$bad" = 0 ] && echo "all ${#results[@]} cases pass" || echo "$bad case(s) failed"
exit "$bad"
