# Shared by run_macos.sh and run_linux.sh (sourced): one tf-mcdma-pair rank, its log and read-only host facts

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# in the repository the scripts sit at tools/zig/mcdma_pair; a bundle (with BUILD-INFO) keeps everything beside them
if [ ! -f "$here/BUILD-INFO" ] && [ -f "$here/../../../build.zig" ]; then
  repo="$(cd "$here/../../.." && pwd)"
  results_default="$repo/build/mcdma-pair/results"
else
  repo=""
  results_default="$here/results"
fi

die() { echo "tf-mcdma-pair: $*" >&2; exit 2; }

sha256() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

# The binary for this host: TF_MCDMA_PAIR_BIN, the bundle's bin/<arch>-<os>/, then the repository's zig-out.
find_bin() {
  local os="$1" arch triple
  arch="$(uname -m)"
  [ "$arch" = arm64 ] && arch=aarch64
  if [ "$os" = macos ]; then triple="$arch-macos"; else triple="$arch-linux-gnu"; fi
  for b in "${TF_MCDMA_PAIR_BIN:-}" "$here/bin/$triple/tf-mcdma-pair" ${repo:+"$repo/zig-out/bin/mcdma-pair/$triple/tf-mcdma-pair"} ${repo:+"$repo/zig-out/bin/tf-mcdma-pair"}; do
    if [ -n "$b" ] && [ -x "$b" ]; then echo "$b"; return 0; fi
  done
  die "no tf-mcdma-pair for $triple: set TF_MCDMA_PAIR_BIN, or build it (zig build tf-mcdma-pair / tf-mcdma-pair-dist)"
}

# One field of a settings JSON (python3 is on the Macs and the Linux hosts alike).
field() {
  python3 -c 'import json,sys
d=json.load(open(sys.argv[1]))
v=d
for k in sys.argv[2].split("."):
    v=v[int(k)] if isinstance(v,list) else v.get(k)
    if v is None: break
print("" if v is None else v)' "$1" "$2"
}

# A copy of the settings with "library" absolute: $BUNDLE/ or a relative path means beside these scripts.
resolve() {
  python3 -c 'import json,os,sys
d=json.load(open(sys.argv[1]))
lib=d.get("library","")
if lib.startswith("$BUNDLE/"): lib=os.path.join(sys.argv[3], lib[len("$BUNDLE/"):])
elif lib and not os.path.isabs(lib): lib=os.path.join(sys.argv[3], lib)
d["library"]=lib
json.dump(d,open(sys.argv[2],"w"),indent=1)' "$1" "$2" "$here"
}

# An offline library's view of the port (the stub's ACTIVE at 4096 unless STUB_PORT_DOWN); fails (1) for a real one.
facts_offline() {
  local lib="$1" device="$2"
  case "$(basename "$lib")" in
    libmcdma-fabric-stub.*)
      if [ "${STUB_PORT_DOWN:-}" = "$device" ]; then echo "port_state DOWN (MCDMA's stub verbs: STUB_PORT_DOWN=$device)"; else echo "port_state ACTIVE (MCDMA's stub verbs)"; fi
      echo "active_mtu 4096 (MCDMA's stub verbs)"
      ;;
    libloop-fabric.*) echo "loop_fabric (not MCDMA): no RDMA port" ;;
    *) return 1 ;;
  esac
}

# "yes" when neighbour lines on stdin (ndp -an, ip -6 neigh show) hold GID at MAC as a permanent entry, else "no".
neighbour_has() {
  python3 -c 'import ipaddress, sys
def mac(m):
    try:
        b = [int(x, 16) for x in m.split(":")]
    except ValueError:
        return None
    return b if len(b) == 6 else None
gid, want = ipaddress.ip_address(sys.argv[1]), mac(sys.argv[2].lower())
for line in sys.stdin:
    w = line.lower().split()
    try:
        if not w or ipaddress.ip_address(w[0].split("%")[0]) != gid:
            continue
    except ValueError:
        continue
    if "permanent" in w and any(mac(x) == want for x in w[1:]):
        print("yes")
        sys.exit()
print("no")' "$1" "$2"
}

# The value of an override if the command line has one, else the settings field.
setting() {
  local file="$1" key="$2" flag="$3"; shift 3
  local prev=""
  for a in "$@"; do
    if [ "$prev" = "$flag" ]; then echo "$a"; return; fi
    prev="$a"
  done
  field "$file" "$key"
}

run_side() {
  local os="$1"; shift
  local pair=0 settings settings1="" bin dir rank label lib device via stamp outer
  if [ "${1:-}" = --pair ]; then
    pair=1; settings="${2:?--pair RANK0.json RANK1.json}"; settings1="${3:?--pair RANK0.json RANK1.json}"; shift 3
  else
    settings="${1:?usage: $(basename "$0") SETTINGS.json [overrides...]}"; shift
  fi
  [ -f "$settings" ] || die "no settings file $settings"
  bin="$(find_bin "$os")"
  rank="$(field "$settings" rank)"
  label="$(setting "$settings" label --label "$@")"
  [ -n "$label" ] || label=run
  device="$(field "$settings" links.0.device)"
  via="$(field "$settings" links.0.via)"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  local tag="rank$rank"
  [ "$pair" = 1 ] && tag=pair
  dir="${TF_MCDMA_PAIR_OUT:-${TF_MCDMA_PAIR_RESULTS:-$results_default}/$stamp-$label-$tag-$(hostname -s)}"
  mkdir -p "$dir"
  resolve "$settings" "$dir/settings.json"
  settings="$dir/settings.json"
  if [ "$pair" = 1 ]; then
    resolve "$settings1" "$dir/settings1.json"
    settings1="$dir/settings1.json"
  fi
  lib="$(field "$settings" library)"
  {
    echo "date_utc $stamp"
    echo "host $(hostname)"
    echo "uname $(uname -a)"
    echo "binary $bin"
    echo "binary_sha256 $(sha256 "$bin")"
    echo "library $lib"
    if [ -f "$lib" ]; then echo "library_sha256 $(sha256 "$lib")"; else echo "library_sha256 MISSING"; fi
    if [ -f "$here/BUILD-INFO" ]; then sed 's/^/build /' "$here/BUILD-INFO"; fi
    if [ -n "$repo" ] && command -v git >/dev/null 2>&1; then echo "commit $(git -C "$repo" rev-parse HEAD 2>/dev/null) $(git -C "$repo" status --porcelain 2>/dev/null | wc -l | tr -d ' ') uncommitted"; fi
    printf 'argv'; printf ' %q' "$bin" "$@"; echo
    "facts_$os" "$settings" "$device" "$via" "$@" 2>&1 | sed 's/^/fact /'
  } >"$dir/meta.txt"
  local warns
  warns="$(grep -c '^fact WARN' "$dir/meta.txt" || true)"
  if [ "${warns:-0}" -gt 0 ]; then
    echo "== tf-mcdma-pair pre-flight: $warns WARN(s) on $(hostname -s); a run with these may fail, or prove less than it says" >&2
    echo "   (compare.py fails the gate on a kext, relaxed ordering or clock WARN):" >&2
    grep '^fact WARN' "$dir/meta.txt" | sed 's/^fact /  /' >&2 || true
  fi
  [ -f "$lib" ] || die "the library $lib is missing (build it with build_mcdma.sh; README.md)"
  outer="${TF_MCDMA_PAIR_OUTER_S:-900}"
  local log="$dir/rank$rank.log" exitf="$dir/rank$rank.exit"
  echo "== tf-mcdma-pair $label ($tag) on $(hostname -s): $dir" >&2
  rm -f "$dir/pid" "$exitf"
  if [ "$pair" = 1 ]; then
    log="$dir/pair.stdout.log"
    {
      "$bin" --pair "$dir" "$settings" "$settings1" "$@" 2>&1 &
      echo $! >"$dir/pid"
      local c=0
      wait $! || c=$?
      echo "$c" >"$dir/pair.exit"
    } | tee "$log" >&2 &
  else
    {
      "$bin" "$settings" "$@" 2>&1 &
      echo $! >"$dir/pid"
      local c=0
      wait $! || c=$?
      echo "$c" >"$exitf"
    } | tee "$log" >&2 &
  fi
  local reader=$! waited=0 done_file="$exitf"
  [ "$pair" = 1 ] && done_file="$dir/pair.exit"
  # the outer guard: only for a process its own watchdog cannot reach (stopped, or stuck in the kernel)
  while [ ! -s "$done_file" ]; do
    sleep 1
    waited=$((waited + 1))
    if [ "$waited" -ge "$outer" ]; then
      local pid
      pid="$(cat "$dir/pid" 2>/dev/null || true)"
      [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null || true
      echo "RESULT status=HANG code=124 error=OuterTimeout after=$waited expected=no mcdma=\"the run script killed it after TF_MCDMA_PAIR_OUTER_S=$outer s\"" >>"$log"
      sleep 1
      [ -s "$done_file" ] || echo 124 >"$done_file"
      break
    fi
  done
  wait "$reader" 2>/dev/null || true
  local code
  code="$(cat "$done_file")"
  echo "== exit $code; results in $dir" >&2
  if [ "$pair" = 1 ]; then
    grep -h '^RESULT' "$dir"/rank*.log 2>/dev/null | sed 's/^/   /' >&2 || true
  else
    grep '^RESULT' "$log" | sed 's/^/   /' >&2 || true
  fi
  echo "$dir"
  return 0
}
