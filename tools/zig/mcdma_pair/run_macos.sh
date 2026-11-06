#!/usr/bin/env bash
# The macOS rank of the #1594 gate: this Mac's side from a settings JSON, with read-only facts (README.md)
set -euo pipefail
# shellcheck source=common.sh
. "$(cd "$(dirname "$0")" && pwd)/common.sh"

# Each MCDMACX5Native instance as one line: its MCDMAAddressInterface, then the properties the gate depends on.
mcdma_ports() {
  ioreg -r -c MCDMACX5Native -l 2>/dev/null | python3 -c 'import re, sys
blocks = []
for line in sys.stdin:
    if re.match(r"^\+-o MCDMACX5Native", line):
        blocks.append({})
        continue
    m = re.search(r"\"(MCDMA\w+)\" = (.*)$", line)
    if m and blocks and m.group(1) not in blocks[-1]:
        blocks[-1][m.group(1)] = m.group(2).strip().strip("\"")
for b in blocks:
    keys = ("PortActive", "GidLive", "RelaxedOrdering", "RelaxedOrderingKeys", "NativeBuild", "EthernetMTU")
    print(b.get("MCDMAAddressInterface", "?"), " ".join("%s=%s" % (k, b.get("MCDMA" + k, "-")) for k in keys),
          "pcie=\"%s\"" % b.get("MCDMAPCIePath", "?"))'
}

facts_macos() {
  local settings="$1" device="$2" via="$3"; shift 3
  local mtu iface lib
  mtu="$(setting "$settings" links.0.mtu --mtu "$@")"
  iface="${via%%[/]*}"
  lib="$(field "$settings" library)"
  echo "macos $(sw_vers -productVersion) build $(sw_vers -buildVersion)"
  echo "boot $(sysctl -n kern.boottime)"
  echo "device $device gid $(field "$settings" links.0.gid) via $via path_mtu ${mtu:-4096}"
  local offline=0
  facts_offline "$lib" "$device" && offline=1
  local kext uuid version expected
  kext="$(kextstat 2>/dev/null | grep -i 'org\.mcdma' || true)"
  if [ -n "$kext" ]; then
    echo "kext $kext"
    uuid="$(echo "$kext" | grep -oE '[0-9A-F]{8}(-[0-9A-F]{4}){3}-[0-9A-F]{12}' | head -1 || true)"
    version="$(echo "$kext" | sed -n 's/.*org\.mcdma\.cx5\.native (\([^)]*\)).*/\1/p' | head -1)"
    echo "kext_uuid ${uuid:-?} kext_version ${version:-?}"
  else
    echo "kext none"
  fi
  # the PCI device's MCDMANativeVersion is a literal in the kext's source, not always its bundle version
  ioreg -l -w0 2>/dev/null | grep -oE '"MCDMANativeVersion" = "[^"]*"' | sort -u | sed 's/.*= "\(.*\)"/native_version \1 (MCDMANativeVersion)/' || true
  expected="$(field "$settings" expected_kext_uuid)"
  if [ -n "$expected" ] && [ "$offline" = 0 ] && [ "${uuid:-}" != "$expected" ]; then
    echo "WARN the loaded MCDMA kext is ${uuid:-none}, not the settings' expected_kext_uuid $expected"
  fi
  case "$device" in
    rdma_mcrdma*)
      [ "$offline" = 1 ] && echo "offline library on a ConnectX device: the facts below are this Mac's, the run is not"
      [ -n "$kext" ] || echo "WARN no MCDMA kext is loaded: a ConnectX link needs org.mcdma.cx5.native"
      local port="${device#rdma_}" flags mine peer_gid peer_mac neigh
      flags="$(ifconfig "$port" 2>/dev/null | head -1 || true)"
      echo "port $flags"
      ifconfig "$port" 2>/dev/null | grep ether | sed 's/^[[:space:]]*/port /' || true
      case "$flags" in *"<UP,"*RUNNING*|*",UP,"*RUNNING*) ;; *) echo "WARN $port is not UP and RUNNING (no carrier, or down): is it the cabled port? then the operator's prep line (restore-rdma.py)";; esac
      mcdma_ports | sed 's/^/ioreg port /' || true
      mine="$(mcdma_ports | awk -v p="$port" '$1 == p' || true)"
      if [ -z "$mine" ]; then
        echo "WARN no MCDMACX5Native instance has MCDMAAddressInterface $port"
      else
        echo "pcie $port ${mine#*pcie=}"
        echo "relaxed $port $(echo "$mine" | grep -oE 'RelaxedOrdering(Keys)?=[^ ]*' | tr '\n' ' ')"
        case "$mine" in *" PortActive=Yes "*) ;; *) echo "WARN MCDMAPortActive is not Yes for $port: its ConnectX port has no link (the cable, the switch port, or the other port is the cabled one: PLAN.txt's Prep says how to switch)";; esac
        if echo "$mine" | grep -qE 'RelaxedOrdering=Yes|RelaxedOrderingKeys=[1-9]'; then
          echo "WARN relaxed ordering is on for $port: a signal written after its data may land before it"
        fi
      fi
      peer_gid="$(field "$settings" peer_gid_addr)"
      peer_mac="$(field "$settings" peer_mac)"
      neigh="$(ndp -an 2>/dev/null | grep -i "%$port " || true)"
      if [ -n "$neigh" ]; then echo "$neigh" | sed 's/^/neighbour /'; fi
      if [ -z "$peer_gid" ] || [ -z "$peer_mac" ]; then
        echo "WARN the settings give no peer_gid_addr and peer_mac: the peer's neighbour on $port is not checked"
      elif [ "$(echo "$neigh" | neighbour_has "$peer_gid" "$peer_mac")" != yes ]; then
        echo "WARN ndp has no permanent neighbour $peer_gid%$port at $peer_mac (the peer's): the operator's prep line (restore-rdma.py)"
      fi
      ;;
    rdma_*) echo "thunderbolt $(ifconfig "${device#rdma_}" 2>/dev/null | head -1 || true)" ;;
  esac
  echo "meeting $(ifconfig "$iface" 2>/dev/null | grep -E 'inet |status' | tr '\n' ' ' || true)"
  local pinned
  if ! ifconfig "$iface" >/dev/null 2>&1; then
    echo "WARN no interface $iface for via $via"
  else
    pinned="$(field "$settings" meeting_addr)"
    if [ -n "$pinned" ] && ! ifconfig "$iface" 2>/dev/null | awk '$1 == "inet" || $1 == "inet6" { sub(/%.*/, "", $2); print $2 }' | grep -qxF "$pinned"; then
      echo "WARN $iface does not hold $pinned, the address the peer's via pins for this Mac (DHCP moved it?): the two ends never meet, and both fail ConnectFailed after connect_timeout_s"
    fi
  fi
  # both clocks place a survivor's failure against the arm (G3) and the drop (G4); sntp with no -s or -S only reads
  local ntp offset
  ntp="$(sntp -t 2 time.apple.com 2>/dev/null | head -1 || true)"
  offset="$(echo "$ntp" | awk '$1 ~ /^[-+]?[0-9.]+$/ { print $1; exit }')"
  if [ -z "$offset" ]; then
    echo "clock offset ? (no answer from sntp -t 2 time.apple.com)"
  else
    echo "clock offset $offset s (sntp: $ntp)"
    if awk -v o="$offset" 'BEGIN { o += 0; exit !(o > 1 || o < -1) }'; then
      echo "WARN the clock is $offset s off time.apple.com: compare.py's cross-host windows (G3, G4) allow only 2 s"
    fi
  fi
  /usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate 2>/dev/null | sed 's/^/firewall /' || true
  if /usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate 2>/dev/null | grep -q enabled; then
    echo "WARN the application firewall is on: allow tf-mcdma-pair's UDP port, or its first run waits on a dialog"
  fi
  [ -f "$lib" ] && otool -L "$lib" 2>/dev/null | tail -n +2 | sed 's/^[[:space:]]*/links /' || true
}

run_side macos "$@"
