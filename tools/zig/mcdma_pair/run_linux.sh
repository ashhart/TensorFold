#!/usr/bin/env bash
# The Linux rank of the #1594 gate: this host's side from a settings JSON, with read-only facts (README.md)
set -euo pipefail
# shellcheck source=common.sh
. "$(cd "$(dirname "$0")" && pwd)/common.sh"

facts_linux() {
  local settings="$1" device="$2" via="$3"; shift 3
  local mtu gid iface sys need memlock lib
  mtu="$(setting "$settings" links.0.mtu --mtu "$@")"
  mtu="${mtu:-4096}"
  gid="$(field "$settings" links.0.gid)"
  iface="${via%%[/]*}"
  lib="$(field "$settings" library)"
  [ -f /etc/os-release ] && echo "os $(. /etc/os-release && echo "$PRETTY_NAME")"
  echo "kernel $(uname -r) page $(getconf PAGESIZE)"
  echo "glibc $(ldd --version 2>&1 | head -1)"
  echo "device $device gid $gid via $via path_mtu $mtu"
  # window (the suite's own, about 16 MiB, or window_mib) plus two ranks' staging, in KiB
  need="$(python3 -c 'import json,sys
d=json.load(open(sys.argv[1]))
w=max(int(d.get("window_mib",0) or 0), 16 + int(d.get("bulk_mib",9)) - 9)
print((w + 2*int(d.get("staging_mib",6)) + 4) * 1024)' "$settings")"
  memlock="$(ulimit -l)"
  echo "memlock $memlock KiB (needs about $need)"
  if [ "$memlock" != unlimited ] && [ "$memlock" -lt "$need" ]; then
    echo "WARN memlock $memlock KiB is below the window plus staging ($need KiB): registration will fail (raise ulimit -l)"
  fi
  sys="${TF_MCDMA_PAIR_SYSFS:-/sys/class/infiniband}/$device"
  if facts_offline "$lib" "$device"; then
    :
  elif [ -d "$sys" ]; then
    echo "port_state $(cat "$sys/ports/1/state" 2>/dev/null) phys $(cat "$sys/ports/1/phys_state" 2>/dev/null) rate $(cat "$sys/ports/1/rate" 2>/dev/null)"
    grep -q ACTIVE "$sys/ports/1/state" 2>/dev/null || echo "WARN $device port 1 is not ACTIVE"
    local gtype gval ndev
    gval="$(cat "$sys/ports/1/gids/$gid" 2>/dev/null || true)"
    gtype="$(cat "$sys/ports/1/gid_attrs/types/$gid" 2>/dev/null || true)"
    ndev="$(cat "$sys/ports/1/gid_attrs/ndevs/$gid" 2>/dev/null || true)"
    echo "gid $gid $gval type ${gtype:-?} netdev ${ndev:-?}"
    [ "$gtype" = "RoCE v2" ] || echo "WARN gid index $gid is '${gtype:-missing}', not RoCE v2"
    if command -v ibv_devinfo >/dev/null 2>&1; then
      local active
      active="$(ibv_devinfo -d "$device" 2>/dev/null | awk '/active_mtu/ {print $2; exit}' || true)"
      echo "active_mtu ${active:-?}"
      [ -n "$active" ] && [ "$active" -lt "$mtu" ] && echo "WARN active_mtu $active is below the path MTU $mtu: open fails with DEVICE"
    fi
    if [ -n "$ndev" ]; then
      ip -br link show "$ndev" 2>/dev/null | sed 's/^/netdev /' || true
      local neigh peer_gid peer_mac
      neigh="$(ip -6 neigh show dev "$ndev" 2>/dev/null || true)"
      if [ -n "$neigh" ]; then echo "$neigh" | sed 's/^/neighbour /'; fi
      peer_gid="$(field "$settings" peer_gid_addr)"
      peer_mac="$(field "$settings" peer_mac)"
      if [ -z "$peer_gid" ] || [ -z "$peer_mac" ]; then
        echo "WARN the settings give no peer_gid_addr and peer_mac: the peer's neighbour on $ndev is not checked"
      elif [ "$(echo "$neigh" | neighbour_has "$peer_gid" "$peer_mac")" != yes ]; then
        echo "WARN ip -6 neigh has no permanent $peer_gid lladdr $peer_mac on $ndev (the peer's): the operator's prep line, sudo ip -6 neigh replace $peer_gid lladdr $peer_mac nud permanent dev $ndev"
      fi
    fi
  else
    echo "WARN no RDMA device $device (have: $(ls /sys/class/infiniband 2>/dev/null | tr '\n' ' '))"
  fi
  local meeting pinned
  if meeting="$(ip -br addr show "$iface" 2>/dev/null)" && [ -n "$meeting" ]; then
    echo "meeting $meeting"
    pinned="$(field "$settings" meeting_addr)"
    if [ -n "$pinned" ] && ! echo "$meeting" | tr -s ' \t' '\n\n' | sed 's#/.*##' | grep -qxF "$pinned"; then
      echo "WARN $iface does not hold $pinned, the address the peer's via pins for this host (DHCP moved it?): the two ends never meet, and both fail ConnectFailed after connect_timeout_s"
    fi
  else
    echo "WARN no interface $iface for via $via"
  fi
  # both hosts' clocks place the survivor's failure against the victim's arm (G3) and the drop (G4): read only
  local synced
  if command -v timedatectl >/dev/null 2>&1; then
    synced="$(timedatectl show -p NTPSynchronized --value 2>/dev/null || true)"
    echo "clock ntp_synchronized ${synced:-?} (timedatectl)"
    if [ "$synced" = no ]; then echo "WARN the clock is not NTP-synchronized (timedatectl): compare.py's cross-host windows (G3, G4) allow only 2 s"; fi
  fi
  if command -v chronyc >/dev/null 2>&1; then
    chronyc tracking 2>/dev/null | python3 -c 'import re, sys
t = sys.stdin.read()
m = re.search(r"System time\s*:\s*([0-9.]+) seconds (fast|slow)", t)
leap = re.search(r"Leap status\s*:\s*(.+)", t)
off = None if m is None else float(m.group(1)) * (1 if m.group(2) == "fast" else -1)
leap = None if leap is None else leap.group(1).strip()
print("clock chrony offset %s s, leap status %s (chronyc tracking)" % ("?" if off is None else "%+.6f" % off, leap or "?"))
if (off is not None and abs(off) > 1) or (leap is not None and leap != "Normal"):
    print("WARN the clock is %s (chronyc tracking): the cross-host windows of compare.py (G3, G4) allow only 2 s" % (
        "not synchronized" if leap not in (None, "Normal") else "%+.3f s off" % off))' || true
  fi
  [ -f "$lib" ] && ldd "$lib" 2>/dev/null | sed 's/^[[:space:]]*/links /' || true
}

run_side linux "$@"
