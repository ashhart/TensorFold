"""Recreate the tf-dev container on a node with one more bind mount, everything else as it was: by default the compose
server's prepared weight folders (PREPARED_DIR) read-only at /prepared, so dev runs read them (tools/dsv41_run2.sh,
notes/dsv41/DEV.md). A container cannot gain a mount in place, so:

1. ``docker inspect tf-dev`` (and its image's) is saved under out/ first; the ``docker run`` flags are rebuilt from it
   (runtime, GPUs, network, IPC, ulimits, devices, capabilities, mounts, environment beyond the image's, command) and
   anything set that this script does not reproduce is named, refusing ``--apply``;
2. ``docker commit`` keeps the container's own layer (the editable install of /tf, whatever was installed since) as
   tf-dev-snapshot:<stamp>, the image the new container runs;
3. the old container is renamed tf-dev-old-<stamp> and stopped, never removed: if the new one does not start or
   cannot import tensorfold, it is removed and the old one renamed back and started.

    python3 tools/dsv41_tfdev_recreate.py aiai            # dry run: the inspect saved, the commands printed
    python3 tools/dsv41_tfdev_recreate.py aiai --apply    # refused while anything but its idle command runs in tf-dev

Run it for both nodes (aiai, aiai2). Afterwards ``docker rm tf-dev-old-<stamp>`` and ``docker rmi
tf-dev-snapshot:<stamp>`` once a dev run has passed (the snapshot stays the container's image until then).
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

NAME = "tf-dev"
PREPARED = "/home/urtho/.cache/tensorfold-prepared"
DEFAULT_SHM = 64 << 20
IDLE = ("sleep infinity",)           # the only processes allowed in tf-dev while it is recreated (besides ps itself)


def run_args(c: dict, image_env: list[str], extra: list[str] = ()) -> tuple[list[str], list[str]]:
    """``docker run`` options (without the image and command) that rebuild container ``c`` (one ``docker inspect``
    entry), plus ``extra`` options; and what ``c`` sets that they do not reproduce."""

    h, cfg = c.get("HostConfig") or {}, c.get("Config") or {}
    out = ["-d", "--name", c.get("Name", "/" + NAME).lstrip("/")]
    missing = []
    if h.get("Runtime") not in (None, "", "runc"):
        out += ["--runtime", h["Runtime"]]
    for req in h.get("DeviceRequests") or []:
        caps = [x for group in (req.get("Capabilities") or []) for x in group]
        if "gpu" not in caps or req.get("Options"):
            missing.append(f"device request {req}")
        elif req.get("DeviceIDs"):
            out += ["--gpus", f"\"device={','.join(req['DeviceIDs'])}\""]
        elif req.get("Count", 0) == -1:
            out += ["--gpus", "all"]
        else:
            out += ["--gpus", str(req.get("Count"))]
    net = h.get("NetworkMode") or "default"
    if net != "default":
        out += ["--network", net]
    if (h.get("IpcMode") or "private") not in ("private", "shareable"):
        out += ["--ipc", h["IpcMode"]]
    if h.get("PidMode"):
        out += ["--pid", h["PidMode"]]
    if h.get("UTSMode"):
        out += ["--uts", h["UTSMode"]]
    if h.get("ShmSize") and h["ShmSize"] != DEFAULT_SHM:
        out += ["--shm-size", str(h["ShmSize"])]
    for u in h.get("Ulimits") or []:
        out += ["--ulimit", f"{u['Name']}={u['Soft']}:{u['Hard']}"]
    for d in h.get("Devices") or []:
        out += ["--device", f"{d['PathOnHost']}:{d['PathInContainer']}:{d.get('CgroupPermissions') or 'rwm'}"]
    for cap in h.get("CapAdd") or []:
        out += ["--cap-add", cap]
    for cap in h.get("CapDrop") or []:
        out += ["--cap-drop", cap]
    if h.get("Privileged"):
        out.append("--privileged")
    for opt in h.get("SecurityOpt") or []:
        out += ["--security-opt", opt]
    for g in h.get("GroupAdd") or []:
        out += ["--group-add", g]
    for e in h.get("ExtraHosts") or []:
        out += ["--add-host", e]
    for k, v in (h.get("Tmpfs") or {}).items():
        out += ["--tmpfs", f"{k}:{v}" if v else k]
    if h.get("Init"):
        out.append("--init")
    rp = h.get("RestartPolicy") or {}
    if rp.get("Name") not in (None, "", "no"):
        out += ["--restart", rp["Name"] + (f":{rp['MaximumRetryCount']}" if rp.get("MaximumRetryCount") else "")]
    if h.get("Memory"):
        out += ["--memory", str(h["Memory"])]
    if h.get("MemorySwap"):
        out += ["--memory-swap", str(h["MemorySwap"])]
    for port, binds in (h.get("PortBindings") or {}).items():
        for b in binds or []:
            out += ["-p", f"{b.get('HostIp') + ':' if b.get('HostIp') else ''}{b.get('HostPort', '')}:{port}"]
    for key in ("VolumesFrom", "Links", "CgroupParent", "CpusetCpus", "NanoCpus", "CpuShares", "PidsLimit",
                "OomKillDisable", "Sysctls", "Dns", "LogConfig"):
        v = h.get(key)
        if key == "LogConfig":
            if v and (v.get("Type") not in (None, "", "json-file") or v.get("Config")):
                missing.append(f"HostConfig.LogConfig={v}")
        elif v not in (None, "", 0, [], {}, False):
            missing.append(f"HostConfig.{key}={v}")
    for m in c.get("Mounts") or []:
        mode = "" if m.get("RW", True) else ":ro"
        if m.get("Type") == "bind":
            out += ["-v", f"{m['Source']}:{m['Destination']}{mode}"]
        elif m.get("Type") == "volume":
            out += ["-v", f"{m['Name']}:{m['Destination']}{mode}"]
        elif m.get("Type") != "tmpfs":
            missing.append(f"mount {m}")
    base = set(image_env)
    for e in cfg.get("Env") or []:
        if e not in base:
            out += ["-e", e]
    if cfg.get("User"):
        out += ["-u", cfg["User"]]
    if cfg.get("WorkingDir"):
        out += ["-w", cfg["WorkingDir"]]
    if cfg.get("Tty"):
        out.append("-t")
    if cfg.get("OpenStdin"):
        out.append("-i")
    labels = cfg.get("Labels") or {}
    if any(k.startswith("com.docker.compose") for k in labels):
        missing.append("a compose-managed container (recreate it with compose)")
    dests = {m.get("Destination") for m in c.get("Mounts") or []}
    for i, x in enumerate(extra):
        if i and extra[i - 1] == "-v" and x.split(":")[1] in dests:
            missing.append(f"{x.split(':')[1]} is mounted already")
    return out + list(extra), missing


def command(c: dict) -> list[str]:
    return list((c.get("Config") or {}).get("Cmd") or [])


def plan(c: dict, image_env: list[str], stamp: str, extra: list[str]) -> tuple[list[list[str]], list[str]]:
    """The docker commands of a recreate, and what is not reproduced."""

    snap, old = f"{NAME}-snapshot:{stamp}", f"{NAME}-old-{stamp}"
    opts, missing = run_args(c, image_env, extra)
    return [["docker", "commit", NAME, snap],
            ["docker", "rename", NAME, old],
            ["docker", "stop", "-t", "10", old],
            ["docker", "run", *opts, snap, *command(c)]], missing


def sh(host: str, cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["ssh", host, cmd], capture_output=True, text=True, timeout=600, check=check)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("host", help="the node (ssh name), e.g. aiai or aiai2")
    ap.add_argument("--mount", default=f"{PREPARED}:/prepared:ro", help="the bind mount to add (src:dst[:ro])")
    ap.add_argument("--apply", action="store_true", help="recreate (default: print the plan)")
    a = ap.parse_args(argv)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    c = json.loads(sh(a.host, f"docker inspect {NAME}").stdout)[0]
    img = json.loads(sh(a.host, f"docker image inspect {shlex.quote(c['Config']['Image'])}").stdout)[0]
    out = Path("out")
    out.mkdir(exist_ok=True)
    saved = out / f"{NAME}-inspect-{a.host}-{stamp}.json"
    saved.write_text(json.dumps({"container": c, "image": img}, indent=1))
    print(f"saved: {saved}")
    src = a.mount.split(":")[0]
    if sh(a.host, f"test -d {shlex.quote(src)}", check=False).returncode:
        print(f"refused: {src} is not a directory on {a.host}")
        return 1
    cmds, missing = plan(c, (img.get("Config") or {}).get("Env") or [], stamp, ["-v", a.mount])
    for cmd in cmds:
        print(" ".join(shlex.quote(x) for x in cmd))
    if missing:
        print("not reproduced (refusing --apply):\n  " + "\n  ".join(missing))
        return 1
    if not a.apply:
        print("dry run: --apply recreates")
        return 0
    procs = sh(a.host, f"docker top {NAME} -eo args", check=False).stdout.splitlines()[1:]
    busy = [p for p in procs if p.strip() and not any(p.strip().endswith(i) for i in IDLE)
            and "nvidia_entrypoint" not in p]
    if busy:
        print("refused: tf-dev is busy:\n  " + "\n  ".join(busy))
        return 1
    size = sh(a.host, f"docker ps -s --filter name=^{NAME}$ --format '{{{{.Size}}}}'", check=False).stdout.strip()
    free = sh(a.host, "df -h --output=avail / | tail -1", check=False).stdout.strip()
    print(f"the container layer ({size}) goes into the snapshot; / has {free} free")
    for i, cmd in enumerate(cmds):
        r = sh(a.host, " ".join(shlex.quote(x) for x in cmd), check=False)
        if r.returncode:
            print(f"failed: {' '.join(cmd[:3])}: {r.stderr.strip()}")
            if i >= 1:
                sh(a.host, f"docker rm -f {NAME} >/dev/null 2>&1; docker rename {NAME}-old-{stamp} {NAME} && "
                   f"docker start {NAME}", check=False)
                print("rolled back: the old tf-dev is back under its name")
            return 1
    dst = a.mount.split(":")[1]
    check = sh(a.host, f"docker exec {NAME} python -c 'import tensorfold' && docker exec {NAME} test -d {dst}",
               check=False)
    if check.returncode:
        print(f"the new tf-dev failed its check ({check.stderr.strip()[:300]}): rolling back")
        sh(a.host, f"docker rm -f {NAME}; docker rename {NAME}-old-{stamp} {NAME} && docker start {NAME}", check=False)
        return 1
    print(f"tf-dev recreated on {a.host} with {a.mount}; the old one is stopped as {NAME}-old-{stamp} "
          f"(rollback: docker rm -f {NAME} && docker rename {NAME}-old-{stamp} {NAME} && docker start {NAME})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
