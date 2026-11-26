"""SSH peer discovery, bootstrap, and TB5 model weight sync for Cluster Studio."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx

from .cluster_telemetry import collect_local_telemetry


def _ssh_base(host: str, user: str | None = None, port: int = 22, key_path: str | None = None) -> list[str]:
    cmd = [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=5",
        "-o", "StrictHostKeyChecking=accept-new",
    ]
    if port and port != 22:
        cmd.extend(["-p", str(port)])
    if key_path:
        cmd.extend(["-i", os.path.expanduser(key_path), "-o", "IdentitiesOnly=yes"])
    target = f"{user}@{host}" if user and "@" not in host else host
    cmd.append(target)
    return cmd


def discover_peers(studio) -> list[dict[str, Any]]:
    """Auto-discover candidate slave peers from local TB5 /30 links and ~/.ssh/config."""
    local = collect_local_telemetry(studio.master_rank, role="master")
    existing_hosts = {n.get("ssh_host") for n in studio._data["nodes"]}
    existing_urls = {n.get("slave_url") for n in studio._data["nodes"]}
    candidates: dict[str, dict[str, Any]] = {}

    # 1. Local TB5 /30 peer IPs (10.10.x.1 <-> 10.10.x.2)
    for port in local.get("rdma") or []:
        pip = port.get("peer_ip")
        if not pip:
            continue
        candidates[pip] = {
            "host": pip,
            "ssh_host": pip,
            "name": f"tb5-{port['iface']}",
            "source": f"Thunderbolt 5 ({port['device']} · {port['subnet']})",
            "tb5_iface": port["iface"],
            "tb5_device": port["device"],
            "already_added": pip in existing_hosts or any(pip in (u or "") for u in existing_urls),
        }

    # 2. ~/.ssh/config host aliases
    ssh_cfg = Path.home() / ".ssh" / "config"
    if ssh_cfg.is_file():
        cur_alias: str | None = None
        cur_hostname: str | None = None
        cur_user: str | None = None
        cur_key: str | None = None

        def flush() -> None:
            if not cur_alias or "*" in cur_alias or cur_alias == local["hostname"]:
                return
            target_ip = cur_hostname or cur_alias
            # Merge with TB5 candidate if same IP!
            if target_ip in candidates:
                candidates[target_ip]["name"] = cur_alias
                candidates[target_ip]["ssh_host"] = cur_alias
                if cur_user:
                    candidates[target_ip]["ssh_user"] = cur_user
                if cur_key:
                    candidates[target_ip]["ssh_key"] = cur_key
                candidates[target_ip]["already_added"] = (
                    cur_alias in existing_hosts or target_ip in existing_hosts or candidates[target_ip]["already_added"]
                )
            else:
                candidates[cur_alias] = {
                    "host": target_ip,
                    "ssh_host": cur_alias,
                    "ssh_user": cur_user,
                    "ssh_key": cur_key,
                    "name": cur_alias,
                    "source": "~/.ssh/config",
                    "already_added": cur_alias in existing_hosts or target_ip in existing_hosts,
                }

        for line in ssh_cfg.read_text().splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split(None, 1)
            if len(parts) < 2:
                continue
            k, v = parts[0].lower(), parts[1].strip()
            if k == "host":
                flush()
                cur_alias = v.split()[0]
                cur_hostname = None
                cur_user = None
                cur_key = None
            elif k == "hostname":
                cur_hostname = v
            elif k == "user":
                cur_user = v
            elif k == "identityfile":
                cur_key = v
        flush()

    def probe_cand(c: dict[str, Any]) -> dict[str, Any]:
        # Quick check if slave daemon already answers on 8091
        c["slave_running"] = False
        c["ssh_reachable"] = False
        c["chip"] = None
        c["ram_gb"] = None
        try:
            r = httpx.get(f"http://{c['host']}:8091/api/slave/health", timeout=0.6, trust_env=False)
            if r.status_code == 200:
                c["slave_running"] = True
        except httpx.HTTPError:
            pass
        # Quick non-interactive SSH check
        cmd = _ssh_base(c["ssh_host"], c.get("ssh_user"), 22, c.get("ssh_key")) + [
            "hostname; sysctl -n machdep.cpu.brand_string hw.memsize 2>/dev/null || true"
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=3.0, check=False)
            if res.returncode == 0:
                lines = [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]
                c["ssh_reachable"] = True
                if lines:
                    c["hostname"] = lines[0].split(".")[0]
                    if c["name"].startswith("tb5-"):
                        c["name"] = c["hostname"]
                if len(lines) >= 2:
                    c["chip"] = lines[1]
                if len(lines) >= 3 and lines[2].isdigit():
                    c["ram_gb"] = round(int(lines[2]) / (1024**3))
        except (OSError, subprocess.SubprocessError):
            pass
        return c

    items = list(candidates.values())
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=min(8, len(items))) as ex:
        results = list(ex.map(probe_cand, items))
    results.sort(key=lambda x: (not x["ssh_reachable"], x["name"]))
    return results


def test_ssh(studio, host: str, user: str | None = None, port: int = 22, key_path: str | None = None) -> dict[str, Any]:
    script = (
        "echo HOST=$(hostname -s); "
        "echo CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -m); "
        "echo RAM=$(sysctl -n hw.memsize 2>/dev/null || echo 0); "
        "echo GPU=$(ioreg -r -d 1 -c IOAccelerator 2>/dev/null | grep -o '\"gpu-core-count\" = [0-9]*' | head -n1 | awk '{print $3}'); "
        "echo RDMA=$(ibv_devinfo -l 2>/dev/null | tail -n +2 | tr -d ' ' | paste -sd, -); "
        "echo NATIVE=$(test -x ~/bin/tensorfold-native && echo yes || echo no); "
        "echo MCDMA=$(test -f ~/lib/libmcdma-fabric.dylib && echo yes || echo no); "
        "echo PY=$(which python3 2>/dev/null || echo '')"
    )
    cmd = _ssh_base(host, user, port, key_path) + [script]
    t0 = time.perf_counter()
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=8.0, check=False)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "SSH connection timed out (8s)"}
    ms = round((time.perf_counter() - t0) * 1000.0, 1)
    if res.returncode != 0:
        err = (res.stderr or res.stdout or "SSH failed").strip()
        return {"ok": False, "latency_ms": ms, "error": err[:300]}
    kv: dict[str, str] = {}
    for line in res.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip()
    ram_bytes = int(kv["RAM"]) if kv.get("RAM", "").isdigit() else 0
    return {
        "ok": True,
        "latency_ms": ms,
        "hostname": kv.get("HOST") or host,
        "chip": kv.get("CHIP") or "Unknown",
        "ram_bytes": ram_bytes,
        "ram_gb": round(ram_bytes / (1024**3)) if ram_bytes else None,
        "gpu_cores": int(kv["GPU"]) if kv.get("GPU", "").isdigit() else None,
        "rdma_devices": [x for x in (kv.get("RDMA") or "").split(",") if x],
        "has_native": kv.get("NATIVE") == "yes",
        "has_mcdma": kv.get("MCDMA") == "yes",
        "python": kv.get("PY") or None,
    }


def bootstrap_slave(studio,
    *,
    host: str,
    user: str | None = None,
    port: int = 22,
    key_path: str | None = None,
    name: str | None = None,
    slave_port: int = 8091,
    slave_host: str | None = None,
) -> dict[str, Any]:
    """1-click SSH bootstrap: sync binaries + web package, start `tensorfold-web --slave`, verify & register."""
    probe = studio.test_ssh(host, user, port, key_path)
    if not probe.get("ok"):
        raise RuntimeError(f"SSH check failed for {host}: {probe.get('error')}")

    ssh_cmd = _ssh_base(host, user, port, key_path)
    target = ssh_cmd[-1]
    scp_opts = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=accept-new"]
    if port and port != 22:
        scp_opts.extend(["-P", str(port)])
    if key_path:
        scp_opts.extend(["-i", os.path.expanduser(key_path), "-o", "IdentitiesOnly=yes"])

    # 1. Ensure remote directories exist
    subprocess.run(ssh_cmd + ["mkdir -p ~/bin ~/lib ~/.config/tensorfold"], check=True, timeout=10)

    # 2. Sync native engine + MCDMA dylib/tools if present locally
    home = Path.home()
    sync_pairs = [
        (home / "bin" / "tensorfold-native", "~/bin/tensorfold-native"),
        (home / "lib" / "libmcdma-fabric.dylib", "~/lib/libmcdma-fabric.dylib"),
        (home / "bin" / "fabric-check", "~/bin/fabric-check"),
        (home / "bin" / "mesh-check", "~/bin/mesh-check"),
    ]
    for src, dst in sync_pairs:
        if src.is_file():
            # Only copy if missing or size differs (fast incremental!)
            chk = subprocess.run(
                ssh_cmd + [f"stat -f%z {dst} 2>/dev/null || stat -c%s {dst} 2>/dev/null || echo 0"],
                capture_output=True, text=True, timeout=6, check=False,
            )
            remote_sz = int(chk.stdout.strip()) if chk.stdout.strip().isdigit() else 0
            if remote_sz != src.stat().st_size:
                subprocess.run(["scp", "-q", *scp_opts, str(src), f"{target}:{dst}"], check=True, timeout=60)
                if "/bin/" in dst:
                    subprocess.run(ssh_cmd + [f"chmod +x {dst}"], check=False, timeout=5)

    # 3. Sync lightweight zero-dependency slave files (works on stock Python 3.9+ with ZERO pip packages!)
    pkg_dir = Path(__file__).resolve().parent
    subprocess.run(ssh_cmd + ["mkdir -p ~/cluster/web_src/tensorfold_web"], check=False, timeout=6)
    for pyfile in ["__init__.py", "cluster_telemetry.py", "slave.py"]:
        p = pkg_dir / pyfile
        if p.is_file():
            subprocess.run(["scp", "-q", *scp_opts, str(p), f"{target}:~/cluster/web_src/tensorfold_web/{pyfile}"], check=True, timeout=20)

    # 4. Restart zero-dependency slave daemon cleanly on remote node
    token_q = shlex.quote(studio.cluster_token)
    launch_script = (
        f"pkill -f 'tensorfold_web\\.slave.*--port {slave_port}' 2>/dev/null || true; "
        f"pkill -f 'tensorfold_web\\.cli.*--slave.*--port {slave_port}' 2>/dev/null || true; "
        f"sleep 0.3; "
        f"PYTHONPATH=~/cluster/web_src nohup /usr/bin/python3 -m tensorfold_web.slave "
        f"--host 0.0.0.0 --port {slave_port} --token {token_q} "
        f"> ~/.config/tensorfold/slave-{slave_port}.log 2>&1 < /dev/null &"
    )
    subprocess.run(ssh_cmd + [launch_script], check=True, timeout=15)

    # Resolve reachable IP/hostname for HTTP calls from Master to Slave
    connect_host = slave_host
    if not connect_host:
        # Resolve via ssh -G hostname so aliases like `node-2` resolve to their TB5 IP `10.10.12.2`!
        g = subprocess.run(["ssh", "-G", host], capture_output=True, text=True, timeout=4, check=False)
        m_hn = re.search(r"^hostname\s+(\S+)", g.stdout, re.MULTILINE)
        connect_host = m_hn.group(1) if m_hn else host
    slave_url = f"http://{connect_host}:{slave_port}"

    # 5. Wait up to 10s for slave telemetry to come online
    tele = None
    last_err = "slave did not respond"
    deadline = time.time() + 10.0
    while time.time() < deadline:
        try:
            r = httpx.get(
                f"{slave_url}/api/slave/telemetry",
                headers=studio._slave_headers(),
                timeout=1.5,
                trust_env=False,
            )
            if r.status_code == 200:
                tele = r.json()
                break
            last_err = f"HTTP {r.status_code}"
        except httpx.HTTPError as exc:
            last_err = str(exc)
        time.sleep(0.5)

    if tele is None:
        log_tail = subprocess.run(
            ssh_cmd + [f"tail -n 20 ~/.config/tensorfold/slave-{slave_port}.log 2>/dev/null || true"],
            capture_output=True, text=True, timeout=5, check=False,
        ).stdout.strip()
        raise RuntimeError(f"Slave started on {host} but telemetry failed ({last_err}). Log:\n{log_tail}")

    node_name = name or probe.get("hostname") or host
    node_rec = studio.register_node(
        name=node_name,
        slave_url=slave_url,
        ssh_host=host,
        ssh_user=user,
        ssh_port=port,
        ssh_key=key_path,
        slave_port=slave_port,
    )
    return {"node": node_rec, "telemetry": tele}


def sync_model_to_slaves(studio, model_path: str, node_ids: list[str] | None = None) -> dict[str, Any]:
    """Fan out model safetensors from Master to Slaves in parallel over their direct TB5 links."""
    src = Path(model_path)
    if not src.is_dir():
        raise ValueError(f"Model directory not found on Master: {model_path}")
    st = studio.state()
    slaves = [n for n in st["nodes"] if n["role"] == "slave" and n["online"]]
    if node_ids:
        slaves = [n for n in slaves if n["id"] in node_ids]
    if not slaves:
        raise ValueError("No online slave nodes selected to sync")

    with studio._lock:
        reg_map = {n["id"]: n for n in studio._data["nodes"]}

    started: dict[str, Any] = {}
    for sl in slaves:
        nid = sl["id"]
        if studio._sync_jobs.get(nid, {}).get("state") == "running":
            started[nid] = studio._sync_jobs[nid]
            continue
        rn = reg_map.get(nid, {})
        ssh_host = rn.get("ssh_host") or sl["hostname"]
        ssh_user = rn.get("ssh_user")
        ssh_port = int(rn.get("ssh_port") or 22)
        ssh_key = rn.get("ssh_key")
        job = {
            "node_id": nid,
            "name": sl["name"],
            "model": src.name,
            "state": "running",
            "pct": 0.0,
            "rate": "",
            "started_at": time.time(),
            "error": None,
        }
        studio._sync_jobs[nid] = job
        started[nid] = job

        def run_sync(j: dict[str, Any], host: str, user: str | None, port: int, key: str | None) -> None:
            try:
                ssh_e = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-T", "-c", "aes128-gcm@openssh.com"]
                if port != 22:
                    ssh_e.extend(["-p", str(port)])
                if key:
                    ssh_e.extend(["-i", os.path.expanduser(key), "-o", "IdentitiesOnly=yes"])
                target = f"{user}@{host}" if user and "@" not in host else host
                subprocess.run([*ssh_e, target, f"mkdir -p {shlex.quote(str(src))}"], check=True, timeout=10)
                # Inspect remote file sizes so we only transfer missing/partial shards
                chk_py = (
                    f"import json,pathlib; p=pathlib.Path({str(src)!r}); "
                    "print(json.dumps({f.name:f.stat().st_size for f in p.iterdir() if f.is_file()}))"
                )
                rmap_res = subprocess.run([*ssh_e, target, f"/usr/bin/python3 -c {shlex.quote(chk_py)}"], capture_output=True, text=True, timeout=10, check=False)
                rmap = json.loads(rmap_res.stdout) if rmap_res.returncode == 0 and rmap_res.stdout.strip() else {}
                local_files = sorted(
                    f for f in src.iterdir()
                    if f.is_file() and not f.name.endswith(".part") and f.name != ".download_status.json"
                )
                todo = [f for f in local_files if rmap.get(f.name) != f.stat().st_size]
                total_bytes = sum(f.stat().st_size for f in todo)
                if not todo:
                    j["state"] = "done"
                    j["pct"] = 100.0
                    j["rate"] = "up to date"
                    j["finished_at"] = time.time()
                    return
                done_bytes = 0
                lock_b = threading.Lock()
                t0 = time.perf_counter()

                def push_file(fp: Path) -> None:
                    nonlocal done_bytes
                    tmp_r = f"{str(src).rstrip('/')}/{fp.name}.syncing"
                    dst_r = f"{str(src).rstrip('/')}/{fp.name}"
                    remote_cmd = f"cat > {shlex.quote(tmp_r)} && mv -f {shlex.quote(tmp_r)} {shlex.quote(dst_r)}"
                    with fp.open("rb") as fin:
                        res = subprocess.run([*ssh_e, target, remote_cmd], stdin=fin, check=False)
                    if res.returncode != 0:
                        raise RuntimeError(f"failed transferring {fp.name} (rc={res.returncode})")
                    with lock_b:
                        done_bytes += fp.stat().st_size
                        dt = max(0.05, time.perf_counter() - t0)
                        mb_s = (done_bytes / (1024 * 1024)) / dt
                        j["pct"] = round(min(100.0, (done_bytes / max(1, total_bytes)) * 100.0), 1)
                        j["rate"] = f"{mb_s / 1024.0:.2f} GB/s" if mb_s >= 1024 else f"{mb_s:.0f} MB/s"

                with ThreadPoolExecutor(max_workers=4) as ex:
                    futs = [ex.submit(push_file, fp) for fp in todo]
                    for fut in futs:
                        fut.result()
                j["state"] = "done"
                j["pct"] = 100.0
            except Exception as exc:  # noqa: BLE001
                j["state"] = "failed"
                j["error"] = str(exc)[:200]
            j["finished_at"] = time.time()

        threading.Thread(
            target=run_sync,
            args=(job, ssh_host, ssh_user, ssh_port, ssh_key),
            name=f"sync-{sl['name']}",
            daemon=True,
        ).start()
    return {"jobs": started}
