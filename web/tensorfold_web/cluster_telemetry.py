"""Local hardware, TB5 RDMA, model cache, and rank process telemetry (shared by Master and Slave)."""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from .cluster_rank import RankManager

__all__ = ["RankManager", "collect_local_telemetry", "probe_rdma_link"]

_CHIP_SPECS: list[tuple[str, float, float]] = [
    ("m4 ultra", 1092.0, 120.0),
    ("m3 ultra", 800.0, 54.0),
    ("m2 ultra", 800.0, 27.2),
    ("m1 ultra", 800.0, 21.0),
    ("m4 max", 546.0, 36.0),
    ("m3 max", 400.0, 28.0),
    ("m2 max", 400.0, 13.6),
    ("m1 max", 400.0, 10.4),
    ("m4 pro", 273.0, 16.0),
    ("m3 pro", 150.0, 12.0),
    ("m2 pro", 200.0, 6.8),
    ("m1 pro", 200.0, 5.2),
    ("m4", 120.0, 8.0),
    ("m3", 100.0, 6.0),
    ("m2", 100.0, 3.6),
    ("m1", 68.0, 2.6),
]

_STATIC_HW: dict[str, Any] | None = None
_NET_PREV: dict[str, tuple[float, int, int]] = {}
_NET_LOCK = threading.Lock()
_START_TIME = time.time()


def _run(cmd: list[str], timeout: float = 3.0) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return out.stdout or ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _chip_bw_tflops(chip: str) -> tuple[float, float]:
    low = chip.lower()
    for name, bw, tflops in _CHIP_SPECS:
        if name in low:
            return bw, tflops
    return 100.0, 10.0


def _static_hardware() -> dict[str, Any]:
    global _STATIC_HW
    if _STATIC_HW is not None:
        return _STATIC_HW

    chip = platform.processor() or platform.machine()
    cpu_cores = os.cpu_count() or 1
    ram_total = 0
    gpu_cores = 0
    os_ver = platform.mac_ver()[0] or platform.release()

    if platform.system() == "Darwin":
        brand = _run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip()
        if brand:
            chip = brand
        mem_s = _run(["sysctl", "-n", "hw.memsize"]).strip()
        if mem_s.isdigit():
            ram_total = int(mem_s)
        ncpu = _run(["sysctl", "-n", "hw.ncpu"]).strip()
        if ncpu.isdigit():
            cpu_cores = int(ncpu)
        ioreg = _run(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"])
        m_gpu = re.search(r'"gpu-core-count"\s*=\s*(\d+)', ioreg)
        if m_gpu:
            gpu_cores = int(m_gpu.group(1))
        m_model = re.search(r'"model"\s*=\s*"([^"]+)"', ioreg)
        if m_model and "apple" in m_model.group(1).lower():
            chip = m_model.group(1)
    else:
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    ram_total = int(line.split()[1]) * 1024
                    break
        except OSError:
            pass

    bw, tflops = _chip_bw_tflops(chip)
    _STATIC_HW = {
        "hostname": socket.gethostname().split(".")[0],
        "fqdn": socket.gethostname(),
        "os": platform.system(),
        "os_version": os_ver,
        "arch": platform.machine(),
        "chip": chip,
        "cpu_cores": cpu_cores,
        "gpu_cores": gpu_cores,
        "ram_total_bytes": ram_total,
        "memory_bandwidth_gbs": bw,
        "peak_tflops_fp16": tflops,
    }
    return _STATIC_HW


def _memory_and_gpu_live(ram_total: int) -> tuple[dict[str, Any], dict[str, Any]]:
    mem = {
        "total_bytes": ram_total,
        "used_bytes": 0,
        "wired_bytes": 0,
        "free_bytes": ram_total,
        "available_bytes": ram_total,
        "pressure_pct": 0.0,
    }
    gpu = {
        "util_pct": 0.0,
        "renderer_pct": 0.0,
        "tiler_pct": 0.0,
        "alloc_bytes": 0,
        "in_use_bytes": 0,
    }

    if platform.system() == "Darwin":
        vm = _run(["vm_stat"])
        page_size = 16384
        m_ps = re.search(r"page size of (\d+) bytes", vm)
        if m_ps:
            page_size = int(m_ps.group(1))
        pages: dict[str, int] = {}
        for line in vm.splitlines():
            m = re.match(r"^([^:]+):\s+(\d+)", line)
            if m:
                pages[m.group(1).strip().lower()] = int(m.group(2))
        free_p = pages.get("pages free", 0)
        spec_p = pages.get("pages speculative", 0)
        inactive_p = pages.get("pages inactive", 0)
        purge_p = pages.get("pages purgeable", 0)
        wired_p = pages.get("pages wired down", 0)
        active_p = pages.get("pages active", 0)
        comp_p = pages.get("pages occupied by compressor", 0)
        avail = (free_p + spec_p + inactive_p + purge_p) * page_size
        wired = wired_p * page_size
        used = max(0, ram_total - avail) if ram_total else (active_p + wired_p + comp_p) * page_size
        mem.update(
            used_bytes=used,
            wired_bytes=wired,
            free_bytes=free_p * page_size,
            available_bytes=avail,
            pressure_pct=round((used / ram_total) * 100.0, 1) if ram_total > 0 else 0.0,
        )

        ioreg = _run(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"])
        m_perf = re.search(r'"PerformanceStatistics"\s*=\s*\{([^}]+)\}', ioreg)
        if m_perf:
            body = m_perf.group(1)
            for key, dst in [
                ("Device Utilization %", "util_pct"),
                ("Renderer Utilization %", "renderer_pct"),
                ("Tiler Utilization %", "tiler_pct"),
                ("Alloc system memory", "alloc_bytes"),
                ("In use system memory", "in_use_bytes"),
            ]:
                mk = re.search(rf'"{re.escape(key)}"\s*=\s*(\d+)', body)
                if mk:
                    val = int(mk.group(1))
                    gpu[dst] = float(val) if dst.endswith("_pct") else val
    return mem, gpu


def _peer_ip_for_slash30(ip: str) -> tuple[str | None, str | None]:
    parts = ip.split(".")
    if len(parts) != 4:
        return None, None
    try:
        a, b, c, d = (int(x) for x in parts)
    except ValueError:
        return None, None
    base = d & ~3
    offset = d - base
    if offset == 1:
        peer = f"{a}.{b}.{c}.{base + 2}"
    elif offset == 2:
        peer = f"{a}.{b}.{c}.{base + 1}"
    else:
        return None, None
    return f"{a}.{b}.{c}.{base}/30", peer


def _rdma_and_tb5_interfaces() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (rdma_ports, lan_ips) with live byte-rate counters."""
    rdma_devices: dict[str, dict[str, Any]] = {}
    ibv = _run(["ibv_devinfo", "-v"])
    cur: str | None = None
    for line in ibv.splitlines():
        m_hca = re.match(r"^hca_id:\s+(\S+)", line.strip())
        if m_hca:
            cur = m_hca.group(1)
            iface = cur[5:] if cur.startswith("rdma_") else cur
            rdma_devices[iface] = {
                "device": cur,
                "iface": iface,
                "state": "UNKNOWN",
                "mtu": 4096,
                "speed_gbps": 80,
                "ip": None,
                "subnet": None,
                "peer_ip": None,
                "rx_bytes": 0,
                "tx_bytes": 0,
                "rx_mbps": 0.0,
                "tx_mbps": 0.0,
            }
            continue
        if cur is None:
            continue
        iface = cur[5:] if cur.startswith("rdma_") else cur
        if "state:" in line:
            m_st = re.search(r"state:\s+(\S+)", line)
            if m_st:
                rdma_devices[iface]["state"] = m_st.group(1)
        elif "active_mtu:" in line:
            m_mtu = re.search(r"active_mtu:\s+(\d+)", line)
            if m_mtu:
                rdma_devices[iface]["mtu"] = int(m_mtu.group(1))

    # Parse netstat -ibn for IPs + live byte counters
    now = time.time()
    lan_ips: list[dict[str, Any]] = []
    ns = _run(["netstat", "-ibn"])
    iface_bytes: dict[str, tuple[int, int]] = {}
    iface_ips: dict[str, list[str]] = {}
    for line in ns.splitlines()[1:]:
        cols = line.split()
        if len(cols) < 10:
            continue
        name = cols[0]
        if name.endswith("*"):
            continue
        net, addr = cols[2], cols[3]
        # Handle Link# rows vs IPv4 rows
        if net.startswith("<Link#"):
            try:
                ibytes = int(cols[6])
                obytes = int(cols[9])
                iface_bytes[name] = (ibytes, obytes)
            except (ValueError, IndexError):
                pass
        elif re.match(r"^\d+\.\d+\.\d+\.\d+$", addr) and not addr.startswith("127.") and not addr.startswith("169.254."):
            iface_ips.setdefault(name, []).append(addr)

    with _NET_LOCK:
        for iface, (ib, ob) in iface_bytes.items():
            rx_mbps = 0.0
            tx_mbps = 0.0
            prev = _NET_PREV.get(iface)
            if prev is not None:
                dt = max(0.05, now - prev[0])
                rx_mbps = round(max(0, ib - prev[1]) / dt / (1024 * 1024), 2)
                tx_mbps = round(max(0, ob - prev[2]) / dt / (1024 * 1024), 2)
            _NET_PREV[iface] = (now, ib, ob)
            if iface in rdma_devices:
                rdma_devices[iface]["rx_bytes"] = ib
                rdma_devices[iface]["tx_bytes"] = ob
                rdma_devices[iface]["rx_mbps"] = rx_mbps
                rdma_devices[iface]["tx_mbps"] = tx_mbps

    for iface, ips in iface_ips.items():
        if iface in rdma_devices:
            # Prefer 10.10.x.y TB5 subnet IP
            chosen = next((ip for ip in ips if ip.startswith("10.")), ips[0])
            subnet, peer = _peer_ip_for_slash30(chosen)
            rdma_devices[iface]["ip"] = chosen
            rdma_devices[iface]["subnet"] = subnet
            rdma_devices[iface]["peer_ip"] = peer
        else:
            for ip in ips:
                kind = "tailscale" if ip.startswith("100.") else "lan"
                lan_ips.append({"iface": iface, "ip": ip, "kind": kind})

    ports = sorted(rdma_devices.values(), key=lambda x: x["iface"])
    return ports, lan_ips


def _scan_models() -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    search_roots = [Path.home() / "models", Path.home() / "tf-models"]
    seen: set[str] = set()
    for root in search_roots:
        if not root.is_dir():
            continue
        for entry in sorted(root.iterdir()):
            if not entry.is_dir():
                continue
            cfg_path = entry / "config.json"
            if not cfg_path.is_file():
                continue
            resolved = str(entry.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                cfg = json.loads(cfg_path.read_text())
            except (OSError, ValueError):
                cfg = {}
            idx_path = entry / "model.safetensors.index.json"
            expected_shards = 0
            if idx_path.is_file():
                try:
                    idx = json.loads(idx_path.read_text())
                    expected_shards = len(set((idx.get("weight_map") or {}).values()))
                except (OSError, ValueError):
                    pass
            st_files = list(entry.glob("*.safetensors"))
            part_files = list(entry.glob("*.part"))
            total_bytes = sum(f.stat().st_size for f in st_files if f.is_file())
            part_bytes = sum(f.stat().st_size for f in part_files if f.is_file())
            if expected_shards == 0:
                expected_shards = len(st_files)
            dl_status = None
            dl_file = entry / ".download_status.json"
            if dl_file.is_file():
                try:
                    dl_status = json.loads(dl_file.read_text())
                except (OSError, ValueError):
                    pass
            complete = expected_shards > 0 and len(st_files) >= expected_shards and not part_files
            if dl_status is not None and dl_status.get("complete") is False:
                complete = False
            qcfg = cfg.get("quantization_config") or cfg.get("quantization") or {}
            fmt = "bf16"
            if isinstance(qcfg, dict):
                qm = str(qcfg.get("quant_method") or "").lower()
                if qm == "fp8":
                    fmt = "fp8-block128"
                elif qm:
                    fmt = qm
                elif qcfg.get("bits"):
                    fmt = f"int{qcfg['bits']}-g{qcfg.get('group_size', 64)}"
            if (entry / "exlr8.manifest.json").is_file():
                fmt = "exlr8"
            models.append({
                "id": entry.name,
                "name": entry.name,
                "path": str(entry),
                "model_type": cfg.get("model_type") or "unknown",
                "format": fmt,
                "num_hidden_layers": cfg.get("num_hidden_layers"),
                "num_attention_heads": cfg.get("num_attention_heads"),
                "n_routed_experts": cfg.get("n_routed_experts"),
                "hidden_size": cfg.get("hidden_size"),
                "max_position_embeddings": cfg.get("max_position_embeddings"),
                "shards_present": len(st_files),
                "shards_expected": expected_shards,
                "bytes_on_disk": total_bytes + part_bytes,
                "complete": complete,
                "download": dl_status,
            })
    return models


def _find_binaries() -> dict[str, str | None]:
    home = Path.home()

    def locate(name: str, extra: list[Path]) -> str | None:
        for p in extra:
            if p.is_file():
                return str(p)
        return shutil.which(name)

    return {
        "tensorfold_native": locate("tensorfold-native", [
            home / "bin" / "tensorfold-native",
            home / "TensorFold" / "zig-out" / "native" / "bin" / "tensorfold-native",
        ]),
        "mcdma_lib": locate("libmcdma-fabric.dylib", [
            home / "lib" / "libmcdma-fabric.dylib",
            home / "MCDMA" / "rpc" / "libmcdma-fabric.dylib",
            Path("/usr/local/lib/libmcdma-fabric.dylib"),
        ]),
        "fabric_check": locate("fabric-check", [
            home / "bin" / "fabric-check",
            home / "MCDMA" / "rpc" / "fabric-check",
        ]),
        "mesh_check": locate("mesh-check", [
            home / "bin" / "mesh-check",
            home / "MCDMA" / "rpc" / "mesh-check",
        ]),
    }


def _process_rss_bytes(pid: int) -> int:
    out = _run(["ps", "-o", "rss=", "-p", str(pid)]).strip()
    return int(out) * 1024 if out.isdigit() else 0


def probe_rdma_link(
    device: str,
    via: str,
    port: int = 19780,
    rank: int = 0,
    rounds: int = 20,
    seconds: int = 1,
    peer_port: int | None = None,
) -> dict[str, Any]:
    """Run fabric-check on one side of a TB5 RDMA link to measure real latency & throughput."""
    bins = _find_binaries()
    fc = bins.get("fabric_check")
    if not fc:
        return {"ok": False, "error": "fabric-check binary not found"}
    env = os.environ.copy()
    env["DYLD_LIBRARY_PATH"] = f"{Path.home() / 'lib'}:{env.get('DYLD_LIBRARY_PATH', '')}".rstrip(":")
    p_port = peer_port or port
    cmd = [
        fc,
        device,
        "auto",
        via,
        str(port),
        f"tfprobe{port}",
        str(rank),
        str(rounds),
        "12288,1048576",
        str(seconds),
        str(p_port),
    ]
    t0 = time.perf_counter()
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=20.0, env=env, check=False)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "RDMA probe timed out"}
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    out = (res.stdout or "") + "\n" + (res.stderr or "")
    lat_us = None
    bw_gbs = None
    m_lat = re.search(r"size=12288[^\n]*?rtt_median_us=([0-9]+(?:\.[0-9]+)?)", out)
    if not m_lat:
        m_lat = re.search(r"rtt_median_us=([0-9]+(?:\.[0-9]+)?)", out)
    if not m_lat:
        m_lat = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:us|µs)", out)
    if m_lat:
        lat_us = round(float(m_lat.group(1)), 1)
    bw_matches = re.findall(r"gb_s=([0-9]+(?:\.[0-9]+)?)", out)
    if not bw_matches:
        bw_matches = re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*GB/s", out, re.IGNORECASE)
    if bw_matches:
        bw_gbs = round(max(float(x) for x in bw_matches), 2)
    return {
        "ok": res.returncode == 0,
        "returncode": res.returncode,
        "elapsed_ms": elapsed_ms,
        "latency_us": lat_us,
        "bandwidth_gbs": bw_gbs,
        "output": out.strip()[-700:],
    }


def collect_local_telemetry(rank_manager: RankManager | None = None, role: str = "master") -> dict[str, Any]:
    hw = _static_hardware()
    mem, gpu = _memory_and_gpu_live(hw["ram_total_bytes"])
    rdma_ports, lan_ips = _rdma_and_tb5_interfaces()
    try:
        load_1m = round(os.getloadavg()[0], 2)
    except OSError:
        load_1m = 0.0
    cpu_pct = round(min(100.0, (load_1m / max(1, hw["cpu_cores"])) * 100.0), 1)
    return {
        "hostname": hw["hostname"],
        "fqdn": hw["fqdn"],
        "role": role,
        "uptime_s": round(time.time() - _START_TIME, 1),
        "timestamp": time.time(),
        "hardware": hw,
        "memory": mem,
        "gpu": gpu,
        "cpu": {"load_1m": load_1m, "util_pct": cpu_pct, "cores": hw["cpu_cores"]},
        "rdma": rdma_ports,
        "network": lan_ips,
        "models": _scan_models(),
        "binaries": _find_binaries(),
        "rank": rank_manager.status() if rank_manager is not None else {"state": "idle"},
    }
