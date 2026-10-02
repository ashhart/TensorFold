"""The RDMA mailbox daemons (`v41rpcd`) a split starts and stops itself.

The stage runs a `v41rpcd spark` daemon for its mailbox, started fresh for every Mac session, so a session that
ended without goodbye leaves no connection behind to refuse the next one. Its RoCE address is found by IP rather
than by GID index (a link that flaps renumbers the GID table). The Mac runs a `v41rpcd mac` daemon toward it with
the RoCE addresses of both ends, and stops it through its control socket when the server exits.

Order matters: a Mac daemon must be gone before the Spark daemon it talks to restarts (MCDMA's warning), so the Mac
stops any daemon it left running before it asks the stage for a new session.
"""

from __future__ import annotations

import atexit
import ipaddress
import os
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

SYSFS = Path("/sys/class/infiniband")


@dataclass
class RoceAddress:
    device: str
    gid_index: int
    ip: str
    prefix: int
    mac: str
    netdev: str


def roce_address(device: str, ip: str = "", sysfs: Path = SYSFS, net: Path = Path("/sys/class/net")) -> RoceAddress:
    """The RoCE v2 GID of an IPv4 address on ``device`` (the first one, or ``ip``), with its netdev's MAC and prefix."""

    port = sysfs / device / "ports" / "1"
    if not port.is_dir():
        raise ValueError(f"no RDMA device {device} (found: {', '.join(p.name for p in sysfs.iterdir()) or 'none'})")
    found = []
    for entry in sorted((port / "gids").iterdir(), key=lambda p: int(p.name)):
        raw = entry.read_text().strip().replace(":", "")
        if not raw.startswith("00000000000000000000ffff"):
            continue
        kind = (port / "gid_attrs" / "types" / entry.name).read_text().strip()
        if "v2" not in kind:
            continue
        address = str(ipaddress.IPv4Address(bytes.fromhex(raw[24:])))
        netdev = (port / "gid_attrs" / "ndevs" / entry.name).read_text().strip()
        found.append((int(entry.name), address, netdev))
    pick = [f for f in found if not ip or f[1] == ip]
    if not pick:
        raise ValueError(f"{device} has no RoCE v2 IPv4 GID{' for ' + ip if ip else ''} "
                         f"(it has: {', '.join(f'{a} at {i}' for i, a, _ in found) or 'none'})")
    index, address, netdev = pick[0]
    mac = (net / netdev / "address").read_text().strip()
    prefix = _prefix(netdev, address)
    return RoceAddress(device, index, address, prefix, mac, netdev)


def _prefix(netdev: str, address: str) -> int:
    """The netdev's IPv4 prefix length (SIOCGIFNETMASK; containers often lack the ip tool); 32 if unknown."""

    try:
        import fcntl
        import struct

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            raw = fcntl.ioctl(s.fileno(), 0x891B, struct.pack("256s", netdev.encode()[:15]))   # SIOCGIFNETMASK
        return bin(int.from_bytes(raw[20:24], "big")).count("1")
    except (OSError, ImportError):
        return 32


def peer_ip(ip: str, prefix: int) -> str:
    """The other host of a point-to-point subnet (/30 or /31), the address a Mac takes toward the stage."""

    net = ipaddress.ip_interface(f"{ip}/{prefix}").network
    hosts = [str(h) for h in (net.hosts() if prefix < 31 else net)]
    others = [h for h in hosts if h != ip]
    if prefix < 30 or len(others) != 1:
        raise ValueError(f"the stage's RoCE address {ip}/{prefix} is not on a point-to-point subnet: give this "
                         "machine's RoCE address with --split-rdma-ip")
    return others[0]


def _die_with_parent() -> None:
    """In the daemon, before exec: SIGTERM it when the stage dies, however it dies (it tears down on SIGTERM)."""

    import ctypes
    import signal

    try:
        ctypes.CDLL(None).prctl(1, signal.SIGTERM)   # PR_SET_PDEATHSIG (Linux)
    except (OSError, AttributeError):
        pass


class SparkDaemon:
    """`v41rpcd spark NAME DEVICE GID MTU PORT SOCKET REQ REP`, restarted for each Mac session."""

    def __init__(self, binary: str, name: str, address: RoceAddress, port: int, mtu: int = 4096,
                 req_mib: int = 64, rep_mib: int = 64, log_path: str = "") -> None:
        self.binary, self.name, self.address, self.port = binary, name, address, port
        self.mtu, self.req_mib, self.rep_mib = mtu, req_mib, rep_mib
        self.sock = f"/dev/shm/v41rpc-{name}.sock"
        self.log_path = log_path or f"/tmp/v41rpcd-spark-{name}.log"
        self.proc: subprocess.Popen | None = None

    def start(self) -> None:
        self.stop()
        # a mailbox an earlier daemon (maybe another user's) left behind: /dev/shm's protected_regular refuses
        # to reopen another user's file for creation, even to root, but lets it be removed
        stale = f"/dev/shm/v41rpc-{self.name}"
        if os.path.exists(stale):
            os.unlink(stale)
        log = open(self.log_path, "ab")
        self.proc = subprocess.Popen([self.binary, "spark", self.name, self.address.device,
                                      str(self.address.gid_index), str(self.mtu), str(self.port), self.sock,
                                      str(self.req_mib), str(self.rep_mib)],
                                     stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                     preexec_fn=_die_with_parent)
        deadline = time.monotonic() + 10
        while not os.path.exists(self.sock):
            if self.proc.poll() is not None:
                raise RuntimeError(f"v41rpcd spark exited ({self.proc.returncode}); see {self.log_path}")
            if time.monotonic() > deadline:
                raise TimeoutError(f"v41rpcd spark did not open {self.sock}; see {self.log_path}")
            time.sleep(0.05)

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()                    # it tears its verbs objects down on SIGTERM
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.proc = None
        if os.path.exists(self.sock):
            os.unlink(self.sock)


def _control(sock_path: str, line: str, timeout: float = 10.0) -> str:
    s = socket.socket(socket.AF_UNIX)
    s.settimeout(timeout)
    try:
        s.connect(sock_path)
        s.sendall((line + "\n").encode())
        data = b""
        while not data.endswith(b"END\n") and not data.endswith(b"BYE\n"):
            chunk = s.recv(4096)
            if not chunk:
                break
            data += chunk
        return data.decode(errors="replace")
    finally:
        s.close()


def shutdown_mac(sock_path: str, timeout: float = 90.0) -> bool:
    """Ask a `v41rpcd mac` on ``sock_path`` to tear down and exit; True once none answers there."""

    if not os.path.exists(sock_path):
        return True
    try:
        _control(sock_path, "SHUTDOWN")
    except OSError:
        return True                                  # a stale socket file, nobody behind it
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            _control(sock_path, "STATUS", timeout=2)
        except OSError:
            return True
        time.sleep(0.2)
    return False


class MacDaemon:
    """`v41rpcd mac SOCKET NAME,HOST,PORT,DEVICE,GID,MTU,REQ,REP` with this machine's and the stage's RoCE addresses."""

    def __init__(self, binary: str, sock_path: str, name: str, host: str, port: int, *, local_ip: str,
                 local_mac: str, peer_mac: str, device: str = "mlx5_0", mtu: int = 4096, req_mib: int = 64,
                 rep_mib: int = 64, log_path: str = "") -> None:
        self.binary, self.sock = binary, sock_path
        self.spec = f"{name},{host},{port},{device},0,{mtu},{req_mib},{rep_mib}"
        self.env = dict(os.environ, MAC_ROCE_IP=local_ip, MAC_ROCE_MAC=local_mac, SPARK_ROCE_MAC=peer_mac)
        self.name = name
        self.log_path = log_path or f"/tmp/v41rpcd-mac-{name}.log"
        self.proc: subprocess.Popen | None = None

    def start(self, timeout: float = 30.0) -> None:
        if not shutdown_mac(self.sock):
            raise RuntimeError(f"a v41rpcd on {self.sock} does not shut down; stop it before starting a split")
        log = open(self.log_path, "ab")
        self.proc = subprocess.Popen([self.binary, "mac", self.sock, self.spec], env=self.env, stdout=log,
                                     stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        atexit.register(self.stop)
        deadline = time.monotonic() + timeout
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f"v41rpcd mac exited ({self.proc.returncode}); see {self.log_path}")
            try:
                status = _control(self.sock, "STATUS", timeout=2)
                if f"PEER {self.name} up" in status:
                    return
            except OSError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError(f"v41rpcd mac did not connect to the stage's mailbox; see {self.log_path}")
            time.sleep(0.1)

    def stop(self) -> None:
        if self.proc is None:
            return
        shutdown_mac(self.sock)                      # never a signal: it must tear its QP down itself
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        self.proc = None
