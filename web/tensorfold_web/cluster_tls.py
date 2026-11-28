"""Zero-dependency TLS CA, worker leaf certificate issuance, and URL policy for Cluster Studio."""

from __future__ import annotations

import ipaddress
import os
import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from urllib.parse import urlsplit

from .cluster_rank import is_loopback_host

_TLS_LOCK = threading.Lock()
_DNS_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*$")


def is_loopback_url(url: str) -> bool:
    """Return True if ``url`` targets a loopback host."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    return bool(parts.hostname) and is_loopback_host(parts.hostname)


def normalize_and_validate_slave_url(url: str, *, allow_schemeless: bool = False) -> str:
    """Normalize a slave control URL and refuse plaintext HTTP to non-loopback hosts."""
    raw = (url or "").strip().rstrip("/")
    if not raw:
        raise ValueError("slave_url is required")
    if "://" not in raw:
        if not allow_schemeless:
            raise ValueError("slave_url must include scheme (https://, or http:// on loopback)")
        host_guess = raw.split("/", 1)[0]
        if host_guess.startswith("["):
            host_only = host_guess[1:].split("]", 1)[0]
        else:
            host_only = host_guess.rsplit(":", 1)[0]
        scheme = "http" if is_loopback_host(host_only) else "https"
        raw = f"{scheme}://{raw}"
    parts = urlsplit(raw)
    if parts.scheme not in ("https", "http"):
        raise ValueError(f"unsupported slave_url scheme: {parts.scheme!r} (expected https)")
    if not parts.hostname:
        raise ValueError("slave_url must include a host")
    if parts.username or parts.password:
        raise ValueError("slave_url must not embed credentials")
    _ = parts.port  # Validates port range if present
    if parts.scheme != "https" and not is_loopback_host(parts.hostname):
        raise ValueError(
            "refusing unencrypted http:// slave URL on a non-loopback host; "
            "use https:// (or tunnel over SSH to 127.0.0.1)"
        )
    return raw


def _find_openssl() -> str:
    found = shutil.which("openssl")
    if found:
        return found
    if Path("/usr/bin/openssl").is_file():
        return "/usr/bin/openssl"
    raise RuntimeError("openssl binary not found; cannot provision Cluster Studio TLS certificates")


def _ensure_private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)


def _slugify(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-") or "worker"


def _san_entries(hosts: list[str]) -> tuple[str, str]:
    entries: list[str] = []
    seen: set[str] = set()
    first_cn: str | None = None
    for raw in hosts:
        h = (raw or "").strip()
        if not h:
            continue
        if h.startswith("[") and h.endswith("]"):
            h = h[1:-1]
        if h in seen:
            continue
        seen.add(h)
        try:
            ipaddress.ip_address(h)
            entries.append(f"IP:{h}")
            first_cn = first_cn or h
            continue
        except ValueError:
            pass
        if _DNS_RE.match(h):
            entries.append(f"DNS:{h}")
            first_cn = first_cn or h
    if not entries or not first_cn:
        raise ValueError(f"cannot build TLS SAN from hosts: {hosts!r}")
    return first_cn[:64], ",".join(entries)


def _atomic_install(src: Path, dst: Path, mode: int) -> None:
    tmp = dst.with_name(f".{dst.name}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    try:
        os.chmod(tmp, mode)
        with os.fdopen(fd, "wb") as fh:
            fh.write(src.read_bytes())
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    os.replace(tmp, dst)
    os.chmod(dst, mode)


def ensure_cluster_ca(tls_dir: Path) -> tuple[Path, Path]:
    """Ensure the Cluster Studio internal CA cert/key pair exists under ``tls_dir``."""
    with _TLS_LOCK:
        _ensure_private_dir(tls_dir)
        ca_crt = tls_dir / "ca.crt"
        ca_key = tls_dir / "ca.key"
        if ca_crt.is_file() and ca_key.is_file():
            os.chmod(ca_key, 0o600)
            return ca_crt, ca_key
        openssl = _find_openssl()
        with tempfile.TemporaryDirectory(prefix="tf-ca-") as td:
            d = Path(td)
            cfg = d / "ca.cnf"
            cfg.write_text(
                "[req]\n"
                "prompt=no\n"
                "distinguished_name=dn\n"
                "x509_extensions=v3_ca\n"
                "[dn]\n"
                "CN=TensorFold Cluster CA\n"
                "[v3_ca]\n"
                "basicConstraints=critical,CA:TRUE,pathlen:0\n"
                "keyUsage=critical,keyCertSign,cRLSign\n"
                "subjectKeyIdentifier=hash\n",
                encoding="utf-8",
            )
            tmp_key = d / "ca.key"
            tmp_crt = d / "ca.crt"
            res = subprocess.run(
                [
                    openssl, "req", "-x509", "-newkey", "rsa:2048", "-nodes",
                    "-days", "3650", "-sha256",
                    "-keyout", str(tmp_key), "-out", str(tmp_crt),
                    "-config", str(cfg),
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if res.returncode != 0:
                raise RuntimeError(f"openssl CA generation failed: {(res.stderr or res.stdout).strip()[:240]}")
            _atomic_install(tmp_key, ca_key, 0o600)
            _atomic_install(tmp_crt, ca_crt, 0o644)
        return ca_crt, ca_key


def issue_worker_cert(tls_dir: Path, slug: str, hosts: list[str]) -> tuple[Path, Path]:
    """Issue a CA-signed server TLS certificate and 0600 private key for a worker node."""
    ca_crt, ca_key = ensure_cluster_ca(tls_dir)
    cn, san = _san_entries(hosts)
    with _TLS_LOCK:
        workers_dir = tls_dir / "workers"
        _ensure_private_dir(workers_dir)
        safe = _slugify(slug)
        cert_path = workers_dir / f"{safe}.crt"
        key_path = workers_dir / f"{safe}.key"
        srl_path = tls_dir / "ca.srl"
        openssl = _find_openssl()
        with tempfile.TemporaryDirectory(prefix="tf-leaf-") as td:
            d = Path(td)
            req_cfg = d / "leaf.cnf"
            req_cfg.write_text(f"[req]\nprompt=no\ndistinguished_name=dn\n[dn]\nCN={cn}\n", encoding="utf-8")
            ext_cfg = d / "leaf.ext"
            ext_cfg.write_text(
                "basicConstraints=critical,CA:FALSE\n"
                "keyUsage=critical,digitalSignature,keyEncipherment\n"
                "extendedKeyUsage=serverAuth\n"
                "subjectKeyIdentifier=hash\n"
                "authorityKeyIdentifier=keyid,issuer\n"
                f"subjectAltName={san}\n",
                encoding="utf-8",
            )
            tmp_key = d / "worker.key"
            tmp_csr = d / "worker.csr"
            tmp_crt = d / "worker.crt"
            r1 = subprocess.run(
                [
                    openssl, "req", "-new", "-newkey", "rsa:2048", "-nodes",
                    "-keyout", str(tmp_key), "-out", str(tmp_csr),
                    "-config", str(req_cfg),
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if r1.returncode != 0:
                raise RuntimeError(f"openssl worker CSR failed: {(r1.stderr or r1.stdout).strip()[:240]}")
            sign_cmd = [
                openssl, "x509", "-req", "-in", str(tmp_csr),
                "-CA", str(ca_crt), "-CAkey", str(ca_key),
                "-out", str(tmp_crt), "-days", "825", "-sha256",
                "-extfile", str(ext_cfg),
            ]
            if srl_path.is_file():
                sign_cmd.extend(["-CAserial", str(srl_path)])
            else:
                sign_cmd.extend(["-CAcreateserial", "-CAserial", str(srl_path)])
            r2 = subprocess.run(sign_cmd, capture_output=True, text=True, timeout=30, check=False)
            if r2.returncode != 0:
                raise RuntimeError(f"openssl worker sign failed: {(r2.stderr or r2.stdout).strip()[:240]}")
            _atomic_install(tmp_key, key_path, 0o600)
            _atomic_install(tmp_crt, cert_path, 0o644)
        return cert_path, key_path
