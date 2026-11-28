"""Managed tensorfold-native rank process lifecycle and log ring buffer."""

from __future__ import annotations

import ipaddress
import json
import os
import re
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path
from typing import Any

_ENV_NAME_RE = re.compile(r"^[A-Z0-9_]+$")
_ALLOWED_ENV_PREFIXES = ("TF_", "MCDMA_")
_ALLOWED_ENV_EXACT = frozenset({"GLM_LOAD_LIMIT_GB"})


def is_loopback_host(host: str) -> bool:
    """Return True only when ``host`` names a loopback address."""
    h = host.strip().lower()
    if h.startswith("[") and h.endswith("]"):
        h = h[1:-1]
    if h in ("localhost", "127.0.0.1", "::1"):
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def _write_secret_file(path: Path, text: str) -> None:
    """Write a secret file atomically with owner-only (0600) permissions."""
    tmp = path.with_name(f".{path.name}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.chmod(tmp, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    os.replace(tmp, path)
    os.chmod(path, 0o600)


def _validate_rank_env(raw_env: Any) -> dict[str, str]:
    """Allow only safe inference/RDMA tuning variables from remote specs."""
    if raw_env is None:
        return {}
    if not isinstance(raw_env, dict):
        raise ValueError("rank env must be a mapping")
    cleaned: dict[str, str] = {}
    for k, v in raw_env.items():
        key = str(k)
        if not _ENV_NAME_RE.match(key) or (
            key not in _ALLOWED_ENV_EXACT and not key.startswith(_ALLOWED_ENV_PREFIXES)
        ):
            raise ValueError(f"disallowed rank environment variable: {key!r}")
        cleaned[key] = str(v)
    return cleaned


def _process_rss_bytes(pid: int) -> int:
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True, timeout=2.0).strip()
        return int(out) * 1024 if out.isdigit() else 0
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0


def _find_rank_binaries() -> dict[str, str | None]:
    from .cluster_telemetry import _find_binaries
    return _find_binaries()


class RankManager:
    """Manages one local `tensorfold-native` rank process (rank 0 on Master or rank k on Slave)."""

    def __init__(self, state_dir: Path | None = None) -> None:
        self.state_dir = state_dir or (Path.home() / ".config" / "tensorfold" / "cluster")
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[str] | None = None
        self._api_key: str | None = None
        self._api_key_path: Path | None = None
        self._log: deque[str] = deque(maxlen=250)
        self._meta: dict[str, Any] = {
            "state": "idle",
            "rank": None,
            "world": None,
            "tp": None,
            "pp": None,
            "tp_rank": None,
            "pp_rank": None,
            "model_path": None,
            "host": None,
            "port": None,
            "api_auth_enabled": False,
            "pid": None,
            "started_at": None,
            "layer_loaded": 0,
            "layer_total": 0,
            "phase": "idle",
            "error": None,
            "settings": None,
        }

    def start(self, spec: dict[str, Any]) -> dict[str, Any]:
        if spec.get("binary"):
            raise ValueError("custom binary path is not allowed")
        bins = _find_rank_binaries()
        binary = bins.get("tensorfold_native")
        if not binary or not Path(binary).is_file():
            raise RuntimeError("tensorfold-native binary not found on this node (expected ~/bin/tensorfold-native)")
        model_path = str(spec.get("model_path") or "").strip()
        if not model_path or model_path.startswith("-") or not Path(model_path).is_dir():
            raise RuntimeError(f"model directory not found on this node: {model_path}")

        rank = int(spec.get("rank", 0))
        world = int(spec.get("world", 1))
        if world < 1 or rank < 0 or rank >= world:
            raise ValueError(f"invalid rank/world: rank={rank}, world={world}")
        port = int(spec.get("port", 8080 if rank == 0 else 8080 + rank))
        if not (1 <= port <= 65535):
            raise ValueError(f"invalid port: {port}")
        host = str(spec.get("host") or "127.0.0.1").strip()
        if not host or host.startswith("-") or any(c.isspace() for c in host):
            raise ValueError(f"invalid bind host: {host!r}")
        raw_api_key = spec.get("api_key")
        api_key = str(raw_api_key).strip() if raw_api_key is not None else ""
        if api_key and any(c.isspace() for c in api_key):
            raise ValueError("api_key must not contain whitespace")
        if not is_loopback_host(host) and not api_key:
            raise ValueError(
                "binding the inference API on a non-loopback address requires an API key; "
                "configure api_key or keep host='127.0.0.1'"
            )
        context = int(spec.get("context", 8192))
        parallel = int(spec.get("parallel", 1))
        if context < 1 or parallel < 1:
            raise ValueError("context and parallel must be positive integers")
        prompt_cache_gib = float(spec.get("prompt_cache_gib", 0))
        if prompt_cache_gib < 0:
            raise ValueError("prompt_cache_gib must be non-negative")
        temperature = float(spec["temperature"]) if spec.get("temperature") is not None else None
        load_limit_gb = float(spec["load_limit_gb"]) if spec.get("load_limit_gb") is not None else None
        if load_limit_gb is not None and load_limit_gb <= 0:
            raise ValueError("load_limit_gb must be positive")
        extra_env = _validate_rank_env(spec.get("env"))
        rank_settings = spec.get("rank_settings")

        self.stop()

        cmd = [
            binary,
            "serve",
            model_path,
            "--host", host,
            "--port", str(port),
            "--context", str(context),
            "--parallel", str(parallel),
            "--prompt-cache-gib", str(prompt_cache_gib),
            "--dashboard",
        ]
        if temperature is not None:
            cmd.extend(["--temperature", str(temperature)])
        key_path: Path | None = None
        if api_key:
            key_path = self.state_dir / f"rank{rank}.api_key"
            _write_secret_file(key_path, f"{api_key}\n")
            cmd.extend(["--api-key-file", str(key_path)])
        settings_path = None
        if rank_settings and world > 1:
            settings_path = self.state_dir / f"rank{rank}.json"
            settings_path.write_text(json.dumps(rank_settings, indent=2))
            cmd.extend(["--speed-up", str(settings_path)])
        elif rank_settings and world == 1 and spec.get("force_settings"):
            settings_path = self.state_dir / f"rank{rank}.json"
            settings_path.write_text(json.dumps(rank_settings, indent=2))
            cmd.extend(["--speed-up", str(settings_path)])

        env = os.environ.copy()
        lib_dir = str(Path.home() / "lib")
        env["DYLD_LIBRARY_PATH"] = f"{lib_dir}:{env.get('DYLD_LIBRARY_PATH', '')}".rstrip(":")
        env.setdefault("TF_GLM_M3_FAST", "1")
        env.setdefault("MCDMA_FABRIC_QOS", "1")
        env.setdefault("MCDMA_FABRIC_QPS", "1")
        if load_limit_gb is not None:
            env["GLM_LOAD_LIMIT_GB"] = str(load_limit_gb)
        env.update(extra_env)

        with self._lock:
            self._log.clear()
            self._log.append(f"[cluster] launch: {' '.join(cmd)}")
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
                start_new_session=True,
            )
            self._proc = proc
            self._api_key = api_key or None
            self._api_key_path = key_path
            layer_total = int(spec.get("layer_total") or 78)
            self._meta = {
                "state": "loading",
                "rank": rank,
                "world": world,
                "tp": int(spec.get("tp", world)),
                "pp": int(spec.get("pp", 1)),
                "tp_rank": int(spec.get("tp_rank", rank)),
                "pp_rank": int(spec.get("pp_rank", 0)),
                "model_path": model_path,
                "host": host,
                "port": port,
                "api_auth_enabled": bool(api_key),
                "pid": proc.pid,
                "started_at": time.time(),
                "layer_loaded": 0,
                "layer_total": layer_total,
                "phase": "initializing",
                "error": None,
                "settings": rank_settings,
            }

        threading.Thread(target=self._pump_output, args=(proc,), name=f"rank-{rank}-log", daemon=True).start()
        return self.status()

    def _pump_output(self, proc: subprocess.Popen[str]) -> None:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip("\r\n")
            with self._lock:
                self._log.append(line)
                m_layer = re.search(r"layer[s]?\s+(\d+)\s*/\s*(\d+)", line, re.IGNORECASE)
                if m_layer:
                    self._meta["layer_loaded"] = int(m_layer.group(1))
                    self._meta["layer_total"] = int(m_layer.group(2))
                    self._meta["phase"] = f"loading layer {m_layer.group(1)}/{m_layer.group(2)}"
                elif "listening" in line.lower() or "serving" in line.lower() or "ready" in line.lower():
                    if self._meta["state"] == "loading":
                        self._meta["state"] = "running"
                        self._meta["layer_loaded"] = self._meta["layer_total"]
                        self._meta["phase"] = "serving"
                elif "loading" in line.lower() and "weight" in line.lower():
                    self._meta["phase"] = "loading weights"
                elif "link" in line.lower() or "rdma" in line.lower() or "mesh" in line.lower():
                    if self._meta["state"] == "loading":
                        self._meta["phase"] = "syncing TB5 RDMA mesh"
                elif "error" in line.lower() or "panic" in line.lower() or "failed" in line.lower():
                    self._meta["error"] = line[:240]
        rc = proc.wait()
        with self._lock:
            if self._proc is proc:
                self._proc = None
                if self._meta["state"] not in ("stopped", "idle"):
                    self._meta["state"] = "failed" if rc != 0 else "stopped"
                    if rc != 0 and not self._meta["error"]:
                        self._meta["error"] = f"rank process exited with code {rc}"
                self._meta["pid"] = None

    @property
    def active_api_key(self) -> str | None:
        with self._lock:
            return self._api_key

    def stop(self) -> dict[str, Any]:
        with self._lock:
            proc = self._proc
            key_path = self._api_key_path
            self._proc = None
            self._api_key = None
            self._api_key_path = None
            self._meta["state"] = "stopped"
            self._meta["phase"] = "stopped"
            self._meta["pid"] = None
            self._meta["api_auth_enabled"] = False
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except OSError:
                try:
                    proc.terminate()
                except OSError:
                    pass
            try:
                proc.wait(timeout=4.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except OSError:
                    proc.kill()
                proc.wait(timeout=2.0)
        if key_path is not None:
            try:
                key_path.unlink(missing_ok=True)
            except OSError:
                pass
        return self.status()

    def logs(self, limit: int = 80) -> list[str]:
        with self._lock:
            return list(self._log)[-limit:]

    def status(self) -> dict[str, Any]:
        with self._lock:
            info = dict(self._meta)
            proc = self._proc
            api_key = self._api_key
            info["log_tail"] = list(self._log)[-15:]
        if proc is not None and proc.poll() is None:
            info["rss_bytes"] = _process_rss_bytes(proc.pid)
            # Scrape local /stats or /health to verify readiness & live tok/s (stdlib only — zero deps!)
            port = info.get("port")
            if port:
                headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
                try:
                    req = urllib.request.Request(f"http://127.0.0.1:{port}/stats", headers=headers)
                    with urllib.request.urlopen(req, timeout=0.6) as resp:
                        if resp.status == 200:
                            info["stats"] = json.loads(resp.read().decode("utf-8"))
                            if info["state"] == "loading":
                                info["state"] = "running"
                                info["phase"] = "serving"
                                info["layer_loaded"] = info["layer_total"]
                except (urllib.error.URLError, OSError, ValueError):
                    try:
                        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=0.4) as rh:
                            if rh.status == 200 and info["state"] == "loading":
                                info["state"] = "running"
                                info["phase"] = "serving"
                                info["layer_loaded"] = info["layer_total"]
                    except (urllib.error.URLError, OSError):
                        pass
        else:
            info["rss_bytes"] = 0
        return info
