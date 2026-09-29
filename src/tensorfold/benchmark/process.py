"""Own a loopback server for one benchmark and stop only the child started here."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import socket
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any
from urllib import error, request

from tensorfold import families, hub


def _complete(path: Path, required: tuple[str, ...] = ()) -> bool:
    tokenizer = ((path / "tokenizer.json").is_file() or (path / "tokenizer.model").is_file()
                 or ((path / "vocab.json").is_file() and (path / "merges.txt").is_file()))
    try:
        return (path / "config.json").is_file() and tokenizer and hub._cached_weights_complete(path, required_files=required)
    except (OSError, ValueError, TypeError, RecursionError):
        return False


def _checkpoint(model: str, download: bool) -> Path:
    path = Path(model).expanduser()
    if path.is_dir():
        return path.resolve()
    if not hub.is_repo_id(model):
        raise ValueError("Model must be a local checkpoint directory or a Hugging Face repo id")
    try:
        found = hub.resolve(model, download=True) if download else hub.cached(model)
    except (OSError, ValueError, ImportError):
        raise ValueError("Checkpoint could not be resolved; run tensorfold pull MODEL and retry") from None
    if found is None or not (found / "config.json").is_file():
        raise ValueError("Checkpoint is not cached; run tensorfold pull MODEL or retry with --download")
    return found.resolve()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _json(base_url: str, route: str) -> dict[str, Any] | None:
    # Bypass inherited HTTP proxies for the benchmark's loopback server.
    opener = request.build_opener(request.ProxyHandler({}))
    try:
        with opener.open(base_url + route, timeout=1) as response:
            body = response.read(65537)
        if len(body) > 65536:
            return None
        value = json.loads(body)
        return value if isinstance(value, dict) else None
    except (OSError, ValueError, error.URLError, TimeoutError):
        return None


class ManagedServer:
    """Start an isolated single-rank server with prompt reuse disabled for repeatable measurements."""

    served_name = ".tensorfold-benchmark"

    def __init__(self, model: str, backend: str = "auto", download: bool = False, context: int | None = None,
                 drafter: str = "auto", startup_timeout: float = 600, output_tokens: int = 256,
                 serial: bool = False):
        self.model = model
        self.backend = ("mlx" if sys.platform == "darwin" else "cuda") if backend == "auto" else backend
        if self.backend not in ("mlx", "cuda"):
            raise ValueError("Benchmark backend must be mlx or cuda")
        if not 0 < startup_timeout <= 3600:
            raise ValueError("Startup timeout must be greater than 0 and at most 3600 seconds")
        if type(output_tokens) is not int or not 1 <= output_tokens <= 32768:
            raise ValueError("Output tokens must be between 1 and 32768")
        if context is not None and (type(context) is not int or context < 0):
            raise ValueError("Context must be a nonnegative token count")
        self.download, self.context, self.drafter = bool(download), context, drafter
        self.startup_timeout, self.output_tokens, self.serial = float(startup_timeout), output_tokens, bool(serial)
        self.base_url = ""
        self._child: subprocess.Popen | None = None
        self._lock: Any = None
        self._log: Any = None
        self._old_sigterm: Any = None
        self._sigterm_installed = False

    def _lock_benchmark(self) -> None:
        directory = Path.home() / ".tensorfold" / "benchmarks"
        # Do not unlink the lock. Every contender must lock the same inode, including after a crash.
        try:
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            fd = os.open(directory / ".lock", os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
        except OSError:
            raise ValueError("Could not create the benchmark lock; check the TensorFold directory permissions") from None
        self._lock = os.fdopen(fd, "w")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock.close()
            self._lock = None
            raise ValueError("Another TensorFold benchmark is running; wait for it to finish") from None

    def _prepare(self) -> tuple[Path, str]:
        path = _checkpoint(self.model, self.download)
        try:
            family = families.detect(path)
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError):
            raise ValueError("This checkpoint has no supported TensorFold family; run tensorfold models") from None
        if self.backend not in families.backends_of(family):
            raise ValueError("This model family has no engine for the selected backend; run tensorfold models")
        if self.backend == "cuda" and family.model_type == "glm5_next":
            raise ValueError("This CUDA model needs two hosts; start its server separately and use --server")
        required = getattr(family.package, "REQUIRED_FILES", {}).get(self.model, ())
        if not _complete(path, required):
            if self.download and hub.is_repo_id(self.model):
                try:
                    path = hub.resolve(self.model, download=True, required_files=required).resolve()
                except (OSError, ValueError, ImportError):
                    raise ValueError("Checkpoint download is incomplete; run tensorfold pull MODEL and retry") from None
            if not _complete(path, required):
                raise ValueError("Checkpoint is incomplete; run tensorfold pull MODEL or retry with --download")
        try:
            families.require_readable(family, families.read_config(path), self.backend)
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError):
            raise ValueError("This checkpoint's weight format is unsupported by this backend; run tensorfold info MODEL") from None
        if self.serial or self.drafter in ("", "none"):
            return path, "none"
        if self.drafter == "auto":
            repo = getattr(family.package, "DRAFTER", "")
            if not repo:
                return path, "none"
            found = hub.cached(repo)
            return path, str(found.resolve()) if found is not None and _complete(found) else "none"
        draft_path = _checkpoint(self.drafter, self.download)
        if not _complete(draft_path):
            raise ValueError("Draft checkpoint is incomplete; run tensorfold pull DRAFTER or retry with --download")
        return path, str(draft_path)

    def _ready(self) -> bool:
        health = _json(self.base_url, "/health")
        if health is None or health.get("warming") is True:
            return False
        if not (health.get("status") == "ok" or health.get("ok") is True):
            return False
        models = _json(self.base_url, "/v1/models")
        entries = models.get("data", []) if models else []
        return isinstance(entries, list) and any(isinstance(model, dict) and model.get("id") == self.served_name
                                                for model in entries)

    def __enter__(self) -> ManagedServer:
        if self._child is not None or self._lock is not None:
            raise ValueError("This benchmark server is already running")
        try:
            self._lock_benchmark()
            try:
                self._old_sigterm = signal.getsignal(signal.SIGTERM)
                signal.signal(signal.SIGTERM, self._terminate_signal)
                self._sigterm_installed = True
            except ValueError:
                pass                         # signal handlers belong to the main thread
            path, drafter = self._prepare()
            port = _free_port()
            self.base_url = f"http://127.0.0.1:{port}"
            command = [sys.executable, "-m", "tensorfold", "serve", str(path), "--host", "127.0.0.1",
                       "--port", str(port), "--name", self.served_name, "--backend", self.backend,
                       "--tp", "1", "--rank", "0", "--snapshot-dir", "none", "--max-snapshots", "0",
                       "--checkpoint-slots", "0", "--prompt-cache-gib", "0", "--spill-gib", "0",
                       "--parallel", "1", "--no-update-check", "--max-tokens", str(self.output_tokens),
                       "--drafter", drafter]
            if self.context is not None:
                command.extend(["--context", str(self.context)])
            if self.serial:
                command.append("--no-drafts")
            env = dict(os.environ)
            env.pop("TENSORFOLD_REQUEST_LOG", None)
            env.pop("TENSORFOLD_BENCHMARK_TOKEN", None)
            env["TENSORFOLD_NO_UPDATE_CHECK"] = "1"
            env["PYTHONUNBUFFERED"] = "1"
            if not self.download:
                env["HF_HUB_OFFLINE"] = "1"
                env["TRANSFORMERS_OFFLINE"] = "1"
            # Private startup output is kept only in an unlinked temporary file, never in receipts or upload errors.
            self._log = tempfile.TemporaryFile(mode="w+b")
            try:
                self._child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=self._log, stderr=self._log,
                                               env=env, start_new_session=True)
            except OSError:
                raise ValueError("Could not start the benchmark server; check the TensorFold Python installation") from None
            deadline = time.monotonic() + self.startup_timeout
            while time.monotonic() < deadline:
                if self._child.poll() is not None:
                    raise ValueError("Benchmark server exited during startup; check the checkpoint and backend toolchain")
                if self._ready():
                    return self
                time.sleep(0.2)
            raise ValueError("Benchmark server startup timed out; check the checkpoint, GPU memory and backend toolchain")
        except OSError:
            self.close()
            raise ValueError("Benchmark server could not start; check local checkpoint and directory permissions") from None
        except BaseException:
            self.close()
            raise

    def _terminate_signal(self, signum: int, frame: Any) -> None:
        raise SystemExit(128 + signum)

    def close(self) -> None:
        child, self._child = self._child, None
        try:
            if child is not None:
                if child.poll() is None:
                    try:
                        child.terminate()
                    except ProcessLookupError:
                        pass
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        child.kill()
                    except ProcessLookupError:
                        pass
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        raise ValueError("Benchmark server did not exit after termination; cleanup is incomplete") from None
        finally:
            if self._log is not None:
                self._log.close()
                self._log = None
            if self._lock is not None:
                fcntl.flock(self._lock, fcntl.LOCK_UN)
                self._lock.close()
                self._lock = None
            if self._sigterm_installed:
                signal.signal(signal.SIGTERM, self._old_sigterm)
                self._sigterm_installed = False

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()
