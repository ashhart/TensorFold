"""What this machine needs for TensorFold, checked one by one with how to fix each, and a report to send.

The checks drive the first-run checklist and the diagnostic in Settings: graphics backend, engine installed, engine
port, model folder, a model downloaded, engine running, memory headroom, who can reach TensorFold Web, and whether it
keeps running. The report adds the machine, the settings, the engine's state and log, the fleet and the usage, with
the keys and tokens taken out: it is meant to be sent to someone helping.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import socket
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlsplit

GB = 1024**3


@dataclass
class Check:
    id: str
    label: str
    status: str            # "pass", "warn", "fail", "skip"
    detail: str
    fix: str = ""          # what to do when it is not a pass

    def to_dict(self) -> dict:
        return asdict(self)


def _amd_gpu() -> str | None:
    """An AMD GPU on Linux, from the kernel's DRM devices (vendor 0x1002), without ROCm tools."""

    for vendor in Path("/sys/class/drm").glob("card*/device/vendor"):
        try:
            if vendor.read_text().strip() == "0x1002":
                return "an AMD GPU"
        except OSError:
            continue
    return None


INSTALL = "curl -fsSL https://tensorfold.dev/install.sh | sh"


def install_hint(os_name: str, arch: str) -> str:
    """TensorFold's own installer (1.0.10 and later): the latest release for an Apple Silicon Mac or NVIDIA Linux
    (x86_64 or aarch64), its sum checked, under ~/.tensorfold, where TensorFold Web looks."""

    return INSTALL


def backend_check(m: dict) -> Check:
    if m.get("backend") == "mlx":
        ceiling = m.get("gpu_ceiling_bytes")
        extra = f"; the GPU may use {ceiling / GB:.0f} GB of {m['ram_bytes'] / GB:.0f}" if ceiling else ""
        return Check("backend", "Graphics backend", "pass", f"{m.get('chip')}, Metal through MLX{extra}")
    if m.get("backend") == "cuda":
        return Check("backend", "Graphics backend", "pass", m.get("backend_note") or "CUDA")
    if sys.platform.startswith("linux") and _amd_gpu():
        return Check("backend", "Graphics backend", "warn", "an AMD GPU; TensorFold's ROCm support is not released "
                     "yet (upstream issue #426)", "watch the TensorFold releases; meanwhile point TensorFold Web at an "
                     "engine on another machine (Settings, Engine elsewhere)")
    return Check("backend", "Graphics backend", "fail", m.get("backend_note") or "no supported GPU",
                 "TensorFold needs Apple Silicon or an NVIDIA GPU of compute capability 8.6 or newer (8.9 on Arm); TensorFold Web can "
                 "still drive an engine on another machine (Settings, Engine elsewhere)")


def grammar_library() -> bool | None:
    """Whether `lib/libtfgrammar` sits beside the native binary's real folder; None when there is nothing to look
    at (no binary, or a release from before 1.0.6, which did not ship it)."""

    from . import native

    binary = native.find_binary()
    if binary is None:
        return None
    try:
        if native.version_tuple(native.version_of(binary)) < (1, 0, 6):
            return None
    except (RuntimeError, ValueError, OSError):
        return None
    lib = Path(os.path.realpath(binary)).parent.parent / "lib"
    return any(lib.glob("libtfgrammar.*"))


def engine_installed_check(m: dict, remote: bool) -> Check:
    if remote:
        return Check("installed", "TensorFold installed here", "skip", "the engine runs on another machine")
    v = m.get("tensorfold_version")
    if v and "Python" not in v:
        grammar = grammar_library()
        if grammar is False:                    # 1.0.6 and later ship it in lib/ beside bin/
            return Check("installed", "TensorFold installed here", "warn",
                         f"native engine {v}, without lib/libtfgrammar: JSON formats and required tool calls get a 400",
                         "unpack the whole release archive, keeping lib/ beside bin/, and link to the binary inside it")
        return Check("installed", "TensorFold installed here", "pass", f"native engine {v}")
    hint = install_hint(m.get("os", ""), m.get("arch", ""))
    if v:
        return Check("installed", "TensorFold installed here", "warn", f"only the frozen Python line ({v})",
                     f"install the native engine: {hint}")
    return Check("installed", "TensorFold installed here", "fail", "no tensorfold-native found on PATH or in "
                 "Homebrew's and the archives' usual places", hint)


def _port_answers(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def port_check(engine_url: str, status: str, remote: bool) -> Check:
    parts = urlsplit(engine_url)
    host, port = parts.hostname or "127.0.0.1", parts.port or 8080
    label = f"Engine port {port}"
    if status in ("running", "starting", "unauthorized"):
        return Check("port", label, "pass", f"TensorFold answers at {engine_url}")
    if remote:
        return Check("port", label, "fail", f"nothing answers at {engine_url}",
                     "start the engine on that machine, or correct its address in Settings")
    if _port_answers(host, port):
        return Check("port", label, "fail", f"another program holds port {port}",
                     f"stop it, or give the engine another port in Settings (Engine address, e.g. http://127.0.0.1:{port + 1})")
    return Check("port", label, "pass", f"free for the engine at {engine_url}")


def folder_check(root: Path, remote: bool) -> Check:
    if remote:
        return Check("folder", "Model folder", "skip", "models live on the engine's machine")
    probe = root if root.exists() else next((p for p in root.parents if p.exists()), Path.home())
    free = shutil.disk_usage(probe).free
    where = f"{root} ({free / GB:.0f} GB free)"
    fix = ("free some disk space, or keep the models on another disk: put HF_HUB_CACHE=/that/folder in web.env "
           "beside the settings and install the service again")
    if free < 5 * GB:
        return Check("folder", "Model folder", "fail", where, fix)
    if free < 30 * GB:
        return Check("folder", "Model folder", "warn", f"{where}: room for one small model at most", fix)
    return Check("folder", "Model folder", "pass", where)


def models_check(installed: list[str], remote: bool) -> Check:
    if remote:
        return Check("model", "A model downloaded", "skip", "models live on the engine's machine")
    if installed:
        more = f" and {len(installed) - 1} more" if len(installed) > 1 else ""
        return Check("model", "A model downloaded", "pass", f"{installed[0]}{more}")
    return Check("model", "A model downloaded", "warn", "none that runs here yet",
                 "pick one in Models: each says whether it fits this machine before you download it")


def engine_check(info: dict) -> Check:
    st = info.get("status")
    if st == "running":
        return Check("engine", "Engine running", "pass", f"serving {info.get('model')}")
    if st == "starting":
        return Check("engine", "Engine running", "warn", f"starting {info.get('model') or ''}".strip(),
                     "wait for it; Monitor shows the engine's log")
    if st == "unauthorized":
        return Check("engine", "Engine running", "fail", info.get("error") or "the engine refuses the key",
                     "set the engine's API key in Settings")
    if st == "failed":
        return Check("engine", "Engine running", "fail", info.get("error") or "the engine stopped with an error",
                     "read the end of its log in Monitor; the diagnostic report carries it too")
    return Check("engine", "Engine running", "warn", "no model is being served",
                 "start one from Models")


def memory_check(info: dict) -> Check:
    mm = info.get("machine_memory")
    if not mm:
        return Check("memory", "Memory headroom", "skip", "not measured for an engine on another machine")
    free = mm["available"] / GB
    if mm.get("headroom") == "low":
        return Check("memory", "Memory headroom", "fail", f"{free:.1f} GB free",
                     "close other applications or serve a smaller model; the engine or your apps may be stopped "
                     "to make room")
    if mm.get("headroom") == "tight":
        return Check("memory", "Memory headroom", "warn", f"{free:.1f} GB free", "close other applications")
    return Check("memory", "Memory headroom", "pass", f"{free:.1f} GB free of {mm['total'] / GB:.0f}")


def access_check(bind: str, token: bool) -> Check:
    if token:
        return Check("access", "Who can reach TensorFold Web", "pass", "its token is required" +
                     ("" if bind in ("127.0.0.1", "localhost", "::1") else f", listening on {bind}"))
    return Check("access", "Who can reach TensorFold Web", "pass", "this machine only, without a token",
                 "to reach it from your other devices: `tailscale serve`, and start it with a token "
                 "(`tensorfold-web service install` makes one)")


def service_check(st: dict | None) -> Check:
    if st is None:
        return Check("service", "Keeps running", "skip", "no service for this system yet")
    if st.get("running") and st.get("lingering", True):
        where = "LaunchAgent" if st["platform"] == "macos" else "systemd user unit"
        return Check("service", "Keeps running", "pass", f"{where}, running")
    if st.get("installed") and st.get("lingering") is False:
        return Check("service", "Keeps running", "warn", "the unit stops with the last session",
                     "sudo loginctl enable-linger $USER")
    return Check("service", "Keeps running", "warn", "started by hand: it will not come back after a restart",
                 "tensorfold-web service install")


# 1.0.11 is Cortheon's security release: before it a page could hide an order from the instruction filter, hold a
# request for minutes or stop the server; 1.0.10 had already limited reads to URLs the conversation has shown
CORTHEON_SAFE = (1, 0, 11)


def cortheon_check(settings: dict, version: str | None, remote: bool) -> Check:
    if not settings.get("cortheon"):
        return Check("cortheon", "Web research", "skip", "off (Settings, Web research)")
    if remote:
        return Check("cortheon", "Web research", "skip", "the engine runs on another machine: its own release decides")
    from . import native

    if not version or "Python" in version or native.version_tuple(version) >= CORTHEON_SAFE:
        return Check("cortheon", "Web research", "pass", f"on, with TensorFold {version}" if version else "on")
    return Check("cortheon", "Web research", "warn", f"on with TensorFold {version}: before 1.0.11 a web page can "
                 "hide orders from Cortheon's instruction filter, hold a request for minutes or stop the engine",
                 f"update TensorFold ({INSTALL}, or brew upgrade tensorfold), or turn Web research off in Settings")


def run(*, machine: dict, info: dict, settings: dict, cache_root: Path, installed: list[str], bind: str,
        token: bool, service_status: dict | None) -> list[Check]:
    remote = bool(settings.get("remote"))
    return [backend_check(machine), engine_installed_check(machine, remote),
            port_check(settings.get("engine_url", ""), info.get("status", ""), remote), folder_check(cache_root, remote),
            models_check(installed, remote), engine_check(info), memory_check(info), access_check(bind, token),
            service_check(service_status), cortheon_check(settings, machine.get("tensorfold_version"), remote)]


SECRET_PATTERNS = (re.compile(r"(?i)(bearer\s+)[^\s\"']+"), re.compile(r"(?i)((?:api[_-]?key|token)[\"']?\s*[:=]\s*[\"']?)[^\s\"',}]+"),
                   re.compile(r"(?i)(\b[a-z][a-z0-9+.-]*://)[^/\s:@\"']+:[^/\s@\"']+@"))   # user:password@ in a URL


def redact(text: str, secrets: list[str]) -> str:
    """Known secrets replaced by a mark, and anything that looks like a key or a bearer token too."""

    for s in secrets:
        if s and len(s) >= 6:
            for form in (s, json.dumps(s)[1:-1]):        # as written, and as it reads inside a JSON string
                text = text.replace(form, "[redacted]")
    for pat in SECRET_PATTERNS:
        text = pat.sub(lambda m: m.group(1) + "[redacted]" + ("@" if m.group(0).endswith("@") else ""), text)
    return text


def report(*, checks: list[Check], machine: dict, info: dict, settings: dict, studio_version: str, log: list[str],
           peers: list[dict], usage: dict, secrets: list[str]) -> dict:
    keep_info = ("status", "model", "error", "startup", "command", "engine_kind", "uptime_s", "context_length",
                 "decode_tps", "prefill_tps", "kv_fill", "prompt_cache", "preemptions", "machine_memory", "findings",
                 "engine_url", "managed", "owned")
    doc = {
        "about": "TensorFold Web diagnostic. It holds this machine's name, paths and the engine's log; keys and "
                 "tokens are taken out.",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "studio_version": studio_version,
        "python": platform.python_version(), "platform": platform.platform(),
        "checks": [c.to_dict() for c in checks], "machine": machine,
        "settings": {k: v for k, v in settings.items() if k != "api_key"},
        "engine": {k: info.get(k) for k in keep_info},
        "engine_log": log[-300:],
        "fleet": [{"name": p.get("name"), "url": p.get("url")} for p in peers],
        "usage": {"models": usage.get("models"), "last_served": usage.get("last_served"), "since": usage.get("since")},
    }
    return json.loads(redact(json.dumps(doc), secrets))
