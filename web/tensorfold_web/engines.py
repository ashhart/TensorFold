"""The engine interface: how TensorFold Web launches an engine, and what it can learn of any engine it reaches.

Two sides. A `Launcher` starts an engine TensorFold Web owns: the native TensorFold for the models qualified on it, the
frozen Python line for the others until they are ported. `identify` names the engine behind any OpenAI-compatible
address (TensorFold, vLLM, SGLang, llama.cpp, Ollama), so the Bench can measure peer engines on the same machine.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Protocol

import httpx

from . import native
from .settings import Settings


@dataclass
class Launch:
    kind: str                        # "native" or "python"
    cmd: list[str]
    env: dict[str, str]
    notes: list[str] = field(default_factory=list)


class Launcher(Protocol):
    kind: str

    def available(self) -> bool: ...

    def launch(self, model: str, s: Settings, drafter: str | None = None, slide: str | None = None) -> Launch: ...


class NativeLauncher:
    kind = "native"

    def available(self) -> bool:
        return native.find_binary() is not None

    def launch(self, model: str, s: Settings, drafter: str | None = None, slide: str | None = None) -> Launch:
        binary = native.find_binary()
        if binary is None:
            raise RuntimeError("TensorFold's native engine is not installed (brew install ashhart/tensorfold/tensorfold)")
        cmd, env, notes = native.build_command(binary, model, s, drafter=drafter, slide=slide)
        return Launch("native", cmd, env, notes)


class PythonLauncher:
    kind = "python"

    def available(self) -> bool:
        from .engine import tensorfold_command

        return tensorfold_command() is not None

    def launch(self, model: str, s: Settings, drafter: str | None = None, slide: str | None = None) -> Launch:
        # the Python line finds a model's drafter itself
        from .engine import build_command

        cmd, env = build_command(model, s)
        _, dropped = native.safe_extra(list(s.extra_args))
        notes = [f"extra flag {f} left out: TensorFold Web sets it itself" for f in dropped]
        return Launch("python", cmd, env, notes)


def choose(native_model: bool) -> Launcher:
    """The launcher for a model: the native engine when the model is qualified there and it is installed.

    A native model never falls back to the Python line when the binary is missing but the line is there: it may
    load there under another family and run inexactly or not at all. A model not yet ported needs the Python line.
    """

    if native_model:
        launcher: Launcher = NativeLauncher()
        if not launcher.available():
            raise RuntimeError("this model runs on TensorFold's native engine, which is not installed "
                               "(brew install ashhart/tensorfold/tensorfold)")
        return launcher
    launcher = PythonLauncher()
    if not launcher.available():
        raise RuntimeError("this model still needs TensorFold's Python line (0.6.6), which is not installed")
    return launcher


# -- engines TensorFold Web only reaches -------------------------------------------------------------------------------

# `owned_by` in /v1/models, as each server fills it
OWNED_BY = {"tensorfold": "TensorFold", "vllm": "vLLM", "sglang": "SGLang", "llamacpp": "llama.cpp",
            "llama.cpp": "llama.cpp", "mlx-lm": "MLX-LM"}


@dataclass
class Endpoint:
    url: str
    engine: str | None = None        # None: nothing answers, or not an OpenAI-compatible server
    version: str | None = None
    models: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _json(client: httpx.Client, url: str, headers: dict) -> dict | None:
    try:
        r = client.get(url, headers=headers)
    except httpx.HTTPError:
        return None
    if r.status_code != 200:
        return None
    try:
        body = r.json()
    except ValueError:
        return None
    return body if isinstance(body, dict) else None


def identify(url: str, *, key: str | None = None, client: httpx.Client | None = None) -> Endpoint:
    """Which engine answers at ``url``: `owned_by` first, then each server's own route (Ollama's `/api/version`,
    llama.cpp's `/props`, SGLang's `/get_server_info`, vLLM's `/version`). Anything else OpenAI-compatible is named so.
    """

    base = url.rstrip("/").removesuffix("/v1")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    own = client or httpx.Client(timeout=3.0)
    try:
        try:
            r = own.get(base + "/v1/models", headers=headers)
        except httpx.HTTPError as exc:
            return Endpoint(base, error=f"nothing answers: {exc}"[:200])
        if r.status_code == 401:
            return Endpoint(base, error="the server wants an API key")
        if r.status_code != 200:
            return Endpoint(base, error=f"/v1/models answered {r.status_code}")
        try:
            data = r.json().get("data") or []
        except (ValueError, AttributeError):
            return Endpoint(base, error="/v1/models is not OpenAI-compatible")
        ep = Endpoint(base, models=[m.get("id") for m in data if isinstance(m, dict) and m.get("id")])
        owners = {str(m.get("owned_by", "")).lower() for m in data if isinstance(m, dict)}
        ep.engine = next((OWNED_BY[o] for o in owners if o in OWNED_BY), None)
        if ep.engine is None and (v := _json(own, base + "/api/version", headers)) and "version" in v:
            ep.engine, ep.version = "Ollama", str(v["version"])
        elif ep.engine is None and (p := _json(own, base + "/props", headers)) and (
                "default_generation_settings" in p or "build_info" in p):
            ep.engine, ep.version = "llama.cpp", p.get("build_info")
        elif ep.engine is None and (g := _json(own, base + "/get_server_info", headers)) and "version" in g:
            ep.engine, ep.version = "SGLang", str(g["version"])
        elif ep.engine is None and (v := _json(own, base + "/version", headers)) and "version" in v:
            ep.engine, ep.version = "vLLM", str(v["version"])
        if ep.engine == "vLLM" and ep.version is None and (v := _json(own, base + "/version", headers)) and v.get("version"):
            ep.version = str(v["version"])
        ep.engine = ep.engine or "OpenAI-compatible"
        return ep
    finally:
        if client is None:
            own.close()
