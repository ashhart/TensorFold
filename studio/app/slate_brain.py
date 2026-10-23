"""Slate's brain, as in Sam Wasserman's Slate (Apache-2.0): a Claude Code or Codex sign-in, or a local server."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

from fastapi import APIRouter

router = APIRouter()
TIMEOUT = 420
CLI_DIRS = ("/opt/homebrew/bin", "/usr/local/bin", str(Path.home() / ".local/bin"), str(Path.home() / ".claude/local"))
TIER_MODEL = {"fast": "haiku", "standard": "sonnet", "top": None}  # top: the sign-in's own default model


def cli(name: str) -> str | None:
    return shutil.which(name, path=os.pathsep.join([os.environ.get("PATH", ""), *CLI_DIRS]))


def endpoint_of(url: str) -> str:
    url = (url or "").strip().rstrip("/")
    if url and not url.startswith(("http://", "https://")):
        url = "http://" + url
    return url if not url or url.endswith("/v1") else url + "/v1"


def local_models(endpoint: str) -> list[str] | None:
    try:
        with urllib.request.urlopen(endpoint + "/models", timeout=2) as answer:
            return [m.get("id") for m in json.loads(answer.read()).get("data", []) if m.get("id")]
    except Exception:
        return None


def find_local(given: str = "") -> tuple[str | None, list[str]]:
    for endpoint in ([endpoint_of(given)] if given.strip() else [f"http://127.0.0.1:{port}/v1" for port in (11434, 1234, 8000, 8080)]):
        models = local_models(endpoint)
        if models is not None:
            return endpoint, models
    return None, []


def extract_json(text: str):
    """The first complete JSON value in a reply that may carry fences or a preamble (Slate's extractJson)."""

    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    for opener, closer in (("{", "}"), ("[", "]")):
        start = cleaned.find(opener)
        if start == -1:
            continue
        depth, quoted, escaped = 0, False, False
        for at in range(start, len(cleaned)):
            ch = cleaned[at]
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                quoted = not quoted
            elif not quoted and ch == opener:
                depth += 1
            elif not quoted and ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(cleaned[start:at + 1])
                    except ValueError:
                        break
    raise ValueError("no valid JSON in the reply")


def ask(backend: str, system: str, prompt: str, tier: str, endpoint: str = "", model: str = "") -> str:
    if backend == "local":
        base, models = find_local(endpoint)
        if not base:
            raise RuntimeError("no local model server answers: start Ollama, LM Studio, vLLM or llama.cpp, or set its address in the Bible")
        body = json.dumps({"model": model or (models[0] if models else ""), "temperature": 0.7,
                           "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}).encode()
        request = urllib.request.Request(base + "/chat/completions", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=TIMEOUT) as answer:
            return json.loads(answer.read())["choices"][0]["message"]["content"]
    tool = cli(backend if backend in ("claude", "codex") else "claude")
    if not tool:
        raise RuntimeError(f"{backend} is not installed on this machine: install it and sign in, or pick another brain in the Bible")
    env = {**os.environ, "PATH": os.pathsep.join([os.environ.get("PATH", ""), *CLI_DIRS])}
    with tempfile.TemporaryDirectory() as scratch:  # an empty folder: the brain writes text, it has no files to act on
        if backend == "codex":
            last = Path(scratch) / "last.txt"
            done = subprocess.run([tool, "exec", "--skip-git-repo-check", "--output-last-message", str(last), "-"],
                                  input=f"{system}\n\n---\n\n{prompt}", capture_output=True, text=True, timeout=TIMEOUT, cwd=scratch, env=env)
            if last.is_file() and last.read_text().strip():
                return last.read_text().strip()
            raise RuntimeError((done.stderr or done.stdout or "codex gave no answer").strip()[-400:])
        args = [tool, "-p", "--output-format", "json", "--tools", "", "--append-system-prompt", system]
        if TIER_MODEL.get(tier):
            args += ["--model", TIER_MODEL[tier]]
        done = subprocess.run(args, input=prompt, capture_output=True, text=True, timeout=TIMEOUT, cwd=scratch, env=env)
    raw = done.stdout.strip()
    try:
        parsed = json.loads(raw)
    except ValueError:
        if raw:
            return raw
        raise RuntimeError((done.stderr or "Claude Code gave no answer").strip()[-400:])
    result = parsed.get("result") if isinstance(parsed, dict) else None
    if isinstance(parsed, dict) and parsed.get("is_error"):
        message = result if isinstance(result, str) else "Claude Code returned an error"
        if re.search(r"authenticat|oauth|401|logged? ?in|revoked", message, re.I):
            message = "Claude Code's sign-in has expired: run `claude auth login` in a terminal on this machine, then retry. " + message
        raise RuntimeError(message)
    return result if isinstance(result, str) else raw


@router.get("/api/slate/brain")
def brain_status(endpoint: str = "") -> dict:
    base, models = find_local(endpoint)
    return {"claude": bool(cli("claude")), "codex": bool(cli("codex")), "local": bool(base), "endpoint": base, "models": models}


@router.post("/api/slate/brain")
def brain(body: dict) -> dict:
    started = time.time()
    try:
        text = ask(body.get("backend") or "claude", body.get("system") or "", body.get("prompt") or "", body.get("tier") or "top",
                   body.get("localEndpoint") or "", body.get("localModel") or "")
        out = {"ok": True, "text": text}
        if body.get("expectJson"):
            out["json"] = extract_json(text)
    except subprocess.TimeoutExpired:
        out = {"ok": False, "text": "", "error": f"the brain did not answer in {TIMEOUT // 60} minutes"}
    except Exception as error:  # the reason goes to the page as it is
        out = {"ok": False, "text": locals().get("text", ""), "error": str(error)[:600]}
    return {**out, "elapsedMs": int(1000 * (time.time() - started))}
