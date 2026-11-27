"""Local native-server transport for the exact draft parity device check."""
import hashlib
import http.client
import json
from pathlib import Path
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request


def require(ok, message):
    if not ok:
        raise ValueError(message)


def unique_object(pairs):
    out = {}
    for key, value in pairs:
        require(key not in out, f"duplicate JSON key: {key}")
        out[key] = value
    return out


def read_json(text):
    def invalid(value):
        raise ValueError(f"nonfinite JSON number: {value}")
    return json.loads(text, object_pairs_hook=unique_object, parse_constant=invalid)


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_identity(path):
    path = Path(path)
    configs = {name: sha256(path / name) for name in ("config.json", "model.safetensors.index.json", "tokenizer.json",
                                                                  "tokenizer_config.json", "generation_config.json",
                                                                  "special_tokens_map.json")
               if (path / name).is_file()}
    weights = []
    for item in sorted(path.glob("*.safetensors")):
        stat = item.stat()
        weights.append([item.name, stat.st_size, stat.st_mtime_ns, stat.st_ino])
    return {"config_sha256": configs, "root_weight_metadata": weights,
            "scope": "config/index hashes and root safetensors metadata; not a full weight digest"}


def unused_port(requested):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", requested))
        return probe.getsockname()[1]


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("local server redirect refused")


def fetch(opener, url, body, timeout):
    data = None if body is None else json.dumps(body, allow_nan=False).encode("utf-8")
    headers = {} if data is None else {"Content-Type": "application/json"}
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with opener.open(request, timeout=timeout) as response:
            data = response.read((16 << 20) + 1)
    except http.client.HTTPException as exc:
        raise OSError(f"native HTTP response incomplete or invalid: {exc}") from exc
    require(len(data) <= 16 << 20, "native response exceeds 16 MiB")
    return read_json(data)


def startup_drafts(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    labels = re.findall(r"\bserving [^\r\n]*; drafts:\s*([^;\r\n]+); context:", text)
    return labels[0].strip() if labels else None


class NativeServer:
    def __init__(self, command, model_name, port, directory, environ, startup_timeout, request_timeout):
        self.command, self.model_name, self.port = command, model_name, port
        self.directory, self.environ = Path(directory), environ
        self.startup_timeout, self.request_timeout = startup_timeout, request_timeout
        self.log_path = self.directory / "server.log"
        self.process = self.log = None
        self.health = None
        self.returncode = None
        self.killed = False
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def alive(self):
        code = self.process.poll()
        if code is not None:
            raise RuntimeError(f"native server exited {code}; see {self.log_path.name}")

    def __enter__(self):
        self.log = self.log_path.open("wb")
        try:
            self.process = subprocess.Popen(self.command, stdout=self.log, stderr=subprocess.STDOUT,
                                            cwd=Path.cwd(), env=self.environ)
            deadline = time.monotonic() + self.startup_timeout
            while time.monotonic() < deadline:
                self.alive()
                try:
                    health = fetch(self.opener, f"http://127.0.0.1:{self.port}/health", None, 1)
                except (urllib.error.URLError, TimeoutError, OSError):
                    time.sleep(0.1)
                    continue
                require(isinstance(health, dict) and health.get("model") == self.model_name,
                        "health does not identify the newly started server")
                if health.get("status") == "ok" and health.get("warming") is False:
                    self.alive()
                    self.health = health
                    return self
                time.sleep(0.1)
            raise TimeoutError("native server startup timed out")
        except BaseException:
            self.close()
            raise

    def request(self, body):
        self.alive()
        result = fetch(self.opener, f"http://127.0.0.1:{self.port}/v1/" + ("chat/completions" if "messages" in body else "completions"),
                       body, self.request_timeout)
        self.alive()
        return result

    def close(self):
        try:
            if self.process is not None:
                if self.process.poll() is None:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        self.killed = True
                        self.process.kill()
                        try:
                            self.process.wait(timeout=10)
                        except subprocess.TimeoutExpired as exc:
                            raise TimeoutError("owned native process did not exit after kill") from exc
                self.returncode = self.process.poll()
        finally:
            if self.log is not None:
                self.log.close()

    def draft_label(self):
        return startup_drafts(self.log_path) if self.log_path.exists() else None

    def __exit__(self, kind, value, traceback):
        try:
            if kind is None:
                self.alive()
        finally:
            self.close()
        return False
