"""One owned native process at a time, with isolated checkpoint copies and retained HTTP receipts."""
import hashlib
import http.client
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def manifest(root):
    if not root.is_dir() or root.is_symlink():
        raise ValueError("checkpoint must be a real directory")
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("checkpoint must not contain symbolic links")
        if path.is_file():
            result[str(path.relative_to(root))] = {"sha256": digest(path), "bytes": path.stat().st_size}
    if not result:
        raise ValueError("checkpoint is empty")
    return result


def verify_copy(source, target, expected):
    if any(p.is_symlink() for p in target.rglob("*")):
        raise ValueError("checkpoint copy contains a symbolic link")
    paths = [p for p in target.rglob("*") if p.is_file()]
    if {str(p.relative_to(target)) for p in paths} != expected.keys():
        raise ValueError("checkpoint copy has a different file set")
    for path in paths:
        relative = str(path.relative_to(target))
        if path.is_symlink() or os.path.samestat((source / relative).stat(), path.stat()):
            raise ValueError("checkpoint copy shares a source inode")
        if path.stat().st_size != expected[relative]["bytes"]:
            raise ValueError("checkpoint copy has a different file size")


def copy_checkpoint(source, target, expected):
    target.mkdir()
    command = (["cp", "-cR"] if sys.platform == "darwin" else ["cp", "-a", "--reflink=auto"])
    command += [str(source) + "/.", str(target)]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        shutil.copytree(source, target, dirs_exist_ok=True)
    verify_copy(source, target, expected)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("local experiment server redirected a request")


class Server:
    def __init__(self, args, binary, root, model, thinking, seeded, slide, graph=None):
        self.args, self.root = args, root
        self.base = "http://127.0.0.1:" + str(args.port)
        self.key = secrets.token_hex(24)
        self.keyfile = root / "key"
        with os.fdopen(os.open(self.keyfile, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
            stream.write(self.key + "\n")
        self.command = [str(binary), "serve", str(model), "--host", "127.0.0.1", "--port", str(args.port),
                        "--api-key-file", str(self.keyfile), "--backend", args.backend,
                        "--slide-graph", str(graph or root / "graph.json"), *args.server_args]
        if slide:
            self.command.append("--slide")
        if seeded:
            self.command.append("--prompt-seed")
        if not thinking:
            self.command.append("--no-thinking")
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        self.process, self.log = None, None

    def __enter__(self):
        try:
            with socket.socket() as reservation:
                reservation.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                reservation.bind(("127.0.0.1", self.args.port))
            write(self.root / "command.json", self.command)
            self.log = (self.root / "server.log").open("wb")
            env = {k: v for k, v in os.environ.items() if not k.startswith("TF_SLIDE_")}
            env["TENSORFOLD_LOG_TIME"] = "1"
            self.process = subprocess.Popen(self.command, env=env, stdout=self.log, stderr=subprocess.STDOUT)
            write(self.root / "process.json", {"pid": self.process.pid})
            deadline = time.monotonic() + self.args.startup_timeout
            while True:
                if self.process.poll() is not None:
                    raise RuntimeError("owned server exited during startup")
                try:
                    self.request("/health", timeout=min(2, self.args.startup_timeout))
                    break
                except (OSError, urllib.error.URLError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("owned server did not become ready")
                    time.sleep(0.05)
            return self
        except BaseException:
            self.close()
            raise

    def request(self, route, body=None, timeout=None):
        raw = None if body is None else json.dumps(body, ensure_ascii=False).encode()
        req = urllib.request.Request(self.base + route, raw, headers={"Authorization": "Bearer " + self.key,
                                                                    "Content-Type": "application/json"})
        try:
            with self.opener.open(req, timeout=timeout or self.args.request_timeout) as response:
                return response.read().decode()
        except urllib.error.HTTPError as error:
            (self.root / "http-error.txt").write_bytes(error.read())
            raise
        except http.client.IncompleteRead as error:
            (self.root / "http-incomplete.txt").write_bytes(error.partial)
            raise

    def json(self, route, body=None):
        return json.loads(self.request(route, body))

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=30)
            write(self.root / "process.json", {"pid": self.process.pid, "exit": self.process.returncode})
        if self.log is not None:
            self.log.close()
        if self.keyfile.exists():
            self.keyfile.unlink()

    def __exit__(self, *_):
        self.close()
