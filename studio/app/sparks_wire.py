"""Sparks: the wire to a worker: its app's JSON API, uploads, ssh commands, rsync and the ssh tunnel."""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

PORT = 7870      # a worker's app, on its own 127.0.0.1
CALL = float(os.environ.get("STUDIO_SPARK_CALL", "10"))    # the longest one request to a worker may take
ADDRESS = re.compile(r"(?:[a-z_][a-z0-9_-]{0,31}@)?[A-Za-z0-9][A-Za-z0-9.-]{0,252}")
FOLDER = re.compile(r"[A-Za-z0-9_+][A-Za-z0-9_.+/-]{0,200}")
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=3"]
# the files that are the app: a worker whose copy of any of them differs from this Spark's is updated before it is used
CODE = ("VERSION", "*.py", "app/*.py", "app/*.json", "app/static/*.js", "app/static/index.html", "spark/*.py", "spark/*.sh",
        "spark/tensorfold_h3/*.py", "scripts/*.sh", "scripts/*.py")


class Refused(Exception):
    """The worker's app answered and said no."""

    def __init__(self, status: int, said: str):
        super().__init__(said)
        self.status, self.said = status, said


def call(base: str, path: str, body=None, method: str | None = None, timeout: float = CALL):
    """One JSON request to an app. Refused when it answers with an error, OSError when it does not answer."""

    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(base + path, data, method=method or ("POST" if data is not None else "GET"),
                                     headers={"Content-Type": "application/json"} if data is not None else {})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as answer:
            return json.loads(answer.read() or b"null")
    except urllib.error.HTTPError as error:
        text = error.read().decode("utf-8", "replace")
        try:
            said = json.loads(text).get("error") or text
        except (ValueError, AttributeError):
            said = text
        raise Refused(error.code, str(said)[:600]) from None
    except ValueError as error:  # an answer that is not JSON: not this app
        raise OSError(f"not a Studio app's answer: {error}") from None


def send(base: str, path: Path, timeout: float = 300) -> str:
    """A file of this studio folder put into the worker's uploads (its /api/upload); the path it has there."""

    mark = uuid.uuid4().hex
    head = (f'--{mark}\r\nContent-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n").encode()
    request = urllib.request.Request(base + "/api/upload", head + path.read_bytes() + f"\r\n--{mark}--\r\n".encode(),
                                     headers={"Content-Type": f"multipart/form-data; boundary={mark}"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as answer:
            return json.loads(answer.read())["path"]
    except urllib.error.HTTPError as error:
        raise Refused(error.code, error.read().decode("utf-8", "replace")[:300]) from None
    except (ValueError, KeyError) as error:
        raise OSError(f"not a Studio app's answer: {error}") from None


def digest(pack: Path) -> dict:
    """The app's files (CODE) as {path: sha1}: what a worker's copy is held against."""

    found = {}
    for pattern in CODE:
        for path in sorted(pack.glob(pattern)):
            if path.is_file():
                found[str(path.relative_to(pack))] = hashlib.sha1(path.read_bytes()).hexdigest()
    return found


def differs(mine: dict, theirs: dict | None) -> list[str]:
    """The files of this Spark's code a worker has otherwise or not at all (what it has besides does not count)."""

    theirs = theirs or {}
    return sorted(name for name, value in mine.items() if theirs.get(name) != value)


def address_of(given: str) -> str:
    """``host`` or ``user@host`` as ssh takes it; ValueError for anything else (nothing ssh reads as an option)."""

    text = (given or "").strip()
    if not ADDRESS.fullmatch(text):
        raise ValueError("a Spark's address is its host name or IP address, as you would type it after ssh (192.0.2.10 or user@spark-5)")
    return text


def folder_of(given: str) -> str:
    """Where the Studio is on a worker, under its home folder."""

    text = (given or "").strip().strip("/")
    if not FOLDER.fullmatch(text) or ".." in text.split("/"):
        raise ValueError("the Studio's folder on a Spark is a path under the home folder there, like slate-studio")
    return text


def shell(address: str, command: str, timeout: float = 30) -> tuple[int, str]:
    """A command on a worker over ssh as this app's user: (status, output); 255 with the reason when ssh fails."""

    try:
        done = subprocess.run([*SSH, address, command], capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return 255, f"no answer over ssh in {timeout:g} s"
    except OSError as error:
        return 255, str(error)
    return done.returncode, (done.stdout + done.stderr).strip()


def copy(pack: Path, address: str, folder: str, timeout: float = 900) -> tuple[int, str]:
    """This Spark's Studio code rsynced to a worker; models, engine and environment are the worker's own install."""

    command = ["rsync", "-a", "--exclude", ".git", "--exclude", "samples", "--exclude", "outputs", "--exclude", "payload",
               "--exclude", "__pycache__", "--exclude", "*.pyc", "-e", " ".join(SSH), f"{pack}/", f"{address}:{folder}/"]
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return 255, f"the copy did not finish in {timeout:g} s"
    except OSError as error:
        return 255, str(error)
    return done.returncode, (done.stdout + done.stderr).strip()[-400:]


class Tunnel:
    """A free local port to 127.0.0.1:PORT on a worker through ssh; the worker's app stays closed to the network."""

    def __init__(self, address: str):
        self.address, self.process, self.port = address, None, None

    def open(self, wait: float = 12) -> str:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.port = probe.getsockname()[1]
        self.process = subprocess.Popen([*SSH, "-o", "ExitOnForwardFailure=yes", "-N", "-L", f"127.0.0.1:{self.port}:127.0.0.1:{PORT}", self.address],
                                        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        until = time.time() + wait
        while time.time() < until:
            if self.process.poll() is not None:
                raise OSError("no tunnel: " + ((self.process.stderr.read() or "").strip()[-200:] or "ssh stopped"))
            try:
                socket.create_connection(("127.0.0.1", self.port), timeout=1).close()
                return f"http://127.0.0.1:{self.port}"
            except OSError:
                time.sleep(0.15)
        self.close()
        raise OSError(f"no tunnel to {self.address} in {wait:g} s")

    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None
