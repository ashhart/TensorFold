"""Sparks: the list of workers and each worker's check, update and start, and stop over ssh."""

from __future__ import annotations

import json
import os
import signal
import threading
import time
from pathlib import Path

from fastapi import HTTPException

from sparks_rules import DISK_GB, LIMIT, MAC
from sparks_wire import CALL, PORT, Refused, call, digest, differs

PREFIX = "$HOME/.local/opt/tensorfold-studio"
THEIR_HOME = "$HOME/TensorFoldStudio"  # a worker's studio folder, and where its app's pid and log are kept
STATES = {"new": "not checked yet", "unreachable": "does not answer over ssh", "bare": "no Studio installed there",
          "stopped": "its Studio app is not running", "update": "its Studio code differs from this Spark's",
          "ready": "idle", "busy": "rendering", "down": "stopped answering"}


class WorkersMixin:
    """The Hub's workers: the list kept in sparks.json, their live state and tunnels, and what is done to them."""

    def config(self) -> dict:
        try:
            found = json.loads(self.book.read_text())
        except (OSError, ValueError):
            found = {}
        found.setdefault("workers", [])
        found.setdefault("sparks", 1)
        return found

    def keep(self, found: dict) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        part = self.book.with_suffix(".part")
        part.write_text(json.dumps(found, indent=1))
        part.replace(self.book)

    def workers(self) -> list[dict]:
        return self.config()["workers"] if self.spark else []

    def default_folder(self) -> str:
        """Where a worker's Studio is unless said otherwise: where this one is, under the home folder."""

        try:
            return str(self.pack.resolve().relative_to(Path.home().resolve()))
        except ValueError:
            return self.pack.name

    def state(self, address: str) -> dict:
        with self.mutex:
            return self.live.setdefault(address, {"state": "new", "why": "", "info": None, "base": None, "tunnel": None, "job": None,
                                                  "working": "", "checked": None, "fails": 0})

    def usable(self, worker: dict) -> str:
        """Why a worker is given no clip now ("" when it may be): off, not up, or rendering for someone else."""

        live = self.state(worker["address"])
        if not worker.get("enabled", True):
            return "switched off"
        if live["working"]:
            return live["working"]
        if live["state"] == "busy" and not live["job"]:
            return "rendering a job that is not this Spark's"
        if live["state"] not in ("ready", "busy"):
            return live["why"] or STATES.get(live["state"], live["state"])
        disk = (live["info"] or {}).get("disk_gb")
        if disk is not None and disk < DISK_GB:
            return f"its disk is nearly full ({disk:g} GB free)"
        return ""

    def shown(self, worker: dict) -> dict:
        live = self.state(worker["address"])
        info, why = live["info"] or {}, self.usable(worker)
        memory = info.get("memory") or {}
        return {"address": worker["address"], "name": worker.get("name") or worker["address"], "folder": worker.get("folder"),
                "enabled": worker.get("enabled", True), "state": live["state"], "says": STATES.get(live["state"], live["state"]),
                "why": live["why"], "usable": not why, "unusable": why, "working": live["working"], "checked": live["checked"],
                "job": live["job"], "version": info.get("version") or live.get("version"),
                "memory_gb": round(memory["available_mb"] / 1000, 1) if memory.get("available_mb") is not None else live.get("memory_gb"),
                "disk_gb": info.get("disk_gb"),
                "guard": bool(info.get("guard")),
                "can": {key: bool(info.get(key)) for key in ("fasth3", "image", "ref2va", "ref2va_turbo", "ffmpeg")} | {"loras": len(info.get("loras") or [])} if info else None}

    def overview(self) -> dict:
        if not self.spark:
            return {"supported": False, "limit": LIMIT, "note": MAC, "available": 1, "listed": 1, "default": 1, "workers": [], "this": {"name": self.name}}
        found = self.config()
        rows = [self.shown(worker) for worker in found["workers"]]
        mine = self.info()
        with self.lock:
            local = next((job["id"] for job in self.jobs.values() if job["status"] == "running" and job.get("spark") == self.name), None)
        return {"supported": True, "limit": LIMIT, "note": "", "available": 1 + sum(1 for row in rows if row["usable"]), "listed": 1 + len(rows),
                "default": max(1, min(int(found.get("sparks") or 1), LIMIT)), "workers": rows, "folder": self.default_folder(),
                "this": {"name": self.name, "version": mine.get("version"), "job": local, "state": "busy" if local else "ready",
                         "memory_gb": round((mine.get("memory") or {}).get("available_mb", 0) / 1000, 1), "disk_gb": mine.get("disk_gb"),
                         "can": {key: bool(mine.get(key)) for key in ("fasth3", "image", "ref2va", "ref2va_turbo", "ffmpeg")} | {"loras": len(mine.get("loras") or [])}}}

    # ---- a worker: check, update and start, stop ----

    def code(self) -> dict:
        if self.mine is None:
            self.mine = digest(self.pack)
        return self.mine

    def connect(self, address: str) -> str:
        """The worker's app as a local address, through a tunnel opened now if none is open."""

        live = self.state(address)
        with self.mutex:
            tunnel = live["tunnel"]
            if tunnel is not None and tunnel.alive() and live["base"]:
                return live["base"]
            if tunnel is not None:
                tunnel.close()
            live["tunnel"], live["base"] = None, None
        tunnel = self.dial(address)
        base = tunnel.open()
        with self.mutex:
            live["tunnel"], live["base"] = tunnel, base
        self.note_tunnels()
        return base

    def hang_up(self, address: str) -> None:
        live = self.state(address)
        with self.mutex:
            tunnel, live["tunnel"], live["base"] = live["tunnel"], None, None
        if tunnel is not None:
            tunnel.close()
        self.note_tunnels()

    def note_tunnels(self) -> None:
        """The open tunnels' process ids, written down: a start after a kill closes what that run left behind."""

        with self.mutex:
            pids = [live["tunnel"].process.pid for live in self.live.values()
                    if live["tunnel"] is not None and getattr(live["tunnel"], "process", None) is not None and live["tunnel"].alive()]
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            (self.folder / "tunnels.json").write_text(json.dumps(pids))
        except OSError:
            pass

    def close(self) -> None:
        """Every tunnel closed: when the app stops."""

        with self.mutex:
            addresses = list(self.live)
        for address in addresses:
            self.hang_up(address)

    def sweep(self) -> None:
        """Tunnels a killed run of this app left open: written-down ssh processes still forwarding to a worker."""

        note = self.folder / "tunnels.json"
        try:
            pids = json.loads(note.read_text())
        except (OSError, ValueError):
            return
        for pid in pids if isinstance(pids, list) else []:
            try:
                words = Path(f"/proc/{int(pid)}/cmdline").read_bytes().split(b"\0")
                if words[:1] == [b"ssh"] and b"-N" in words and any(word.endswith(f":127.0.0.1:{PORT}".encode()) for word in words):
                    os.kill(int(pid), signal.SIGTERM)
            except (OSError, ValueError):
                pass
        note.unlink(missing_ok=True)

    def probe(self, worker: dict) -> dict:
        """What ssh finds on a worker: name, Studio and environment, memory, an app on 127.0.0.1:PORT and if ours."""

        where = f'"$HOME/{worker.get("folder") or self.default_folder()}"'
        code, said = self.shell(worker["address"], (
            f'F={where}; P="${{PREFIX:-{PREFIX}}}"; H="{THEIR_HOME}"; echo "host $(hostname)"; echo "version $(cat "$F/VERSION" 2>/dev/null)"; '
            '[ -f "$F/app/server.py" ] && echo "pack yes"; [ -x "$P/venv/bin/python" ] && echo "install yes"; '
            'grep -E "^(MemTotal|MemAvailable)" /proc/meminfo; '
            f'(exec 3<>/dev/tcp/127.0.0.1/{PORT}) 2>/dev/null && echo "app yes"; '
            '[ -f "$H/app.pid" ] && kill -0 "$(cat "$H/app.pid")" 2>/dev/null && echo "ours yes"; true'), 25)
        found = {"reached": code == 0, "said": said}
        for line in said.splitlines() if code == 0 else []:
            key, _, value = line.partition(" ")
            if key in ("host", "version", "pack", "install", "app", "ours"):
                found[key] = value.strip()
            elif key.rstrip(":") in ("MemTotal", "MemAvailable"):
                found[key.rstrip(":")] = int(value.split()[0]) // 1024
        return found

    def check(self, worker: dict) -> dict:
        """Look at a worker and say what state it is in. Nothing on it is started, stopped or changed."""

        address, live = worker["address"], self.state(worker["address"])
        found = self.probe(worker)
        set_to = lambda state, why="", **more: live.update(state=state, why=why, checked=time.time(), fails=0, **more)
        if not found["reached"]:
            self.hang_up(address)
            set_to("unreachable", "ssh as this user did not get in (a key login is needed): " + found["said"][-160:], info=None)
            return live
        if found.get("host") and worker.get("name") in (None, "", address):
            self.rename(address, found["host"])
        live["version"] = found.get("version") or None
        live["memory_gb"] = round(found["MemAvailable"] / 1000, 1) if "MemAvailable" in found else None
        if not found.get("pack") or not found.get("install"):
            self.hang_up(address)
            set_to("bare", ("No Studio code in " + (worker.get("folder") or self.default_folder()) + " there. " if not found.get("pack") else "")
                   + ("" if found.get("install") else "No Studio environment there (models, ComfyUI, the engine): run oneshot-setup-spark.sh on that Spark once; "
                      "“Update and start” sends the code, not the models."), info=None)
            return live
        if not found.get("app"):
            self.hang_up(address)
            set_to("stopped", "Its Studio app is not running: “Update and start” starts it.", info=None)
            return live
        try:
            info = call(self.connect(address), "/api/spark/self", timeout=CALL)
        except Refused:
            set_to("update", f"Its app is Studio {found.get('version') or '?'}, from before the Sparks setting and the memory guard: “Update and start” "
                   "brings it to this Spark's code.", info=None)
            return live
        except OSError as error:
            self.hang_up(address)
            set_to("down", "Its app does not answer through the tunnel: " + str(error)[-160:], info=None)
            return live
        changed = differs(self.code(), info.get("code"))
        if changed or not info.get("guard"):
            set_to("update", f"Its Studio code differs from this Spark's in {len(changed)} file{'s' if len(changed) != 1 else ''} ({', '.join(changed[:3])}"
                   + (", …" if len(changed) > 3 else "") + "): “Update and start” brings it level.", info=info)
            return live
        if info.get("active") and not live["job"]:
            set_to("busy", "It is rendering a job that is not this Spark's.", info=info)
        else:
            set_to("busy" if live["job"] else "ready", "", info=info)
        return live

    def rename(self, address: str, name: str) -> None:
        found = self.config()
        taken = {worker.get("name") for worker in found["workers"] if worker["address"] != address} | {self.name}
        for worker in found["workers"]:
            if worker["address"] == address:
                worker["name"] = name if name not in taken else f"{name} ({address})"
        self.keep(found)

    def update(self, worker: dict) -> dict:
        """Code rsynced to a worker, its app started, worker checked; a rendering or foreign app is left alone."""

        address, live = worker["address"], self.state(worker["address"])
        folder = worker.get("folder") or self.default_folder()
        found = self.probe(worker)
        if not found["reached"]:
            return self.check(worker)
        if found.get("app"):
            try:
                if any(job.get("status") in ("queued", "running") for job in call(self.connect(address), "/api/jobs", timeout=CALL)):
                    live.update(why="It is rendering: update it when its job is done.", checked=time.time())
                    return live
            except (OSError, Refused):
                pass
        code, said = self.copy(self.pack, address, folder)
        if code:
            live.update(state="unreachable" if code == 255 else live["state"], why="The code could not be sent (rsync): " + said[-200:], checked=time.time())
            return live
        self.hang_up(address)
        if found.get("app") and not found.get("ours"):
            live.update(state="update", why="The code is there. An app that this Studio did not start is running on that Spark: stop it there, "
                        "then press “Update and start” again.", checked=time.time(), info=None)
            return live
        code, said = self.shell(address, self.stop_command(folder, comfy=False) + "; " + self.start_command(folder), 40)
        if code:
            live.update(state="stopped", why="Its app could not be started: " + said[-200:], checked=time.time(), info=None)
            return live
        for _ in range(30):  # the app needs a few seconds to come up
            time.sleep(1)
            if self.probe(worker).get("app"):
                break
        return self.check(worker)

    @staticmethod
    def start_command(folder: str) -> str:
        # in a subshell of its own with every stream closed off: ssh returns at once and the app outlives it
        return (f'cd "$HOME/{folder}" && mkdir -p "{THEIR_HOME}" && (STUDIO_PIDFILE="{THEIR_HOME}/app.pid" HOST=127.0.0.1 PORT={PORT} '
                f'nohup setsid bash scripts/app.sh > "{THEIR_HOME}/app.log" 2>&1 < /dev/null &) && echo started')

    @staticmethod
    def stop_command(folder: str, comfy: bool = True) -> str:
        """Stop the app a head started on a worker (by its checked pid file) and, with ``comfy``, its ComfyUI server."""

        stop = (f'H="{THEIR_HOME}"; if [ -f "$H/app.pid" ] && grep -qa "app/server.py" "/proc/$(cat "$H/app.pid")/cmdline" 2>/dev/null; then '
                'kill "$(cat "$H/app.pid")"; for i in 1 2 3 4 5 6 7 8 9 10; do kill -0 "$(cat "$H/app.pid")" 2>/dev/null || break; sleep 0.5; done; '
                'rm -f "$H/app.pid"; echo "app stopped"; else echo "no app of ours"; fi')
        return stop + (f'; cd "$HOME/{folder}" && bash spark/comfy.sh stop' if comfy else "")

    def stop(self, worker: dict) -> dict:
        """A worker's app (the one a head started) and its ComfyUI server stopped: its memory is free again."""

        address, live = worker["address"], self.state(worker["address"])
        if live["job"]:
            live.update(why="It is rendering one of this Spark's clips: cancel the job first.")
            return live
        found = self.probe(worker)
        if found["reached"] and found.get("app") and not found.get("ours"):
            live.update(why="The app on that Spark was not started by this Studio: it is left running.", checked=time.time())
            return live
        self.hang_up(address)
        self.shell(address, self.stop_command(worker.get("folder") or self.default_folder()), 40)
        return self.check(worker)

    def act(self, address: str, what: str) -> dict:
        """Check, update or stop a worker in the background; the page reads the result from the list."""

        worker = next((one for one in self.workers() if one["address"] == address), None)
        if worker is None:
            raise HTTPException(404, "no such Spark in the list")
        live = self.state(address)
        with self.mutex:
            if live["working"]:
                return self.shown(worker)
            live["working"] = {"check": "checking", "update": "updating and starting", "stop": "stopping"}[what]

        def go() -> None:
            try:
                {"check": self.check, "update": self.update, "stop": self.stop}[what](worker)
            except Exception as error:  # noqa: BLE001 - said beside the worker, never raised into a thread's void
                live.update(why=f"{what} failed: {str(error)[-200:]}", checked=time.time())
            finally:
                live["working"] = ""
                self.kick.set()

        threading.Thread(target=go, daemon=True).start()
        return self.shown(worker)
