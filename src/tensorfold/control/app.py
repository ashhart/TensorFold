"""Keyboard-driven control room. Blocking OS/network calls never run on the terminal event loop."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
import sys
import time
from typing import Any

from prompt_toolkit.application import Application
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout as TerminalLayout
from prompt_toolkit.layout import Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.output import ColorDepth

from .config import Profile, Store
from .demo import demo_view
from .launchd import Manager
from .logs import Tail
from .safety import ControlError, redact
from .telemetry import Client, Rates
from .view import ACTIONS, View, Node, console_frame


class ControlApp:
    def __init__(self, *, manager: Manager | None = None, urls: list[str] | None = None,
                 profile: str | None = None, demo: bool = False, interval: float = 2,
                 token: str | None = None, color: str = "auto", input=None, output=None):
        if not 0.5 <= interval <= 30:
            raise ControlError("poll interval must be 0.5 through 30 seconds")
        self.manager = manager or Manager()
        self.store = Store(self.manager.paths)
        self.interval, self.token = interval, token
        self.color = color
        self.view = demo_view() if demo else View()
        self.urls = list(urls or [])
        self.clients: dict[str, Client] = {}
        self.rates: dict[str, Rates] = {}
        self.tails: dict[str, Tail] = {}
        self.tick = 0
        self._epoch = 0
        self._last_profiles = 0.0
        if not demo:
            self.reload_profiles()
        if profile:
            found = next((i for i, n in enumerate(self.view.nodes) if n.name == profile), None)
            if found is None:
                raise ControlError(f"unknown profile: {profile}")
            self.view.selected = found
        self.bindings = self._bindings()
        control = FormattedTextControl(self._text, focusable=True, show_cursor=False)
        depth = {"truecolor": ColorDepth.DEPTH_24_BIT, "256": ColorDepth.DEPTH_8_BIT,
                 "mono": ColorDepth.DEPTH_1_BIT}.get(color)
        self.application: Application = Application(
            layout=TerminalLayout(Window(control, wrap_lines=False, always_hide_cursor=True)),
            full_screen=True, key_bindings=self.bindings, color_depth=depth,
            input=input, output=output, min_redraw_interval=0.1,
            erase_when_done=True, mouse_support=False,
        )

    def _text(self):
        size = self.application.output.get_size()
        frame, _ = console_frame(self.view, size.columns, size.rows,
                                 color=self.color != "mono" and "NO_COLOR" not in os.environ)
        return ANSI(frame)

    def reload_profiles(self, result=None) -> None:
        previous = {(n.name, n.endpoint): n for n in self.view.nodes}
        selected = self.view.node.name if self.view.node else None
        profiles, errors = self.store.list() if result is None else result
        nodes = []
        for p in profiles:
            node = previous.get((p.name, p.endpoint)) or Node(p.name, p.model, p.endpoint, True)
            node.model = p.model
            nodes.append(node)
        for i, url in enumerate(self.urls):
            client = self.clients.setdefault(url, Client(url, self.token, timeout=min(2, self.interval)))
            name = f"remote:{i + 1}"
            nodes.append(previous.get((name, client.endpoint)) or Node(name, "", client.endpoint))
        self.view.nodes = nodes
        self.view.selected = next((i for i, n in enumerate(nodes) if n.name == selected), 0)
        if errors:
            self.view.notice = "Profile error: " + "; ".join(errors)
        self._last_profiles = time.monotonic()

    async def refresh(self) -> None:
        if self.view.paused:
            return
        if self.view.demo:
            self.tick += 1
            current = demo_view(self.tick)
            self.view.nodes[0] = current.nodes[0]
            return
        if not self.view.busy and time.monotonic() - self._last_profiles >= 10:
            profiles = await asyncio.to_thread(self.store.list)
            self.reload_profiles(profiles)
        node = self.view.node
        if node is None:
            return
        client = self.clients.get(node.endpoint)
        if client is None:
            client = self.clients[node.endpoint] = Client(node.endpoint, self.token, timeout=min(2, self.interval))
        epoch = self._epoch
        sample = await asyncio.to_thread(client.sample)
        if self._epoch != epoch or self.view.busy or node not in self.view.nodes:
            return
        node.sample = sample
        tracker = self.rates.setdefault(node.name, Rates(max_gap=max(8, self.interval * 4)))
        node.rates = tracker.update(sample)
        node.series = (node.series + [node.rates.get("generation")])[-120:]
        if node.managed:
            try:
                # Only the selected job is inspected each poll: no O(N) subprocess storm.
                status = await asyncio.to_thread(self.manager.status, node.name)
                if self._epoch != epoch:
                    return
                node.state, node.pid, node.last_exit = status.state, status.pid, status.last_exit
                node.status_error = ""
            except (ControlError, OSError) as exc:
                node.status_error = redact(str(exc), 240)
            try:
                tail = self.tails.setdefault(node.name, Tail(self.manager.paths.log(node.name)))
                node.logs = await asyncio.to_thread(tail.read)
            except (ControlError, OSError) as exc:
                node.logs = ["[control] log unavailable: " + redact(str(exc))]

    async def _poll(self) -> None:
        while True:
            try:
                await self.refresh()
            except (ControlError, OSError, ValueError) as exc:
                self.view.notice = redact(str(exc), 400)
            self.application.invalidate()
            await asyncio.sleep(self.interval)

    async def run_async(self) -> None:
        def start() -> None:
            self.application.create_background_task(self._poll())
        await self.application.run_async(pre_run=start)

    def run(self) -> None:
        asyncio.run(self.run_async())

    def _invalidate(self) -> None:
        self.application.invalidate()

    def _quit(self) -> None:
        if self.view.busy:
            self.view.notice = "Wait for the bounded service operation to finish before exiting."
        else:
            self.application.exit()

    @property
    def normal(self) -> bool:
        return not (self.view.confirm or self.view.editor is not None or self.view.palette or self.view.help)

    def choose(self, step: int) -> None:
        if self.view.palette:
            self.view.palette_index = (self.view.palette_index + step) % len(ACTIONS)
        elif self.normal and self.view.nodes:
            self.view.selected = (self.view.selected + step) % len(self.view.nodes)
            self.view.log_offset = 0
        self._invalidate()

    def request(self, action: str) -> None:
        self.view.palette = False
        if action in {"logs", "overview"}:
            self.view.tab = action
        elif action == "help":
            self.view.help = True
        elif action == "new":
            if self.view.demo or self.manager.platform != "darwin":
                self.view.notice = "Service installation is disabled in demo / non-macOS monitoring."
            elif not self.view.busy:
                ports = {p.port for p in self.store.list()[0]}
                port = next((p for p in range(8080, 9000) if p not in ports), 8080)
                self.view.editor = {"name": "default", "model": "", "port": str(port)}
                self.view.editor_index = 0
        elif action in {"start", "stop", "restart"}:
            node = self.view.node
            if self.view.demo or not node or not node.managed or self.manager.platform != "darwin":
                self.view.notice = "Monitor-only endpoint: no service or remote process will be changed."
            elif self.view.busy:
                self.view.notice = "A service operation is already running."
            elif action == "start":
                self.view.busy = True
                self.application.create_background_task(self.operate(action, node.name))
            else:
                self.view.confirm = action
                self.view.confirm_target = node.name
        self._invalidate()

    async def operate(self, action: str, name: str, profile: Profile | None = None) -> None:
        self._epoch += 1
        self.view.busy = True
        self.view.notice = f"{action} {name} …"
        try:
            method = getattr(self.manager, action)
            status = await asyncio.to_thread(method, profile if profile else name)
            self.view.notice = f"{name}: {status.state}. {status.detail}"
            profiles = await asyncio.to_thread(self.store.list)
            self.reload_profiles(profiles)
            for node in self.view.nodes:
                if node.name == name:
                    node.state, node.pid, node.last_exit = status.state, status.pid, status.last_exit
                    node.status_error = ""
                    node.sample = None
                    self.rates.pop(name, None)
                    self.view.selected = self.view.nodes.index(node)
        except (ControlError, OSError, ValueError) as exc:
            self.view.notice = "Operation failed: " + redact(str(exc), 600)
        finally:
            self._epoch += 1
            self.view.busy = False
            self._invalidate()

    def _accept(self) -> None:
        if self.view.confirm:
            action, name = self.view.confirm, self.view.confirm_target
            self.view.confirm = None
            if not self.view.busy:
                self.view.busy = True
                self.application.create_background_task(self.operate(action, name))
        elif self.view.palette:
            self.request(ACTIONS[self.view.palette_index][0])
        elif self.view.editor is not None:
            fields = self.view.editor
            if "filter" in fields:
                self.view.log_filter = fields["filter"]
                self.view.log_offset = 0
                self.view.editor = None
            else:
                try:
                    model = fields["model"]
                    local = Path(model).expanduser()
                    if local.is_dir():
                        model = str(local.absolute())
                    profile = Profile(fields["name"], model, port=int(fields["port"]))
                    self.view.editor = None
                    self.view.busy = True
                    self.application.create_background_task(self.operate("install", profile.name, profile))
                except (ControlError, ValueError) as exc:
                    self.view.notice = redact(str(exc), 400)
        self._invalidate()

    def _close(self) -> None:
        self.view.confirm = None
        self.view.editor = None
        self.view.palette = False
        self.view.help = False
        self._invalidate()

    def _bindings(self) -> KeyBindings:
        keys = KeyBindings()
        normal = Condition(lambda: self.normal)
        editing = Condition(lambda: self.view.editor is not None)
        confirming = Condition(lambda: self.view.confirm is not None)
        @keys.add("q", filter=normal)
        @keys.add("c-c")
        def quit_(event):
            self._quit()
        @keys.add("escape")
        def escape(event):
            self._close()
        @keys.add("down", filter=~editing & ~confirming)
        @keys.add("j", filter=normal)
        def down(event):
            self.choose(1)
        @keys.add("up", filter=~editing & ~confirming)
        @keys.add("k", filter=normal)
        def up(event):
            self.choose(-1)
        for key, action in (("s", "start"), ("x", "stop"), ("r", "restart"), ("n", "new"),
                            ("l", "logs"), ("d", "overview"), ("?", "help")):
            def handler(event, action=action):
                self.request(action)
            keys.add(key, filter=normal)(handler)
        @keys.add("/", filter=normal)
        def palette(event):
            self.view.palette = True
            self._invalidate()
        @keys.add(" ", filter=normal)
        def pause(event):
            self.view.paused = not self.view.paused
            self.view.notice = "Monitoring paused; inference continues." if self.view.paused else "Monitoring resumed."
            self.rates.clear()  # first sample after any pause is a baseline, not a fabricated rate
            self._invalidate()
        @keys.add("tab")
        def tab(event):
            if self.view.editor is not None:
                self.view.editor_index = (self.view.editor_index + 1) % len(self.view.editor)
            elif self.normal:
                self.view.tab = "logs" if self.view.tab == "overview" else "overview"
            self._invalidate()
        @keys.add("enter")
        @keys.add("y", filter=confirming)
        def accept(event):
            self._accept()
        @keys.add("n", filter=confirming)
        def no(event):
            self._close()
        @keys.add("f", filter=normal)
        def search(event):
            self.view.editor = {"filter": self.view.log_filter}
            self.view.editor_index = 0
            self.view.tab = "logs"
        @keys.add("pageup", filter=normal)
        def older(event):
            maximum = max(0, len(self.view.node.logs) - 1) if self.view.node else 0
            self.view.log_offset = min(maximum, self.view.log_offset + 10)
            self._invalidate()
        @keys.add("pagedown", filter=normal)
        def newer(event):
            self.view.log_offset = max(0, self.view.log_offset - 10)
            self._invalidate()
        @keys.add("end", filter=normal)
        def end(event):
            self.view.log_offset = 0
            self._invalidate()
        @keys.add("backspace", filter=editing)
        def backspace(event):
            field = list(self.view.editor)[self.view.editor_index]
            self.view.editor[field] = self.view.editor[field][:-1]
        @keys.add("c-u", filter=editing)
        def clear(event):
            self.view.editor[list(self.view.editor)[self.view.editor_index]] = ""
        @keys.add("<any>", filter=editing)
        def type_(event):
            field = list(self.view.editor)[self.view.editor_index]
            data = event.data
            if data and all(ord(c) >= 32 and c != "\x7f" for c in data):
                self.view.editor[field] = (self.view.editor[field] + data)[:512]
        @keys.add("<bracketed-paste>", filter=editing)
        def paste(event):
            # Never interpret pasted escape sequences or submit pasted newlines as keystrokes.
            field = list(self.view.editor)[self.view.editor_index]
            data = "".join(c for c in event.data if ord(c) >= 32 and c != "\x7f")
            self.view.editor[field] = (self.view.editor[field] + data)[:512]
        return keys
