"""Pure terminal rendering: no I/O, no mutable widget globals, no markup from telemetry or logs."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from importlib.resources import files
import io
import json
import math
from typing import Any

from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .safety import clean, redact
from .telemetry import Sample

BG = "#0a0d17"
PANEL = "#101624"
EDGE = "#29324a"
FG = "#e4ecfa"
MUTED = "#8591ad"
CYAN = "#60e1ed"
PINK = "#f27dce"
VIOLET = "#ac9aff"
GREEN = "#78e6b0"
AMBER = "#f1c784"
RED = "#ff8495"


@dataclass
class Node:
    name: str
    model: str
    endpoint: str
    managed: bool = False
    state: str = "unknown"
    pid: int | None = None
    last_exit: int | None = None
    sample: Sample | None = None
    rates: dict[str, float | None] = field(default_factory=dict)
    series: list[float | None] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)
    status_error: str = ""


@dataclass
class View:
    nodes: list[Node] = field(default_factory=list)
    selected: int = 0
    tab: str = "overview"
    paused: bool = False
    demo: bool = False
    busy: bool = False
    notice: str = "Ready. The dashboard never changes inference settings."
    confirm: str | None = None
    confirm_target: str = ""
    palette: bool = False
    palette_index: int = 0
    editor: dict[str, str] | None = None
    editor_index: int = 0
    log_filter: str = ""
    log_offset: int = 0
    help: bool = False

    @property
    def node(self) -> Node | None:
        return self.nodes[self.selected % len(self.nodes)] if self.nodes else None


@lru_cache(maxsize=4)
def logo(width: int = 30) -> Text:
    data = json.loads(files("tensorfold.control.assets").joinpath("logo-pixels.json").read_text("utf-8"))
    image = data["versions"][str(width)]
    text = Text(no_wrap=True)
    def color(pixel: list[int]) -> str:
        return "#" + "".join(f"{v:02x}" for v in pixel)
    for row in range(0, image["height"], 2):
        for a, b in zip(image["pixels"][row], image["pixels"][row + 1]):
            text.append("▀", f"{color(a)} on {color(b)}")
        if row < image["height"] - 2:
            text.append("\n")
    return text


def literal(value: Any, style: str = FG) -> Text:
    return Text(clean(value), style=style, overflow="ellipsis", no_wrap=True)


def number(value: float | None, suffix: str = "", places: int = 1) -> str:
    return "—" if value is None or not math.isfinite(value) else f"{value:,.{places}f}{suffix}"


def panel(content, title: str, *, border: str = EDGE, subtitle: str | None = None) -> Panel:
    return Panel(content, title=Text(title, style=f"bold {FG}"), title_align="left",
                 subtitle=Text(subtitle, style=MUTED) if subtitle else None, subtitle_align="left",
                 padding=(0, 1), border_style=border, style=f"{FG} on {PANEL}", box=box.ROUNDED)


def card(title: str, value: str, detail: str, color: str):
    return panel(Group(Text(value, style=f"bold {color}"), Text(detail, style=MUTED, overflow="ellipsis")), title)


def sparkline(values: list[float | None], width: int) -> Text:
    values = values[-width:]
    finite = [v for v in values if v is not None and math.isfinite(v)]
    top = max(finite, default=1) or 1
    output = Text(" " * max(0, width - len(values)), no_wrap=True)
    levels = "▁▂▃▄▅▆▇█"
    for value in values:
        output.append("·" if value is None else levels[min(7, int(max(0, value) / top * 7))],
                      MUTED if value is None else CYAN)
    return output


def chart(node: Node | None, width: int):
    series = node.series if node else []
    columns = max(10, width - 7)
    values = series[-columns:]
    top = max((v for v in values if v is not None), default=1) or 1
    rows = []
    for level in (3, 2, 1, 0):
        text = Text(" " * max(0, columns - len(values)), no_wrap=True)
        for value in values:
            if value is None:
                text.append("·" if level == 0 else " ", MUTED)
            else:
                h = max(0, min(4, value / top * 4))
                coverage = max(0, min(1, h - level))
                ch = " " if coverage == 0 else "▁▂▃▄▅▆▇█"[min(7, max(0, math.ceil(coverage * 8) - 1))]
                text.append(ch, CYAN if level < 2 else VIOLET)
        rows.append(text)
    rows.append(Text(f"0  {'─' * max(1, columns - 19)}  peak {number(top if values else None)}", style=MUTED))
    return panel(
        Group(*rows), "OUTPUT HISTORY",
        subtitle="aggregate tokens/s · 10s rolling counter rate · gaps = unknown")


def sidebar(view: View, width: int):
    mark = 30 if width >= 34 else 24
    brand = Text("T E N S O R F O L D", style=f"bold {FG}", justify="center")
    label = Text("CONTROL ROOM", style=f"bold {PINK}", justify="center")
    elements: list[Any] = [Align.center(logo(mark)), brand, label, Text(""), Text(" MODEL SERVICES", style=MUTED)]
    if not view.nodes:
        elements += [Text(" No profiles yet", style=MUTED), Text(" n  create a local service", style=CYAN)]
    else:
        start = max(0, view.selected - 4)
        for index, node in list(enumerate(view.nodes))[start:start + 7]:
            online = node.sample is not None and node.sample.online
            color = (
                AMBER if online and node.sample.phase == "warming"
                else GREEN if online
                else AMBER if node.pid else MUTED)
            line = Text(" ▸ " if index == view.selected else "   ", style=PINK)
            line.append("● ", color)
            line.append(clean(node.name, 28), f"bold {FG}" if index == view.selected else MUTED)
            elements.append(line)
            if index == view.selected:
                elements.append(Text(
                    "     " + ("launchd / local" if node.managed else "HTTP / monitor only"),
                    style=MUTED))
    elements += [Text(""), Text(" NO ENGINE PATCHES", style=f"bold {MUTED}"),
                 Text(" Observe. Control. Keep exact.", style=MUTED)]
    return Panel(Group(*elements), box=box.ROUNDED, border_style=EDGE, style=f"{FG} on {BG}", padding=(0, 1))


def session(node: Node | None, *, compact: bool = False):
    table = Table.grid(padding=(0, 1), expand=True)
    table.add_column(style=MUTED, width=13)
    table.add_column(style=FG, overflow="fold")
    if node:
        sample = node.sample
        rows = [
            ("profile", node.name),
            ("HTTP", sample.phase if sample else "not sampled"),
            ("model", sample.model if sample and sample.model else node.model),
            ("endpoint", node.endpoint),
            ("launchd", node.state if node.managed else "monitor only"),
            ("process", str(node.pid) if node.pid else "—"),
            ("last exit", str(node.last_exit) if node.last_exit is not None else "—"),
            ("context", number(sample.context, " tokens", 0) if sample and sample.online else "—"),
            ("KV peak pool", number(sample.kv_ratio * 100, "%")
             if sample and sample.online and sample.kv_ratio is not None else "—"),
            ("draft accept", number(sample.acceptance * 100, "%")
             if sample and sample.online and sample.acceptance is not None else "—"),
            ("TTFT mean", number(sample.ttft_mean, "s", 2) if sample and sample.online else "—")]
        if compact:
            rows = [item for item in rows if item[0] in {"profile", "HTTP", "endpoint", "launchd", "process"}]
        for label, value in rows:
            table.add_row(Text(label), Text(redact(value, 240)))
    else:
        table.add_row("setup", "Press n to install a cached model as a LaunchAgent.")
    return panel(table, "SESSION", subtitle="acceptance / TTFT: cumulative completed requests")


def logs(view: View, height: int):
    node = view.node
    lines = node.logs if node else []
    if view.log_filter:
        lines = [line for line in lines if view.log_filter.casefold() in line.casefold()]
    end = max(0, len(lines) - view.log_offset)
    rows = lines[max(0, end - max(1, height - 3)):end]
    if not rows:
        rows = [
            "No local logs yet." if not node or node.managed
            else "Remote log streaming is not exposed by this endpoint."]
    content = []
    for line in rows:
        color = RED if any(x in line.lower() for x in ("error", "failed", "traceback")) else MUTED
        content.append(Text(redact(line, 4096), style=color, overflow="ellipsis", no_wrap=True))
    title = "LIVE LOG" if view.log_offset == 0 else "LOG / SCROLLED"
    subtitle = "redacted view · f search · PgUp/PgDn scroll · End follow"
    if view.log_filter:
        subtitle = "filter: " + clean(view.log_filter, 60)
    return panel(Group(*content), title, subtitle=subtitle)


ACTIONS = [("start", "Start selected service"), ("stop", "Stop and disable selected service"),
           ("restart", "Gracefully restart selected service"), ("new", "Install a cached model"),
           ("logs", "Open local logs"), ("overview", "Show overview"), ("help", "Keyboard help")]


def overlay(view: View):
    if view.confirm:
        return panel(Group(Text("CONFIRM SERVICE CHANGE", style=f"bold {AMBER}"), Text(""),
                           Text(f"{view.confirm.upper()}  {clean(view.confirm_target)}", style=f"bold {FG}"),
                           Text(""),
                           Text("Active requests may be interrupted. "
                                "Model files are never removed.", style=MUTED),
                           Text("Enter / y  confirm     Esc / n  cancel", style=CYAN)), "SERVICE CONTROL", border=AMBER)
    if view.editor is not None:
        content = [Text("Cached models only. No weights are downloaded by this form.", style=MUTED), Text("")]
        for i, (key, value) in enumerate(view.editor.items()):
            active = i == view.editor_index
            content.append(Text(
                ("▸ " if active else "  ") + key.ljust(10)
                + clean(value, 180) + ("▏" if active else ""),
                style=f"bold {CYAN}" if active else FG))
        content += [Text(""), Text("Tab next field · Enter install / apply · Esc cancel", style=MUTED)]
        return panel(Group(*content), "NEW SERVICE" if "model" in view.editor else "LOG SEARCH", border=CYAN)
    if view.palette:
        content = [Text("SELECT A COMMAND", style=MUTED), Text("")]
        for index, (_, title) in enumerate(ACTIONS):
            content.append(Text(("▸ " if index == view.palette_index else "  ") + title,
                                style=f"bold {CYAN}" if index == view.palette_index else FG))
        content += [Text(""), Text("↑ ↓ select · Enter run · Esc close", style=MUTED)]
        return panel(Group(*content), "COMMAND PALETTE", border=VIOLET)
    return panel(Group(Text("KEYBOARD", style=f"bold {PINK}"), Text(""),
                       Text("j / k / ↑ / ↓     select model service"),
                       Text("s                 start selected service"),
                       Text("x / r             stop / restart (confirmation)"),
                       Text("n                 install a cached model profile"),
                       Text("Tab / l / d       switch view / logs / dashboard"),
                       Text("/                 command palette"),
                       Text("f                 filter local logs"),
                       Text("PgUp / PgDn / End scroll / follow logs"),
                       Text("Space             pause monitoring, not inference"),
                       Text("q / Ctrl+C        leave TUI; service keeps running"), Text(""),
                       Text("Esc closes this panel. Unknown metrics are shown as —.",
                            style=MUTED)), "CONTROL ROOM", border=PINK)


def render(view: View, width: int, height: int) -> Layout | Panel:
    if width < 72 or height < 23:
        return panel(Group(Text("TENSORFOLD", style=f"bold {CYAN}"),
                           Text("This dashboard needs at least 72 × 23 terminal cells."),
                           Text(f"Current size: {width} × {height}. Resize, or press q to leave.")), "TERMINAL SIZE")
    root = Layout()
    root.split_column(Layout(name="header", size=3), Layout(name="body"), Layout(name="notice", size=1),
                      Layout(name="keys", size=1))
    mode = "DEMO / SIMULATED" if view.demo else "PAUSED" if view.paused else "LIVE / READ-ONLY TELEMETRY"
    title = Text(" TENSORFOLD ", style=f"bold {FG}")
    title.append("/ CONTROL", VIOLET)
    header = Table.grid(expand=True)
    header.add_column(ratio=1)
    header.add_column(justify="right")
    header.add_row(title, Text(mode + "  ", style=AMBER if view.demo or view.paused else CYAN))
    root["header"].update(Panel(header, box=box.HORIZONTALS, border_style=EDGE, style=f"on {BG}"))
    side = 34 if width >= 112 else 28
    root["body"].split_row(Layout(sidebar(view, side), size=side), Layout(name="main"))
    if view.confirm or view.editor is not None or view.palette or view.help:
        root["main"].update(Align.center(overlay(view), vertical="middle"))
    else:
        node = view.node
        sample = node.sample if node else None
        usable = sample is not None and sample.online
        rates = node.rates if node else {}
        cards = Layout(size=5)
        generation = rates.get("generation") if usable else None
        prompt = rates.get("prompt") if usable else None
        rate_source = sample.sources.get("generation", "no counter") if usable else "waiting for telemetry"
        queue = f"{number(sample.running, places=0)} / {number(sample.waiting, places=0)}" if usable else "— / —"
        cards.split_row(Layout(card("OUTPUT TOK/S", number(generation), rate_source, CYAN)),
                        Layout(card("PROMPT TOK/S", number(prompt), "completed prompt tokens", VIOLET)),
                        Layout(card("ACTIVE / WAIT", queue, "requests, not GPU lanes", PINK)))
        if width >= 126:
            memory = sample.memory / 1024**3 if usable and sample.memory is not None else None
            cards.add_split(Layout(card("MLX ACTIVE", number(memory, " GiB"), "GPU buffers only", GREEN)))
        if view.tab == "logs":
            root["main"].split_column(cards, Layout(logs(view, height - 10)))
        else:
            history = (
                panel(sparkline(node.series if node else [], width - side - 6),
                      "OUTPUT HISTORY") if height < 32
                else chart(node, width - side))
            root["main"].split_column(cards, Layout(history, size=4 if height < 32 else 8), Layout(name="lower"))
            if width >= 126:
                root["lower"].split_row(
                    Layout(session(node, compact=height < 32)),
                    Layout(logs(view, height - (14 if height < 32 else 18))))
            else:
                root["lower"].update(session(node, compact=height < 32))
    node = view.node
    failure = node.sample.error if node and node.sample and node.sample.error else ""
    warning = node.sample.warning if node and node.sample else ""
    interacting = view.editor is not None or view.confirm or view.busy or view.notice.startswith("Operation failed:")
    line = view.notice if interacting else failure or (node.status_error if node else "") or warning or view.notice
    root["notice"].update(Text(" " + redact(line, width - 2), style=AMBER if failure or warning else MUTED,
                              no_wrap=True, overflow="ellipsis"))
    keys = " j/k select   s start   x stop   r restart   n new   / commands   Tab view   ? help   q quit"
    if view.busy:
        keys = " Service operation in progress · UI stays responsive · wait before exiting"
    root["keys"].update(Text(keys, style=f"{CYAN} on {BG}", no_wrap=True, overflow="ellipsis"))
    return root


def console_frame(view: View, width: int, height: int, *, color: bool = True,
                  record: bool = False) -> tuple[str, Console]:
    output = io.StringIO()
    console = Console(file=output, width=width, height=height, force_terminal=True,
                      color_system="truecolor" if color else None, record=record,
                      style=f"{FG} on {BG}", markup=False, highlight=False, legacy_windows=False)
    console.print(render(view, width, height), end="")
    return output.getvalue(), console
