"""Pure terminal rendering: no I/O, no mutable widget globals, no markup from telemetry or logs."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from importlib.resources import files
import io
import json
import math
import os
from typing import Any

from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..i18n import t
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
# A narrower header turns the ribbon into blocks, so the wordmark replaces it.
MIN_LOGO_COLUMNS = 24
WORDMARK = "TensorFold"
# Palette action ids, in order; labels are chosen at render time.
ACTIONS = ("start", "stop", "restart", "new", "logs", "overview", "help")
_ACTION_TITLES = {"start": "Start selected service", "stop": "Stop and disable selected service",
                  "restart": "Gracefully restart selected service", "new": "Install a cached model",
                  "logs": "Open local logs", "overview": "Show overview", "help": "Keyboard help"}
# The compact session table keeps these rows: the first entry of each is its stable id.
_COMPACT_ROWS = {"profile", "HTTP", "endpoint", "launchd", "process"}
# Keyboard help: one key column, 17 cells wide, and one localized description column.
_HELP_ROWS = (("j / k / ↑ / ↓", "select model service"), ("s", "start selected service"),
              ("x / r", "stop / restart (confirmation)"), ("n", "install a cached model profile"),
              ("Tab / l / d", "switch view / logs / dashboard"), ("/", "command palette"),
              ("f", "filter local logs"), ("PgUp / PgDn / End", "scroll / follow logs"),
              ("Space", "pause monitoring, not inference"), ("q / Ctrl+C", "leave TUI; service keeps running"))


def actions() -> list[tuple[str, str]]:
    """Command palette entries: the action id and its localized label."""

    return [(name, t(_ACTION_TITLES[name])) for name in ACTIONS]


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
    notice: str = field(default_factory=lambda: t("Ready. The dashboard never changes inference settings."))
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


def use_truecolor(mode: str = "auto") -> bool:
    """24-bit logo only when the terminal says it can show it, or the flag asks for it."""
    if "NO_COLOR" in os.environ or mode in {"mono", "256"}:
        return False
    if mode == "truecolor":
        return True
    return os.environ.get("COLORTERM", "") in {"truecolor", "24bit"}


def _wordmark() -> Text:
    return Text(WORDMARK, style=f"bold {FG}", no_wrap=True, justify="center")


@lru_cache(maxsize=1)
def _logo_table() -> dict:
    return json.loads(files("tensorfold.control.assets").joinpath("logo-pixels.json").read_text("utf-8"))


def _half_blocks(image: dict) -> Text:
    """One cell, two pixel rows: upper half block, foreground on top, background underneath."""
    text = Text(no_wrap=True, overflow="crop")
    rows = image["pixels"]

    def color(pixel: list[int]) -> str:
        return "#" + "".join(f"{int(v):02x}" for v in pixel)

    for row in range(0, image["height"], 2):
        if row:
            text.append("\n")
        below = rows[row + 1]
        for top, bottom in zip(rows[row], below):
            text.append("▀", f"{color(top)} on {color(bottom)}")
    return text


def logo_image(columns: int) -> Text:
    return _half_blocks(_logo_table()["versions"][str(columns)])


def logo(columns: int, *, truecolor: bool) -> Text:
    table = _logo_table()["versions"]
    if not truecolor or columns < MIN_LOGO_COLUMNS or str(columns) not in table:
        return _wordmark()
    return _half_blocks(table[str(columns)])


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
    rows.append(Text(t("0  {rule}  peak {peak}", rule="─" * max(1, columns - 19),
                       peak=number(top if values else None)), style=MUTED))
    return panel(
        Group(*rows), t("OUTPUT HISTORY"),
        subtitle=t("decode tokens/s · the server live rate when reported · gaps = unknown"))


def sidebar(view: View, width: int, *, truecolor: bool):
    # Rounded border plus one column of padding on each side.
    columns = width - 4
    mark = logo(columns, truecolor=truecolor)
    brand = Text("T E N S O R F O L D", style=f"bold {FG}", justify="center")
    label = Text(t("CONTROL ROOM"), style=f"bold {PINK}", justify="center")
    elements: list[Any] = [Align.center(mark)]
    if "▀" in mark.plain:
        elements.append(brand)
    elements += [label, Text(""), Text(" " + t("MODEL SERVICES"), style=MUTED)]
    if not view.nodes:
        elements += [Text(" " + t("No profiles yet"), style=MUTED),
                     Text(" " + t("n  create a local service"), style=CYAN)]
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
                    "     " + t("launchd / local" if node.managed else "HTTP / monitor only"),
                    style=MUTED))
    elements += [Text(""), Text(" " + t("NO ENGINE PATCHES"), style=f"bold {MUTED}"),
                 Text(" " + t("Observe. Control. Keep exact."), style=MUTED)]
    return Panel(Group(*elements), box=box.ROUNDED, border_style=EDGE, style=f"{FG} on {BG}", padding=(0, 1))


def session(node: Node | None, *, compact: bool = False):
    table = Table.grid(padding=(0, 1), expand=True)
    table.add_column(style=MUTED, width=13)
    table.add_column(style=FG, overflow="fold")
    if node:
        sample = node.sample
        rows = [
            ("profile", node.name),
            ("HTTP", t(sample.phase) if sample else t("not sampled")),
            ("model", sample.model if sample and sample.model else node.model),
            ("endpoint", node.endpoint),
            ("launchd", t(node.state) if node.managed else t("monitor only")),
            ("process", str(node.pid) if node.pid else "—"),
            ("last exit", str(node.last_exit) if node.last_exit is not None else "—"),
            ("context", number(sample.context, " " + t("tokens"), 0) if sample and sample.online else "—"),
            ("KV peak pool", number(sample.kv_ratio * 100, "%")
             if sample and sample.online and sample.kv_ratio is not None else "—"),
            ("draft accept", number(sample.acceptance * 100, "%")
             if sample and sample.online and sample.acceptance is not None else "—"),
            ("TTFT mean", number(sample.ttft_mean, "s", 2) if sample and sample.online else "—")]
        if compact:
            rows = [row for row in rows if row[0] in _COMPACT_ROWS]
        for name, value in rows:
            table.add_row(Text(t(name)), Text(redact(value, 240)))
    else:
        table.add_row(t("setup"), t("Press n to install a cached model as a LaunchAgent."))
    return panel(table, t("SESSION"), subtitle=t("acceptance / TTFT: cumulative completed requests"))


def logs(view: View, height: int):
    node = view.node
    lines = node.logs if node else []
    if view.log_filter:
        lines = [line for line in lines if view.log_filter.casefold() in line.casefold()]
    end = max(0, len(lines) - view.log_offset)
    rows = lines[max(0, end - max(1, height - 3)):end]
    if not rows:
        rows = [
            t("No local logs yet.") if not node or node.managed
            else t("Remote log streaming is not exposed by this endpoint.")]
    content = []
    for line in rows:
        color = RED if any(x in line.lower() for x in ("error", "failed", "traceback")) else MUTED
        content.append(Text(redact(line, 4096), style=color, overflow="ellipsis", no_wrap=True))
    title = t("LIVE LOG") if view.log_offset == 0 else t("LOG / SCROLLED")
    subtitle = t("redacted view · f search · PgUp/PgDn scroll · End follow")
    if view.log_filter:
        subtitle = t("filter: {value}", value=clean(view.log_filter, 60))
    return panel(Group(*content), title, subtitle=subtitle)


def overlay(view: View):
    if view.confirm:
        return panel(Group(Text(t("CONFIRM SERVICE CHANGE"), style=f"bold {AMBER}"), Text(""),
                           Text(f"{t(view.confirm.upper())}  {clean(view.confirm_target)}", style=f"bold {FG}"),
                           Text(""),
                           Text(t("Active requests may be interrupted. Model files are never removed."),
                                style=MUTED),
                           Text(t("Enter / y  confirm     Esc / n  cancel"), style=CYAN)),
                     t("SERVICE CONTROL"), border=AMBER)
    if view.editor is not None:
        content = [Text(t("Cached models only. No weights are downloaded by this form."), style=MUTED), Text("")]
        for i, (key, value) in enumerate(view.editor.items()):
            active = i == view.editor_index
            content.append(Text(
                ("▸ " if active else "  ") + key.ljust(10)
                + clean(value, 180) + ("▏" if active else ""),
                style=f"bold {CYAN}" if active else FG))
        content += [Text(""), Text(t("Tab next field · Enter install / apply · Esc cancel"), style=MUTED)]
        return panel(Group(*content), t("NEW SERVICE") if "model" in view.editor else t("LOG SEARCH"), border=CYAN)
    if view.palette:
        content = [Text(t("SELECT A COMMAND"), style=MUTED), Text("")]
        for index, (_, title) in enumerate(actions()):
            content.append(Text(("▸ " if index == view.palette_index else "  ") + title,
                                style=f"bold {CYAN}" if index == view.palette_index else FG))
        content += [Text(""), Text(t("↑ ↓ select · Enter run · Esc close"), style=MUTED)]
        return panel(Group(*content), t("COMMAND PALETTE"), border=VIOLET)
    help_rows = [Text(f"{keys:<17} {t(description)}") for keys, description in _HELP_ROWS]
    return panel(Group(Text(t("KEYBOARD"), style=f"bold {PINK}"), Text(""), *help_rows, Text(""),
                       Text(t("Esc closes this panel. Unknown metrics are shown as —."),
                            style=MUTED)), t("CONTROL ROOM"), border=PINK)


def render(view: View, width: int, height: int, *, truecolor: bool = False) -> Layout | Panel:
    if width < 72 or height < 23:
        return panel(Group(Text("TENSORFOLD", style=f"bold {CYAN}"),
                           Text(t("This dashboard needs at least 72 × 23 terminal cells.")),
                           Text(t("Current size: {width} × {height}. Resize, or press q to leave.",
                                  width=width, height=height))), t("TERMINAL SIZE"))
    root = Layout()
    root.split_column(Layout(name="header", size=3), Layout(name="body"), Layout(name="notice", size=1),
                      Layout(name="keys", size=1))
    mode = t("DEMO / SIMULATED" if view.demo else "PAUSED" if view.paused else "LIVE / READ-ONLY TELEMETRY")
    title = Text(" TENSORFOLD ", style=f"bold {FG}")
    title.append(t("/ CONTROL"), VIOLET)
    header = Table.grid(expand=True)
    header.add_column(ratio=1)
    header.add_column(justify="right")
    header.add_row(title, Text(mode + "  ", style=AMBER if view.demo or view.paused else CYAN))
    root["header"].update(Panel(header, box=box.HORIZONTALS, border_style=EDGE, style=f"on {BG}"))
    side = 34 if width >= 112 else 28
    root["body"].split_row(Layout(sidebar(view, side, truecolor=truecolor), size=side), Layout(name="main"))
    if view.confirm or view.editor is not None or view.palette or view.help:
        root["main"].update(Align.center(overlay(view), vertical="middle"))
    else:
        node = view.node
        sample = node.sample if node else None
        usable = sample is not None and sample.online
        rates = node.rates if node else {}
        cards = Layout(size=5)
        live = sample.live if usable else {}
        decode, prefill = live.get("decode_tokens_per_second"), live.get("prefill_tokens_per_second")
        generation = decode if decode is not None else (rates.get("generation") if usable else None)
        prompt = prefill if prefill is not None else (rates.get("prompt") if usable else None)
        rate_source = (t("server, live") if decode is not None
                       else t(sample.sources.get("generation", "no counter"))) \
            if usable else t("waiting for telemetry")
        if "connections" in live:
            queue = f"{number(live['connections'], places=0)} / {number(live.get('waiting'), places=0)}"
        else:
            queue = f"{number(sample.running, places=0)} / {number(sample.waiting, places=0)}" if usable else "— / —"
        cards.split_row(Layout(card(t("DECODE TOK/S"), number(generation), rate_source, CYAN)),
                        Layout(card(t("PREFILL TOK/S"), number(prompt),
                                    t("server, live") if prefill is not None else t("completed prompt tokens"),
                                    VIOLET)),
                        Layout(card(t("CONNECTIONS / WAIT"), queue, t("open requests"), PINK)))
        if width >= 126:
            memory = sample.memory / 1024**3 if usable and sample.memory is not None else None
            cards.add_split(Layout(card(t("MLX ACTIVE"), number(memory, " GiB"), t("GPU buffers only"), GREEN)))
        if view.tab == "logs":
            root["main"].split_column(cards, Layout(logs(view, height - 10)))
        else:
            history = (
                panel(sparkline(node.series if node else [], width - side - 6),
                      t("OUTPUT HISTORY")) if height < 32
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
    interacting = (view.editor is not None or view.confirm or view.busy
                   or view.notice.startswith(t("Operation failed:")))
    line = view.notice if interacting else failure or (node.status_error if node else "") or warning or view.notice
    root["notice"].update(Text(" " + redact(line, width - 2), style=AMBER if failure or warning else MUTED,
                              no_wrap=True, overflow="ellipsis"))
    keys = " " + t("j/k select   s start   x stop   r restart   n new   / commands   Tab view   ? help   q quit")
    if view.busy:
        keys = " " + t("Service operation in progress · UI stays responsive · wait before exiting")
    root["keys"].update(Text(keys, style=f"{CYAN} on {BG}", no_wrap=True, overflow="ellipsis"))
    return root


def console_frame(view: View, width: int, height: int, *, color: bool = True,
                  truecolor: bool | None = None, record: bool = False) -> tuple[str, Console]:
    if truecolor is None:
        truecolor = bool(color) and use_truecolor("auto")
    output = io.StringIO()
    console = Console(file=output, width=width, height=height, force_terminal=True,
                      color_system="truecolor" if color else None, record=record,
                      style=f"{FG} on {BG}", markup=False, highlight=False, legacy_windows=False)
    console.print(render(view, width, height, truecolor=bool(truecolor)), end="")
    return output.getvalue(), console
