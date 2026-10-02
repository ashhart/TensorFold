import asyncio
import math
from pathlib import Path

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.cells import cell_len

from tensorfold.control.app import ControlApp
from tensorfold.control.demo import demo_view
from tensorfold.control.telemetry import Sample
from tensorfold.control.view import View, Node, console_frame


async def until(condition, timeout=4):
    async def wait():
        while not condition():
            await asyncio.sleep(0.02)
    await asyncio.wait_for(wait(), timeout)


@pytest.mark.parametrize("width,height", [(72,23), (80,24), (100,32), (144,42), (220,60), (50,15)])
@pytest.mark.parametrize("tab", ["overview", "logs"])
def test_responsive_frame_has_no_overflow(width, height, tab):
    view = demo_view()
    view.tab = tab
    text, _ = console_frame(view, width, height, color=False)
    lines = text.splitlines()
    assert len(lines) <= height
    assert max(map(cell_len, lines)) <= width
    assert "TENSORFOLD" in text


def test_snapshot_uses_actual_logo_resource():
    from importlib.resources import files
    import json
    pixels = json.loads(
        files("tensorfold.control.assets").joinpath("logo-pixels.json").read_text())
    image = pixels["versions"]["30"]
    crop = pixels["crop"]
    aspect = (crop[2] - crop[0]) / (crop[3] - crop[1])
    assert image["width"] == 30 and image["height"] % 2 == 0
    assert abs(image["width"] / image["height"] - aspect) / aspect < 0.03
    assert all(len(row) == 30 for row in image["pixels"])
    text, _ = console_frame(demo_view(), 144, 42, truecolor=True)
    assert "▀" in text and "DEMO / SIMULATED" in text
    assert "TensorFold" not in text


def test_logo_falls_back_to_the_wordmark(monkeypatch):
    from tensorfold.control.view import logo
    monkeypatch.delenv("COLORTERM", raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    plain, _ = console_frame(demo_view(), 144, 42)
    assert "TensorFold" in plain and "▀" not in plain
    assert logo(16, truecolor=True).plain == "TensorFold"
    assert "▀" not in logo(23, truecolor=True).plain
    assert "▀" in logo(30, truecolor=True).plain


def test_small_logo_escape_snapshot():
    import io
    from rich.console import Console
    from tensorfold.control.view import _logo_table, logo_image
    image = _logo_table()["versions"]["8"]
    text = logo_image(8)
    assert text.plain.count("▀") == image["width"] * (image["height"] // 2)
    buf = io.StringIO()
    Console(file=buf, width=image["width"], force_terminal=True, color_system="truecolor",
            legacy_windows=False).print(text, end="")
    escapes = buf.getvalue()
    snapshot = Path(__file__).with_name("logo-8.escapes")
    assert escapes == snapshot.read_text()
    # Foreground is the top pixel and background is the bottom pixel.
    top, bottom = image["pixels"][0][0], image["pixels"][1][0]
    assert f"38;2;{top[0]};{top[1]};{top[2]};48;2;{bottom[0]};{bottom[1]};{bottom[2]}m▀" in escapes


def test_unsafe_remote_text_is_literal():
    view = demo_view()
    view.nodes[0].logs = ["\x1b]52;c;steal\x07[bold red]literal HF_TOKEN=hidden"]
    view.nodes[0].model = "[blink]literal model"
    view.nodes[0].sample.model = "[blink]literal model"
    text, _ = console_frame(view, 160, 42, color=False)
    assert "[bold red]literal" in text
    assert "[blink]literal model" in text
    assert "hidden" not in text and "steal" not in text and "\x1b" not in text


def test_real_keyboard_navigation_and_palette(manager):
    async def body():
        m, fake = manager
        with create_pipe_input() as pipe:
            ui = ControlApp(manager=m, demo=True, input=pipe, output=DummyOutput(), interval=0.5)
            task = asyncio.create_task(ui.run_async())
            await until(lambda: ui.application.is_running)
            pipe.send_text("j")
            await until(lambda: ui.view.selected == 1)
            pipe.send_text("k")
            await until(lambda: ui.view.selected == 0)
            pipe.send_text("l")
            await until(lambda: ui.view.tab == "logs")
            pipe.send_text(" ")
            await until(lambda: ui.view.paused)
            pipe.send_text("/")
            await until(lambda: ui.view.palette)
            pipe.send_text("\x1b[B")
            await until(lambda: ui.view.palette_index == 1)
            pipe.send_text("\r")
            await until(lambda: not ui.view.palette)
            assert not fake.calls  # even demo controls go through the no-side-effects guard
            pipe.send_text("q")
            await asyncio.wait_for(task, 3)

    asyncio.run(body())


def test_confirmation_is_required_and_target_is_stable(manager, profile, monkeypatch):
    async def body():
        m, fake = manager
        m.install(profile, start=True)
        monkeypatch.setattr("tensorfold.control.app.Client.sample", lambda _: Sample(1, True, "ready"))
        with create_pipe_input() as pipe:
            ui = ControlApp(manager=m, input=pipe, output=DummyOutput(), interval=0.5)
            task = asyncio.create_task(ui.run_async())
            await until(lambda: ui.application.is_running)
            before = sum(c[1] == "bootout" for c in fake.calls)
            pipe.send_text("x")
            await until(lambda: ui.view.confirm == "stop")
            pipe.send_text("n")
            await until(lambda: ui.view.confirm is None)
            assert sum(c[1] == "bootout" for c in fake.calls) == before
            pipe.send_text("r")
            await until(lambda: ui.view.confirm == "restart")
            assert ui.view.confirm_target == profile.name
            pipe.send_text("\r")
            await until(lambda: sum(c[1] == "bootout" for c in fake.calls) == before + 1 and not ui.view.busy)
            assert fake.loaded
            pipe.send_text("q")
            await asyncio.wait_for(task, 3)

    asyncio.run(body())


def test_new_service_key_refuses_the_smoke_name(manager, monkeypatch):
    async def body():
        m, fake = manager
        monkeypatch.setattr("tensorfold.control.app.Client.sample", lambda _: Sample(1))
        with create_pipe_input() as pipe:
            ui = ControlApp(manager=m, input=pipe, output=DummyOutput(), interval=0.5)
            task = asyncio.create_task(ui.run_async())
            await until(lambda: ui.application.is_running)
            pipe.send_text("n")
            await until(lambda: ui.view.editor is not None)
            pipe.send_text("\x15control-smoke\tOrg/Model\r")
            await until(lambda: "reserved" in ui.view.notice)
            assert ui.view.editor is not None and m.store.list()[0] == [] and not fake.calls
            pipe.send_text("\x1b")
            await until(lambda: ui.view.editor is None)
            pipe.send_text("q")
            await asyncio.wait_for(task, 3)

    asyncio.run(body())


def test_new_service_form_and_paste_are_real_key_events(manager, monkeypatch):
    async def body():
        m, fake = manager
        monkeypatch.setattr("tensorfold.control.app.Client.sample", lambda _: Sample(1))
        with create_pipe_input() as pipe:
            ui = ControlApp(manager=m, input=pipe, output=DummyOutput(), interval=0.5)
            task = asyncio.create_task(ui.run_async())
            await until(lambda: ui.application.is_running)
            pipe.send_text("n")
            await until(lambda: ui.view.editor is not None)
            pipe.send_text("\tOrg/Model")
            await until(lambda: ui.view.editor["model"] == "Org/Model")
            pipe.send_text("\r")
            await until(lambda: ui.view.editor is None and not ui.view.busy and len(ui.view.nodes) == 1)
            assert m.store.get("default").model == "Org/Model"
            assert not fake.loaded  # install is not a surprise model launch
            pipe.send_text("f")
            await until(lambda: ui.view.editor is not None)
            pipe.send_text("\x1b[200~error\n\x1b[201~")
            await until(lambda: ui.view.editor.get("filter") == "error")
            pipe.send_text("\r")
            await until(lambda: ui.view.log_filter == "error")
            pipe.send_text("q")
            await asyncio.wait_for(task, 3)

    asyncio.run(body())


def test_poll_failure_and_recovery_do_not_retain_fake_rates(manager, profile, monkeypatch):
    async def body():
        m, fake = manager
        m.install(profile)
        samples = iter([Sample(1, True, "ready", counters={"generation": 10}),
                        Sample(2, True, "ready", counters={"generation": 40}),
                        Sample(3, error="offline"), Sample(4, True, "ready", counters={"generation": 90})])
        monkeypatch.setattr("tensorfold.control.app.Client.sample", lambda _: next(samples))
        with create_pipe_input() as pipe:
            ui = ControlApp(manager=m, input=pipe, output=DummyOutput())
            await ui.refresh()
            assert ui.view.node.rates["generation"] is None
            await ui.refresh()
            assert ui.view.node.rates["generation"] == 30
            await ui.refresh()
            assert ui.view.node.rates["generation"] is None
            await ui.refresh()
            assert ui.view.node.rates["generation"] is None

    asyncio.run(body())


def test_demo_and_remote_actions_are_disabled(manager):
    m, fake = manager
    with create_pipe_input() as pipe:
        ui = ControlApp(manager=m, demo=True, input=pipe, output=DummyOutput())
        for action in ["start", "stop", "restart", "new"]:
            ui.request(action)
        assert not fake.calls and ui.view.confirm is None and ui.view.editor is None


def test_no_profile_is_a_useful_empty_state():
    text, _ = console_frame(View(), 144, 32, color=False)
    assert "No profiles yet" in text and "install" in text


def test_in_flight_telemetry_cannot_resurrect_stopped_service(manager, profile, monkeypatch):
    async def body():
        import threading
        m, _ = manager
        m.install(profile, start=True)
        began, finish = threading.Event(), threading.Event()
        def delayed(_):
            began.set()
            assert finish.wait(4)
            return Sample(1, True, "ready", counters={"generation": 9000})
        monkeypatch.setattr("tensorfold.control.app.Client.sample", delayed)
        with create_pipe_input() as pipe:
            ui = ControlApp(manager=m, input=pipe, output=DummyOutput())
            polling = asyncio.create_task(ui.refresh())
            await until(began.is_set)
            await ui.operate("stop", profile.name)
            finish.set()
            await polling
            assert ui.view.node.sample is None
            assert ui.view.node.state == "stopped"
            assert ui.view.node.pid is None

    asyncio.run(body())


def test_warming_visible_in_session():
    view = demo_view()
    view.nodes[0].sample.phase = "warming"
    text, _ = console_frame(view, 144, 36, color=False)
    assert "warming" in text


def test_cards_prefer_the_health_live_block():
    from tensorfold.control.telemetry import normalize
    sample = normalize(1, {"status": "ok", "model": "Fixture", "live": {
        "connections": 3, "waiting": 1, "decode_tokens_per_second": 142.4,
        "prefill_tokens_per_second": 1210,
    }}, {"tensorfold:requests_running": [9], "tensorfold:requests_waiting": [8]})
    node = Node("local", "Fixture", "http://127.0.0.1:8080", True, "running",
                sample=sample, rates={"generation": 17.5, "prompt": 6.5})
    text, _ = console_frame(View([node]), 144, 42, color=False)
    assert "DECODE TOK/S" in text and "142.4" in text and "server, live" in text
    assert "PREFILL TOK/S" in text and "1,210.0" in text
    assert "CONNECTIONS / WAIT" in text and "3 / 1" in text
    assert "17.5" not in text and "6.5" not in text
    fallback = normalize(1, {"status": "ok", "model": "Fixture"}, {
        "tensorfold:requests_running": [2], "tensorfold:requests_waiting": [4],
    })
    assert fallback.live == {}
    node.sample = fallback
    text, _ = console_frame(View([node]), 144, 42, color=False)
    assert "17.5" in text and "6.5" in text and "2 / 4" in text
    assert "completed prompt tokens" in text
