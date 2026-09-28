"""MLX's freed buffers go back only from the engine's thread, once no stream is left to want them."""

from __future__ import annotations

import threading
import time
from typing import Any

from tensorfold.server.app import ChatApp
from tests.lane_fakes import FakeEngine
from tests.test_lane_server import FakeTokenizer
from tests.test_memory_window import Runtime, SizedFamily


def served(clears: list[tuple[str, int]]) -> ChatApp:
    model = SizedFamily(delay=0.004)
    model.args = type("Args", (), {"num_attention_heads": 1, "head_dim": 128})()
    runtime = Runtime(2**30)
    app = ChatApp(model, FakeTokenizer(), served_name="fake", lanes=2, default_max_tokens=16, checkpoint_slots=4,
                  use_proposer=False, engine_factory=FakeEngine, memory_budget_bytes=int(1.5 * 2**30),
                  memory_overhead_bytes=0, memory_runtime=runtime, context_window=8192)

    def clear() -> None:
        clears.append((threading.current_thread().name, app.scheduler.engine.active_count))
        runtime.cache = 0

    runtime.clear_cache = clear
    return app


def settle(app: ChatApp, clears: list[Any], count: int) -> None:
    deadline = time.perf_counter() + 2
    while (len(clears) < count or app.scheduler.active) and time.perf_counter() < deadline:
        time.sleep(0.01)


def test_a_stream_finishing_beside_another_leaves_the_cache_until_the_engine_is_idle() -> None:
    clears: list[tuple[str, int]] = []
    app = served(clears)
    try:
        clears.clear()                                   # the startup probe's own clears
        replies: dict[str, dict[str, Any]] = {}

        def ask(name: str, tokens: int) -> None:
            replies[name] = app.chat([{"role": "user", "content": name}], max_tokens=tokens)

        long = threading.Thread(target=ask, args=("long", 60))
        long.start()
        while not app.scheduler.active:
            time.sleep(0.001)
        ask("short", 2)
        assert app.scheduler.active == 1                 # the long reply is still decoding
        assert clears == []
        long.join(timeout=30)
        settle(app, clears, 1)
        assert clears == [("tensorfold-engine", 0)]      # once, from the engine's thread, with nothing decoding
        checkpoints = len(app.checkpoints)
        ask("short", 2)
        settle(app, clears, 2)
        assert clears == [("tensorfold-engine", 0)] * 2
        assert replies["long"]["content"] and len(app.checkpoints) >= checkpoints     # retained prefixes stay
    finally:
        app.close()
