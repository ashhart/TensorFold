"""The Python server in-process on the scripted engine: ChatApp and its real HTTP handler, nothing faked above them."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fake_text import FakeTokenizer
from script_scheduler import scheduler_class


class HfTokenizer:
    """A Hugging Face tokenizer as the server reads it: ids from apply_chat_template, end ids from config.json."""

    def __init__(self, directory: Path) -> None:
        from transformers import AutoTokenizer

        self._t = AutoTokenizer.from_pretrained(str(directory))
        config = json.loads((directory / "config.json").read_text())
        eos = config.get("eos_token_id")
        if eos is None:
            eos = (config.get("text_config") or {}).get("eos_token_id")
        self.eos_token_ids = set(eos if isinstance(eos, list) else [eos] if isinstance(eos, int) else [self._t.eos_token_id])

    def __getattr__(self, name: str) -> Any:
        return getattr(self._t, name)

    def __len__(self) -> int:
        return len(self._t)

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("return_dict", False)
        return self._t.apply_chat_template(messages, **kwargs)


class FakeMemory:
    """/health's memory snapshot: zeros, as the fake Zig engine reports."""

    def memory_snapshot(self, reset_peak: bool) -> dict[str, int]:
        return {"active": 0, "cache": 0, "peak": 0}


def start(fixtures: Path, *, context: int, keys: list[str] | None = None, key_file: str | None = None,
          environment: str = "", metrics_open: bool = False, tokenizer_dir: Path | None = None,
          scripts_file: str = "scripts.json") -> tuple[Any, int]:
    """A served ChatApp on a free port: the settings fake_serve gets from its flags."""
    from tensorfold.server import app as app_module
    from tensorfold.server.authentication import KeyStore
    from tensorfold.server.http import Server, make_handler

    pieces = json.loads((fixtures / "vocab.json").read_text())["pieces"]
    scripts = json.loads((fixtures / scripts_file).read_text())
    tokenizer = HfTokenizer(tokenizer_dir) if tokenizer_dir else FakeTokenizer(pieces)
    app_module.Scheduler = scheduler_class(tokenizer, scripts)
    app = app_module.ChatApp(None, tokenizer, served_name="fake-model", model_aliases=["fake-alias"],
                             engine_factory=lambda *a, **k: SimpleNamespace(round_stats=[]), lanes=4,
                             default_max_tokens=64, context_window=context, enable_thinking=True,
                             checkpoint_slots=0, use_proposer=True, max_rows=16, max_draft=8)
    app.prompt_memory = FakeMemory()
    store = KeyStore(keys or [], key_file=key_file, environment=environment, metrics_open=metrics_open)
    app.auth = store if store.enabled else None
    server = Server(("127.0.0.1", 0), make_handler(app))
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
    return server, server.server_port
