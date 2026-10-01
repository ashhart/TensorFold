"""Explicit synthetic telemetry for UI testing and preview; never represented as a benchmark."""
import math
import time

from .telemetry import Sample
from .view import Node, View


def demo_view(tick: int = 0) -> View:
    rates = [110 + 24 * math.sin(i / 6) + 7 * math.sin(i * 1.7) for i in range(tick, tick + 70)]
    sample = Sample(time.monotonic(), True, "ready", "Qwen · demo fixture", {"generation": 24000, "prompt": 72000,
                    "drafted": 30000, "accepted": 24400}, {"generation": "simulated live counter"},
                    running=3, waiting=1, memory=21.6 * 1024**3, cache=2.4 * 1024**3, peak=24.0 * 1024**3,
                    context=32768, kv_ratio=0.42, acceptance=0.813, ttft_mean=0.38)
    node = Node("qwen-local", "Qwen · demo fixture", "http://127.0.0.1:8080", True, "running", 4271, 0,
                sample, {"generation": rates[-1], "prompt": 1482.0}, rates)
    node.logs = ["12:04:21 [control] service started · offline cache only",
                 "12:04:23 [tensorfold] model loaded; serving on loopback",
                 "12:04:25 [tensorfold] prompt cache hit · 6,144 tokens",
                 "12:04:25 [tensorfold] request admitted · slot 0",
                 "12:04:26 [tensorfold] 3 requests sharing a decode round",
                 "12:04:27 [tensorfold] request finished · length",
                 "12:04:28 [control] DEMO: these values are synthetic"]
    return View(
        [node, Node("glm-studio", "GLM · demo fixture", "http://127.0.0.1:8081", True, "stopped"),
         Node("spark-remote", "Remote · demo fixture", "http://192.0.2.10:8080", False, "monitor-only")],
        demo=True,
        notice=("DEMO MODE · simulated telemetry · "
                "service controls disabled · no network requests"))
