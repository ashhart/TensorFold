from tensorfold_web.engine import totals_of
from tensorfold_web.usage import HOUR, Usage

T0 = 1_791_600_000 // HOUR * HOUR          # the start of an hour


def reading(model="Nemotron", prompt=0, completion=0, requests=0, cached=0, decode=0.0, prefill=0.0, status="running",
            connections=0):
    return {"status": status, "model": model, "decode_tps": decode, "prefill_tps": prefill, "connections": connections,
            "totals": {"prompt": prompt, "completion": completion, "cached": cached, "requests": requests}}


def test_totals_from_metrics():
    m = {"prompt_tokens_total": 19.0, "generation_tokens_total": 8.0, "prompt_tokens_cached_total": 0.0,
         "request_latency_seconds_count": 1.0}
    assert totals_of(m) == {"prompt": 19, "completion": 8, "cached": 0, "requests": 1}
    assert totals_of({"requests_running": 0.0}) is None


def test_growth_restarts_and_models(tmp_path):
    u = Usage(tmp_path / "usage.sqlite")
    u.record(reading(prompt=1000, completion=500, requests=3), T0 + 10)       # baseline only: no earlier reading
    u.record(reading(prompt=1500, completion=900, requests=5, decode=150.0), T0 + 20)
    # a TensorFold Web restart: a new instance reads the baseline the previous one kept
    u = Usage(tmp_path / "usage.sqlite")
    u.record(reading(prompt=1600, completion=1000, requests=6, prefill=6900.0), T0 + 30)
    u.record(reading(prompt=40, completion=20, requests=1), T0 + HOUR + 5)    # the engine restarted: all of it new
    u.record(reading(model="Qwen", prompt=10, completion=5, requests=1), T0 + HOUR + 10)   # another model
    u.record(reading(status="stopped"), T0 + HOUR + 20)                       # nothing served: nothing kept

    s = u.summary(days=2, hours=48, now=T0 + 2 * HOUR)
    models = {m["model"]: m for m in s["models"]}
    assert models["Nemotron"] == {"model": "Nemotron", "prompt": 640, "completion": 520, "cached": 0, "requests": 4}
    assert models["Qwen"]["completion"] == 5
    assert sum(d["requests"] for d in s["days"]) == 5
    assert max(d["peak_decode"] for d in s["days"]) == 150.0
    assert max(d["peak_prefill"] for d in s["days"]) == 6900.0
    assert [h["hour"] for h in s["hours"]] == [T0, T0 + HOUR]
    assert s["hours"][0]["models"]["Nemotron"] == {"prompt": 600, "completion": 500}
    assert s["last_served"] == {"t": T0 + HOUR + 10, "model": "Qwen"}


def test_idle_readings_do_not_move_last_served(tmp_path):
    u = Usage(tmp_path / "usage.sqlite")
    u.record(reading(completion=10, requests=1), T0)
    u.record(reading(completion=20, requests=2), T0 + 5)
    u.record(reading(completion=20, requests=2), T0 + 600)
    assert u.last_served() == {"t": T0 + 5, "model": "Nemotron"}
    u.record(reading(completion=20, requests=2, connections=1), T0 + 700)     # a request still running counts
    assert u.last_served()["t"] == T0 + 700


def test_a_restart_that_passes_the_old_totals_is_still_a_restart(tmp_path):
    u = Usage(tmp_path / "usage.sqlite")
    first = {**reading(completion=100, requests=1), "started_at": T0 - 600}
    u.record(first, T0)
    u.record({**reading(completion=120, requests=2), "started_at": T0 - 600}, T0 + 5)          # +20
    u.record({**reading(completion=150, requests=3), "started_at": T0 + 60}, T0 + 120)          # restarted: +150
    assert u.summary(days=1, now=T0 + 200)["models"][0]["completion"] == 170
