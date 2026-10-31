import time

from tensorfold_web.history import FIELDS, History, row_of

GIB = 1024**3
RUNNING = {"status": "running", "decode_tps": 31.5, "prefill_tps": 900.0, "connections": 1, "kv_fill": 0.02,
           "memory": {"footprint": 20 * GIB}, "machine_memory": {"available": 5 * GIB, "engine": 20 * GIB}}


def test_a_reading_becomes_a_row_and_a_stopped_engine_a_gap():
    assert row_of(RUNNING) == {"decode_tps": 31.5, "prefill_tps": 900.0, "connections": 1, "kv_fill": 0.02,
                               "engine_bytes": 20 * GIB, "available_bytes": 5 * GIB}
    assert row_of({"status": "stopped"}) == dict.fromkeys(FIELDS)


def test_rows_are_kept_for_the_window_and_dropped_past_the_retention(tmp_path):
    h = History(tmp_path / "history.sqlite", lambda: RUNNING, keep=3600)
    now = time.time()
    h.add({"status": "stopped"}, now - 7200)           # older than the retention once a newer row arrives
    h.add(RUNNING, now - 600)
    h.add({"status": "stopped"}, now - 300)
    h.add(RUNNING, now - 10)
    got = h.since(1800)
    assert len(got["t"]) == 3 and got["decode_tps"] == [31.5, None, 31.5] and got["every"] == 5.0
    assert len(h.since(8 * 3600)["t"]) == 3                 # the two-hour-old row went


def test_the_sampler_reads_on_its_own(tmp_path):
    h = History(tmp_path / "history.sqlite", lambda: RUNNING, every=0.05)
    h.start()
    end = time.time() + 5
    while len(h.since(60)["t"]) < 3 and time.time() < end:
        time.sleep(0.05)
    h.stop()
    assert len(h.since(60)["t"]) >= 3
