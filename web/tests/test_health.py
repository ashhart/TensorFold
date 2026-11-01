from tensorfold_web.health import GIB, Checker


def _ids(found):
    return [f.id for f in found]


def test_a_quiet_engine_has_no_findings():
    info = {"status": "running", "machine_memory": {"total": 32 * GIB, "available": 12 * GIB, "engine": 18 * GIB,
                                                     "headroom": "ok"}, "kv_fill": 0.1, "waiting": 0}
    assert Checker().check(info, {"mac_pressure": "normal", "thermal": "nominal"}) == []


def test_each_finding_has_its_evidence_and_what_to_try():
    info = {"status": "running", "machine_memory": {"total": 32 * GIB, "available": int(1.5 * GIB),
                                                     "engine": 25 * GIB, "headroom": "low"},
            "kv_fill": 0.95, "waiting": 2, "connections": 8}
    found = Checker().check(info, {"mac_pressure": "critical", "thermal": "serious",
                                   "gpu_slowdown": ["power cap", "thermal (hardware)"]})
    assert _ids(found) == ["memory-headroom", "mac-pressure", "kv-full", "waiting", "thermal", "gpu-slowdown"]
    mem = found[0]
    assert mem.level == "bad" and mem.evidence == "1.5 GB free of 32 GB; the engine holds 25.0 GB" and mem.hint
    assert found[-1].evidence == "thermal (hardware)"           # a power cap alone is not overheating


def test_counters_count_from_the_first_reading_and_start_over_when_they_go_back():
    c = Checker()
    assert c.check({"status": "running", "preemptions": 5.0}, {}) == []        # a baseline, not news
    found = c.check({"status": "running", "preemptions": 8.0}, {})
    assert _ids(found) == ["preemptions"] and found[0].evidence.startswith("3 since ")
    assert c.check({"status": "running", "preemptions": 1.0}, {}) == []        # the engine restarted
    assert _ids(c.check({"status": "running", "preemptions": 2.0}, {})) == ["preemptions"]


def test_a_failed_engine_and_a_refused_key():
    assert _ids(Checker().check({"status": "failed", "error": "exit code 1"}, {})) == ["engine-failed"]
    assert _ids(Checker().check({"status": "unauthorized", "error": "x"}, {})) == ["key"]


def test_an_engine_killed_from_outside_is_said_as_the_system_freeing_memory():
    found = Checker().check({"status": "failed", "error": "exit code -9"}, {})
    assert _ids(found) == ["engine-killed"] and "memory" in found[0].hint


def test_a_prompt_cache_without_memory_is_said():
    found = Checker().check({"status": "running", "startup": {"prompt_cache_gib": 0.0, "in_use_gib": 23.6}}, {})
    assert _ids(found) == ["prompt-cache-none"] and "23.6 GiB in use" in found[0].evidence
    assert Checker().check({"status": "running", "startup": {"prompt_cache_gib": 1.8}}, {}) == []


def test_the_memory_budget_is_advised_only_where_it_applies():
    for kind, said in (("python", True), ("native", False), (None, False)):
        found = Checker().check({"status": "failed", "error": "exit code -9", "engine_kind": kind}, {})
        assert ("in Settings" in found[0].hint) == said, kind
