"""Whole-request admission, drain and retry behavior without GPU dependencies."""

from concurrent.futures import ThreadPoolExecutor
import threading
import time

import pytest

from tensorfold.server.lifecycle import Lifecycle, LifecycleError


def lifecycle(**overrides):
    calls = []
    callbacks = {name: lambda name=name: calls.append(name) for name in ("release", "restore", "cleanup")}
    return Lifecycle(**(callbacks | overrides)), calls


def test_sleep_wake_are_idempotent_and_keep_admission_closed_until_restore_finishes():
    entered, finish = threading.Event(), threading.Event()

    def restore():
        entered.set()
        assert finish.wait(5)

    state, calls = lifecycle(restore=restore)
    assert state.snapshot()["state"] == "awake"
    assert state.sleep()["state"] == "sleeping"
    assert state.sleep()["state"] == "sleeping"
    assert calls == ["release"]
    with ThreadPoolExecutor() as pool:
        wake = pool.submit(state.wake_up)
        assert entered.wait(5)
        assert state.snapshot()["state"] == "waking"
        with pytest.raises(LifecycleError) as refused:
            with state.admit():
                pytest.fail("request entered during wake")
        assert refused.value.status == 503
        with pytest.raises(LifecycleError) as conflict:
            state.sleep()
        assert conflict.value.status == 409
        finish.set()
        assert wake.result(5)["state"] == "awake"
    assert state.wake_up()["state"] == "awake"
    with state.admit():
        assert state.snapshot()["active_requests"] == 1
    assert state.snapshot()["active_requests"] == 0


def test_drain_waits_for_accepted_request_and_allows_its_nested_translation():
    state, calls = lifecycle()
    with ThreadPoolExecutor() as pool:
        with state.admit():
            sleep = pool.submit(state.sleep)
            deadline = time.monotonic() + 5
            while state.snapshot()["state"] != "draining" and time.monotonic() < deadline:
                threading.Event().wait(0.001)
            assert state.snapshot()["state"] == "draining"
            assert not calls
            # Responses/Anthropic translate to another POST on the same request thread.
            with state.admit():
                assert state.snapshot()["active_requests"] == 1
            def newcomer():
                with state.admit():
                    pytest.fail("new request admitted after the drain barrier")
            with pytest.raises(LifecycleError):
                pool.submit(newcomer).result(5)
            assert not sleep.done()
        assert sleep.result(5)["state"] == "sleeping"
    assert calls == ["release"]


def test_drain_timeout_restores_admission_without_releasing_runtime():
    state, calls = lifecycle(drain_timeout=0.01)
    with ThreadPoolExecutor() as pool, state.admit():
        with pytest.raises(LifecycleError, match="drain") as error:
            pool.submit(state.sleep).result(5)
        assert error.value.status == 409
        assert state.snapshot()["state"] == "awake"
    assert not calls
    with state.admit():
        pass


def test_failed_wake_drops_exception_frames_before_cleanup_and_can_retry():
    import gc
    import weakref

    references, failures = [], [True]

    class Allocation:
        pass

    def restore():
        allocation = Allocation()
        references.append(weakref.ref(allocation))
        if failures[0]:
            raise MemoryError("not enough memory")

    def cleanup():
        gc.collect()
        assert all(reference() is None for reference in references)

    state, calls = lifecycle(restore=restore, cleanup=cleanup)
    state.sleep()
    with pytest.raises(LifecycleError, match="not enough memory") as error:
        state.wake_up()
    assert error.value.status == 503
    assert state.snapshot()["state"] == "sleeping"
    assert state.snapshot()["last_error"]
    failures[0] = False
    assert state.wake_up()["state"] == "awake"
    assert state.snapshot()["last_error"] is None


@pytest.mark.parametrize("phase", ["release", "cleanup"])
def test_unrecoverable_failure_never_admits_or_attempts_another_wake(phase):
    def broken():
        raise RuntimeError("cleanup failed")

    overrides = {phase: broken}
    if phase == "cleanup":
        overrides["restore"] = broken
    state, calls = lifecycle(**overrides)
    with pytest.raises(LifecycleError):
        state.sleep()
        state.wake_up()
    assert state.snapshot()["state"] == "error"
    with pytest.raises(LifecycleError):
        state.wake_up()
    with pytest.raises(LifecycleError):
        with state.admit():
            pytest.fail("admitted after unrecoverable cleanup")


def test_request_failure_releases_lease_and_sleep_from_inside_lease_is_refused():
    state, calls = lifecycle()
    with pytest.raises(ValueError):
        with state.admit():
            with pytest.raises(LifecycleError, match="request"):
                state.sleep()
            raise ValueError("client disconnected")
    assert state.snapshot()["active_requests"] == 0
    assert state.sleep()["state"] == "sleeping"


@pytest.mark.parametrize("level", [1, 0, 3, True, "2", 2.0])
def test_unsupported_levels_never_release(level):
    state, calls = lifecycle()
    with pytest.raises(LifecycleError) as error:
        state.sleep(level)
    assert error.value.status == 400
    assert state.snapshot()["state"] == "awake"
    assert not calls


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_invalid_drain_timeout_is_refused(timeout):
    with pytest.raises(ValueError, match="timeout"):
        lifecycle(drain_timeout=timeout)


def test_failed_preflight_keeps_the_original_runtime_and_allows_requests():
    def changed_checkpoint():
        raise ValueError("checkpoint changed")

    state, calls = lifecycle(preflight=changed_checkpoint)
    with pytest.raises(LifecycleError, match="checkpoint changed") as error:
        state.sleep()
    assert error.value.status == 409
    assert state.snapshot()["state"] == "awake"
    assert not calls
    with state.admit():
        pass
