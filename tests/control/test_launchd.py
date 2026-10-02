from dataclasses import replace
import plistlib
import sys

import pytest

from tensorfold.control.config import Profile
from tensorfold.control.launchd import Manager, Result, parse_status
from tensorfold.control.safety import ControlError, atomic_write


def test_preview_has_no_side_effects(tmp_path, profile):
    from tensorfold.control.config import Paths
    m = Manager(Paths(tmp_path), run=lambda *_: pytest.fail("must not call launchctl"), platform="linux")
    p = plistlib.loads(m.preview(profile))
    assert p["KeepAlive"] == {"SuccessfulExit": False}
    assert p["RunAtLoad"] is True and p["ThrottleInterval"] == 30
    assert p["ProgramArguments"][:4] == [sys.executable, "-u", "-m", "tensorfold.control.runner"]
    assert p["Umask"] == 0o077
    assert not list(tmp_path.iterdir())


def test_install_start_stop_start_restart_uninstall(manager, profile):
    m, fake = manager
    assert m.install(profile).state == "installed"
    assert not fake.loaded
    assert m.start(profile.name).pid == 321
    first_bootstraps = sum(a[1] == "bootstrap" for a in fake.calls)
    m.start(profile.name)
    assert sum(a[1] == "bootstrap" for a in fake.calls) == first_bootstraps
    m.stop(profile.name)
    assert not fake.loaded and f"{m.domain}/{profile.label}" in fake.disabled
    m.start(profile.name)
    assert fake.loaded and not fake.disabled
    m.restart(profile.name)
    assert fake.loaded
    m.paths.log(profile.name).write_text("keep me")
    m.uninstall(profile.name)
    assert not m.paths.plist(profile.name).exists()
    assert not m.paths.profile(profile.name).exists()
    assert m.paths.log(profile.name).read_text() == "keep me"
    assert all("-k" not in a for a in fake.calls)
    target = f"{m.domain}/{profile.label}"
    assert target not in fake.disabled
    bootout = max(i for i, call in enumerate(fake.calls) if call[1] == "bootout")
    enable = max(i for i, call in enumerate(fake.calls) if call[1] == "enable")
    assert enable > bootout and fake.calls[enable][2] == target


def test_delayed_bootout_is_waited_for(manager, profile):
    m, fake = manager
    m.install(profile, start=True)
    fake.delay = 4
    m.restart(profile.name)
    bootout = max(i for i, a in enumerate(fake.calls) if a[1] == "bootout")
    bootstrap = max(i for i, a in enumerate(fake.calls) if a[1] == "bootstrap")
    assert sum(a[1] == "print" for a in fake.calls[bootout + 1:bootstrap]) >= 4


def test_stop_timeout_never_starts_second_process(manager, profile):
    m, fake = manager
    m.install(profile, start=True)
    before = sum(a[1] == "bootstrap" for a in fake.calls)
    fake.delay = 10000
    with pytest.raises(ControlError, match="still unloading"):
        m.restart(profile.name)
    assert sum(a[1] == "bootstrap" for a in fake.calls) == before


def test_tampered_plist_is_not_controlled(manager, profile):
    m, fake = manager
    m.install(profile)
    data = plistlib.loads(m.paths.plist(profile.name).read_bytes())
    data["ProgramArguments"] = ["/bin/sh", "-c", "unrelated"]
    atomic_write(m.paths.plist(profile.name), plistlib.dumps(data))
    with pytest.raises(ControlError, match="differs"):
        m.start(profile.name)
    assert not fake.loaded


def test_unrelated_loaded_label_is_not_adopted(manager, profile):
    m, fake = manager
    fake.loaded[f"{m.domain}/{profile.label}"] = {"path": "/some/other.plist"}
    with pytest.raises(ControlError, match="already loaded"):
        m.install(profile)
    assert not m.paths.profile(profile.name).exists()


def test_running_job_cannot_be_replaced(manager, profile):
    m, _ = manager
    m.install(profile, start=True)
    with pytest.raises(ControlError, match="stop"):
        m.install(replace(profile, model="Different/Model"), replace=True)
    assert m.store.get(profile.name).model == profile.model


def test_replace_stopped_profile(manager, profile):
    m, _ = manager
    m.install(profile)
    changed = replace(profile, port=8090)
    m.install(changed, replace=True)
    assert m.store.get(profile.name) == changed


def test_duplicate_ports_refused(manager, profile):
    m, _ = manager
    m.install(profile)
    with pytest.raises(ControlError, match="port"):
        m.install(replace(profile, name="other"))


def test_permissions_are_not_misreported_as_stopped(manager, profile):
    m, fake = manager
    m.install(profile)
    fake.fail = "print"
    with pytest.raises(ControlError, match="inspect"):
        m.status(profile.name)


def test_start_failure_keeps_recoverable_configuration(manager, profile):
    m, fake = manager
    m.install(profile)
    fake.fail = "bootstrap"
    with pytest.raises(ControlError):
        m.start(profile.name)
    assert m.store.get(profile.name) == profile
    assert m.paths.plist(profile.name).exists()


def test_uninstall_enables_the_label(manager, profile):
    m, fake = manager
    m.install(profile)
    m.uninstall(profile.name)
    target = f"{m.domain}/{profile.label}"
    assert target not in fake.disabled
    disable = max(i for i, call in enumerate(fake.calls) if call[1] == "disable")
    enable = max(i for i, call in enumerate(fake.calls) if call[1] == "enable")
    assert enable > disable and fake.calls[enable][2] == target
    m.install(profile)
    assert target not in fake.disabled


@pytest.mark.parametrize("platform,uid", [("linux", 501), ("win32", 501), ("darwin", 0), ("darwin", -1)])
def test_platform_and_root_guard(tmp_path, platform, uid, profile):
    from tensorfold.control.config import Paths
    m = Manager(Paths(tmp_path), run=lambda *_: pytest.fail("must not execute"), platform=platform, uid=uid)
    with pytest.raises(ControlError):
        m.install(profile)
    assert not list(tmp_path.iterdir())


def test_nested_fields_do_not_replace_job_status():
    state = parse_status("default", "job = {\n\tpath = /a.plist\n\tstate = waiting\n"
                         "\tlast exit code = 7\n\tenv = {\n\t\tpid = 12\n\t\tstate = running\n\t}\n}")
    assert state.state == "waiting" and state.pid is None and state.last_exit == 7


def test_partial_install_rolls_back(manager, profile, monkeypatch):
    m, _ = manager
    import tensorfold.control.launchd as mod
    original = mod.atomic_write
    def fail(path, data):
        if path.suffix == ".plist":
            raise OSError("simulated disk error")
        return original(path, data)
    monkeypatch.setattr(mod, "atomic_write", fail)
    with pytest.raises(OSError):
        m.install(profile)
    assert not m.paths.profile(profile.name).exists()
    assert not m.paths.plist(profile.name).exists()
