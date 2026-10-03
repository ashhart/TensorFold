from pathlib import Path
import sys

import pytest

from tensorfold.control.config import Paths, Profile
from tensorfold.control.launchd import Manager, Result


class FakeLaunchctl:
    """An explicitly fake launchd transport, with persistent enable/disable and delayed removal."""
    def __init__(self):
        self.calls = []
        self.loaded = {}
        self.disabled = set()
        self.fail = None
        self.delay = 0

    def __call__(self, argv, timeout):
        self.calls.append(list(argv))
        assert argv[0] == "/bin/launchctl"
        verb, *args = argv[1:]
        if self.fail == verb:
            return Result(5, stderr="simulated failure")
        if verb == "print" and args[0].count("/") == 1:
            return Result(0, "gui domain")
        if verb == "print":
            target = args[0]
            if target in self.loaded:
                item = self.loaded[target]
                if item.get("unloading"):
                    self.delay -= 1
                    if self.delay <= 0:
                        del self.loaded[target]
                if target in self.loaded:
                    return Result(0, f'{target} = {{\n\tpath = {item["path"]}\n\tstate = running\n'
                                  f'\tpid = 321\n\tlast exit code = 0\n}}\n')
            return Result(113, stderr="Could not find service")
        if verb == "enable":
            self.disabled.discard(args[0])
        elif verb == "disable":
            self.disabled.add(args[0])
        elif verb == "bootstrap":
            domain, filename = args
            import plistlib
            p = plistlib.loads(Path(filename).read_bytes())
            target = domain + "/" + p["Label"]
            if target in self.loaded:
                return Result(37, stderr="operation already in progress")
            self.loaded[target] = {"path": filename}
        elif verb == "bootout":
            if self.delay:
                self.loaded[args[0]]["unloading"] = True
            else:
                self.loaded.pop(args[0], None)
        return Result(0)


@pytest.fixture
def profile():
    return Profile("default", "Org/Model", python=sys.executable)


@pytest.fixture
def manager(tmp_path):
    clock = [0.0]
    def sleep(seconds):
        clock[0] += seconds
    transport = FakeLaunchctl()
    manager = Manager(Paths(tmp_path), run=transport, platform="darwin", uid=501,
                      clock=lambda: clock[0], sleep=sleep)
    return manager, transport


def pytest_configure(config):
    config.addinivalue_line("markers", "macos: real logged-in macOS launchd lifecycle (explicit opt-in)")
