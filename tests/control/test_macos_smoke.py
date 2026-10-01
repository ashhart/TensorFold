"""Opt-in real launchd lifecycle; --help exits before loading a model or touching GPU memory."""
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

import pytest

from tensorfold.control.config import Paths, Profile
from tensorfold.control.launchd import Manager

SMOKE_NAME = "control-smoke"


@pytest.mark.macos
@pytest.mark.skipif(sys.platform != "darwin" or os.environ.get("TENSORFOLD_TEST_LAUNCHD") != "1",
                    reason="requires explicit TENSORFOLD_TEST_LAUNCHD=1 in a logged-in macOS session")
def test_real_launchd_lifecycle_without_model_load(monkeypatch):
    # Files stay in a temp home. The label is the one reserved name.
    import tensorfold.control.config as config
    original = config.name_of

    def allow(name: str) -> str:
        if name == SMOKE_NAME:
            return name
        return original(name)

    monkeypatch.setattr(config, "name_of", allow)
    with tempfile.TemporaryDirectory(prefix=".tensorfold-control-smoke-", dir=Path.home()) as directory:
        manager = Manager(Paths(Path(directory)))
        name = SMOKE_NAME
        profile = Profile(name, "NO_MODEL_IS_LOADED", python=sys.executable, args=("--help",))
        installed = False
        try:
            manager.install(profile)
            installed = True
            manager.start(name)
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                log = manager.paths.log(name)
                if log.exists() and "server exited status=0" in log.read_text():
                    break
                time.sleep(0.2)
            else:
                pytest.fail("launchd runner did not complete the real TensorFold --help child successfully")
            assert manager.status(name).loaded
            manager.stop(name)
            assert not manager.status(name).loaded
            manager.start(name)
            assert manager.status(name).loaded
        finally:
            if installed:
                manager.uninstall(name)
                assert not manager.paths.plist(name).exists()
                printed = subprocess.run(
                    ["/bin/launchctl", "print-disabled", f"gui/{os.getuid()}"],
                    capture_output=True, text=True, check=False)
                assert printed.returncode == 0, printed.stderr
                label = f"dev.tensorfold.{SMOKE_NAME}"
                assert not re.search(rf'"?{re.escape(label)}"?\s*=>\s*disabled\b', printed.stdout)
