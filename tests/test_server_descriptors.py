"""The server raises its open-file limit at start, and says once when accepts fail for want of descriptors."""

import os
import subprocess
import sys

import pytest

resource = pytest.importorskip("resource")


def _run(script: str) -> str:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60, env=env,
                          check=False)
    assert done.returncode == 0, done.stderr
    return done.stdout


def test_the_soft_open_file_limit_rises_to_the_hard_limit():
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if hard != resource.RLIM_INFINITY and hard <= 256:
        pytest.skip("the hard open-file limit leaves nothing to raise")
    out = _run(
        "import resource\n"
        "_, hard = resource.getrlimit(resource.RLIMIT_NOFILE)\n"
        "resource.setrlimit(resource.RLIMIT_NOFILE, (256, hard))\n"
        "from tensorfold.server.http import Server\n"
        "server = Server(('127.0.0.1', 0), lambda *args: None)\n"
        "soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)\n"
        "print(soft, hard == resource.RLIM_INFINITY)\n"
        "server.server_close()\n")
    soft, unlimited = out.split()
    assert int(soft) > 256 and (unlimited == "True" or int(soft) == hard)


def test_an_accept_refused_for_want_of_descriptors_is_said_once_a_run():
    out = _run(
        "import os, resource, socket\n"
        "from tensorfold.server.http import Server\n"
        "server = Server(('127.0.0.1', 0), lambda *args: None)\n"
        "server.timeout = 2\n"
        "clients = [socket.create_connection(server.server_address) for _ in range(3)]\n"
        "_, hard = resource.getrlimit(resource.RLIMIT_NOFILE)\n"
        "def squeeze():\n"
        "    free = os.dup(0)\n"
        "    os.close(free)\n"
        "    resource.setrlimit(resource.RLIMIT_NOFILE, (free, hard))   # the next descriptor can't be had\n"
        "squeeze()\n"
        "server.handle_request()\n"
        "server.handle_request()\n"
        "resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))\n"
        "server.handle_request()                                        # accepted: the run ends\n"
        "squeeze()\n"
        "server.handle_request()\n"
        "resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))\n"
        "server.server_close()\n")
    assert out.count("out of file descriptors") == 2, out
