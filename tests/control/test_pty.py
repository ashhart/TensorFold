"""Real POSIX pseudo-terminal smoke: alternate screen, user logo, keyboard exit and terminal restoration."""
import os
import select
import subprocess
import sys
import time

import pytest


@pytest.mark.skipif(os.name != "posix", reason="POSIX pseudo-terminal check")
def test_actual_terminal_entry_and_restoration():
    import fcntl
    import pty
    import struct
    import termios
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 36, 144, 0, 0))
    env = {key: value for key, value in os.environ.items() if key != "NO_COLOR"}
    env.update(TERM="xterm-256color", COLORTERM="truecolor")
    process = subprocess.Popen([sys.executable, "-m", "tensorfold.control", "tui", "--demo"],
                               stdin=slave, stdout=slave, stderr=slave, env=env)
    os.close(slave)
    collected = b""
    sent_quit = False
    deadline = time.monotonic() + 10
    try:
        while time.monotonic() < deadline:
            if select.select([master], [], [], 0.1)[0]:
                try:
                    data = os.read(master, 65536)
                except OSError:
                    break
                if not data:
                    break
                collected += data
                if b"\x1b[6n" in data:
                    os.write(master, b"\x1b[1;1R")
                if b"TENSORFOLD" in collected and not sent_quit:
                    os.write(master, b"q")
                    sent_quit = True
            if process.poll() is not None:
                # Drain the final renderer cleanup sequence before closing the PTY.
                while select.select([master], [], [], 0.1)[0]:
                    try:
                        data = os.read(master, 65536)
                        if not data:
                            break
                        collected += data
                    except OSError:
                        break
                break
        assert process.wait(timeout=2) == 0
        assert b"\x1b[?1049h" in collected and b"\x1b[?1049l" in collected
        assert b"DEMO" in collected and "▀".encode() in collected
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)
        os.close(master)
