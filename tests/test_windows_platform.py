"""The host-platform answers that differ between Windows and the two Unixes, pinned down without a Windows PC.

Each test hands a module the answer a Windows host would give - a kernel32 that answers or refuses, a socket whose
peer hung up, a page-pinning answer that says no - and says what the engine must then do. A change that only works
on one platform has to fail here on the other two; a change that bends the Unix answer to fit Windows fails here too."""

import ctypes
import os
import socket
import sys
import types
from unittest import mock

import pytest

RAM = 16 * 2**30 - 4096          # what GlobalMemoryStatusEx would answer for a 16 GiB host


class Kernel32:
    """kernel32's GlobalMemoryStatusEx as a stand-in: writes the field the real one writes, then answers."""

    def __init__(self, answer=True):
        self.answer = answer

    def GlobalMemoryStatusEx(self, memory):
        memory._obj.ullTotalPhys = RAM
        return self.answer


def test_windows_dumps_stacks_through_a_python_handler(monkeypatch):
    from tensorfold.server import stacks

    seen = {}
    monkeypatch.setattr(stacks, "faulthandler", types.SimpleNamespace(dump_traceback=lambda **kw: seen.update(kw)))
    monkeypatch.setattr(stacks.signal, "signal", lambda signum, handler: seen.update(armed=(signum, handler)))
    monkeypatch.setattr(stacks, "_started", True)
    stacks.arm()
    signum, handler = seen["armed"]
    handler(signum, None)
    assert signum == stacks.DUMP and seen["all_threads"] is True


def test_windows_sizes_ram_by_api_and_refuses_to_guess():
    from tensorfold.server import memory_budget

    with mock.patch.object(os, "name", "nt"), mock.patch.object(ctypes, "WinDLL", lambda *a, **k: Kernel32(),
                                                                create=True):
        assert memory_budget.physical_memory_bytes() == RAM
    with mock.patch.object(os, "name", "nt"), mock.patch.object(ctypes, "WinDLL", lambda *a, **k: Kernel32(False),
                                                                create=True):
        with pytest.raises(RuntimeError, match="GlobalMemoryStatusEx"):
            memory_budget.physical_memory_bytes()


def test_the_unix_answer_stays_sysconf():
    from tensorfold.server import memory_budget

    assert memory_budget.physical_memory_bytes() == os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


def test_a_hung_up_client_reads_as_cancelled():
    from tensorfold.server.cancellation import socket_cancellation

    listen, other = socket.socketpair()
    other.close()                              # the client hung up: the next peek says so, on either OS
    try:
        assert socket_cancellation(listen).cancelled
    finally:
        listen.close()


def test_a_live_connection_never_reads_as_cancelled():
    from tensorfold.server.cancellation import socket_cancellation

    listen, other = socket.socketpair()
    with listen, other:                        # a byte to peek is not a hang-up, and either answer must not block
        other.sendall(b"e")
        assert not socket_cancellation(listen).cancelled


def test_a_windows_read_stages_through_a_pinned_block(tmp_path):
    torch = pytest.importorskip("torch")
    from tensorfold.cuda import direct_read

    data = bytes(range(256)) * 4
    path = tmp_path / "weights.bin"
    path.write_bytes(data)
    reader = direct_read.Reader()
    reader.direct, reader.staged = False, True  # a Windows Reader: no O_DIRECT to use, so reads stage

    raw = reader.read(path, 256, 512, "cuda" if torch.cuda.is_available() else "cpu", pinned=True)
    assert raw.cpu().tolist() == list(data[256:768])


def test_tensor_parallel_is_refused_on_windows_by_name():
    pytest.importorskip("torch")
    from tensorfold.cuda import comm

    with mock.patch.object(os, "name", "nt"):
        with pytest.raises(RuntimeError, match="does not run tensor-parallel"):
            comm._library()


def test_an_sm75_card_is_refused_by_name_at_startup():
    from tensorfold.cuda import build

    gpu = types.SimpleNamespace(
        version=types.SimpleNamespace(cuda="13.2"),
        cuda=types.SimpleNamespace(is_available=lambda: True,
                                   get_device_capability=lambda: (7, 5),
                                   get_device_name=lambda: "NVIDIA GeForce RTX 2080 Ti"),
    )
    with mock.patch.dict(sys.modules, {"torch": gpu}):
        with pytest.raises(ValueError, match="2080 Ti"):
            build.refuse_old_gpu(build.MIN_CAPABILITY)


def test_windows_pins_table_pages_and_gives_them_up_when_refused():
    numpy = pytest.importorskip("numpy")
    from tensorfold.families.qwen4_exp import host_table

    class Kernel32:
        def __init__(self, answers):
            self.answers, self.locked = list(answers), []

        def VirtualLock(self, address, size):
            answer = self.answers.pop(0)
            if answer:
                self.locked.append(address.value)
            return answer

        def VirtualUnlock(self, address, size):
            self.locked.remove(address.value)   # on Windows an unmade pin means nothing stayed pinned either

    tables = [numpy.zeros(4096, numpy.uint8), numpy.zeros(8192, numpy.uint8)]
    pinned = [array.ctypes.data for array in tables]

    keep_all = Kernel32([True, True])
    assert host_table.windows_lock_pages(tables, keep_all)
    assert keep_all.locked == pinned            # the same addresses, in the order the tables were built

    keep_some = Kernel32([True, False])
    assert not host_table.windows_lock_pages(tables, keep_some)
    assert keep_some.locked == []               # what was pinned came back out with the refused read
