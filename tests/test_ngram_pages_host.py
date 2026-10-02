"""Page-rounded pin budgets, contiguous runs and failed locks use deterministic host syscall fixtures."""

from types import SimpleNamespace

import pytest

from tensorfold.cuda import ngram_pages as pages


@pytest.fixture(autouse=True, params=[4096, 16384])
def page_size(monkeypatch, request):
    monkeypatch.setattr(pages, "PAGE", request.param)


class Rows:
    def __init__(self, address, rows, width=128):
        self.address, self.shape, self.strides = address, (rows, width), (width, 1)
        self.nbytes, self.ctypes = rows * width, SimpleNamespace(data=address)
        self.flags = SimpleNamespace(c_contiguous=True)

    def __getitem__(self, value):
        start, end, _ = value.indices(self.shape[0])
        return Rows(self.address + start * self.strides[0], end - start, self.strides[0])


class API:
    def __init__(self, fail=None):
        self.locked, self.unlocked, self.fail = [], [], fail

    def mlock(self, address, size):
        self.locked.append((address, size))
        return -1 if len(self.locked) == self.fail else 0

    def munlock(self, address, size):
        self.unlocked.append((address, size))
        return 0


@pytest.fixture
def api(monkeypatch):
    result = API()
    monkeypatch.setattr(pages, "libc", lambda: result)
    return result


def test_page_rounding_refuses_a_raw_byte_budget_before_syscalls(api):
    words = Rows(pages.PAGE + 17, 8)
    pins = pages.Pins()
    assert pins.runs([words], words.nbytes) == 0
    assert pins.nbytes == 0 and api.locked == []
    assert pins.runs([words], pages.PAGE) == pages.PAGE
    assert api.locked == [(pages.PAGE, pages.PAGE)]


def test_one_gib_runs_are_not_shrunk_to_use_a_remaining_fragment(api):
    words = Rows(pages.PAGE, (2 << 30) // 128)
    pins = pages.Pins()
    assert pins.runs([words], (1 << 30) + pages.PAGE) == 1 << 30
    assert api.locked == [(pages.PAGE, 1 << 30)]


def test_repeated_calls_and_adjacent_rows_do_not_charge_the_same_page_twice(api):
    words = Rows(pages.PAGE + 17, 20, 122)
    pins = pages.Pins()
    assert pins.runs([words], pages.PAGE, 10 * 122) == pages.PAGE
    assert pins.runs([words], pages.PAGE, 10 * 122) == 0
    assert pins.nbytes == pages.PAGE and len(api.locked) == 1


def test_failed_whole_lock_rolls_back_new_pages(api):
    api.fail = 2
    words = [Rows(pages.PAGE, 16), Rows(4 * pages.PAGE, 16)]
    pins = pages.Pins()
    assert not pins.all(words)
    assert pins.nbytes == 0
    assert api.unlocked == [(pages.PAGE, pages.PAGE)]


def test_partial_failure_retains_only_successful_previous_runs(api):
    api.fail = 2
    pins = pages.Pins()
    assert pins.runs([Rows(pages.PAGE, 2 * pages.PAGE // 128)], 2 * pages.PAGE, pages.PAGE) == pages.PAGE
    assert pins.nbytes == pages.PAGE and pins.ranges == [(pages.PAGE, 2 * pages.PAGE)]


def test_failed_unlock_remains_charged_instead_of_claiming_zero(api):
    api.fail = 2
    api.munlock = lambda *args: -1
    pins = pages.Pins()
    assert not pins.all([Rows(pages.PAGE, 16), Rows(4 * pages.PAGE, 16)])
    assert pins.nbytes == pages.PAGE


def test_two_tables_share_the_actual_page_budget(api):
    budget, charged = 2 * pages.PAGE - 1, 0
    for words in ([Rows(pages.PAGE + 17, 8)], [Rows(4 * pages.PAGE + 13, 8)]):
        charged += pages.Pins().runs(words, budget - charged)
    assert charged == pages.PAGE and charged <= budget


def test_noncontiguous_rows_are_refused(api):
    words = Rows(pages.PAGE, 16)
    words.flags.c_contiguous = False
    with pytest.raises(ValueError, match="contiguous"):
        pages.Pins().runs([words], 2 * pages.PAGE)


def test_windows_and_missing_lock_api_pin_nothing(monkeypatch):
    monkeypatch.setattr(pages, "os", SimpleNamespace(name="nt"))
    pages.libc.cache_clear()
    assert pages.libc() is None
    assert pages.Pins().runs([Rows(pages.PAGE, 16)], 2 * pages.PAGE) == 0
    pages.libc.cache_clear()


def test_empty_and_overlapping_arrays_charge_their_union():
    table = SimpleNamespace(words=[Rows(pages.PAGE + 1, 8), Rows(pages.PAGE + 2, 8), Rows(pages.PAGE + 900, 0)], scales=[], biases=[])
    assert pages.lock_bytes(table) == pages.PAGE
