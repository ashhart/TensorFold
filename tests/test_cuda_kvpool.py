"""The page pool: lowest pages first, reservations, shared pages with holders, copy-on-write adoption, the null page."""

from __future__ import annotations

import numpy as np
import pytest

from tensorfold.cuda.kvpool import PagePool, Plane, pages_for
from tensorfold.cuda.memory_gate import NoRoom

PLANES = (Plane("kv", 4), Plane("half", 2, 2), Plane("block", 8, 16))


def pool(pages=8, page=16, device=None):
    return PagePool(PLANES, pages, page, device)


def test_planes_hold_page_tokens_over_their_ratio():
    p = pool()
    assert [p.rows_per_page(n) for n in ("kv", "half", "block")] == [16, 8, 1]
    assert p.view("half").shape == (9 * 8, 2) and p.page_bytes() == 16 * 4 + 8 * 2 + 8
    with pytest.raises(ValueError, match="divide"):
        PagePool([Plane("odd", 4, 3)], 4, 16)


def test_pages_are_handed_out_lowest_first_and_come_back():
    p = pool()
    a, b = p.table(64), p.table(64)
    a.ensure(20)
    b.ensure(1)
    a.ensure(40)
    assert a.pages == [0, 1, 3] and b.pages == [2]
    assert list(a.device[:4]) == [0, 1, 3, p.null]
    assert a.truncate(16) == 2 and a.pages == [0] and list(a.device[:3]) == [0, p.null, p.null]
    b.release()
    a.ensure(48)
    assert a.pages == [0, 1, 2]


def test_a_lane_never_takes_another_lanes_reserved_pages():
    p = pool(pages=6)
    a, b = p.table(96), p.table(96)
    a.reserve(4)
    assert p.available() == 2
    b.ensure(32)
    with pytest.raises(NoRoom):
        b.ensure(33)  # past every unreserved page
    a.ensure(64)
    assert a.pages == [2, 3, 4, 5] and p.available() == 0
    with pytest.raises(ValueError, match="past the lane"):
        a.ensure(97)


def test_shared_pages_return_when_the_last_holder_lets_go():
    p = pool()
    a = p.table(64)
    a.ensure(40)
    kept = a.share(40)
    assert kept == [0, 1, 2] and p.holders(1) == 2
    assert a.release() == 0 and p.free == 5
    assert p.drop(kept) == 3 and p.free == 8


def test_adopting_an_entry_shares_full_pages_and_copies_the_partial_one():
    p = pool()
    a = p.table(64)
    a.ensure(40)
    rows = p.view("kv")
    rows[:48] = np.arange(48 * 4, dtype=np.uint8).reshape(48, 4)
    kept = a.share(40)
    a.release()
    b = p.table(64)
    b.adopt(kept, 40)
    assert b.pages[:2] == kept[:2] and b.pages[2] != kept[2]
    assert np.array_equal(rows[b.pages[2] * 16 : (b.pages[2] + 1) * 16], rows[32:48])
    b.ensure(41)
    c = p.table(64)
    c.adopt(kept, 32)  # page-aligned: nothing to copy
    assert c.pages == kept[:2] and p.holders(0) == 3
    with pytest.raises(ValueError, match="maps none"):
        c.adopt(kept, 32)


def test_pages_round_trip_through_the_host_and_the_null_page_stays_zero():
    p = pool()
    a = p.table(64)
    a.ensure(32)
    for name in ("kv", "half", "block"):
        p.view(name)[: 2 * p.rows_per_page(name)] = 7
    data = p.read_pages(a.pages)
    b = p.table(64)
    b.ensure(64)
    p.write_pages(b.pages[2:], data)
    for name in ("kv", "half", "block"):
        n = p.rows_per_page(name)
        assert (p.view(name)[b.pages[2] * n : (b.pages[3] + 1) * n] == 7).all()
    assert p.null_clean()
    p.view("kv")[p.null * 16] = 1
    assert not p.null_clean()
    with pytest.raises(ValueError, match="rows for"):
        p.write_pages(b.pages[:1], data)


def test_pages_for_rounds_up():
    assert [pages_for(t, 16) for t in (0, 1, 16, 17)] == [0, 1, 1, 2]


def test_device_tables_are_tensors_at_a_fixed_address():
    torch = pytest.importorskip("torch")
    p = pool(device="cpu")
    a = p.table(64)
    at = a.device.data_ptr()
    a.ensure(40)
    a.truncate(0)
    assert a.device.dtype == torch.int32 and a.device.data_ptr() == at
    assert a.device.tolist() == [p.null] * 4
    a.ensure(20)
    kept = a.share(20)
    data = p.read_pages(kept)
    assert data["kv"].shape == (32, 4) and isinstance(data["kv"], np.ndarray)
    p.write_pages(kept, data)
