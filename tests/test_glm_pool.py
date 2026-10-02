"""GLM's shared per-token pool (``glm5_next.cuda.pool``) on CPU: extents are aligned, never overlap, grow in place
only into free rows and move only into free ones; an arena's views of an extent are its planes' rows (pooled keys at
base / 4 with their pad), and copies between extents keep every plane's rows, overlapping or not."""

from __future__ import annotations

import pytest

from tensorfold.families.glm5_next.cuda.pool import ALIGN, Arena, Plane, Pool, align_up

pytestmark = pytest.mark.torch


def test_extents_are_placed_lowest_first_and_never_overlap():
    p = Pool(8 * ALIGN)
    a = p.alloc(1)                                  # rounds up to one aligned block
    b = p.alloc(ALIGN + 1)
    c = p.alloc(ALIGN)
    assert (a.base, a.size, b.base, b.size, c.base) == (0, ALIGN, ALIGN, 2 * ALIGN, 3 * ALIGN)
    assert p.free_rows() == 4 * ALIGN
    p.remove(b)
    assert p.gaps() == [(ALIGN, 2 * ALIGN), (4 * ALIGN, 4 * ALIGN)]
    d = p.alloc(3 * ALIGN)                          # the hole is too small: after c
    assert d.base == 4 * ALIGN
    assert p.alloc(2 * ALIGN).base == ALIGN          # the hole fits exactly
    assert p.alloc(2 * ALIGN) is None               # one block left
    with pytest.raises(ValueError, match="overlaps"):
        p.add(0, ALIGN)
    with pytest.raises(ValueError, match="aligned"):
        p.add(7 * ALIGN + 1, ALIGN)


def test_growth_in_place_and_moves():
    p = Pool(6 * ALIGN)
    a, b = p.alloc(ALIGN), p.alloc(ALIGN)
    assert p.room_after(a) == 0 and p.room_after(b) == 4 * ALIGN
    with pytest.raises(ValueError, match="in place"):
        p.resize(a, 2 * ALIGN)
    p.resize(b, 3 * ALIGN)
    assert b.size == 3 * ALIGN and p.room_after(b) == 2 * ALIGN
    assert p.place(3 * ALIGN) is None and p.place(3 * ALIGN, ignore=[b]) == ALIGN
    p.remove(a)
    old = p.move(b, 0)                              # slides down over its own old rows
    assert old == ALIGN and (b.base, b.size) == (0, 3 * ALIGN)
    with pytest.raises(ValueError, match="taken"):
        p.move(p.alloc(ALIGN), ALIGN)
    p.resize(b, ALIGN)                              # shrink from the end
    assert p.gaps()[0] == (ALIGN, 2 * ALIGN)
    assert align_up(1) == ALIGN and align_up(ALIGN) == ALIGN and align_up(ALIGN + 1) == 2 * ALIGN


def _arena(torch, rows: int) -> Arena:
    lat = torch.arange(rows * 3, dtype=torch.int64).view(rows, 3)
    pk = -torch.arange((rows // 4 + 2) * 2, dtype=torch.int64).view(rows // 4 + 2, 2)
    return Arena(rows, [Plane(lat, 1, 0), Plane(pk, 4, 2)])


def test_views_are_the_extents_rows():
    import torch

    a = _arena(torch, 4 * ALIGN)
    lat, pk = a.view(0, ALIGN, 2 * ALIGN), a.view(1, ALIGN, 2 * ALIGN)
    assert lat.shape == (2 * ALIGN, 3) and pk.shape == (2 * ALIGN // 4 + 2, 2)
    assert lat.data_ptr() == a.planes[0].tensor[ALIGN].data_ptr()
    assert pk.data_ptr() == a.planes[1].tensor[ALIGN // 4].data_ptr()
    assert a.view(1, 3 * ALIGN, ALIGN).shape[0] == ALIGN // 4 + 2      # the last extent's pad is the arena's
    with pytest.raises(ValueError, match="outside"):
        a.view(0, 3 * ALIGN, 2 * ALIGN)
    with pytest.raises(ValueError, match="plane"):
        Arena(ALIGN, [Plane(torch.zeros(ALIGN // 4, 2), 4, 2)])


@pytest.mark.parametrize("src, dst, n", [(0, 3 * 2048, 2048), (2048, 0, 3000), (0, 2048, 5000), (4096, 2048, 4000),
                                         (0, 0, 100)])
def test_copies_keep_every_plane_even_when_overlapping(src, dst, n):
    import torch

    a = _arena(torch, 4 * ALIGN)
    before = [p.tensor.clone() for p in a.planes]
    a.copy(src, dst, n)
    lat, pk = (p.tensor for p in a.planes)
    assert torch.equal(lat[dst:dst + n], before[0][src:src + n])
    m = -(-n // 4)
    assert torch.equal(pk[dst // 4:dst // 4 + m], before[1][src // 4:src // 4 + m])
    outside = torch.ones(lat.shape[0], dtype=torch.bool)
    outside[dst:dst + n] = False
    assert torch.equal(lat[outside], before[0][outside])            # nothing else is written


def test_copies_stay_inside_the_arena():
    import torch

    a = _arena(torch, 2 * ALIGN)
    with pytest.raises(ValueError, match="outside"):
        a.copy(ALIGN, 0, ALIGN + 1)
