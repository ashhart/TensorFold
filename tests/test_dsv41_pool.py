"""The shared cache pool's extents: aligned, first fit, no overlap, freed rows reused."""

import pytest

from tensorfold.families.deepseek_v41.cuda.pool import ALIGN, Pool, align_up


def test_align_up():
    assert align_up(1) == ALIGN and align_up(ALIGN) == ALIGN and align_up(ALIGN + 1) == 2 * ALIGN


def test_first_fit_and_reuse():
    p = Pool(8 * ALIGN)
    a = p.place(2 * ALIGN, 1)
    b = p.place(3 * ALIGN, 2)
    assert (a.base, b.base) == (0, 2 * ALIGN)
    assert p.place(4 * ALIGN, 3) is None                      # 3 rows of ALIGN left
    c = p.place(3 * ALIGN, 3)
    assert c.base == 5 * ALIGN and p.free_rows() == 0
    p.release(1)                                              # a's 2 rows come back at the bottom
    assert p.gaps() == [(0, 2 * ALIGN)] and p.largest_gap() == 2 * ALIGN
    d = p.place(ALIGN, 4)
    assert d.base == 0 and p.find(4) is d


def test_add_checks():
    p = Pool(4 * ALIGN)
    p.add(ALIGN, ALIGN, 1)
    with pytest.raises(ValueError):
        p.add(0, 2 * ALIGN, 2)                                # overlaps
    with pytest.raises(ValueError):
        p.add(3 * ALIGN, 2 * ALIGN, 2)                        # past the end
    with pytest.raises(ValueError):
        p.place(100, 3)                                       # not aligned
