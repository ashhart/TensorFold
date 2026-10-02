"""The shared cache pool: aligned first fit, growth in place, moves, and two pools replaying one op log equal."""

import random

import pytest
import torch

from tensorfold.families.deepseek_v41.cuda.pool import ALIGN, Pool, align_up, move_rows


def test_place_first_fit_and_release():
    p = Pool(8 * ALIGN)
    a = p.add(p.place(2 * ALIGN), 2 * ALIGN, owner=1)
    b = p.add(p.place(3 * ALIGN), 3 * ALIGN, owner=2)
    assert (a.base, b.base) == (0, 2 * ALIGN)
    p.remove(a)
    assert p.place(2 * ALIGN) == 0 and p.place(4 * ALIGN) is None
    assert p.free_rows() == 5 * ALIGN and p.largest_gap() == 3 * ALIGN


def test_add_checks():
    p = Pool(4 * ALIGN)
    p.add(0, ALIGN)
    with pytest.raises(ValueError):
        p.add(ALIGN // 2, ALIGN)
    with pytest.raises(ValueError):
        p.add(0, ALIGN)
    with pytest.raises(ValueError):
        p.add(ALIGN, ALIGN, eid=5)                     # ranks out of step
    p.add(ALIGN, ALIGN, eid=1)


def test_resize_and_move():
    p = Pool(8 * ALIGN)
    a = p.add(0, ALIGN, owner=1)
    b = p.add(2 * ALIGN, ALIGN, owner=2)
    assert p.room_after(a) == ALIGN
    p.resize(a, 2 * ALIGN)
    with pytest.raises(ValueError):
        p.resize(a, 3 * ALIGN)
    assert p.gaps(ignore=[b]) == [(2 * ALIGN, 6 * ALIGN)]
    old = p.move(a, p.place(4 * ALIGN, ignore=[a]), 4 * ALIGN)
    assert old == 0 and a.base == 3 * ALIGN
    with pytest.raises(ValueError):
        p.move(b, 4 * ALIGN)
    p.move(b, ALIGN)                                    # overlapping its own old rows is fine
    assert [x.eid for x in p.extents] == [1, 0]


def test_random_ops_replay_equal():
    rng = random.Random(7)
    log, p = [], Pool(64 * ALIGN)
    for step in range(3000):
        op = rng.random()
        if op < 0.4:
            size = align_up(rng.randrange(1, 12 * ALIGN))
            base = p.place(size)
            if base is not None:
                log.append(("add", base, size, p.next_eid))
                p.add(base, size, eid=p.next_eid)
        elif op < 0.6 and p.extents:
            x = rng.choice(p.extents)
            log.append(("remove", x.eid))
            p.remove(x)
        elif op < 0.8 and p.extents:
            x = rng.choice(p.extents)
            size = x.size + ALIGN * rng.randrange(-1, 3)
            if 0 < size and size - x.size <= p.room_after(x):
                log.append(("resize", x.eid, size))
                p.resize(x, size)
        elif p.extents:
            x = rng.choice(p.extents)
            base = p.place(x.size, ignore=[x])
            if base is not None:
                log.append(("move", x.eid, base))
                p.move(x, base)
        xs = p.extents
        assert all(a.end <= b.base for a, b in zip(xs, xs[1:]))
        assert all(x.base % ALIGN == 0 and x.size % ALIGN == 0 and x.end <= p.rows for x in xs)
        assert p.free_rows() == sum(n for _, n in p.gaps())
    q = Pool(64 * ALIGN)
    for op in log:
        if op[0] == "add":
            q.add(op[1], op[2], eid=op[3])
        elif op[0] == "remove":
            q.remove(q.get(op[1]))
        elif op[0] == "resize":
            q.resize(q.get(op[1]), op[2])
        else:
            q.move(q.get(op[1]), op[2])
    assert q.digest() == p.digest()


@pytest.mark.parametrize("a,b,m", [(0, 3, 10), (5, 1, 10), (0, 20, 10), (7, 7, 5), (2, 4, 9)])
def test_move_rows(a, b, m):
    t = torch.arange(40 * 3).view(40, 3)
    want = t.clone()
    want[b:b + m] = t[a:a + m].clone()
    move_rows(t, a, b, m)
    assert torch.equal(t, want)
