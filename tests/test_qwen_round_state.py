"""Committed shared rounds release their rollback records."""

from types import SimpleNamespace
import weakref

import pytest

pytest.importorskip("mlx.core")

from tensorfold.families.qwen3_5.family import Qwen35Family  # noqa: E402


class Record:
    pass


@pytest.mark.parametrize("count", [1, 2, 4])
@pytest.mark.parametrize("keep", [1, 3, [0, 2]])
def test_shared_commit_releases_records_and_keeps_the_path(count, keep):
    family = Qwen35Family.__new__(Qwen35Family)
    caches = [[SimpleNamespace(tokens=[], state=None)] for _ in range(count)]

    def commit(layers, records, paths, widths, starts):
        assert widths == [3] * count
        assert starts == [len(cache[0].tokens) for cache in layers]
        for cache, record, path in zip(layers, records, paths):
            cache[0].tokens.extend(record.tokens[r] for r in path)
            cache[0].state = record.states[path[-1]]

    family._commit_streams = commit
    path = list(range(keep)) if isinstance(keep, int) else keep
    for _ in range(2):
        records = [Record() for _ in caches]
        for i, record in enumerate(records):
            record.tokens = [i * 3 + r for r in range(3)]
            record.states = [object() for _ in range(3)]
        refs = [weakref.ref(record) for record in records]
        expected_tokens = [c[0].tokens + [r.tokens[k] for k in path] for c, r in zip(caches, records)]
        expected_states = [r.states[path[-1]] for r in records]
        starts = [len(c[0].tokens) for c in caches]
        family._last = {id(c): (r, 3, st, i * 3) for i, (c, r, st) in enumerate(zip(caches, records, starts))}
        family._shared = list(zip(records, [3] * count, starts))
        del records, record
        family.keep_rows_streams(caches, [3] * count, [keep] * count)
        assert [c[0].tokens for c in caches] == expected_tokens
        assert [c[0].state for c in caches] == expected_states
        assert not family._shared
        assert not family._last
        assert all(ref() is None for ref in refs)
