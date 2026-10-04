"""Production tree-attention plans cover each key chunk once under exact and padded group policies."""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import pytest


def _policy(minimum):
    path = Path(__file__).resolve().parents[1] / "src/tensorfold/cuda/kernels/attention.py"
    source = ast.parse(path.read_text())
    constants = {"CHUNK", "GROUP", "SPAN", "MIN_GROUPED", "MAX_NODES", "QUERY_TILE"}
    methods = {"groups", "slots", "plan_host", "padded_host"}
    body = [node for node in source.body if isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id in constants for target in node.targets)]
    body.extend(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in methods)
    namespace = {"Sequence": Sequence}
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(path), "exec"), namespace)
    namespace["MIN_GROUPED"] = minimum
    return SimpleNamespace(**namespace)


def _writers(policy, flat, width, rows, p, items):
    stream = flat[width:width + 4]
    count = (p + rows + policy.CHUNK - 1) // policy.CHUNK
    groups = (count - stream[3]) // (policy.GROUP - 1)
    assert groups == policy.groups(p, rows)
    writes, chunks = {}, []
    codes = [flat[width + 4 + item * 3 + 2] for item in range(items)]
    for code in dict.fromkeys(codes):
        whole = code >= 0
        chunk = code * policy.GROUP if whole else groups * policy.GROUP - 1 - code
        length = policy.GROUP if whole else 1
        slot = code if whole else chunk - groups * (policy.GROUP - 1)
        if (chunk + length) * policy.CHUNK <= p and code < groups:
            assert slot not in writes
            writes[slot] = list(range(chunk, chunk + length))
            chunks.extend(writes[slot])
    for chunk in range(p // policy.CHUNK, count):
        slot = chunk - groups * (policy.GROUP - 1)
        assert slot not in writes
        writes[slot] = [chunk]
        chunks.append(chunk)
    assert sorted(chunks) == list(range(count))
    assert sorted(writes) == list(range(policy.slots(p, rows)))
    return writes


@pytest.mark.parametrize("minimum", [0, 64, 10**9])
@pytest.mark.parametrize("rows", [1, 3, 16, 64, 128])
@pytest.mark.parametrize("p", [0, 511, 512, 2047, 2048, 4095, 8192, 9000, 32768])
def test_exact_and_padded_plans_write_every_partial_slot_once(minimum, rows, p):
    policy = _policy(minimum)
    parents = list(range(-1, rows - 1))
    flat, items, capacity = policy.plan_host([parents], [p], 6)
    exact = _writers(policy, flat, rows, rows, p, items)
    assert capacity == policy.slots(p, rows)
    context = ((p + rows + 2047) // 2048) * 2048
    padded, items, capacity = policy.padded_host(parents, context, 6)
    padded[rows + 2], padded[rows + 3] = p, policy.slots(p, rows)
    assert _writers(policy, padded, rows, rows, p, items) == exact
    assert capacity >= len(exact)


@pytest.mark.parametrize("minimum", [0, 64, 10**9])
def test_multi_stream_offsets_use_per_stream_slot_counts(minimum):
    policy = _policy(minimum)
    widths, lengths = [16, 1, 128], [32768, 9000, 2047]
    trees = [list(range(-1, width - 1)) for width in widths]
    flat, _, most = policy.plan_host(trees, lengths, 6)
    width = sum(widths)
    streams = [flat[width + 4 * index:width + 4 * (index + 1)] for index in range(3)]
    assert [stream[0] for stream in streams] == [0, 16, 17]
    assert [stream[3] for stream in streams] == [policy.slots(p, rows) for p, rows in zip(lengths, widths)]
    assert most == max(stream[3] for stream in streams)
