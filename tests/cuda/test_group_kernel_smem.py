"""EXL3 grouping opts in to the device limit for large launches and preserves exact membership order."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")

PREFILL_ROWS = 2048     # the engine's prompt-chunk row count
SLOTS = 9               # top_k + 1
EXPERTS = 288           # a 288-expert MoE layer


@pytest.mark.parametrize("rows", [PREFILL_ROWS, 8, 512])
def test_group_succeeds_cold_at_prefill_scale(rows):
    """The first parameter is a 72-KiB launch, before any small launch can opt this kernel in."""
    from tensorfold.cuda.exl3 import experts

    ext = experts._ext()
    maxu = min(rows * SLOTS, EXPERTS)
    pick = torch.randint(0, EXPERTS, (rows, SLOTS), dtype=torch.int32, device="cuda")
    uids = torch.zeros((maxu,), dtype=torch.int32, device="cuda")
    ucount = torch.zeros((1,), dtype=torch.int32, device="cuda")
    members = torch.full((maxu * rows,), -1, dtype=torch.int32, device="cuda").view(maxu, rows)

    ext.group(pick, uids, ucount, members, rows, SLOTS, EXPERTS)
    torch.cuda.synchronize()
    distinct = ucount[0].item()
    assert 0 < distinct <= min(rows * SLOTS, EXPERTS)
    _assert_members(pick, uids, ucount, members, EXPERTS)


def test_group_output_is_consistent_across_call_order():
    """Small grouping calls stay deterministic after a large call has configured the kernel."""
    from tensorfold.cuda.exl3 import experts

    ext = experts._ext()
    R, slots, E = 8, 9, 288
    maxu = min(R * slots, E)
    g = torch.Generator(device="cuda").manual_seed(11)
    pick = torch.randint(0, E, (R, slots), dtype=torch.int32, device="cuda", generator=g)

    results = []
    for trial in range(2):
        ids = torch.zeros((maxu,), dtype=torch.int32, device="cuda")
        count = torch.zeros((1,), dtype=torch.int32, device="cuda")
        members = torch.full((maxu * R,), -1, dtype=torch.int32, device="cuda").view(maxu, R)
        ext.group(pick, ids, count, members, R, slots, E)
        torch.cuda.synchronize()
        results.append((count[0].item(), members.clone()))
    assert results[0][0] == results[1][0] and torch.equal(results[0][1], results[1][1])


def _assert_members(pick, ids, count, members, experts):
    cpu = pick.cpu()
    wanted = torch.unique(cpu[(cpu >= 0) & (cpu < experts)], sorted=True)
    n = int(count.item())
    assert n == len(wanted) and torch.equal(ids[:n].cpu(), wanted)
    actual = members[:n].cpu()
    for i, expert in enumerate(wanted):
        at = (cpu == expert).nonzero()
        expected = (at[:, 0] * 32 + at[:, 1]).to(torch.int32)
        assert len(expected) <= members.shape[1]
        assert torch.equal(actual[i, :len(expected)], expected)
        assert torch.all(actual[i, len(expected):] == -1)


def _launch(rows, slots=SLOTS, device="cuda"):
    from tensorfold.cuda.exl3 import experts

    g = torch.Generator().manual_seed(31)
    pick = torch.randint(0, EXPERTS, (rows, slots), generator=g, dtype=torch.int32).to(device)
    n = min(rows * slots, EXPERTS)
    ids = torch.zeros(n, dtype=torch.int32, device=device)
    count = torch.zeros(1, dtype=torch.int32, device=device)
    members = torch.full((n, rows), -1, dtype=torch.int32, device=device)
    experts._ext().group(pick, ids, count, members, rows, slots, EXPERTS)
    torch.cuda.synchronize(device)
    _assert_members(pick, ids, count, members, EXPERTS)


def test_device_limit_allows_more_than_a_hardcoded_96_kib_when_available():
    limit = torch.cuda.get_device_properties(0).shared_memory_per_block_optin - 128
    rows = limit // (SLOTS * 4)
    if rows * SLOTS * 4 <= 96 * 1024:
        pytest.skip("this device has no grouping launch between 96 KiB and its opt-in limit")
    _launch(rows)


def test_oversized_launch_refuses_before_launch_and_a_small_one_still_works():
    limit = torch.cuda.get_device_properties(0).shared_memory_per_block_optin
    with pytest.raises(RuntimeError, match="EXL3 grouping needs"):
        _launch(limit // (SLOTS * 4) + 1)
    _launch(8)


def test_large_group_opt_in_is_per_device():
    if torch.cuda.device_count() < 2:
        pytest.skip("two local CUDA devices are needed for the per-device attribute check")
    if torch.cuda.get_device_capability(0) != torch.cuda.get_device_capability(1):
        pytest.skip("the extension builds for one architecture; this check needs matching devices")
    _launch(PREFILL_ROWS, device="cuda:0")
    _launch(PREFILL_ROWS, device="cuda:1")
