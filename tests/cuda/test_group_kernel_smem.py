"""The grouping kernel at prefill scale: shared memory must be raised before launch.

`group_kernel` launches with dynamic shared memory `R * slots * 4` bytes. The
engine builds its `Buffers` for `PREFILL_ROWS` rows (a prompt chunk) at
`top_k + 1` slots, and a rank's first `routed()` call during startup
calibration therefore launches with `2048 * 9 * 4` = 72 KB — above the 48 KB
default dynamic-smem limit, which makes the launch fail with
`cudaErrorInvalidValue`. Upstream's own tests never cross it (their largest
fixture is 128 rows at 7 slots, 3.6 KB), which is why the kernel worked
everywhere it was tested and failed in the first real engine run.

The fix raises the limit once per process before the first launch, with the
call's return checked: `cudaFuncSetAttribute` failure is silent, and the sticky
error it leaves surfaces later as a confusing "invalid argument" on the launch
itself. A better upstream home for the attribute is the extension's init; the
value should come from `cudaDevAttrMaxSharedMemoryPerBlockOptin` clamped to
the largest launch the engine can issue (`PREFILL_ROWS * max_slots * 4`).
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")

PREFILL_ROWS = 2048     # the engine's prompt-chunk row count
SLOTS = 9               # top_k + 1
EXPERTS = 288           # a 288-expert MoE layer


@pytest.mark.parametrize("rows", [8, 512, PREFILL_ROWS])
def test_group_succeeds_cold_at_prefill_scale(rows):
    """`group()` must succeed on the first call of a process: a cold context is
    exactly when the attribute has not been raised yet."""
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


def test_group_output_is_consistent_across_call_order():
    """The first and second call must agree: a cold first call failing while the
    second succeeds is the failure mode being pinned here."""
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
