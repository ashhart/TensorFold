"""A CUDA page pool's spill and resume copies: exact, and no device-side temporary the size of a plane."""

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

from tensorfold.cuda.kvpool import PagePool, Plane  # noqa: E402


def test_pages_cross_to_the_host_and_back_exactly_without_a_device_temporary():
    pool = PagePool([Plane("kv", 4096), Plane("c", 2048, per_tokens=4)], 64, 64, device="cuda")  # 256 KiB pages
    g = torch.Generator(device="cuda").manual_seed(3)
    for buf in pool.buffers.values():
        buf.copy_(torch.randint(0, 256, buf.shape, dtype=torch.uint8, device="cuda", generator=g))
    pages = [3, 4, 5, 9, 10, 40]
    want = {k: torch.cat([b[p * pool.rows_per_page(k):(p + 1) * pool.rows_per_page(k)] for p in pages]).cpu().numpy()
            for k, b in pool.buffers.items()}
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    got = pool.read_pages(pages)
    pool.write_pages([20, 21, 22, 50, 51, 60], got)
    torch.cuda.synchronize()
    assert torch.cuda.max_memory_allocated() - base < 64 * 4096  # less than one page: no plane-sized gather
    assert all((got[k] == want[k]).all() for k in want)
    assert all((pool.read_pages([20, 21, 22, 50, 51, 60])[k] == want[k]).all() for k in want)
