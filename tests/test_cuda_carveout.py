"""Display carveout settings and DRM ioctl numbers, without a GPU or a DRM card."""

import types

from tensorfold.cuda import capacity, carveout


def test_ioctl_numbers_match_the_kernel_uapi():
    # include/uapi/drm/drm.h: DRM_IOWR(0xB2..0xB4, struct drm_mode_{create,map,destroy}_dumb)
    assert carveout.CREATE_DUMB == 0xC02064B2
    assert carveout.MAP_DUMB == 0xC01064B3
    assert carveout.DESTROY_DUMB == 0xC00464B4


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("TF_CARVEOUT", raising=False)
    assert not carveout.enabled()
    assert carveout.requested_bytes() == 0
    assert carveout.get() is None


def test_size_rounds_down_to_whole_framebuffer_rows(monkeypatch):
    monkeypatch.setenv("TF_CARVEOUT", "1")
    monkeypatch.delenv("TF_CARVEOUT_BYTES", raising=False)
    assert carveout.requested_bytes() == 1792 << 20
    monkeypatch.setenv("TF_CARVEOUT_BYTES", str((1 << 30) + 1000))
    assert carveout.requested_bytes() == 1 << 30


def test_reserve_override(monkeypatch):
    total = 121 * capacity.GIB
    monkeypatch.delenv("TENSORFOLD_MEMORY_RESERVE_GIB", raising=False)
    assert capacity.reserve_bytes(total, host=True) == total // 10
    monkeypatch.setenv("TENSORFOLD_MEMORY_RESERVE_GIB", "2.5")
    assert capacity.reserve_bytes(total, host=True) == int(2.5 * capacity.GIB)


def test_carveout_adds_to_the_unified_budget(monkeypatch):
    gib = capacity.GIB
    monkeypatch.setenv("TENSORFOLD_MEMORY_RESERVE_GIB", "2")
    monkeypatch.setattr(capacity, "_meminfo", lambda: {"MemTotal": 121 * gib, "MemAvailable": 6 * gib})
    torch = types.SimpleNamespace(cuda=types.SimpleNamespace(
        mem_get_info=lambda: (5 * gib, 121 * gib),
        get_device_properties=lambda _: types.SimpleNamespace(is_integrated=True)))
    assert capacity.available_bytes(torch) == 4 * gib
    assert capacity.available_bytes(torch, carveout=gib) == 5 * gib
