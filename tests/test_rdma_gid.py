"""The RoCE all-gather finds its source GID by type and address, not by a fixed index."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from tensorfold.cuda.rdma import gid_index, roce_v2_ipv4_gids

ZERO = "0000:0000:0000:0000:0000:0000:0000:0000"
LINK = "fe80:0000:0000:0000:a2ad:9fff:fedc:aec8"
V4 = "0000:0000:0000:0000:0000:ffff:0a2a:0002"          # 10.42.0.2


def _port(tmp_path, entries):
    """entries: index -> (gid, type or None for a slot whose type read fails)."""

    base = tmp_path / "ib" / "dev0" / "ports" / "1"
    (base / "gids").mkdir(parents=True)
    (base / "gid_attrs" / "types").mkdir(parents=True)
    for i in range(8):
        gid, kind = entries.get(i, (ZERO, None))
        (base / "gids" / str(i)).write_text(gid + "\n")
        if kind is not None:
            (base / "gid_attrs" / "types" / str(i)).write_text(kind + "\n")
    return str(tmp_path / "ib")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("TF_RDMA_GID_INDEX", "TF_RDMA_ADDR_RANGE", "NCCL_IB_ADDR_RANGE", "NCCL_IB_GID_INDEX"):
        monkeypatch.delenv(name, raising=False)


def test_usual_layout_is_index_3(tmp_path):
    root = _port(tmp_path, {0: (LINK, "IB/RoCE v1"), 1: (LINK, "RoCE v2"), 2: (V4, "IB/RoCE v1"), 3: (V4, "RoCE v2")})
    assert roce_v2_ipv4_gids("dev0", root=root) == [(3, "10.42.0.2")]
    assert gid_index("dev0", root=root) == 3


def test_moved_gid_is_found(tmp_path, monkeypatch):
    # the address re-added while a QP held slot 3: v1 back at 2, v2 at 4, slot 3 stale (type read fails)
    monkeypatch.setenv("NCCL_IB_GID_INDEX", "3")                   # a stale hard-coded index loses to the scan
    root = _port(tmp_path, {0: (LINK, "IB/RoCE v1"), 1: (LINK, "RoCE v2"), 2: (V4, "IB/RoCE v1"), 4: (V4, "RoCE v2")})
    assert gid_index("dev0", root=root) == 4


def test_overrides_and_ranges(tmp_path, monkeypatch):
    other = "0000:0000:0000:0000:0000:ffff:c0a8:0101"             # 192.168.1.1
    root = _port(tmp_path, {1: (LINK, "RoCE v2"), 3: (other, "RoCE v2"), 5: (V4, "RoCE v2")})
    assert gid_index("dev0", root=root) == 3
    monkeypatch.setenv("NCCL_IB_ADDR_RANGE", "10.42.0.0/15")
    assert gid_index("dev0", root=root) == 5
    monkeypatch.setenv("TF_RDMA_GID_INDEX", "7")
    assert gid_index("dev0", root=root) == 7


def test_no_gid(tmp_path, monkeypatch):
    root = _port(tmp_path, {0: (LINK, "IB/RoCE v1"), 1: (LINK, "RoCE v2")})
    with pytest.raises(RuntimeError, match="no RoCE v2 IPv4 GID"):
        gid_index("dev0", root=root)
    monkeypatch.setenv("NCCL_IB_GID_INDEX", "3")
    assert gid_index("dev0", root=root) == 3
