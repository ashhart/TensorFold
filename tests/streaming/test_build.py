"""The host-sync extension builds with the nanobind of the installed MLX, and says which one when it can't."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.streaming import build  # noqa: E402


@pytest.mark.parametrize("mlx, nanobind", [("0.32.2", "2.15.0"), ("0.32.3", "3.0.1")])
def test_each_mlx_gets_the_nanobind_it_was_built_with(monkeypatch, mlx, nanobind):
    monkeypatch.setattr(mx, "__version__", mlx)
    assert build.nanobind_for_mlx() == nanobind


def test_an_unknown_mlx_names_the_ones_it_knows(monkeypatch):
    monkeypatch.setattr(mx, "__version__", "0.99.0")
    with pytest.raises(RuntimeError, match=r"MLX 0\.32\.2, 0\.32\.3, not 0\.99\.0"):
        build.nanobind_for_mlx()


def test_a_build_key_changes_with_mlx(monkeypatch):
    keys = set()
    for mlx in build.NANOBIND:
        monkeypatch.setattr(mx, "__version__", mlx)
        keys.add(build._key())
    assert len(keys) == len(build.NANOBIND)


def test_the_wrong_nanobind_names_the_right_one(monkeypatch, tmp_path):
    nanobind = pytest.importorskip("nanobind")
    monkeypatch.setattr(mx, "__version__", "0.32.3")
    monkeypatch.setattr(nanobind, "__version__", "2.15.0")
    with pytest.raises(RuntimeError, match=r"pip install nanobind==3\.0\.1"):
        build._build(tmp_path / "hostsync")
