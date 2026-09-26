"""The CLI's CUDA path: backend choice and argument checks that run before any GPU work (any machine)."""

import argparse
from types import SimpleNamespace

import pytest

from tensorfold import cli


def _family(**members):
    return SimpleNamespace(title="Test family", package=SimpleNamespace(**members))


def test_auto_backend_follows_the_platform(monkeypatch):
    both = _family(load=lambda *a, **k: None, cuda_engine=lambda *a, **k: None)
    monkeypatch.setattr(cli.sys, "platform", "darwin")
    assert cli._backend("auto", both) == "mlx"
    monkeypatch.setattr(cli.sys, "platform", "linux")
    assert cli._backend("auto", both) == "cuda"


def test_a_family_serves_only_the_backends_it_has():
    with pytest.raises(ValueError, match="no CUDA engine"):
        cli._backend("cuda", _family(load=lambda *a, **k: None))
    with pytest.raises(ValueError, match="NVIDIA GPUs only"):
        cli._backend("mlx", _family(cuda_engine=lambda *a, **k: None))


def test_two_gpus_need_a_master_before_anything_loads(tmp_path):
    called = []
    family = _family(cuda_engine=lambda *a, **k: called.append(k))
    args = argparse.Namespace(tp=2, rank=0, master="", master_port=29551, no_drafts=True, drafter="none",
                              mtp_drafts=None, name="", model=str(tmp_path))
    with pytest.raises(ValueError, match="--master"):
        cli._serve_cuda(args, family, tmp_path)
    args.tp, args.rank = 1, 1
    with pytest.raises(ValueError, match="--rank 1 needs --tp 2"):
        cli._serve_cuda(args, family, tmp_path)
    assert not called


def test_serve_parses_the_cuda_flags():
    args = cli.build_parser().parse_args(["serve", "owner/model", "--tp", "2", "--rank", "1", "--master", "10.1.1.1"])
    assert (args.backend, args.tp, args.rank, args.master, args.master_port) == ("auto", 2, 1, "10.1.1.1", 29551)
