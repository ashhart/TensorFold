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


def test_no_cuda_engine_serves_one_token_a_round_by_default(tmp_path, monkeypatch):
    """Everything on the lanes: a CUDA engine whose drafter is missing refuses to start rather than decode one token
    a round, and names the fix; --no-drafts (the serial reference) still starts."""

    import json

    from tensorfold.families import glm5_next, qwen3_5, qwen4_exp
    from tensorfold.families.glm5_next.cuda import engine as glm_engine
    from tensorfold.families.qwen3_5.cuda import engine as q27_engine
    from tensorfold.families.qwen4_exp.cuda import engine as fn_engine

    made = []
    stub = lambda *a, **k: made.append(k) or SimpleNamespace(**k)      # noqa: E731
    monkeypatch.setattr(q27_engine, "Qwen27Engine", stub)
    monkeypatch.setattr(fn_engine, "FlashNextEngine", stub)
    monkeypatch.setattr(glm_engine, "GlmEngine", stub)

    # the 27B drafts with DFlash2: without it, only the serial reference
    with pytest.raises(ValueError, match="tensorfold pull z-lab/Qwen3.8-27B-DFlash2"):
        qwen3_5.cuda_engine(tmp_path, drafter="")
    assert qwen3_5.cuda_engine(tmp_path, drafter="", no_drafts=True).allow_copy is False
    assert qwen3_5.cuda_engine(tmp_path, drafter=str(tmp_path)).max_rows == 12

    # Flash Next drafts with the checkpoint's MTP head: a checkpoint without it serves only the serial reference
    index = {"weight_map": {"model.layers.0.mlp.gate.weight": "model.safetensors"}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="no MTP head"):
        qwen4_exp.cuda_engine(tmp_path)
    assert qwen4_exp.cuda_engine(tmp_path, no_drafts=True).depth == 0
    index["weight_map"]["mtp.fc.weight"] = "model.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    assert qwen4_exp.cuda_engine(tmp_path).depth == 6

    # GLM: --mtp-drafts 0 with the DFlash2 drafter still drafts (DFlash2 alone); without it, the serial reference
    glm = dict(tp=2, master="10.0.0.1")
    assert glm5_next.cuda_engine(tmp_path, drafter=str(tmp_path), mtp_drafts=0, **glm).policy == "fc5:0.3"
    assert glm5_next.cuda_engine(tmp_path, mtp_drafts=0, **glm).policy == "0"
    assert glm5_next.cuda_engine(tmp_path, drafter=str(tmp_path), **glm).policy == "auto"
    assert glm5_next.cuda_engine(tmp_path, mtp_drafts=2, **glm).policy == "2"
