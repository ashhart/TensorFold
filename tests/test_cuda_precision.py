"""--precision: checkpoint by default, set before loading and shown; each GPU runs the math it has, none refused."""

from types import SimpleNamespace

import pytest

from tensorfold import cli
from tensorfold.cuda import precision
from tests.test_cuda_cli import _family


def test_the_flag_parses_and_defaults_to_the_checkpoints_math():
    parser = cli.build_parser()
    assert getattr(parser.parse_args(["serve", "owner/model"]), "precision", None) is None
    assert parser.parse_args(["serve", "owner/model", "--precision", "full"]).precision == "full"
    with pytest.raises(SystemExit):
        parser.parse_args(["serve", "owner/model", "--precision", "fp4"])
    serve = next(a for a in parser._actions if a.dest == "command").choices["serve"]
    flag = next(a for a in serve._actions if a.dest == "precision")
    assert "the default" in flag.help and "never change" in flag.help
    assert precision.mode() == precision.CHECKPOINT


@pytest.mark.parametrize("flags,mode,asked", [([], "checkpoint", False), (["--precision", "full"], "full", True),
                                              (["--precision", "checkpoint"], "checkpoint", True)])
def test_the_mode_is_set_before_loading_and_shown(tmp_path, monkeypatch, capsys, flags, mode, asked):
    import tensorfold.cuda.server as server

    seen = []

    def engine(*a, **k):
        seen.append((precision.mode(), precision.asked()))
        return SimpleNamespace(max_len=8192, w=SimpleNamespace(fast_prefill=False, precision=mode))

    family = _family(cuda_engine=engine)
    family.model_type = "test"
    monkeypatch.setattr(server, "App", lambda *a, **k: SimpleNamespace(effective_context_window=8185))
    monkeypatch.setattr(server, "serve", lambda *a: None)
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--backend", "cuda", "--no-drafts"] + flags)
    try:
        assert cli._serve_cuda(args, family, tmp_path, 8192) == 0
        out = capsys.readouterr().out
        assert seen == [(mode, asked)]
        assert ("prompts: the checkpoint math" in out) is (mode == "checkpoint")
    finally:
        precision.set_mode(precision.CHECKPOINT)


def test_prefill_fp8_is_refused_under_the_checkpoints_math(tmp_path, monkeypatch):
    import tensorfold.cuda.server as server

    family = _family(cuda_engine=lambda *a, **k: SimpleNamespace(
        max_len=8192, w=SimpleNamespace(fast_prefill=False, precision="checkpoint")))
    family.model_type = "test"
    monkeypatch.setattr(server, "serve", lambda *a: None)
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--backend", "cuda", "--no-drafts", "--prefill-fp8"])
    try:
        with pytest.raises(ValueError, match="--precision full"):
            cli._serve_cuda(args, family, tmp_path, 8192)
    finally:
        precision.set_mode(precision.CHECKPOINT)
        from tensorfold.cuda import prompt_precision

        prompt_precision.set_fp8(prompt_precision.FP8_BY_DEFAULT)


CHIPS = [((8, 9), False, True),        # RTX 40 (Ada): the e4m3 mma, no block-scaled FP4
         ((9, 0), False, True),        # H100, H200
         ((10, 0), False, True),       # B200 (its FP4 runs on tcgen05, which these kernels don't use)
         ((12, 0), True, True),        # RTX 50, RTX PRO 6000 Blackwell
         ((12, 1), True, True)]        # GB10 (DGX Spark)


@pytest.mark.parametrize("capability,nvfp4,fp8", CHIPS + [((8, 6), False, False), ((8, 0), False, False)])
def test_each_chip_runs_the_formats_it_has_the_mma_for(capability, nvfp4, fp8):
    assert precision.own_math(capability) == {"nvfp4": nvfp4, "fp8": fp8}


@pytest.mark.torch
@pytest.mark.parametrize("capability,nvfp4,fp8", CHIPS)
@pytest.mark.parametrize("asked", [False, True])
def test_no_chip_is_refused_and_the_startup_line_names_the_math(monkeypatch, capability, nvfp4, fp8, asked):
    import torch

    from tensorfold.families.qwen3_5.cuda import nvfp4_load

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: capability)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "NVIDIA TEST")
    with precision.using(precision.CHECKPOINT, asked=asked):
        own, line = nvfp4_load.maths()
    assert own == {"nvfp4": nvfp4, "fp8": fp8}
    assert "its FP8 layers FP8 x FP8" in line
    if nvfp4:
        assert line.startswith("checkpoint (") and "its NVFP4 layers FP4 x FP4" in line
    else:
        assert f"SM {capability[0]}.{capability[1]}" in line and "its NVFP4 layers W4A16" in line
    with precision.using(precision.FULL, asked=True):
        assert nvfp4_load.maths() == ({"nvfp4": False, "fp8": False}, nvfp4_load.FULL_LINE)
