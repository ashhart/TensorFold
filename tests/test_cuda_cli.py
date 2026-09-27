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
    args = cli.build_parser().parse_args(["serve", "owner/model", "--tp", "2", "--rank", "1", "--master", "192.0.2.11"])
    assert (args.backend, args.tp, args.rank, args.master, args.master_port) == ("auto", 2, 1, "192.0.2.11", 29551)


@pytest.mark.parametrize("override, expected", [(None, 128), (0, 0), (64, 64)])
def test_cuda_dispatch_keeps_the_resolved_context(tmp_path, monkeypatch, override, expected):
    import json

    from tensorfold import families, hub

    (tmp_path / "config.json").write_text(json.dumps({"max_position_embeddings": 128}))
    family = _family(cuda_engine=lambda *a, **k: None)
    family.model_type = "test"
    monkeypatch.setattr(families, "detect", lambda path: family)
    monkeypatch.setattr(families, "require_readable", lambda *a: None)
    monkeypatch.setattr(hub, "resolve", lambda *a, **k: tmp_path)
    seen = []
    monkeypatch.setattr(cli, "_serve_cuda", lambda args, found, path, context: seen.append(context) or 0)
    command = ["serve", str(tmp_path), "--backend", "cuda", "--no-update-check"]
    if override is not None:
        command += ["--context", str(override)]
    assert cli.cmd_serve(cli.build_parser().parse_args(command)) == 0
    assert seen == [expected]


def test_cuda_admission_metadata_does_not_enlarge_the_engine_cache(tmp_path, monkeypatch, capsys):
    import tensorfold.cuda.server as server

    made, served = [], []
    engine = SimpleNamespace(max_len=8192)
    family = _family(cuda_engine=lambda *a, **k: made.append(k) or engine)
    family.model_type = "test"
    monkeypatch.setattr(server, "App", lambda *a, **k: served.append(k) or
                        SimpleNamespace(effective_context_window=8185))
    monkeypatch.setattr(server, "serve", lambda *a: None)
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--backend", "cuda", "--no-drafts"])
    assert cli._serve_cuda(args, family, tmp_path, 262144) == 0
    assert made[0]["context"] == 262144
    assert made[0]["context_explicit"] is False
    assert served[0]["context_window"] == 262144
    assert "context: 8185" in capsys.readouterr().out


def test_serve_parses_the_kv_cache_flag():
    plain = cli.build_parser().parse_args(["serve", "owner/model"])
    assert plain.kv_dtype == "bf16"                        # the cache stays bf16 unless it is asked for
    assert cli.build_parser().parse_args(["serve", "owner/model", "--kv-dtype", "int8"]).kv_dtype == "int8"
    assert cli.build_parser().parse_args(["serve", "owner/model", "--kv-dtype", "int4"]).kv_dtype == "int4"
    assert cli.build_parser().parse_args(["serve", "owner/model", "--mtp-confidence", "0.6"]).mtp_confidence == 0.6
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["serve", "owner/model", "--kv-dtype", "fp8"])


@pytest.mark.torch
def test_kv_dtype_reaches_only_the_families_that_declare_it(tmp_path, monkeypatch):
    """``CUDA_KV_DTYPES`` is the gate: a family that does not list a dtype is refused before anything loads, and
    a family that does gets it through its engine."""

    import json

    from tensorfold.families import glm5_next, qwen3_5, qwen4_exp
    from tensorfold.families.qwen4_exp.cuda import engine as fn_engine

    made = []
    monkeypatch.setattr(fn_engine, "FlashNextEngine", lambda *a, **k: made.append(k) or SimpleNamespace(**k))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"mtp.fc.weight": "x"}}))

    assert qwen4_exp.CUDA_KV_DTYPES == ("bf16", "int8", "int4")
    assert qwen4_exp.cuda_engine(tmp_path, kv_dtype="int8").kv_dtype == "int8"
    assert made[-1]["kv_dtype"] == "int8"
    with pytest.raises(ValueError, match="kv-dtype"):
        qwen4_exp.cuda_engine(tmp_path, kv_dtype="fp8")
    for module in (qwen3_5, glm5_next):
        args = argparse.Namespace(tp=1, rank=0, master="", master_port=29551, no_drafts=True, drafter="none",
                                  mtp_drafts=None, name="", model=str(tmp_path), kv_dtype="int8")
        with pytest.raises(ValueError, match="KV cache, not --kv-dtype int8"):
            cli._serve_cuda(args, SimpleNamespace(title=module.TITLE, package=module), tmp_path)
    assert not made[1:]


@pytest.mark.torch
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
    glm = dict(tp=2, master="192.0.2.10")
    assert glm5_next.cuda_engine(tmp_path, drafter=str(tmp_path), mtp_drafts=0, **glm).policy == "fc5:0.3"
    assert glm5_next.cuda_engine(tmp_path, mtp_drafts=0, **glm).policy == "0"
    assert glm5_next.cuda_engine(tmp_path, drafter=str(tmp_path), **glm).policy == "auto"
    assert glm5_next.cuda_engine(tmp_path, mtp_drafts=2, **glm).policy == "2"


@pytest.mark.parametrize("flag, streams", [(None, None), ("auto", None), ("1", None), ("4", 4)])
def test_cuda_parallel_is_one_request_at_a_time_unless_a_number_asks(tmp_path, monkeypatch, flag, streams):
    import tensorfold.cuda.server as server

    made = []
    engine = SimpleNamespace(context_window=4096)
    family = _family(cuda_engine=lambda *a, **k: made.append(k) or engine)
    family.model_type = "test"
    monkeypatch.setattr(server, "App", lambda *a, **k: SimpleNamespace(effective_context_window=4096))
    monkeypatch.setattr(server, "serve", lambda *a: None)
    command = ["serve", str(tmp_path), "--backend", "cuda", "--no-drafts"] + (["--parallel", flag] if flag else [])
    assert cli._serve_cuda(cli.build_parser().parse_args(command), family, tmp_path, 4096) == 0
    assert made[0].get("parallel") == streams
