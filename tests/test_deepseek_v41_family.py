"""The DeepSeek-V4.1-Flash family: discovery without torch, check()'s refusals, cuda_engine's settings and options."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorfold import cli, families, serve_options
from tensorfold.families import deepseek_v41

FIXTURE = Path(__file__).parent / "fixtures" / "deepseek_v41" / "config.json"
ENGINE = "tensorfold.families.deepseek_v41.cuda.engine"
RANKS = {"tp": 2, "master": "192.0.2.10"}


def _write(folder: Path, edit=None, names=None) -> Path:
    """The pack's config.json, edited; with ``names``, an index of those tensors too (a downloaded checkpoint)."""

    config = json.loads(FIXTURE.read_text())
    if edit is not None:
        edit(config)
    (folder / "config.json").write_text(json.dumps(config))
    if names is not None:
        index = {"weight_map": {name: "model-00001-of-00001.safetensors" for name in names}}
        (folder / "model.safetensors.index.json").write_text(json.dumps(index))
    return folder


NAMES = ["embed.weight", "lm_head.weight", "lm_head.weight_scale", "layers.0.ffn.experts.0.w1.trellis",
         "layers.39.ffn.experts.383.w2.trellis", "mtp.0.ffn.experts.0.w1.weight", "mtp.0.ffn.experts.0.w1.scale"]


@pytest.fixture
def engine(monkeypatch):
    """The engine class stubbed: cuda_engine's arguments without a GPU."""

    made = []

    def stub(*args, **kwargs):
        made.append(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setitem(sys.modules, ENGINE, SimpleNamespace(DeepSeekV41Engine=stub))
    return made


def test_discovery_and_cuda_engine_refusals_never_import_torch():
    script = ("import sys, tensorfold.families as f\n"
              "assert f.families()['deepseek_v41'].lanes\n"
              "from tensorfold.families import deepseek_v41\n"
              "for bad in ({'tp': 1}, {'tp': 2}, {'tp': 2, 'master': 'm', 'drafter': 'd'},\n"
              "            {'tp': 2, 'master': 'm', 'mtp_drafts': 6}):\n"
              "    try:\n"
              "        deepseek_v41.cuda_engine('.', **bad)\n"
              "    except ValueError:\n"
              "        pass\n"
              "    else:\n"
              "        raise SystemExit(f'not refused: {bad}')\n"
              "assert 'torch' not in sys.modules, 'torch imported'\n")
    src = str(Path(families.__file__).parents[2])
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [src, os.environ.get("PYTHONPATH")]))}
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, check=False)
    assert run.returncode == 0, run.stderr


def test_the_pack_config_is_detected_and_accepted(tmp_path, capsys):
    folder = _write(tmp_path)
    family = families.detect(folder)
    assert family.module == "tensorfold.families.deepseek_v41" and family.title == "DeepSeek-V4.1-Flash"
    families.require_readable(family, families.read_config(folder), "cuda")
    deepseek_v41.check(folder)
    assert "two DGX Sparks" in capsys.readouterr().out
    deepseek_v41.check(_write(tmp_path, names=NAMES))                  # and once downloaded


def _set(path: str, value):
    def edit(config):
        *parents, key = path.split(".")
        block = config
        for parent in parents:
            block = block[parent]
        if value is None:
            del block[key]
        else:
            block[key] = value
    return edit


@pytest.mark.parametrize("path, value, message", [
    ("quantization_config.quant_method", "modelopt", "reads EXL3 routed experts"),
    ("quantization_config.non_routed_quantization.quant_method", "fp8", "DeepSeek FP8"),
    ("quantization_config.non_routed_quantization.fmt", "e5m2", "fmt e5m2"),
    ("quantization_config.non_routed_quantization.scale_fmt", "fp32", "scale_fmt fp32"),
    ("quantization_config.non_routed_quantization.weight_block_size", [128, 128], r"weight_block_size \[128, 128\]"),
    ("quantization_config.mtp_experts", "exl3", "mtp_experts 'exl3'"),
    ("text_config.index_topk", None, "has no index_topk"),
    ("text_config.rope_scaling", None, "has no rope_scaling"),
    ("text_config.num_attention_heads", 63, "63 heads"),
    ("text_config.vocab_size", 129281, "vocabulary 129281"),
    ("text_config.moe_intermediate_size", 2176, "intermediate size 2176"),
    ("text_config.candidate_topk_blocks", 64, "64 candidate blocks of 8 cannot hold the 512"),
])
def test_check_refuses_what_the_engine_does_not_read(tmp_path, path, value, message):
    with pytest.raises(ValueError, match=message):
        deepseek_v41.check(_write(tmp_path, _set(path, value)))


@pytest.mark.parametrize("names, message", [
    ([n for n in NAMES if n != "lm_head.weight_scale"], "MXFP8 LM head"),
    (NAMES + ["mtp.0.ffn.experts.0.w1.trellis"], "mtp.0.ffn.experts.0.w1.trellis first"),
    (NAMES + ["layers.0.attn.wq_a.trellis"], "layers.0.attn.wq_a.trellis first"),
])
def test_check_refuses_a_downloaded_layout_it_does_not_read(tmp_path, names, message):
    with pytest.raises(ValueError, match=message):
        deepseek_v41.check(_write(tmp_path, names=names))


@pytest.mark.parametrize("options, message", [
    ({"tp": 1}, "needs two GPUs"), ({"tp": 2}, "needs --master"),
    ({**RANKS, "drafter": "/drafter"}, "own DSpark stages"),
    ({**RANKS, "mtp_drafts": 6}, "--mtp-drafts 6"), ({**RANKS, "mtp_drafts": -1}, "--mtp-drafts -1")])
def test_cuda_engine_refuses_settings_before_building(tmp_path, engine, options, message):
    with pytest.raises(ValueError, match=message):
        deepseek_v41.cuda_engine(tmp_path, **options)
    assert not engine


def test_cuda_engine_passes_its_policy_and_context(tmp_path, engine, capsys):
    built = deepseek_v41.cuda_engine(tmp_path, rank=1, master_port=29600, context=4096, context_explicit=True, **RANKS)
    assert built.policy == (3, None) and built.serial_only is False                  # three drafts a round
    assert (built.rank, built.master, built.port, built.context, built.context_explicit) == \
        (1, "192.0.2.10", 29600, 4096, True)
    assert deepseek_v41.cuda_engine(tmp_path, mtp_drafts=0, **RANKS).policy == (0, None)   # serial, as GLM
    assert deepseek_v41.cuda_engine(tmp_path, mtp_drafts=5, **RANKS).policy == (5, None)
    assert deepseek_v41.cuda_engine(tmp_path, mtp_confidence=0.6, **RANKS).policy == (3, 0.6)
    assert deepseek_v41.cuda_engine(tmp_path, no_drafts=True, **RANKS).serial_only is True
    deepseek_v41.cuda_engine(tmp_path, parallel=4, **RANKS)
    assert "--parallel 4 is ignored" in capsys.readouterr().out
    assert "parallel" not in engine[-1]


def test_cuda_app_is_imported_on_first_use(monkeypatch):
    app = type("DeepSeekV41App", (), {})
    monkeypatch.setitem(sys.modules, "tensorfold.families.deepseek_v41.cuda.app", SimpleNamespace(DeepSeekV41App=app))
    assert deepseek_v41.CUDA_APP is app
    with pytest.raises(AttributeError):
        deepseek_v41.NO_SUCH_NAME      # noqa: B018


def test_serve_options_refuse_prefill_fp8_and_take_mtp_confidence(tmp_path):
    """No CUDA_PREFILL_FP8: prompts always run bf16 activations; --mtp-confidence reaches cuda_engine."""

    family = SimpleNamespace(title=deepseek_v41.TITLE, package=deepseek_v41, model_type="deepseek_v41")
    assert not hasattr(deepseek_v41, "CUDA_PREFILL_FP8")
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--prefill-fp8"])
    with pytest.raises(ValueError, match="--prefill-fp8 .* DeepSeek-V4.1-Flash on CUDA has none"):
        serve_options.check(args, family, "cuda")
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--mtp-confidence", "0.6"])
    assert serve_options.check(args, family, "cuda") is None
