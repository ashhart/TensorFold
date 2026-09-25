"""Models by Hugging Face repo id (resolved in a local cache, no network here) and the families' checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorfold import families, hub
from tensorfold.cli import main

COMMIT = "0123456789abcdef0123456789abcdef01234567"


def fake_repo(cache: Path, repo_id: str, files: dict[str, str]) -> Path:
    """A repo in Hugging Face's cache layout: refs/main naming a snapshot folder."""

    folder = cache / f"models--{repo_id.replace('/', '--')}"
    (folder / "refs").mkdir(parents=True)
    (folder / "refs" / "main").write_text(COMMIT)
    snapshot = folder / "snapshots" / COMMIT
    snapshot.mkdir(parents=True)
    for name, text in files.items():
        (snapshot / name).write_text(text)
    return snapshot


def test_repo_ids_and_local_directories(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert hub.is_repo_id("Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP")
    assert not hub.is_repo_id("just-a-name") and not hub.is_repo_id("a/b/c")
    (tmp_path / "local" / "model").mkdir(parents=True)
    assert not hub.is_repo_id("local/model")                 # an existing directory is a directory
    assert hub.resolve("local/model", download=False) == Path("local/model")


def test_cached_repos_resolve_and_missing_ones_say_how_to_pull(tmp_path):
    snapshot = fake_repo(tmp_path, "owner/model", {"config.json": "{}"})
    assert hub.resolve("owner/model", download=False, cache_dir=tmp_path) == snapshot
    assert hub.cached("owner/other", cache_dir=tmp_path) is None
    with pytest.raises(FileNotFoundError, match="tensorfold pull owner/other"):
        hub.resolve("owner/other", download=False, cache_dir=tmp_path)


def test_a_cache_without_refs_still_resolves(tmp_path):
    snapshot = fake_repo(tmp_path, "owner/model", {"config.json": "{}"})
    (tmp_path / "models--owner--model" / "refs" / "main").unlink()
    assert hub.cached("owner/model", cache_dir=tmp_path) == snapshot


def test_quantization_is_read_from_the_config():
    assert families.quantization({"quantization": {"bits": 4, "group_size": 32}}) == (4, 32)
    assert families.quantization({"text_config": {"quantization_config": {"bits": 8}}}) == (8, 64)
    assert families.quantization({}) == (None, None)


def write_checkpoint(folder: Path, bits: int, group: int, mtp: bool) -> Path:
    folder.mkdir(parents=True)
    (folder / "config.json").write_text(json.dumps(
        {"model_type": "qwen4_exp", "quantization": {"bits": bits, "group_size": group}}))
    names = ["language_model.model.embed_tokens.weight"] + (["language_model.mtp.fc_hidden.weight"] if mtp else [])
    (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {n: "model.safetensors"
                                                                                    for n in names}}))
    return folder


def test_flash_next_refuses_other_quantizations_and_notes_a_missing_mtp_head(tmp_path, capsys):
    from tensorfold.families import qwen4_exp

    good = write_checkpoint(tmp_path / "good", 4, 32, mtp=True)
    qwen4_exp.check(good)
    assert qwen4_exp.has_mtp(good)
    with pytest.raises(ValueError, match="Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP"):
        qwen4_exp.check(write_checkpoint(tmp_path / "eight", 8, 64, mtp=True))
    plain = write_checkpoint(tmp_path / "plain", 4, 32, mtp=False)
    qwen4_exp.check(plain)
    assert not qwen4_exp.has_mtp(plain) and "no MTP head" in capsys.readouterr().out


def test_models_lists_the_tested_checkpoints(capsys):
    assert main(["models"]) == 0
    out = capsys.readouterr().out
    for repo in ("Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP", "Vontra/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit",
                 "Vontra/Qwen3.8-27B-MLX-4bit", "z-lab/Qwen3.8-27B-DFlash2"):
        assert repo in out


def test_info_reads_a_local_config(tmp_path, capsys):
    folder = write_checkpoint(tmp_path / "flash", 4, 32, mtp=True)
    assert main(["info", str(folder)]) == 0
    assert "Qwen3.8 Flash Next" in capsys.readouterr().out
    assert main(["info", str(write_checkpoint(tmp_path / "eight", 8, 64, mtp=True))]) == 1
