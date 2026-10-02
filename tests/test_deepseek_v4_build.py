"""Verified source reuse, isolated build/preflight, and installation failure boundaries."""

import hashlib
import subprocess
from unittest.mock import patch

import pytest

from tensorfold.families.deepseek_v4.cuda import build_runtime as b
from tensorfold.families.deepseek_v4.cuda import sources as s


def args(tmp_path, *extra):
    return b.parser().parse_args(
        [
            "--gguf",
            str(tmp_path / "target.gguf"),
            "--model-dir",
            str(tmp_path / "model"),
            "--companion-reserve-gib",
            "0",
            *extra,
        ]
    )


def test_preflight_read_only_and_invalid_budget_before_inspection(tmp_path):
    with (
        patch.object(b, "inspect_inputs", return_value={"artifact_valid": True}),
        patch.object(b, "install_candidate", side_effect=AssertionError("installation reached")),
    ):
        assert b.execute(args(tmp_path, "--preflight-only"))["artifact_valid"]
        assert not (tmp_path / "model").exists()
    for option in ("nan", "inf", "-1"):
        with (
            patch.object(b, "inspect_inputs", side_effect=AssertionError("inspection reached")),
            pytest.raises(ValueError),
        ):
            b.execute(args(tmp_path, "--companion-reserve-gib", option))


def test_install_failure_leaves_no_prepared_model(tmp_path):
    report = {"artifact_valid": True}
    with (
        patch.object(b, "inspect_inputs", return_value=report),
        patch.object(b, "install_candidate", side_effect=RuntimeError("wheel failed")),
        pytest.raises(RuntimeError, match="wheel failed"),
    ):
        b.execute(args(tmp_path))
    assert not (tmp_path / "model/config.json").exists()


def test_success_and_preparation_publish_installed_library(tmp_path):
    library = tmp_path / "installed.so"
    source = tmp_path / "target.gguf"
    source.write_bytes(bytes(100))
    report = {
        "artifact_valid": True,
        "source": str(tmp_path / "target.gguf"),
        "arch": {},
        "source_size": 100,
        "header_sha256": "header",
        "source_identity": {**b.stamp(source), "sha256_scope": "gguf-header"},
        "tokenizer": {},
        "external_tokenizer": {},
    }
    with (
        patch.object(b, "inspect_inputs", return_value=report),
        patch.object(b, "install_candidate", return_value=(library, {"sha256": "wheel"})),
        patch.object(b, "prepare_candidate", side_effect=lambda model, **_: model.mkdir()) as prepare,
    ):
        result = b.execute(args(tmp_path))
    assert prepare.call_args.kwargs["native_library"] == library
    assert prepare.call_args.kwargs["source_identity"]["sha256_scope"] == "gguf-header"
    assert result["wheel"]["sha256"] == "wheel"


def test_aligned_budget_adds_dense_q8_but_not_replaced_expert_weights():
    from types import SimpleNamespace as T

    dense = T(name="blk.0.attn_q_a.weight", type_id=8, shape=(1024, 2048), size=1024 * 2048 // 32 * 34)
    iq2 = T(name="blk.0.ffn_gate_exps.weight", type_id=16, shape=(1024, 2048, 256), size=1024 * 2048 * 256 // 256 * 66)
    q2 = T(name="blk.0.ffn_down_exps.weight", type_id=10, shape=(2048, 4096, 256), size=2048 * 4096 * 256 // 256 * 84)
    assert b.aligned_artifact_extra_bytes([dense, iq2, q2]) == dense.size
    dense.name = "token_embd.weight"
    assert b.aligned_artifact_extra_bytes([dense, iq2, q2]) == 0


def test_verified_library_reused_without_source_acquisition(tmp_path):
    import json

    library = tmp_path / "libtensorfold_ds4.so"
    library.write_bytes(b"verified library fixture")
    receipt = {
        "abi": 2,
        "revision": b.PIN,
        "backend": "cuda",
        "cuda_arch": "sm_121a",
        "source_sha256": b.source_manifest()["sha256"],
        "shim_sha256": b.digest(b.PACKAGE / "native/shim.c"),
        "sampling_hook_sha256": b.digest(b.PACKAGE / "sampling_hook.py"),
        "library_sha256": b.digest(library),
    }
    (tmp_path / "native-build.json").write_text(json.dumps(receipt))
    with patch.object(b, "build", side_effect=AssertionError("source acquisition reached")):
        assert b.native_library(args(tmp_path, "--native-library", str(library), "--offline")) == library


@pytest.fixture
def local_donor(tmp_path, monkeypatch):
    repository = tmp_path / "donor"
    repository.mkdir()
    (repository / "ds4.c").write_text("pinned native input\n")
    subprocess.run(["git", "init", "--quiet", str(repository)], check=True)
    subprocess.run(["git", "-C", str(repository), "add", "ds4.c"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "pin",
        ],
        check=True,
    )
    pin = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    manifest = {
        "repository": str(repository),
        "revision": pin,
        "sha256": {"ds4.c": hashlib.sha256((repository / "ds4.c").read_bytes()).hexdigest()},
    }
    monkeypatch.setattr(s, "PIN", pin)
    monkeypatch.setattr(s, "source_manifest", lambda: manifest)
    monkeypatch.delenv("TENSORFOLD_DS4_SOURCE", raising=False)
    return repository


def test_packaged_pin_and_license():
    manifest = s.source_manifest()
    assert manifest["revision"] == s.PIN
    assert len(manifest["sha256"]) == 42
    assert (
        manifest["sha256"]["LICENSE"]
        == hashlib.sha256((s.PACKAGE.parents[4] / "LICENSES/ds4.txt").read_bytes()).hexdigest()
    )


def test_local_and_cached_sources_never_fetch(local_donor, tmp_path):
    cache = tmp_path / "cache"
    with patch.object(s.subprocess, "run", side_effect=AssertionError("Git reached")):
        assert s.resolve_sources(local_donor, cache_dir=cache, offline=True) == cache
        assert s.resolve_sources(cache_dir=cache) == cache
    (cache / "ds4.c").write_text("damaged cache")
    with pytest.raises(ValueError, match="missing or changed"):
        s.resolve_sources(cache_dir=cache)


def test_modified_local_checkout_uses_pinned_object_without_fetch(local_donor, tmp_path):
    (local_donor / "ds4.c").write_text("local edits must remain intact")
    original_run = subprocess.run

    def no_fetch(command, **kwargs):
        assert "fetch" not in command
        return original_run(command, **kwargs)

    with patch.object(s.subprocess, "run", side_effect=no_fetch):
        cache = s.resolve_sources(local_donor, cache_dir=tmp_path / "cache", offline=True)
    assert (cache / "ds4.c").read_text() == "pinned native input\n"
    assert (local_donor / "ds4.c").read_text() == "local edits must remain intact"


def test_missing_source_fetches_once_and_offline_refuses(local_donor, tmp_path):
    cache = tmp_path / "cache"
    with pytest.raises(RuntimeError, match="cache missing"):
        s.resolve_sources(cache_dir=cache, offline=True)
    with patch.object(s.subprocess, "run", wraps=subprocess.run) as run:
        s.resolve_sources(cache_dir=cache)
        s.resolve_sources(cache_dir=cache)
    assert sum("fetch" in call.args[0] for call in run.call_args_list) == 1
    s.verify_sources(cache)
