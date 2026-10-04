"""The native engine's contract: JSON schemas, gate manifest, weight formats, bundle identity, build hook, gate step."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from tensorfold import native
from tensorfold.native import contract

REPO = Path(__file__).resolve().parents[1]
CAPS = {"schema": 1, "engine": "zig", "version": "0.6.6", "chip": "apple-m5", "backends": ["metal"],
        "families": {"nemotron_h": ["mlx-q4g64"]},
        "serve": {"flags": {"--port": {}, "--backend": {"values": ["auto", "mlx"]}}},
        "env": ["TENSORFOLD_NO_LIVE", "TENSORFOLD_NO_UPDATE_CHECK"]}
ENTRY = {"family": "nemotron_h", "format": "mlx-q4g64", "backend": "metal", "chip": "apple-m5", "run": "rel066a",
         "bundle": "a" * 64}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_binary(path: Path, body: str) -> Path:
    if " " in sys.executable:
        pytest.skip("a script's #! line can't name an interpreter path with spaces")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!{sys.executable}\nimport json, sys\n{body}\n")
    path.chmod(0o755)
    return path


def test_the_repo_manifest_is_the_empty_one_the_gate_step_writes(tmp_path):
    shipped = REPO / "zig" / "gate.json"
    contract.check(json.loads(shipped.read_text()), contract.schema("gate"))
    gate = _load("zig_gate", REPO / "tools" / "zig_gate.py")
    assert gate.write(tmp_path / "gate.json", []) == {"schema": 1, "entries": []}
    assert (tmp_path / "gate.json").read_bytes() == shipped.read_bytes()


@pytest.mark.parametrize("change, where", [
    ({"schema": 2}, "$.schema"), ({"schema": True}, "$.schema"), ({"engine": "rust"}, "$.engine"),
    ({"version": "0.6"}, "$.version"), ({"chip": "gpu-0"}, "$.chip"), ({"chip": "m5"}, "$.chip"),
    ({"backends": ["metal", "metal"]}, "$.backends"), ({"backends": ["vulkan"]}, "$.backends[0]"),
    ({"families": {"Nemotron": []}}, "$.families.Nemotron"), ({"serve": {"flags": {"-p": {}}}}, "$.serve.flags.-p"),
    ({"serve": {"flags": {"--port": {"default": 8080}}}}, "$.serve.flags.--port"), ({"env": ["PATH"]}, "$.env[0]"),
    ({"extra": 1}, "$"),
])
def test_the_capabilities_schema_names_each_break(change, where):
    contract.check(CAPS, contract.schema("capabilities"))
    contract.check({**CAPS, "chip": None}, contract.schema("capabilities"))      # no usable GPU here
    with pytest.raises(ValueError) as caught:
        contract.check({**CAPS, **change}, contract.schema("capabilities"))
    assert str(caught.value).startswith(where + ":")


@pytest.mark.parametrize("change", [{"bundle": "A" * 64}, {"bundle": "a" * 63}, {"run": "../elsewhere"},
                                    {"backend": "mlx"}, {"chip": "sm_12.1"}, {"format": "MLX 4-bit"}, {"note": "x"}])
def test_the_gate_schema_takes_whole_public_entries_only(change):
    contract.check({"schema": 1, "entries": [ENTRY]}, contract.schema("gate"))
    with pytest.raises(ValueError):
        contract.check({"schema": 1, "entries": [{**ENTRY, **change}]}, contract.schema("gate"))
    for key in ENTRY:
        with pytest.raises(ValueError, match=f"missing {key}"):
            contract.check({"schema": 1, "entries": [{k: v for k, v in ENTRY.items() if k != key}]},
                           contract.schema("gate"))


def test_the_checker_runs_only_the_keywords_it_implements():
    with pytest.raises(ValueError, match="unsupported schema keywords"):
        contract.check(3, {"type": "integer", "minimum": 0})
    with pytest.raises(ValueError):
        contract.check(True, {"type": "integer"})
    with pytest.raises(ValueError):
        contract.check(True, {"const": 1})
    contract.check(1, {"type": "integer", "const": 1})


@pytest.mark.parametrize("config, expected", [
    ({"quantization": {"group_size": 64, "bits": 4}}, "mlx-q4g64"),
    ({"quantization": {"group_size": 32, "bits": 4}, "quantization_config": {"group_size": 32, "bits": 4}},
     "mlx-q4g32"),
    ({"quantization": {"group_size": 64, "bits": 4, "model.layers.3.mlp": {"bits": 8, "group_size": 64},
                       "model.layers.9.mlp": {"bits": 8, "group_size": 64}}}, "mlx-q4g64-q8g64"),
    ({"quantization": {"group_size": 64, "bits": 4, "lm_head": False}}, "mlx-q4g64-bf16"),
    ({"quantization": {"group_size": 64, "bits": 4, "model.layers.0.mlp": {"mode": "mxfp4"}}}, "mlx-q4g64-mxfp4g32"),
    ({"text_config": {"model_type": "x"}}, "unquantized"),
    ({"quantization_config": {"quant_method": "exl3", "bits": 4}}, "exl3-b4"),
    ({"quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"}}, "modelopt-nvfp4"),
])
def test_weight_formats_differ_wherever_the_kernels_would(config, expected):
    name = contract.weight_format(config)
    assert name == expected
    contract.check(name, contract.schema("gate")["properties"]["entries"]["items"]["properties"]["format"])


def test_the_bundle_digest_covers_the_bundle_files_and_nothing_else(tmp_path, monkeypatch):
    monkeypatch.setattr(native, "ROOT", tmp_path)
    empty = native.bundle_digest()
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "tensorfold-native").write_bytes(b"\xcf\xfa\xed\xfe engine")
    (tmp_path / "kernels").mkdir()
    (tmp_path / "kernels" / "lane.metallib").write_bytes(b"kernels")
    first = native.bundle_digest()
    for name in ("__init__.py", "switch.py", "gate.json", "gate.schema.json"):
        (tmp_path / name).write_text("{}")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "switch.cpython-314.pyc").write_bytes(b"bytecode")
    assert native.bundle_digest() == first != empty
    (tmp_path / "kernels" / "lane.metallib").write_bytes(b"kernelz")
    assert native.bundle_digest() != first
    (tmp_path / "kernels" / "lane.metallib").write_bytes(b"kernels")
    (tmp_path / "kernels" / "lane.metallib").rename(tmp_path / "kernels" / "other.metallib")
    assert native.bundle_digest() != first
    assert native.binary() is None                                   # present but not executable
    (tmp_path / "bin" / "tensorfold-native").chmod(0o755)
    assert native.binary() == tmp_path / "bin" / "tensorfold-native"


def test_a_missing_manifest_is_empty_and_a_broken_one_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(native, "ROOT", tmp_path)
    assert contract.manifest() == {"schema": 1, "entries": []}
    (tmp_path / "gate.json").write_text(json.dumps({"schema": 1, "entries": [{**ENTRY, "chip": "anything"}]}))
    with pytest.raises(ValueError, match="chip"):
        contract.manifest()
    (tmp_path / "gate.json").write_text("{")
    with pytest.raises(ValueError):
        contract.manifest()


def test_the_capabilities_probe_returns_only_a_valid_answer(tmp_path):
    good = fake_binary(tmp_path / "good", f"print(json.dumps({CAPS!r}))")
    assert contract.capabilities(good) == CAPS
    for name, body, reason in (("fails", "sys.exit(3)", "status 3"), ("junk", "print('ready')", "not JSON"),
                               ("old", f"print(json.dumps({{**{CAPS!r}, 'schema': 0}}))", "schema")):
        with pytest.raises(ValueError, match=reason):
            contract.capabilities(fake_binary(tmp_path / name, body))
    with pytest.raises(ValueError, match="could not run"):
        contract.capabilities(tmp_path / "absent")


def test_the_build_hook_is_inert_without_its_flag(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "setuptools", SimpleNamespace(setup=lambda **kw: calls.append(kw)))
    monkeypatch.delenv("TENSORFOLD_NATIVE_BUNDLE", raising=False)
    _load("tensorfold_setup", REPO / "setup.py").main()
    assert calls == [{}]                         # exactly the setup() setuptools runs with no setup.py at all


def test_the_build_hook_takes_only_a_bundle_for_the_wheels_platform(tmp_path, monkeypatch):
    hook = _load("tensorfold_setup", REPO / "setup.py")
    binary = tmp_path / "bin" / "tensorfold-native"
    with pytest.raises(SystemExit, match="missing"):
        hook.bundle_files(tmp_path, "macosx_14_0_arm64")
    binary.parent.mkdir()
    binary.write_bytes(b"\xcf\xfa\xed\xfe" + b"\0" * 16)
    with pytest.raises(SystemExit, match="not executable"):
        hook.bundle_files(tmp_path, "macosx_14_0_arm64")
    binary.chmod(0o755)
    with pytest.raises(SystemExit, match="manylinux_2_28_aarch64 executable"):
        hook.bundle_files(tmp_path, "manylinux_2_28_aarch64")
    (tmp_path / "kernels").mkdir()
    (tmp_path / "kernels" / "lane.metallib").write_bytes(b"kernels")
    (tmp_path / "README").write_text("not part of the bundle")
    assert hook.bundle_files(tmp_path, "macosx_14_0_arm64") == [Path("bin/tensorfold-native"),
                                                                Path("kernels/lane.metallib")]
    assert hook.bundle_files(tmp_path, "macosx_14_0_arm64") == native.bundle_files(tmp_path)
    monkeypatch.setenv("TENSORFOLD_NATIVE_PLATFORM", "manylinux_2_28_x86_64")
    assert hook.platform_tag() == "manylinux_2_28_x86_64"


def test_the_gate_step_writes_sorted_checked_entries_for_the_bundles_asked(tmp_path):
    gate = _load("zig_gate", REPO / "tools" / "zig_gate.py")
    other = {**ENTRY, "chip": "apple-m3", "run": "rel066b", "bundle": "b" * 64}
    (tmp_path / "one.json").write_text(json.dumps(other))
    (tmp_path / "two.json").write_text(json.dumps({"schema": 1, "entries": [ENTRY, other]}))
    written = gate.write(tmp_path / "gate.json", [tmp_path / "one.json", tmp_path / "two.json"])
    assert written["entries"] == [ENTRY, other]
    assert json.loads((tmp_path / "gate.json").read_text()) == written
    assert gate.write(tmp_path / "gate.json", [tmp_path / "two.json"], ["b" * 64])["entries"] == [other]
    (tmp_path / "bad.json").write_text(json.dumps({**ENTRY, "run": "box name"}))
    with pytest.raises(ValueError):
        gate.write(tmp_path / "gate.json", [tmp_path / "bad.json"])


def test_the_gate_step_records_this_installs_cell_and_verifies_it(tmp_path, monkeypatch, capsys):
    gate = _load("zig_gate", REPO / "tools" / "zig_gate.py")
    root, model = tmp_path / "native", tmp_path / "model"
    monkeypatch.setattr(native, "ROOT", root)
    fake_binary(root / "bin" / "tensorfold-native", f"print(json.dumps({CAPS!r}))")
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "nemotron_h",
                                                   "quantization": {"group_size": 64, "bits": 4}}))
    entry = gate.cell("rel066a", model, "mlx")
    assert entry == {**ENTRY, "bundle": native.bundle_digest()}
    assert gate.verify() == 1                                         # no manifest names this bundle yet
    (tmp_path / "cell.json").write_text(json.dumps(entry))
    gate.write(root / "gate.json", [tmp_path / "cell.json"])
    capsys.readouterr()
    assert gate.verify() == 0
    assert json.loads(capsys.readouterr().out)["this_bundle"] == [entry]


def test_every_schema_keyword_is_one_the_checker_runs():
    def walk(rules):
        assert set(rules) <= contract._KEYWORDS, set(rules) - contract._KEYWORDS
        for inner in rules.get("properties", {}).values():
            walk(inner)
        for key in ("items", "additionalProperties", "propertyNames"):
            if isinstance(rules.get(key), dict):
                walk(rules[key])

    for name in ("capabilities", "gate"):
        walk(contract.schema(name))
