from __future__ import annotations

import io
import json
import struct
import tempfile
import unittest
from contextlib import redirect_stdout
from contextlib import redirect_stderr
import importlib
from pathlib import Path
import subprocess
import sys
from unittest import mock

import tensorfold.cli
from tensorfold.cli import main


def write_toy_safetensors(path: Path) -> None:
    tensors = {
        "model.embed_tokens.weight": ("F32", [2, 4], bytes(range(0, 32))),
        "model.layers.0.self_attn.q_proj.weight": ("F32", [4, 4], bytes(range(32, 96))),
    }
    offset = 0
    header: dict[str, object] = {"__metadata__": {"format": "toy-tensorfold"}}
    payload = bytearray()
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
        payload.extend(data)

    raw_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + payload)


def write_toy_expert_safetensors(path: Path) -> None:
    data = bytes(range(32))
    header = {
        "__metadata__": {"format": "toy-tensorfold-moe"},
        "model.layers.0.mlp.experts.gate_up_proj.weight": {
            "dtype": "F32",
            "shape": [2, 4],
            "data_offsets": [0, len(data)],
        },
    }
    raw_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + data)


def write_toy_glm_model(model_dir: Path) -> None:
    model_dir.mkdir()
    tensors: dict[str, tuple[str, list[int], bytes]] = {}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        tensors[f"model.layers.8.mlp.switch_mlp.{proj}.weight"] = ("U8", [2, 2, 4], bytes(range(16)))
        tensors[f"model.layers.8.mlp.switch_mlp.{proj}.scales"] = ("F32", [2, 2], bytes(range(16)))
        tensors[f"model.layers.8.mlp.switch_mlp.{proj}.biases"] = ("F32", [2, 2], bytes(range(16)))

    offset = 0
    header: dict[str, object] = {"__metadata__": {"format": "toy-glm"}}
    payload = bytearray()
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
        payload.extend(data)

    raw_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    shard = model_dir / "model-00001-of-00001.safetensors"
    shard.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + payload)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "glm_moe_dsa",
                "architectures": ["GlmMoeDsaForCausalLM"],
                "hidden_size": 4,
                "moe_intermediate_size": 2,
                "n_routed_experts": 2,
                "num_experts_per_tok": 1,
                "quantization": {"bits": 8, "group_size": 2},
            }
        )
    )
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard.name for name in tensors}})
    )


class TensorFoldCliTests(unittest.TestCase):
    def test_help_uses_tensorfold_product_name(self) -> None:
        stdout = io.StringIO()

        with self.assertRaises(SystemExit) as raised, redirect_stdout(stdout):
            main(["--help"])

        self.assertEqual(raised.exception.code, 0)
        text = stdout.getvalue()
        self.assertIn("TensorFold Runtime", text)
        self.assertIn("doctor", text)
        self.assertIn("inspect", text)
        self.assertIn("serve", text)

    def test_doctor_runs_without_loading_model(self) -> None:
        with mock.patch("smarttensor.cli.main") as smart_main:
            exit_code = main(["doctor"])

        self.assertEqual(exit_code, 0)
        smart_main.assert_not_called()

    def test_demo_create_writes_installable_no_model_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            demo_dir = Path(tmp) / "demo"
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(["demo", "create", str(demo_dir)])

            toy_file = demo_dir / "toy.safetensors"
            moe_shard = demo_dir / "toy-moe" / "model-00001-of-00001.safetensors"
            toy_exists = toy_file.is_file()
            moe_exists = moe_shard.is_file()

        self.assertEqual(exit_code, 0)
        self.assertTrue(toy_exists)
        self.assertTrue(moe_exists)
        self.assertIn("tensorfold inspect", stdout.getvalue())
        self.assertIn("tensorfold pack", stdout.getvalue())

    def test_demo_create_refuses_existing_directory_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            demo_dir = Path(tmp) / "demo"
            demo_dir.mkdir()
            (demo_dir / "keep.txt").write_text("do not overwrite\n")
            stderr = io.StringIO()

            with redirect_stderr(stderr):
                exit_code = main(["demo", "create", str(demo_dir)])

        self.assertEqual(exit_code, 2)
        self.assertIn("--force", stderr.getvalue())

    def test_inspect_delegates_to_smarttensor_manifest_reader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            toy = Path(tmp) / "toy.safetensors"
            write_toy_safetensors(toy)
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(["inspect", str(toy), "--json"])

        self.assertEqual(exit_code, 0)
        self.assertIn("model.layers.0.self_attn.q_proj.weight", stdout.getvalue())

    def test_serve_help_delegates_to_runtime_options(self) -> None:
        stdout = io.StringIO()

        with self.assertRaises(SystemExit) as raised, redirect_stdout(stdout):
            main(["serve", "--help"])

        self.assertEqual(raised.exception.code, 0)
        text = stdout.getvalue()
        self.assertIn("usage: tensorfold serve", text)
        self.assertNotIn("usage: smarttensor serve", text)
        self.assertIn("--resident-budget", text)
        self.assertIn("--pack-dir", text)
        self.assertIn("--pack-read-workers", text)

    def test_pack_help_uses_tensorfold_command_name(self) -> None:
        stdout = io.StringIO()

        with self.assertRaises(SystemExit) as raised, redirect_stdout(stdout):
            main(["pack", "--help"])

        self.assertEqual(raised.exception.code, 0)
        text = stdout.getvalue()
        self.assertIn("usage: tensorfold pack", text)
        self.assertNotIn("pack-experts", text)
        self.assertIn("--out", text)
        self.assertIn("--layers", text)

    def test_pack_builds_expert_pack_for_tiny_moe_shard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "model"
            model_dir.mkdir()
            write_toy_expert_safetensors(model_dir / "model-00001-of-00001.safetensors")
            pack_dir = root / "packs"
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(["pack", str(model_dir), "--out", str(pack_dir)])

            packs = sorted(pack_dir.glob("*.pack"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(packs), 1)
        self.assertIn("wrote 1 pack", stdout.getvalue())

    def test_pack_layers_filters_expert_tensors_by_layer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "glm"
            write_toy_glm_model(model_dir)
            pack_dir = root / "packs-layer8"
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(["pack", str(model_dir), "--out", str(pack_dir), "--layers", "8"])

            packs = sorted(pack_dir.glob("*.pack"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(packs), 1)
        self.assertIn("wrote 1 pack", stdout.getvalue())

    def test_pack_layers_returns_no_pack_when_filter_excludes_all_experts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "glm"
            write_toy_glm_model(model_dir)
            pack_dir = root / "packs-layer7"
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(["pack", str(model_dir), "--out", str(pack_dir), "--layers", "7"])

            packs = sorted(pack_dir.glob("*.pack"))

        self.assertEqual(exit_code, 1)
        self.assertEqual(packs, [])
        self.assertIn("no expert-axis tensors found", stdout.getvalue())

    def test_frontier_profile_sizes_glm_moe_without_loading_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "glm"
            write_toy_glm_model(model_dir)
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(["frontier-profile", str(model_dir), "--budget", "200B", "--json"])

        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["model_type"], "glm_moe_dsa")
        self.assertTrue(payload["shards"]["complete"])
        self.assertEqual(payload["routed_layer_count"], 1)
        self.assertEqual(payload["bytes_per_expert"], 72)
        self.assertEqual(payload["budget_fits"][0]["experts_per_layer"], 2)

    def test_public_toy_moe_example_exercises_inspect_and_pack(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "toy-moe"
            completed = subprocess.run(
                [sys.executable, "examples/create_toy_moe_model.py", str(model_dir)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            shard = model_dir / "model-00001-of-00001.safetensors"
            inspect_stdout = io.StringIO()

            with redirect_stdout(inspect_stdout):
                inspect_code = main(["inspect", str(shard)])

            pack_dir = root / "packs"
            pack_stdout = io.StringIO()
            with redirect_stdout(pack_stdout):
                pack_code = main(["pack", str(model_dir), "--out", str(pack_dir)])

            packs = sorted(pack_dir.glob("*.pack"))

        self.assertEqual(inspect_code, 0)
        self.assertIn("layers:       1", inspect_stdout.getvalue())
        self.assertEqual(pack_code, 0)
        self.assertEqual(len(packs), 1)
        self.assertIn("wrote 1 pack", pack_stdout.getvalue())

    def test_canary_reports_source_checkout_requirement_when_tools_package_is_unavailable(self) -> None:
        original_import_module = importlib.import_module

        def import_or_block(name: str, package: str | None = None):
            if name == "tools.run_qwen_frontier_canary_pipeline":
                exc = ModuleNotFoundError("No module named 'tools'")
                exc.name = "tools"
                raise exc
            return original_import_module(name, package)

        stderr = io.StringIO()
        with mock.patch.object(tensorfold.cli.importlib, "import_module", side_effect=import_or_block):
            with redirect_stderr(stderr):
                exit_code = main(["canary", "qwen-frontier", "--dry-run"])

        self.assertEqual(exit_code, 2)
        self.assertIn("source checkout", stderr.getvalue())

    def test_canary_reports_source_checkout_requirement_when_runner_is_unavailable(self) -> None:
        original_import_module = importlib.import_module

        def import_or_block(name: str, package: str | None = None):
            if name == "tools.run_qwen_frontier_canary_pipeline":
                raise ModuleNotFoundError(name)
            return original_import_module(name, package)

        stderr = io.StringIO()
        with mock.patch.object(tensorfold.cli.importlib, "import_module", side_effect=import_or_block):
            with redirect_stderr(stderr):
                exit_code = main(["canary", "qwen-frontier", "--dry-run"])

        self.assertEqual(exit_code, 2)
        self.assertIn("source checkout", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
