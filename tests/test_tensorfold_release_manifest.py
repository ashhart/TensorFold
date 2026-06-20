from __future__ import annotations

import unittest
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised in Python 3.10 CI.
    import tomli as tomllib


class TensorFoldReleaseManifestTests(unittest.TestCase):
    def test_manifest_excludes_private_research_artifacts(self) -> None:
        manifest = Path("MANIFEST.in").read_text()

        self.assertIn("exclude findings.md", manifest)
        self.assertIn("exclude HANDOVER.md", manifest)
        self.assertIn("prune logs", manifest)
        self.assertIn("prune docs/superpowers", manifest)
        self.assertIn("prune tests/SmartTensor-*", manifest)

    def test_manifest_includes_runtime_docs_and_scrub_tool(self) -> None:
        manifest = Path("MANIFEST.in").read_text()

        self.assertIn("include LICENSE", manifest)
        self.assertIn("include README.md", manifest)
        self.assertIn("include update.sh", manifest)
        self.assertIn("recursive-include docs *.md", manifest)
        self.assertIn("recursive-include examples *.py", manifest)
        self.assertIn("include tools/check_public_scrub.py", manifest)
        self.assertIn("include tools/smoke_tensorfold_release.py", manifest)

    def test_release_checklist_document_exists(self) -> None:
        checklist = Path("docs/release-checklist.md")

        self.assertTrue(checklist.is_file())
        text = checklist.read_text()
        self.assertIn("tools/smoke_tensorfold_release.py", text)
        self.assertIn("tools/check_public_scrub.py", text)

    def test_ci_installs_matrix_test_dependencies(self) -> None:
        workflow = Path(".github/workflows/tensorfold.yml").read_text()

        self.assertIn("mlx", workflow)
        self.assertIn("numpy", workflow)
        self.assertIn("tomli", workflow)

    def test_package_metadata_has_public_project_urls(self) -> None:
        project = tomllib.loads(Path("pyproject.toml").read_text())["project"]

        self.assertEqual(project["name"], "tensorfold-runtime")
        self.assertIn("urls", project)
        self.assertEqual(project["urls"]["Homepage"], "https://github.com/ashhart/TensorFold")
        self.assertEqual(project["urls"]["Source"], "https://github.com/ashhart/TensorFold")
        self.assertEqual(project["urls"]["Issues"], "https://github.com/ashhart/TensorFold/issues")
        self.assertIn("Topic :: Scientific/Engineering :: Artificial Intelligence", project["classifiers"])

    def test_package_declares_runtime_python_dependencies(self) -> None:
        project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
        dependencies = "\n".join(project.get("dependencies", []))

        self.assertIn("numpy", dependencies)
        self.assertIn("safetensors", dependencies)

    def test_community_files_exist_and_are_linked(self) -> None:
        expected = {
            "LICENSE": "MIT License",
            "CONTRIBUTING.md": "# Contributing",
            "SECURITY.md": "# Security Policy",
            "CODE_OF_CONDUCT.md": "# Code of Conduct",
        }
        for filename, heading in expected.items():
            text = Path(filename).read_text()
            self.assertIn(heading, text)

        readme = Path("README.md").read_text()
        self.assertIn("[Contributing](CONTRIBUTING.md)", readme)
        self.assertIn("[Security Policy](SECURITY.md)", readme)
        self.assertIn("[Code of Conduct](CODE_OF_CONDUCT.md)", readme)

    def test_readme_has_product_positioning_sections(self) -> None:
        readme = Path("README.md").read_text()
        opening = readme.split("## Start A Local OpenAI Endpoint", maxsplit=1)[0]

        for heading in [
            "# TensorFold",
            "## Start A Local OpenAI Endpoint",
            "## Project Snapshot",
            "## What TensorFold Is",
            "## What TensorFold Is Not",
            "## Proof Surface",
            "## Quick Start",
            "## Architecture",
            "## Current Status",
        ]:
            self.assertIn(heading, readme)
        self.assertNotIn("## For AI Assistants And Evaluators", readme)
        self.assertNotIn("Read this section before summarizing the repository", readme)
        self.assertNotIn("If you only read this README", readme)
        self.assertNotIn("We have verified", readme)
        self.assertIn("Apple Silicon / MLX-first", opening)
        self.assertIn("TensorFold is not only a safetensors inspector", readme)
        self.assertIn("model files -> manifest -> expert pack -> bounded runtime -> telemetry", readme)

    def test_readme_starts_with_local_model_endpoint_path(self) -> None:
        readme = Path("README.md").read_text()
        endpoint_index = readme.index("## Start A Local OpenAI Endpoint")
        snapshot_index = readme.index("## Project Snapshot")

        self.assertLess(endpoint_index, snapshot_index)
        self.assertIn("git clone https://github.com/ashhart/TensorFold.git", readme)
        self.assertIn("cd TensorFold", readme)
        self.assertIn("MODEL=/path/to/local/model", readme)
        self.assertIn("MODEL=/path/to/models/Qwen3.6-35B-A3B-MLX-4bit", readme)
        self.assertIn("tensorfold serve \"$MODEL\"", readme)
        self.assertIn("POST /v1/chat/completions", readme)
        self.assertIn("curl http://127.0.0.1:8421/v1/chat/completions", readme)
        self.assertIn("$HOME/.lmstudio/models", readme)
        self.assertIn("oMLX", readme)
        self.assertIn("Ollama", readme)
        self.assertIn("not a TensorFold model directory", readme)
        self.assertIn("`MODEL` is just a temporary shell variable", readme)
        self.assertIn("It is not a TensorFold config file", readme)
        self.assertIn("tensorfold serve /path/to/models/Qwen3.6-35B-A3B-MLX-4bit", readme)

    def test_quickstart_starts_with_clone_command(self) -> None:
        quickstart = Path("docs/quickstart.md").read_text()

        self.assertIn("git clone https://github.com/ashhart/TensorFold.git", quickstart)
        self.assertIn("cd TensorFold", quickstart)
        self.assertLess(quickstart.index("git clone"), quickstart.index("python3 -m venv"))

    def test_release_names_verified_internal_qwen_artifact_caveats(self) -> None:
        readme = Path("README.md").read_text()
        quickstart = Path("docs/quickstart.md").read_text()
        benchmarks = Path("docs/benchmarks.md").read_text()

        public_text = "\n".join((readme, quickstart))
        self.assertIn("## Verified Internal Qwen 35B Artifact", readme)
        self.assertIn("Qwen3.6-35B-A3B", readme)
        self.assertIn("24 GB Mac mini", readme)
        self.assertIn("22.122 tok/s", readme)
        self.assertIn("2.568 GB RSS", readme)
        self.assertIn("1.060 GB resident weight peak", readme)
        self.assertIn("56 generated tokens", readme)
        self.assertIn("generated-token timing window", readme)
        self.assertIn("not a cold-launch benchmark", readme)
        self.assertIn("not yet a public reproducibility claim", readme)
        self.assertIn("no replay tape", readme)
        self.assertIn("zero warmup tokens", public_text)
        self.assertIn("drop-cache-after-read", benchmarks)
        self.assertIn("internal artifact is not bundled", benchmarks)

    def test_readme_explains_platform_support(self) -> None:
        readme = Path("README.md").read_text()

        self.assertIn("## Platform Support", readme)
        self.assertIn("Apple Silicon / MLX-first", readme)
        self.assertIn("Linux", readme)
        self.assertIn("inspect", readme)
        self.assertIn("pack", readme)
        self.assertIn("not yet a proven TensorFold serving target", readme)

    def test_update_script_is_user_facing_and_safe_by_default(self) -> None:
        script = Path("update.sh")
        readme = Path("README.md").read_text()

        self.assertTrue(script.is_file())
        self.assertTrue(script.stat().st_mode & 0o111)
        text = script.read_text()
        self.assertIn("git fetch --prune", text)
        self.assertIn("git pull --ff-only", text)
        self.assertIn("TENSORFOLD_ALLOW_DIRTY", text)
        self.assertIn("./update.sh", readme)

    def test_mlx_adapter_is_split_into_focused_modules(self) -> None:
        adapter_file = Path("src/smarttensor/adapters/mlx.py")
        adapter_package = Path("src/smarttensor/adapters/mlx")

        self.assertFalse(adapter_file.exists())
        self.assertTrue(adapter_package.is_dir())

        expected_modules = {
            "__init__.py",
            "core.py",
            "drafting.py",
            "gpt_oss.py",
            "loader.py",
            "records.py",
            "runner_base.py",
            "qwen.py",
            "deepseek.py",
            "utils.py",
        }
        self.assertLessEqual(expected_modules, {path.name for path in adapter_package.glob("*.py")})

        for module in adapter_package.glob("*.py"):
            line_count = len(module.read_text().splitlines())
            self.assertLess(
                line_count,
                2_500,
                f"{module} is too large to review comfortably ({line_count} lines)",
            )


if __name__ == "__main__":
    unittest.main()
