"""Discovery / pinning tests for the DeepSeek-V4-Flash T01 audit.

These are CPU-only by design: they import no torch and no mlx. They guard the
evidence manifest (docs/evidence/deepseek-v4-t01-evidence.json) against the
actual repos, so the recorded revisions, licenses, module paths and
unresolved-decision flags stay reproducible (R7).

They are read-only against the donor repos and this checkout: git rev-parse,
file reads and a grep for quant symbols only. No launcher is modified.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "evidence" / "deepseek-v4-t01-evidence.json"

DONOR_DS4_C = Path("/home/josh/code/ds4")
DONOR_DS4_SPARK = Path("/home/josh/Documents/Projects/ds4")


def git_head(path: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(path)).decode().strip()


def load_manifest() -> dict:
    with MANIFEST.open() as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def manifest():
    return load_manifest()


def test_manifest_exists_and_valid_json():
    assert MANIFEST.exists()
    with MANIFEST.open() as fh:
        data = json.load(fh)          # raises if not valid JSON
    assert data["schema_version"] == "1.0"
    assert data["manifest_id"].startswith("deepseek-v4-t01")


def test_tensorfold_local_head_pinned(manifest):
    src = manifest["sources"]["tensorfold_local"]
    assert git_head(ROOT) == src["revision"]
    assert src["license"] == "MIT"


def test_donor_ds4_c_engine_revision_pinned(manifest):
    src = manifest["sources"]["donor_ds4_c_engine"]
    assert git_head(DONOR_DS4_C) == src["revision"]
    assert src["license"].startswith("MIT")        # donor LICENSE: MIT (ggml authors)


def test_donor_ds4_on_spark_revision_pinned(manifest):
    src = manifest["sources"]["donor_ds4_on_spark"]
    assert git_head(DONOR_DS4_SPARK) == src["revision"]
    assert src["license"] == "MIT"


def test_donor_quant_coverage_symbols_present():
    # The manifest records that the C donor implements the 0731 mixed-quant set.
    quants_h = DONOR_DS4_C / "gguf-tools" / "quants.h"
    assert quants_h.exists()
    text = quants_h.read_text()
    for symbol in ("IQ2_XXS", "Q2_K", "Q8_0"):
        assert symbol in text, f"donor quants.h missing {symbol}"
    assert "F16" in text and "F32" in text


def test_donor_module_paths_recorded_exist(manifest):
    modules = manifest["sources"]["donor_ds4_c_engine"]["modules"]
    for entry in modules:
        spec = entry.split("#")[0].strip()
        # glob pattern (e.g. cuda/mmq/ds4_ggml_stubs.*): at least one real match.
        if "*" in spec:
            assert list(DONOR_DS4_C.glob(spec)), f"no donor file matches {spec}"
            continue
        # multi-file shorthand (e.g. "a / b"): check each basename exists somewhere.
        if "/" in spec and not spec.startswith("cuda/mmq") and "gguf-tools" not in spec:
            for part in spec.split("/"):
                part = part.strip()
                if part and part not in ("ds4.c", "ds4.h", "quants.h", "quants.c"):
                    assert (DONOR_DS4_C / part).exists() or list(DONOR_DS4_C.glob(f"**/{part}")), part
            continue
        # single basename (no slash): exists at root or under a subdir.
        if "/" not in spec:
            assert (DONOR_DS4_C / spec).exists() or list(DONOR_DS4_C.glob(f"**/{spec}")), spec
        else:
            assert (DONOR_DS4_C / spec).exists(), spec


def test_pr119_glm5_converter_absent_from_tree(manifest):
    # PR #119 (feni6:glm-8bit-q8_0) is open and unmerged upstream; its proposed
    # tools/glm5_q8_0_gguf_to_mlx.py must be absent from this checkout.
    assert not (ROOT / "tools" / "glm5_q8_0_gguf_to_mlx.py").exists()
    assert manifest["sources"]["pr119"]["head"].startswith("feni6:")


def test_unresolved_decision_flags_recorded(manifest):
    flags = manifest["unresolved_upstream_decisions"]
    assert isinstance(flags, list) and len(flags) >= 3
    for item in flags:
        assert item["status"].startswith("UNRESOLVED")
        assert item["id"] in {"U1", "U2", "U3", "U4"}
    # U1 must record the issue-#14 close commit (Mac shipped, CUDA pending).
    u1 = next(i for i in flags if i["id"] == "U1")
    close = manifest["sources"]["tensorfold_upstream_main"]["revision_at_issue14_close"]
    assert close in u1["evidence"]
