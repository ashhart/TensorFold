"""The LoRA hub's presets and its bundle path: python3 -m pytest tests/test_loras.py (no network, no GPU)."""
import hashlib
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
import loras  # noqa: E402

PRESETS = json.loads((ROOT / "app/lora-presets.json").read_text())
GROUPS = {key for key, _ in loras.GROUPS}


def fake(path: Path) -> str:
    """A tiny file with the layout of a MiniMax H3 LoRA."""
    head = json.dumps({"diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight": {"dtype": "F16", "shape": [1, 1], "data_offsets": [0, 2]},
                       "diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight": {"dtype": "F16", "shape": [1, 1], "data_offsets": [2, 4]}}).encode()
    data = struct.pack("<Q", len(head)) + head + b"\0\0\0\0"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def test_presets_are_well_formed():
    ids = [r["id"] for r in PRESETS]
    assert len(ids) == len(set(ids)) == 22
    for r in PRESETS:
        assert r["name"] and r["author"] and r["page"] and r["licence"], r["id"]
        assert set(r["category"]) <= GROUPS and r["category"], r["id"]
        assert r["status"] in ("ready", "gated", "unsupported", "action"), r["id"]  # action: used from a Library clip
        assert len(r["sha256"]) == 64 and r["url"].startswith("https://"), r["id"]
        assert r["needs"] in ("text", "first-last", "reference", "source-video", "workflow", "image", "action"), r["id"]
        assert r["status"] == "ready" or r["why"], r["id"]
        assert r["range"][0] <= r["strength"] <= r["range"][1], r["id"]
    for group in ("camera", "transform", "character", "motion", "style", "finish", "fun"):
        assert sum(group in r["category"] for r in PRESETS) >= 3, group
    assert [r["id"] for r in PRESETS if len(r["category"]) == 2] == ["person-remover", "natural-face-speech", "motion-continuity-repair"]


def test_install_reads_a_bundle_and_checks_it(tmp_path, monkeypatch):
    recipe = dict(next(r for r in PRESETS if r["id"] == "camera-motion"))
    name = recipe["url"].rsplit("/", 1)[-1]
    recipe["sha256"] = fake(tmp_path / "bundle/camera-motion" / name)
    recipe["url"] = "https://invalid.invalid/" + name  # a dead author link is harmless with a bundle
    monkeypatch.setenv("LORA_BUNDLE", str(tmp_path / "bundle"))
    target = tmp_path / "loras/camera-motion.safetensors"
    target.parent.mkdir()
    loras.download(recipe, target, {"done": 0, "of": 0})
    assert target.is_file() and not list(target.parent.glob("*.part"))
    recipe["sha256"] = "0" * 64
    other = tmp_path / "loras/other.safetensors"
    try:
        loras.download(recipe, other, {"done": 0, "of": 0})
        raise AssertionError("a file that does not match its sha256 was installed")
    except RuntimeError:
        assert not other.exists()


def test_strength_and_status_rules(tmp_path, monkeypatch):
    monkeypatch.delenv("LORA_BUNDLE", raising=False)
    monkeypatch.setenv("STUDIO_HOME", str(tmp_path))
    _, resolve, lead = loras.attach(tmp_path, ROOT, True)
    for lora_id in ("anime-slider", "camera-motion", "combat-base-v2"):
        fake(tmp_path / "loras" / f"{lora_id}.safetensors")
    got = resolve([{"id": "anime-slider", "strength": -3}, {"id": "camera-motion"}, {"id": "combat-base-v2", "style": "Intense"}])
    assert [g["strength"] for g in got] == [-3, 0.9, 1.0]
    assert lead("A fox runs.", got) == "camera motion, prfight2. A fox runs."
    for bad in ([{"id": "anime-slider", "strength": 0}], [{"id": "camera-motion", "strength": -1}], [{"id": "camera-motion", "strength": 2.5}], [{"id": "orbit-360"}]):
        try:
            resolve(bad)
            raise AssertionError(f"not refused: {bad}")
        except loras.HTTPException:
            pass
