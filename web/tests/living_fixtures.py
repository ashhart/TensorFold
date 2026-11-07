import json

import pytest

from tensorfold_web import settings as settings_mod


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("TENSORFOLD_WEB_LIVING", str(tmp_path / "living"))
    monkeypatch.setattr(settings_mod, "config_dir", lambda: tmp_path / "conf")
    return tmp_path

def _snapshot(tmp_path, family="nemotron_h"):
    """A Hugging Face snapshot as the cache keeps it: files that are links to blobs."""

    blobs, snap = tmp_path / "blobs", tmp_path / "snap"
    blobs.mkdir()
    snap.mkdir()
    for name, body in (("config.json", json.dumps({"model_type": family})), ("model.safetensors", "weights")):
        (blobs / name).write_text(body)
        (snap / name).symlink_to(blobs / name)
    return snap
