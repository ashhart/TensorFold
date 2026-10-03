"""Is a model directory a GGUF checkpoint? Kept apart from ``gguf_load`` (which imports the weights and torch) so
engine admission and family dispatch can ask without loading the GGUF path on every platform."""

from __future__ import annotations

from pathlib import Path


def gguf_file(model_dir: Path) -> Path | None:
    """The one ``*.gguf`` in ``model_dir`` when it has no safetensors, else None."""

    files = sorted(Path(model_dir).glob("*.gguf"))
    if not files or any(Path(model_dir).glob("*.safetensors")):
        return None
    if len(files) > 1:
        raise ValueError(f"{model_dir} holds {len(files)} GGUF files; keep one (split files are not read)")
    return files[0]
