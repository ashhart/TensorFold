"""Models from Hugging Face: a repo id (``Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP``) or a local directory.

A repo id resolves to its snapshot in the Hugging Face cache (``~/.cache/huggingface/hub`` unless ``HF_HOME`` or
``HF_HUB_CACHE`` say otherwise), downloading it first when it is not there yet.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

_REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


def is_repo_id(name: str) -> bool:
    """``owner/name`` that is not an existing local path."""

    return bool(_REPO_ID.match(str(name))) and not Path(str(name)).expanduser().exists()


def cached(repo_id: str, *, cache_dir: Any = None) -> Path | None:
    """The repo's snapshot in the local cache, or None when it has not been downloaded. A cache written without
    ``refs/main`` (some download tools skip it) falls back to its newest snapshot holding a config.json."""

    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(repo_id, local_files_only=True, cache_dir=cache_dir))
    except Exception:  # noqa: BLE001 - not cached, or cached without a ref: look at the snapshots themselves
        pass
    if cache_dir is None:
        from huggingface_hub import constants

        cache_dir = constants.HF_HUB_CACHE
    snapshots = Path(cache_dir) / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    found = [s for s in snapshots.glob("*") if (s / "config.json").is_file()] if snapshots.is_dir() else []
    return max(found, key=lambda s: s.stat().st_mtime) if found else None


def pull(repo_id: str, *, cache_dir: Any = None) -> Path:
    """Download (or finish downloading) a repo into the cache; returns its snapshot directory."""

    from huggingface_hub import snapshot_download

    print(f"[tensorfold] downloading {repo_id} from Hugging Face", flush=True)
    return Path(snapshot_download(repo_id, cache_dir=cache_dir))


def _cached_weights_complete(snapshot: Path) -> bool:
    """A config-only snapshot is enough for ``info``, but not for ``serve``.

    Hugging Face's local-only snapshot lookup also returns partial snapshots, so check the model weights before
    skipping the full download. The supported checkpoints use either a safetensors index, numbered model shards,
    or a single model.safetensors file.
    """

    index = snapshot / "model.safetensors.index.json"
    if index.is_file():
        try:
            weight_map = json.loads(index.read_text())["weight_map"]
        except (OSError, ValueError, KeyError, TypeError):
            return False
        if not isinstance(weight_map, dict) or not weight_map:
            return False
        try:
            files = set(weight_map.values())
        except TypeError:
            return False
        return all(isinstance(name, str) and not Path(name).is_absolute() and ".." not in Path(name).parts
                   and (snapshot / name).is_file() for name in files)

    shards = list(snapshot.glob("model-*-of-*.safetensors"))
    if shards:
        matches = [re.fullmatch(r"model-(\d+)-of-(\d+)\.safetensors", path.name) for path in shards]
        if not all(matches):
            return False
        totals = {int(match.group(2)) for match in matches}
        if len(totals) != 1:
            return False
        total = totals.pop()
        return len(shards) == total and {int(match.group(1)) for match in matches} == set(range(1, total + 1))

    return (snapshot / "model.safetensors").is_file()


def resolve(name: str, *, download: bool = True, cache_dir: Any = None) -> Path:
    """A model directory for ``name``: the directory itself, or a repo id's snapshot (downloaded if needed)."""

    path = Path(str(name)).expanduser()
    if path.is_dir():
        return path
    if not is_repo_id(str(name)):
        raise FileNotFoundError(f"{name} is neither a directory nor a Hugging Face repo id (owner/name)")
    found = cached(str(name), cache_dir=cache_dir)
    if found is not None and (found / "config.json").is_file() and (
        not download or _cached_weights_complete(found)
    ):
        return found
    if not download:
        raise FileNotFoundError(f"{name} is not in the Hugging Face cache; run: tensorfold pull {name}")
    return pull(str(name), cache_dir=cache_dir)


def size_of(directory: Path) -> int:
    """Bytes of the files under ``directory`` (following the cache's symlinks)."""

    return sum(p.stat().st_size for p in Path(directory).rglob("*") if p.is_file())
