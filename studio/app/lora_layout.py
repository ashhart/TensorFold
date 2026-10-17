"""The key layouts MiniMax H3 LoRAs are published in (comfy, peft, kohya, runner), and the rewrite to comfy."""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path

# ComfyUI reads the comfy layout only: a peft file loaded as it is matches no key and changes nothing, silently
PEFT = re.compile(r"^(?P<module>.+)\.lora_(?P<side>[AB])\.(?P<adapter>[A-Za-z0-9_-]+)\.weight$")
PLAIN = re.compile(r"^(?P<module>.+)\.(?:lora_[AB]|lora_down|lora_up)\.weight$")
KOHYA = re.compile(r"^lora_unet_(?P<flat>[A-Za-z0-9_]+)\.(?P<part>lora_down\.weight|lora_up\.weight|alpha)$")
FLAT = re.compile(r"^(?P<refiner>token_refiner_)?blocks_(?P<n>\d+)_(?P<group>attn|mlp|adaln_proj)_(?P<leaf>qkv_proj|out_proj|fc[12]|linear)$")
# the linear layers of the H3 transformer a LoRA can sit on, under the checkpoint's names
MODULE = re.compile(r"^(?:token_refiner\.)?blocks\.\d+\.(?:attn\.(?:qkv_proj|out_proj)|mlp\.fc[12]|adaln_proj\.linear)$"
                    r"|^final_layer\.(?:adaln_proj\.linear|video_out|audio_out)$"
                    r"|^(?:video_patch_proj|audio_patch_proj|condition_proj|time_embedder\.proj_(?:in|out))$")


def header(path: Path) -> tuple[dict, dict, int]:
    """(tensors, metadata, where the payload starts) of a safetensors file."""

    with open(path, "rb") as handle:
        head = handle.read(8)
        size = struct.unpack("<Q", head)[0] if len(head) == 8 else 0
        if not 0 < size < 50_000_000:
            raise ValueError("not a safetensors file")
        tensors = json.loads(handle.read(size))
    return tensors, tensors.pop("__metadata__", None) or {}, 8 + size


def kohya_module(flat: str) -> str | None:
    """blocks_0_attn_qkv_proj -> blocks.0.attn.qkv_proj: the checkpoint's name of a kohya-named layer, or None."""

    hit = FLAT.match(flat)
    if not hit:
        return {"final_layer_adaln_proj_linear": "final_layer.adaln_proj.linear"}.get(flat)
    return f"{'token_refiner.' if hit['refiner'] else ''}blocks.{hit['n']}.{hit['group']}.{hit['leaf']}"


def layout(names) -> str:
    """comfy, peft, kohya, runner or unknown, from a file's tensor names."""

    names = [n for n in names if n != "__metadata__"]
    if any(KOHYA.match(n) for n in names):
        return "kohya"
    if any(PEFT.match(n) for n in names):
        return "peft"
    low = [n for n in names if PLAIN.match(n)]
    if not low:
        return "unknown"
    return "comfy" if all(n.startswith("diffusion_model.") for n in low) else "runner"


def modules(names) -> list[str]:
    """The adapted modules under checkpoint names, whatever the layout."""

    found = set()
    for name in names:
        flat = KOHYA.match(name)
        if flat:
            found.add(kohya_module(flat["flat"]) or "lora_unet_" + flat["flat"])
            continue
        hit = PEFT.match(name) or PLAIN.match(name)
        if hit:
            module = hit["module"]
            found.add(module[len("diffusion_model."):] if module.startswith("diffusion_model.") else module)
    return sorted(found)


def comfy_names(names) -> dict[str, str]:
    """{name in a peft file: name in the comfy layout}; a name that is not a LoRA matrix of the transformer raises."""

    adapters = {PEFT.match(n)["adapter"] for n in names if PEFT.match(n)}
    if len(adapters) != 1:
        raise ValueError(f"expected one PEFT adapter in the file, found {sorted(adapters) or 'none'}")
    renamed = {}
    for name in names:
        hit = PEFT.match(name)
        if not hit:
            raise ValueError(f"unsupported tensor {name}: a low-rank LoRA with lora_A / lora_B is read")
        module = hit["module"]
        module = module[len("diffusion_model."):] if module.startswith("diffusion_model.") else module
        if not MODULE.match(module):
            raise ValueError(f"{module} is not a layer of MiniMax H3's transformer")
        renamed[name] = f"diffusion_model.{module}.lora_{hit['side']}.weight"
    if len(set(renamed.values())) != len(renamed):
        raise ValueError("two tensors would take the same name")
    return renamed


def kohya_names(names) -> dict[str, str]:
    """{name in a kohya file: name in the comfy layout}, the .alpha of each pair kept beside it."""

    ends = {"lora_down.weight": "lora_A.weight", "lora_up.weight": "lora_B.weight", "alpha": "alpha"}
    renamed = {}
    for name in names:
        hit = KOHYA.match(name)
        module = kohya_module(hit["flat"]) if hit else None
        if module is None:
            raise ValueError(f"{name} is not a LoRA matrix on a layer of MiniMax H3's transformer")
        renamed[name] = f"diffusion_model.{module}.{ends[hit['part']]}"
    return renamed


ADALN = re.compile(r"(?:^|\.)adaln_proj\.linear$")
# MiniMax's own Ref2VA transformer: the time embedder sits in the first 64 MB of its first file
OFFICIAL = "https://huggingface.co/MiniMaxAI/MiniMax-H3/resolve/main/Ref2VA/transformer/model-00001-of-00013.safetensors"
EMBEDDER = ("time_embedder.proj_in.weight", "time_embedder.proj_in.bias", "time_embedder.proj_out.weight", "time_embedder.proj_out.bias")


def _tensor(handle, start: int, entry: dict):
    import numpy as np

    kinds = {"F32": np.float32, "F16": np.float16, "F64": np.float64, "BF16": np.uint16}
    begin, end = entry["data_offsets"]
    handle.seek(start + begin)
    flat = np.frombuffer(handle.read(end - begin), dtype=kinds[entry["dtype"]])
    if entry["dtype"] == "BF16":  # bfloat16 is the top half of a float32
        flat = (flat.astype(np.uint32) << 16).view(np.float32)
    return flat.astype(np.float32).reshape(entry["shape"])


def embedder(local: Path | None = None, url: str = OFFICIAL) -> dict:
    """The four tensors of H3's time embedder: from the local model file, else a ranged read of its first 64 MB."""

    import io
    import urllib.request

    if local is not None and Path(local).is_file():
        tensors, _, start = header(Path(local))
        with open(local, "rb") as handle:
            return {name: _tensor(handle, start, tensors[name]) for name in EMBEDDER}

    def part(first: int, last: int) -> bytes:
        request = urllib.request.Request(url, headers={"Range": f"bytes={first}-{last}", "User-Agent": "TensorFoldStudio"})
        with urllib.request.urlopen(request, timeout=300) as reply:
            return reply.read()

    size = struct.unpack("<Q", part(0, 7))[0]
    if not 0 < size < 50_000_000:
        raise ValueError("the model file's header is not a safetensors header")
    tensors = json.loads(part(8, 7 + size))
    first = min(tensors[name]["data_offsets"][0] for name in EMBEDDER)
    last = max(tensors[name]["data_offsets"][1] for name in EMBEDDER)
    blob = io.BytesIO(part(8 + size + first, 8 + size + last - 1))
    return {name: _tensor(blob, -first, tensors[name]) for name in EMBEDDER}


def curve_basis(table, weights: dict):
    """(basis, offset, fit error) with silu(time_embedder(t)) = table(t) @ basis + offset, by least squares."""

    import numpy as np

    table = np.asarray(table, dtype=np.float64)
    grid = np.arange(table.shape[0], dtype=np.float64) / (table.shape[0] - 1)
    half = weights[EMBEDDER[0]].shape[1] // 2
    angles = grid[:, None] * np.exp(-np.log(10000.0) * np.arange(half, dtype=np.float64) / half)[None]
    silu = lambda x: x / (1.0 + np.exp(-x))  # noqa: E731
    hidden = silu(np.concatenate([np.cos(angles), np.sin(angles)], axis=-1) @ weights[EMBEDDER[0]].astype(np.float64).T + weights[EMBEDDER[1]])
    wanted = silu(hidden @ weights[EMBEDDER[2]].astype(np.float64).T + weights[EMBEDDER[3]])
    design = np.concatenate([table, np.ones((table.shape[0], 1))], axis=1)
    solved = np.linalg.lstsq(design, wanted, rcond=None)[0]
    error = float(np.linalg.norm(design @ solved - wanted) / np.linalg.norm(wanted))
    return solved[:-1].astype(np.float32), solved[-1].astype(np.float32), error


def curve_table(checkpoint: Path):
    """`adaln_t_table` of a ComfyUI H3 checkpoint, or None when the file keeps the full time embedder."""

    tensors, _, start = header(Path(checkpoint))
    if "adaln_t_table" not in tensors:
        return None
    with open(checkpoint, "rb") as handle:
        return _tensor(handle, start, tensors["adaln_t_table"])


def to_comfy(source: Path, target: Path, basis=None, offset=None) -> dict:
    """Write `source` (peft, kohya or comfy) as `target` in comfy layout; adaln pairs moved by basis/offset if given."""

    tensors, metadata, start = header(source)
    kind = layout(list(tensors))
    if kind == "peft":
        renamed = comfy_names(list(tensors))
    elif kind == "comfy":
        renamed = {name: name for name in tensors}
    elif kind == "kohya":
        renamed = kohya_names(list(tensors))
    else:
        raise ValueError(f"a {kind} file is not rewritten for ComfyUI")
    carried = sorted(name[: -len(".lora_A.weight")] for name in renamed.values()
                     if name.endswith(".lora_A.weight") and ADALN.search(name[: -len(".lora_A.weight")])) if basis is not None else []
    back = {new: old for old, new in renamed.items()}
    out, blobs, at = {}, [], 0

    def put(name: str, dtype: str, shape: list, data: bytes) -> None:
        nonlocal at
        out[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [at, at + len(data)]}
        blobs.append(data)
        at += len(data)

    with open(source, "rb") as handle:
        for new in sorted(renamed.values()):
            entry = tensors[back[new]]
            module = new.rsplit(".lora_", 1)[0]
            if module in carried:
                if new.endswith(".lora_A.weight"):
                    import numpy as np

                    a = _tensor(handle, start, entry)
                    b = _tensor(handle, start, tensors[back[module + ".lora_B.weight"]])
                    alpha = tensors.get(back.get(module + ".alpha", ""), None)
                    scale = float(_tensor(handle, start, alpha).reshape(-1)[0]) / a.shape[0] if alpha else 1.0
                    moved = (a.astype(np.float64) @ basis.astype(np.float64).T).astype(np.float32)
                    bias = (scale * (b.astype(np.float64) @ (a.astype(np.float64) @ offset.astype(np.float64)))).astype(np.float32)
                    put(new, "F32", moved.shape, moved.tobytes())
                    put(module + ".diff_b", "F32", bias.shape, bias.tobytes())
                    continue
            begin, end = entry["data_offsets"]
            handle.seek(start + begin)
            put(new, entry["dtype"], entry["shape"], handle.read(end - begin))
    out["__metadata__"] = {**{k: str(v) for k, v in metadata.items()}, "converted_from": Path(source).name,
                           "source_layout": kind, "adaln": "curve" if carried else "as published", "format": "pt"}
    blob = json.dumps(out, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    part = Path(str(target) + ".part")
    with open(part, "wb") as writer:
        writer.write(struct.pack("<Q", len(blob)))
        writer.write(blob)
        for data in blobs:
            writer.write(data)
    part.replace(target)
    return {"modules": sum(1 for name in renamed.values() if name.endswith(".lora_A.weight")), "adaln": len(carried), "layout": kind}
