"""How GLM-5.3-Flash's checkpoint splits between the two ranks, and an optional one-time split to disk.

The engine reads each rank's share straight from the checkpoint ``tensorfold pull`` downloaded (``RankReader``).
Per tensor a rule picks the split, and the rank's part is sliced as bytes (no value is converted):

  row   first-axis halves: output rows of column-parallel projections (heads, expert and MLP width), per-head
        vectors (A_log, dt_bias, the conv weights)
  col   last-axis halves of a 2-D tensor: input columns of row-parallel projections (o_proj, down_proj); packed
        words hold 8 inputs and groups hold 64, so every split lands on a group boundary
  rep   both ranks whole (norms, hyper-connections, router, indexer, MLA down-projections, embeddings, head)
  drop  the vision tower

A machine short on disk can write its rank's share once and serve that folder instead (85 GB against the
checkpoint's 182 GB); the engine reads either:

    python -m tensorfold.families.glm5_next.cuda.split MODEL_DIR --rank R OUT
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

ROW = (
    r"\.mlp\.experts\.\d+\.(gate|up)_proj\.",
    r"\.mlp\.shared_experts\.(gate|up)_proj\.",
    r"\.mlp\.(gate|up)_proj\.",
    r"\.self_attn\.(q|k|v)_proj\.",
    r"\.self_attn\.(q|k|v)_conv1d\.",
    r"\.self_attn\.(f_b|g_b|b)_proj\.",
    r"\.self_attn\.(A_log|dt_bias)$",
    r"\.self_attn\.(q_b|kv_b)_proj\.",
)
COL = (
    r"\.mlp\.experts\.\d+\.down_proj\.",
    r"\.mlp\.shared_experts\.down_proj\.",
    r"\.mlp\.down_proj\.",
    r"\.self_attn\.o_proj\.",
)
REP = (
    r"^lm_head\.", r"embed_tokens\.", r"^model\.language_model\.norm\.weight$",
    r"_layernorm\.weight$", r"\.hc_(attn|ffn)_(fn|base|scale)$", r"\.mlp\.gate\.(weight|e_score_correction_bias)$",
    r"\.self_attn\.indexer\.", r"\.self_attn\.(q_a_proj|kv_a_proj_with_mqa)\.", r"\.self_attn\.(f_a|g_a)_proj\.",
    r"\.self_attn\.o_norm\.weight$", r"\.(eh_proj)\.", r"\.(enorm|hnorm)\.weight$", r"\.shared_head\.norm\.weight$",
)
DTYPE_BYTES = {"U32": 4, "I32": 4, "F32": 4, "BF16": 2, "F16": 2, "U8": 1, "I8": 1, "I64": 8, "F64": 8}
# the files a rank folder needs besides its weights (the tokenizer, chat template and configs)
SMALL = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
         "processor_config.json", "model.safetensors.index.json")


def rule(name: str) -> str:
    if name.startswith("model.visual."):
        return "drop"
    hits = [kind for kind, pats in (("row", ROW), ("col", COL), ("rep", REP)) if any(re.search(p, name) for p in pats)]
    if len(hits) != 1:
        raise ValueError(f"{name}: split rule is ambiguous or missing ({hits})")
    return hits[0]


def read_header(path: str | Path) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header, 8 + n


def split_bytes(raw: np.ndarray, shape: list[int], itemsize: int, kind: str, rank: int) -> tuple[np.ndarray, list[int]]:
    """A tensor's bytes -> rank's part of them and its shape."""

    if kind == "rep":
        return raw, list(shape)
    if kind == "row":
        rows = shape[0]
        if rows % 2:
            raise ValueError(f"row split of odd leading dim {shape}")
        per = raw.size // rows
        half = rows // 2
        return raw[rank * half * per:(rank + 1) * half * per], [half] + list(shape[1:])
    if kind == "col":
        if len(shape) != 2 or shape[1] % 2:
            raise ValueError(f"column split needs an even 2-D shape, got {shape}")
        view = raw.reshape(shape[0], shape[1] * itemsize)
        half = shape[1] // 2
        part = np.ascontiguousarray(view[:, rank * half * itemsize:(rank + 1) * half * itemsize])
        return part.reshape(-1), [shape[0], half]
    raise ValueError(kind)


def rank_files(model_dir: str | Path, rank: int) -> list[Path]:
    return sorted(Path(model_dir).glob(f"*.rank{rank}.safetensors"))


class RankReader:
    """One rank's tensors by checkpoint name (CPU tensors in the stored dtype): sliced from the full checkpoint,
    or read from a folder ``split`` wrote for this rank."""

    def __init__(self, model_dir: str | Path, rank: int) -> None:
        self.dir, self.rank = Path(model_dir), rank
        self.handles: dict[str, object] = {}
        self.index: dict[str, object] = {}
        mine, other = rank_files(self.dir, rank), rank_files(self.dir, 1 - rank)
        if other and not mine:
            raise ValueError(f"{self.dir} holds rank {1 - rank}'s share: give rank {rank} its own folder or the "
                             "full checkpoint")
        self.split = bool(mine)
        if self.split:
            from safetensors import safe_open

            for path in mine:
                h = safe_open(str(path), framework="pt", device="cpu")
                self.handles[str(path)] = h
                for k in h.keys():
                    self.index[k] = h
            return
        index = self.dir / "model.safetensors.index.json"
        if index.exists():
            names = json.loads(index.read_text())["weight_map"]
        else:
            names = {k: p.name for p in sorted(self.dir.glob("*.safetensors")) for k in read_header(p)[0]
                     if k != "__metadata__"}
        self.files: dict[str, tuple[dict, int]] = {}
        self.maps: dict[str, np.memmap] = {}
        self.index = dict(names)

    def get(self, name: str):
        import torch

        if self.split:
            return self.index[name].get_tensor(name)
        file = str(self.dir / self.index[name])
        if file not in self.files:
            self.files[file] = read_header(file)
            self.maps[file] = np.memmap(file, dtype=np.uint8, mode="r")
        header, base = self.files[file]
        info = header[name]
        kind = rule(name)
        if kind == "drop":
            raise KeyError(f"{name} is not used by the engine")
        a, b = info["data_offsets"]
        itemsize = DTYPE_BYTES[info["dtype"]]
        data, shape = split_bytes(self.maps[file][base + a:base + b], info["shape"], itemsize, kind, self.rank)
        dtype = {"U32": torch.uint32, "I32": torch.int32, "F32": torch.float32, "BF16": torch.bfloat16,
                 "F16": torch.float16, "U8": torch.uint8, "I8": torch.int8, "I64": torch.int64,
                 "F64": torch.float64}[info["dtype"]]
        return torch.from_numpy(np.array(data, copy=True)).view(dtype).reshape(shape)


def write(path: str, tensors: list[tuple[str, str, list[int], np.ndarray]], metadata: dict | None) -> None:
    header: dict = {}
    offset = 0
    for name, dtype, shape, data in tensors:
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + data.size]}
        offset += data.size
    if metadata:
        header["__metadata__"] = metadata
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for _, _, _, data in tensors:
            f.write(memoryview(data))
    os.replace(tmp, path)


def split_file(src: str | Path, out: str | Path, rank: int) -> dict:
    """One checkpoint file -> OUT/<stem>.rank<R>.safetensors with the rank's part of every tensor it keeps."""

    header, base = read_header(src)
    metadata = header.pop("__metadata__", None)
    mm = np.memmap(src, dtype=np.uint8, mode="r")
    stem = os.path.basename(str(src)).replace(".safetensors", "")
    summary = {"rep": 0, "row": 0, "col": 0, "drop": 0}
    part = []
    for name in sorted(header, key=lambda k: header[k]["data_offsets"][0]):
        info = header[name]
        kind = rule(name)
        summary[kind] += 1
        if kind == "drop":
            continue
        a, b = info["data_offsets"]
        itemsize = DTYPE_BYTES[info["dtype"]]
        data, shape = split_bytes(mm[base + a:base + b], info["shape"], itemsize, kind, rank)
        if int(np.prod(shape)) * itemsize != data.size:
            raise ValueError(f"{name}: {shape} does not match {data.size} bytes")
        part.append((name, info["dtype"], shape, data))
    os.makedirs(out, exist_ok=True)
    if part:
        write(os.path.join(str(out), f"{stem}.rank{rank}.safetensors"), part, metadata)
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model_dir", type=Path, help="the checkpoint (e.g. the snapshot `tensorfold pull` downloaded)")
    p.add_argument("--rank", type=int, choices=(0, 1), required=True, help="the rank this machine serves")
    p.add_argument("out", type=Path, help="the folder to write (then: tensorfold serve OUT --tp 2 --rank R ...)")
    args = p.parse_args(argv)
    files = sorted(args.model_dir.glob("model-*.safetensors"))
    if not files:
        raise SystemExit(f"{args.model_dir}: no model-*.safetensors files")
    args.out.mkdir(parents=True, exist_ok=True)
    for name in SMALL:
        if (args.model_dir / name).exists():
            shutil.copyfile(args.model_dir / name, args.out / name)
    for src in files:
        stem = src.name.replace(".safetensors", "")
        if (args.out / f"{stem}.rank{args.rank}.safetensors").exists():
            continue
        print(src.name, split_file(src, args.out, args.rank), flush=True)
    print(f"rank {args.rank}'s share of {len(files)} files in {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
