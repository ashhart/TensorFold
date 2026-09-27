"""Exercise native schema rejection through real indexes and sparse checkpoint files.

Only metadata is inspected by the native command. Payloads in mutated shards are
sparse holes; original checkpoints are linked read-only and never modified.
"""
import argparse
import copy
import json
import struct
import subprocess
from pathlib import Path

from export_native_schemas import RECIPES, header


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("--model-root", type=Path, default=Path("build/models"))
    parser.add_argument("--output", type=Path, default=Path("build/native-checks/schema-failures"))
    args = parser.parse_args()
    count = 0
    for kind, name, prefix in RECIPES:
        source = (args.model_root / name).resolve()
        specs = json.loads(Path(f"native/schemas/{kind}.json").read_text())
        key = prefix + specs[0]["name"]
        index_path = source / "model.safetensors.index.json"
        index = json.loads(index_path.read_text()) if index_path.exists() else None
        filename = index["weight_map"][key] if index else "model.safetensors"
        cases = ["missing-file", "truncated", "missing-tensor", "dtype", "rank", "shape"]
        if index:
            cases += ["missing-index-entry", "invalid-index-type", "unsafe-shard", "wrong-shard"]
        if kind in ("nemotron", "flash"):
            cases += ["missing-mtp"]
        for case in cases:
            directory = args.output / kind / case
            directory.mkdir(parents=True, exist_ok=True)
            # Refresh generated links from earlier runs without touching their targets.
            for path in directory.iterdir():
                if path.is_symlink() or path.suffix in (".json", ".safetensors"):
                    path.unlink()
            for path in source.glob("*.safetensors"):
                (directory / path.name).symlink_to(path)
            changed = copy.deepcopy(index)
            expected = "MissingWeight"
            target = directory / filename
            if case == "missing-file":
                target.unlink()
                expected = "FileNotFound"
            elif case == "missing-mtp" and kind == "nemotron":
                (directory / "mtp-4bit.safetensors").unlink()
                expected = "MissingDraftHead"
            elif case in ("missing-index-entry", "invalid-index-type", "unsafe-shard", "wrong-shard"):
                if case == "missing-index-entry":
                    del changed["weight_map"][key]
                elif case == "invalid-index-type":
                    changed["weight_map"][key] = 123
                    expected = "InvalidWeightIndex"
                elif case == "unsafe-shard":
                    changed["weight_map"][key] = "../outside.safetensors"
                    expected = "InvalidShardName"
                else:
                    changed["weight_map"][key] = next(n for n in set(index["weight_map"].values()) if n != filename)
            else:
                if case == "missing-mtp":
                    key_mtp = prefix + next(s["name"] for s in specs if s["name"].startswith("mtp."))
                    filename_mtp = index["weight_map"][key_mtp]
                    target = directory / filename_mtp
                    metadata = header(source / filename_mtp)
                    metadata["unused.removed_mtp"] = metadata.pop(key_mtp)
                    expected = "MissingDraftHead"
                else:
                    metadata = header(source / filename)
                    if case == "missing-tensor":
                        metadata["unused.removed_tensor"] = metadata.pop(key)
                    elif case == "dtype":
                        dtype = metadata[key]["dtype"]
                        metadata[key]["dtype"] = {"U32": "F32", "BF16": "F16", "F32": "U32", "I64": "F64"}[dtype]
                        expected = "InvalidTensorDType"
                    elif case == "rank":
                        metadata[key]["shape"].insert(0, 1)
                        expected = "InvalidTensorShape"
                    elif case == "shape":
                        shape = metadata[key]["shape"]
                        shape[0], shape[-1] = shape[-1], shape[0]
                        if shape == specs[0]["shape"]:
                            shape[:] = [1, *shape]
                        expected = "InvalidTensorShape"
                    elif case == "truncated":
                        expected = "InvalidTensorOffsets"
                target.unlink()
                encoded = json.dumps(metadata, separators=(",", ":")).encode()
                payload_size = max(v["data_offsets"][1] for k, v in metadata.items() if k != "__metadata__")
                with target.open("wb") as stream:
                    stream.write(struct.pack("<Q", len(encoded)))
                    stream.write(encoded)
                    stream.truncate(8 + len(encoded) + payload_size - (case == "truncated"))
            if changed is not None:
                (directory / "model.safetensors.index.json").write_text(json.dumps(changed))
            result = subprocess.run([str(args.executable.resolve()), "check-model-schema", kind, str(directory)],
                                    text=True, capture_output=True)
            if result.returncode != 1 or f"error: {expected}" not in result.stderr:
                raise AssertionError(f"{kind}/{case}: expected {expected}, got {result.returncode}\n{result.stderr}")
            count += 1
            print(f"PASS: {kind}/{case}: {expected}", flush=True)
    print(f"PASS: {count} checkpoint metadata failure cases")


if __name__ == "__main__":
    main()
