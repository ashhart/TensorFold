"""GLP control vectors in Zig against weightless's container reader: the same layers, numbers and digest."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_reference(reference: Path):
    """weightless's container module, read from the checkout; torch is stubbed when it is absent."""
    path = reference / "vllm-plugin" / "weightless_steer" / "container.py"
    if not path.is_file():
        raise SystemExit(f"{path} is not there: point --reference at a weightless checkout")
    try:
        import torch  # noqa: F401
    except ImportError:
        # The module imports torch at the top; the parse below needs numpy alone.
        sys.modules["torch"] = types.ModuleType("torch")
    spec = importlib.util.spec_from_file_location("weightless_container", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def direction_ids(tensors: dict) -> list[int]:
    """The layer ids the file's tensor names carry, ascending."""
    ids = []
    for name in tensors:
        if not name.startswith("direction."):
            continue
        ids.append(int(name.split(".")[1]))
    return sorted(ids)


def payload_digest(tensors: dict) -> str:
    """SHA-256 over the tensor payloads in the order the file holds them."""
    h = hashlib.sha256()
    for values in tensors.values():
        h.update(values.astype("<f4").tobytes())
    return h.hexdigest()


def compare(meta: dict, tensors: dict, ours: dict, blob: bytes, problems: list[str]) -> None:
    """The reference's view against ours: the ids, the metadata and every direction's bytes."""
    their_ids = direction_ids(tensors)
    if their_ids != ours["layers"]:
        problems.append(f"layer ids differ: reference {their_ids[:4]}..{their_ids[-1:]} against ours {ours['layers'][:4]}..{ours['layers'][-1:]}")
        return
    for key, field in (("glp.mode", "mode"), ("glp.hook_point", "hook_point"), ("glp.method", "method"),
                       ("glp.contrast", "contrast"), ("glp.base_model", "base_model"), ("glp.base_revision", "base_revision")):
        if key in meta and meta[key] != ours[field]:
            problems.append(f"{key} is {meta[key]!r} for the reference and {ours[field]!r} for us")
    if "glp.alpha_default" in meta and float(meta["glp.alpha_default"]) != float(ours["alpha"]):
        problems.append(f"glp.alpha_default is {meta['glp.alpha_default']} for the reference and {ours['alpha']} for us")
    if "glp.layer_ids_zero_based" in meta:
        declared = [int(part) for part in str(meta["glp.layer_ids_zero_based"]).split(",")]
        if declared != their_ids:
            problems.append(f"the file names layers {declared[:4]}..{declared[-1:]} and holds {their_ids[:4]}..{their_ids[-1:]}")

    step = len(tensors[next(iter(tensors))].astype("<f4").tobytes()) if tensors else 0
    for i, layer in enumerate(ours["layers"]):
        their_bytes = tensors[f"direction.{layer}"].astype("<f4").tobytes()
        our_bytes = blob[i * step:(i + 1) * step]
        if their_bytes != our_bytes:
            differing = sum(1 for a, b in zip(their_bytes, our_bytes) if a != b)
            problems.append(f"direction.{layer} differs in {differing} of {len(their_bytes)} bytes")
            break

    claimed = meta.get("glp.content_sha256")
    if claimed:
        if payload_digest(tensors) != claimed:
            problems.append("the reference's digest over the payloads does not match the file's claim")
        if ours["sha256"] != claimed:
            problems.append(f"our digest {ours['sha256'][:16]} does not match the file's claim {claimed[:16]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reference", required=True, help="a weightless checkout")
    ap.add_argument("--vector", required=True, action="append", help="a control vector file to compare")
    ap.add_argument("--exe", default=str(ROOT / "zig-out" / "bin" / "tf-glp-check"), help="tf-glp-check")
    ap.add_argument("--work", default="/tmp/tf-glp-parity", help="folder for the receipts")
    args = ap.parse_args()

    container = load_reference(Path(args.reference))
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []
    checked = 0

    for vector in args.vector:
        path = Path(vector)
        if not path.is_file():
            problems.append(f"{path} is not there")
            continue
        meta, tensors = container.read_gguf_cvec(str(path))

        receipt = work / "ours.json"
        blob_path = work / "ours.bin"
        read = subprocess.run([args.exe, "read", str(path), str(receipt), str(blob_path)], capture_output=True, text=True)
        if read.returncode != 0:
            # The reference took the file and this reader did not: a difference the pair has to name.
            problems.append(f"{path.name}: the reference reads it, this reader refuses it ({read.stdout.strip()})")
            print(f"{path.name}: the reference reads it, this reader refuses it")
            continue
        ours = json.loads(receipt.read_text())
        blob = blob_path.read_bytes()

        compare(meta, tensors, ours, blob, problems)

        # The file read in Zig and written out again, then read by the reference.
        rewritten = work / "rewritten.gguf"
        subprocess.run([args.exe, "rewrite", str(path), str(rewritten)], check=True)
        back_meta, back_tensors = container.read_gguf_cvec(str(rewritten))
        if direction_ids(back_tensors) != direction_ids(tensors):
            problems.append(f"{path.name}: the rewrite holds different layers")
        for layer in direction_ids(tensors):
            if back_tensors[f"direction.{layer}"].astype("<f4").tobytes() != tensors[f"direction.{layer}"].astype("<f4").tobytes():
                problems.append(f"{path.name}: direction.{layer} did not survive the rewrite")
                break
        if back_meta.get("glp.mode") != meta.get("glp.mode"):
            problems.append(f"{path.name}: the rewrite changed the mode to {back_meta.get('glp.mode')!r}")
        if back_meta.get("glp.hook_point") != meta.get("glp.hook_point"):
            problems.append(f"{path.name}: the rewrite changed the hook point")
        if abs(float(back_meta.get("glp.alpha_default", 0)) - float(meta.get("glp.alpha_default", 0))) > 1e-6:
            problems.append(f"{path.name}: the rewrite changed the strength")

        print(f"{path.name}: {len(tensors)} layers, {ours['width']} wide, mode {ours['mode']}, "
              f"alpha {ours['alpha']}, digest {ours['sha256'][:12]}: reference and Zig agree")
        checked += 1

    for problem in problems:
        print(f"differs: {problem}")
    print(f"glp parity: {checked - len(problems)}/{checked} files agree" if problems else f"glp parity: {checked}/{checked} files agree")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
