"""app/lora_layout.py: H3 LoRA layouts, the rewrite for ComfyUI, and an adaln pair moved onto a pruned checkpoint."""

import json
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import lora_layout as L  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"


def save(path: Path, tensors: dict, metadata: dict | None = None) -> None:
    head, blobs, at = {}, [], 0
    for name, array in tensors.items():
        raw = np.ascontiguousarray(array).tobytes()
        kind = {"float32": "F32", "float16": "F16"}[str(array.dtype)]
        head[name] = {"dtype": kind, "shape": list(array.shape), "data_offsets": [at, at + len(raw)]}
        blobs.append(raw)
        at += len(raw)
    if metadata:
        head["__metadata__"] = metadata
    blob = json.dumps(head).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"".join(blobs))


def load(path: Path) -> tuple[dict, dict]:
    tensors, metadata, start = L.header(path)
    with open(path, "rb") as handle:
        return {name: L._tensor(handle, start, entry) for name, entry in tensors.items()}, metadata


def test_the_real_lineart_file_is_peft_and_every_key_lands_on_the_checkpoint():
    """DiffSynth-Studio's LineartAnime LoRA key list against the linear layers of ComfyUI's pruned Ref2VA checkpoint."""

    keys = json.loads((DATA / "lineart_anime_keys.json").read_text())
    linears = json.loads((DATA / "ref2va_pruned_linear_shapes.json").read_text())
    assert L.layout(list(keys)) == "peft"
    renamed = L.comfy_names(list(keys))
    assert len(renamed) == 518 and all(new.startswith("diffusion_model.") for new in renamed.values())
    assert not any(".default." in new for new in renamed.values())
    adaln = 0
    for old, new in renamed.items():
        if "lora_A" not in old:
            continue
        module = new[len("diffusion_model."): -len(".lora_A.weight")]
        assert module + ".weight" in linears, module
        rows, columns = keys[old.replace("lora_A", "lora_B")][1][0], keys[old][1][1]
        if L.ADALN.search(module):
            adaln += 1  # full width in the LoRA, the 8-number time code in the pruned checkpoint: curve_basis's job
            assert columns == 2688 and linears[module + ".weight"] == [rows, 8]
        else:
            assert linears[module + ".weight"] == [rows, columns], module
    assert adaln == 51 and len(L.modules(list(keys))) == 259


def test_layouts_are_told_apart():
    assert L.layout(["diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight", "diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight"]) == "comfy"
    assert L.layout(["diffusion_model.blocks.0.mlp.fc1.lora_down.weight", "diffusion_model.blocks.0.mlp.fc1.lora_up.weight", "diffusion_model.blocks.0.mlp.fc1.alpha"]) == "comfy"
    assert L.layout(["blocks.0.attn.qkv_proj.lora_A.weight", "blocks.0.attn.qkv_proj.lora_B.weight"]) == "runner"
    assert L.layout(["blocks.0.attn.qkv_proj.lora_A.default.weight", "blocks.0.attn.qkv_proj.lora_B.default.weight"]) == "peft"
    assert L.layout(["transformer.h.0.weight"]) == "unknown"


def test_a_peft_file_becomes_comfy_with_the_same_bytes():
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as scratch:
        source, target = Path(scratch) / "peft.safetensors", Path(scratch) / "comfy.safetensors"
        tensors = {"blocks.3.attn.qkv_proj.lora_A.default.weight": rng.standard_normal((4, 16), dtype=np.float32),
                   "blocks.3.attn.qkv_proj.lora_B.default.weight": rng.standard_normal((48, 4), dtype=np.float32),
                   "token_refiner.blocks.1.mlp.fc2.lora_A.default.weight": rng.standard_normal((4, 16), dtype=np.float32).astype(np.float16),
                   "token_refiner.blocks.1.mlp.fc2.lora_B.default.weight": rng.standard_normal((16, 4), dtype=np.float32).astype(np.float16)}
        save(source, tensors)
        done = L.to_comfy(source, target)
        assert done == {"modules": 2, "adaln": 0, "layout": "peft"}
        got, metadata = load(target)
        assert sorted(got) == ["diffusion_model.blocks.3.attn.qkv_proj.lora_A.weight", "diffusion_model.blocks.3.attn.qkv_proj.lora_B.weight",
                               "diffusion_model.token_refiner.blocks.1.mlp.fc2.lora_A.weight", "diffusion_model.token_refiner.blocks.1.mlp.fc2.lora_B.weight"]
        for old, array in tensors.items():
            assert np.array_equal(got[L.comfy_names(list(tensors))[old]], array.astype(np.float32))
        assert metadata["source_layout"] == "peft" and L.layout(list(got)) == "comfy"
        assert L.header(target)[0]["diffusion_model.token_refiner.blocks.1.mlp.fc2.lora_A.weight"]["dtype"] == "F16"


def test_what_is_not_a_lora_of_the_transformer_is_refused():
    for names in (["blocks.0.attn.qkv_proj.lora_A.default.weight", "blocks.0.attn.qkv_proj.lora_B.other.weight"],
                  ["text_encoder.layers.0.lora_A.default.weight", "text_encoder.layers.0.lora_B.default.weight"],
                  ["blocks.0.attn.qkv_proj.lora_A.default.weight", "blocks.0.attn.qkv_proj.weight"]):
        try:
            L.comfy_names(names)
        except ValueError:
            continue
        raise AssertionError(names)


def toy_embedder(rng, freq=16, hidden=24, out=12) -> dict:
    return {L.EMBEDDER[0]: rng.standard_normal((hidden, freq), dtype=np.float32), L.EMBEDDER[1]: rng.standard_normal(hidden, dtype=np.float32),
            L.EMBEDDER[2]: rng.standard_normal((out, hidden), dtype=np.float32) * 0.3, L.EMBEDDER[3]: rng.standard_normal(out, dtype=np.float32)}


def toy_curve(weights: dict, grid: int = 65) -> np.ndarray:
    """silu(time_embedder(t)) on the grid, as MiniMax's model computes it."""

    t = np.arange(grid, dtype=np.float64) / (grid - 1)
    half = weights[L.EMBEDDER[0]].shape[1] // 2
    angles = t[:, None] * np.exp(-np.log(10000.0) * np.arange(half) / half)[None]
    silu = lambda x: x / (1.0 + np.exp(-x))  # noqa: E731
    hidden = silu(np.concatenate([np.cos(angles), np.sin(angles)], -1) @ weights[L.EMBEDDER[0]].T.astype(np.float64) + weights[L.EMBEDDER[1]])
    return silu(hidden @ weights[L.EMBEDDER[2]].T.astype(np.float64) + weights[L.EMBEDDER[3]])


def test_an_adaln_pair_gives_the_same_output_on_the_pruned_checkpoint():
    """A full-weight adaln LoRA moved with curve_basis adds the same: B A z(t) = B A' c(t) + diff_b."""

    rng = np.random.default_rng(1)
    weights = toy_embedder(rng)
    curve = toy_curve(weights)                                # [grid, 12]
    mean = curve.mean(0)
    u, s, vt = np.linalg.svd(curve - mean, full_matrices=False)
    k = 8
    table = ((curve - mean) @ vt[:k].T).astype(np.float32)    # what a pruned checkpoint stores: k numbers a timestep
    basis, offset, error = L.curve_basis(table, weights)
    lost = float(np.linalg.norm((curve - mean) - table @ vt[:k]) / np.linalg.norm(curve))
    assert error <= lost + 1e-6                               # as good as the table allows
    with tempfile.TemporaryDirectory() as scratch:
        source, target = Path(scratch) / "peft.safetensors", Path(scratch) / "comfy.safetensors"
        a, b = rng.standard_normal((3, 12), dtype=np.float32), rng.standard_normal((20, 3), dtype=np.float32)
        qa, qb = rng.standard_normal((3, 16), dtype=np.float32), rng.standard_normal((48, 3), dtype=np.float32)
        save(source, {"blocks.0.adaln_proj.linear.lora_A.default.weight": a, "blocks.0.adaln_proj.linear.lora_B.default.weight": b,
                      "blocks.0.attn.qkv_proj.lora_A.default.weight": qa, "blocks.0.attn.qkv_proj.lora_B.default.weight": qb})
        done = L.to_comfy(source, target, basis, offset)
        assert done == {"modules": 2, "adaln": 1, "layout": "peft"}
        got, metadata = load(target)
        assert metadata["adaln"] == "curve"
        moved, bias = got["diffusion_model.blocks.0.adaln_proj.linear.lora_A.weight"], got["diffusion_model.blocks.0.adaln_proj.linear.diff_b"]
        assert moved.shape == (3, k) and bias.shape == (20,)
        assert np.array_equal(got["diffusion_model.blocks.0.adaln_proj.linear.lora_B.weight"], b)
        assert np.array_equal(got["diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight"], qa)  # other pairs untouched
        full = curve @ a.T.astype(np.float64) @ b.T.astype(np.float64)               # on MiniMax's weights
        pruned = table.astype(np.float64) @ moved.T @ b.T.astype(np.float64) + bias  # on the pruned ones
        assert np.linalg.norm(full - pruned) / np.linalg.norm(full) < max(5 * lost, 1e-4)
        # without a basis the pair is written as published
        plain = L.to_comfy(source, target)
        assert plain["adaln"] == 0 and load(target)[0]["diffusion_model.blocks.0.adaln_proj.linear.lora_A.weight"].shape == (3, 12)


def test_a_comfy_file_with_alpha_keeps_its_scale_in_the_bias():
    rng = np.random.default_rng(2)
    weights = toy_embedder(rng)
    curve = toy_curve(weights)
    mean = curve.mean(0)
    vt = np.linalg.svd(curve - mean, full_matrices=False)[2]
    table = ((curve - mean) @ vt[:8].T).astype(np.float32)
    basis, offset, _ = L.curve_basis(table, weights)
    with tempfile.TemporaryDirectory() as scratch:
        source, target = Path(scratch) / "in.safetensors", Path(scratch) / "out.safetensors"
        a, b = rng.standard_normal((4, 12), dtype=np.float32), rng.standard_normal((20, 4), dtype=np.float32)
        name = "diffusion_model.final_layer.adaln_proj.linear"
        save(source, {name + ".lora_A.weight": a, name + ".lora_B.weight": b, name + ".alpha": np.array(2.0, dtype=np.float32)})
        L.to_comfy(source, target, basis, offset)
        got, _ = load(target)
        scale = 2.0 / 4
        full = scale * (curve @ a.T.astype(np.float64) @ b.T.astype(np.float64))
        pruned = scale * (table.astype(np.float64) @ got[name + ".lora_A.weight"].T @ b.T.astype(np.float64)) + got[name + ".diff_b"]
        assert np.linalg.norm(full - pruned) / np.linalg.norm(full) < 1e-3
        assert float(got[name + ".alpha"]) == 2.0




def test_a_kohya_file_is_named_by_the_checkpoint_and_rewritten_with_its_alpha():
    """kohya's and musubi-tuner's saves (SOLRICKS' 3D Animation Style): lora_unet_ names with _ for every dot."""

    rng = np.random.default_rng(3)
    tensors = {}
    for flat in ("blocks_0_attn_qkv_proj", "blocks_12_attn_out_proj", "blocks_49_mlp_fc1", "blocks_49_mlp_fc2"):
        tensors[f"lora_unet_{flat}.lora_down.weight"] = rng.standard_normal((4, 8)).astype(np.float32)
        tensors[f"lora_unet_{flat}.lora_up.weight"] = rng.standard_normal((6, 4)).astype(np.float32)
        tensors[f"lora_unet_{flat}.alpha"] = np.asarray(4.0, dtype=np.float32)
    assert L.layout(list(tensors)) == "kohya"
    assert L.modules(list(tensors)) == ["blocks.0.attn.qkv_proj", "blocks.12.attn.out_proj", "blocks.49.mlp.fc1", "blocks.49.mlp.fc2"]
    assert all(L.MODULE.match(m) for m in L.modules(list(tensors)))
    assert L.kohya_module("token_refiner_blocks_1_attn_qkv_proj") == "token_refiner.blocks.1.attn.qkv_proj"
    assert L.kohya_module("blocks_3_adaln_proj_linear") == "blocks.3.adaln_proj.linear"
    assert L.kohya_module("double_blocks_0_img_attn_qkv") is None
    with tempfile.TemporaryDirectory() as scratch:
        source, target = Path(scratch) / "k.safetensors", Path(scratch) / "c.safetensors"
        save(source, tensors)
        assert L.to_comfy(source, target) == {"modules": 4, "adaln": 0, "layout": "kohya"}
        made, _ = load(target)
        assert L.layout(list(made)) == "comfy"
        assert np.array_equal(made["diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight"], tensors["lora_unet_blocks_0_attn_qkv_proj.lora_down.weight"])
        assert np.array_equal(made["diffusion_model.blocks.49.mlp.fc2.lora_B.weight"], tensors["lora_unet_blocks_49_mlp_fc2.lora_up.weight"])
        assert float(made["diffusion_model.blocks.12.attn.out_proj.alpha"]) == 4.0


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
