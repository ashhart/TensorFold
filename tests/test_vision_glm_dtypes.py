"""GLM's CUDA vision tower keeps each tensor's stored dtype: the loader, the mixed-dtype forward and the byte count."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tensorfold.vision.glm_cuda import checkpoint_vision

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")

VISION = {"model_type": "glm5_next_vision", "hidden_size": 32, "out_hidden_size": 32, "depth": 2, "patch_size": 4,
          "temporal_patch_size": 2, "spatial_merge_size": 2, "in_channels": 3, "intermediate_size": 64,
          "num_heads": 2, "projection_intermediate_size": 48, "attention_bias": True}
F16_LINEAR = "merger.down_proj.weight"            # the last linear: its output keeps F16 precision
F32_BIAS = "blocks.1.attn.proj.bias"           # beside a BF16 weight: the bias-add cast site


def stored_dtype(name, mode):
    """spec: F32 norms, one F16 linear, BF16 elsewhere. bias: also an F32 bias beside a BF16 weight. wide: F16 in
    every linear (as a checkpoint converted from F16 would be), where the cast to bf16 costs the most."""

    if "norm" in name or name.endswith("layernorm.weight"):
        return torch.float32
    if name == F16_LINEAR or (mode == "wide" and name.startswith(("blocks", "merger")) and "norm" not in name):
        return torch.float16
    return torch.float32 if mode == "bias" and name == F32_BIAS else torch.bfloat16


def write_checkpoint(path, mode="spec"):
    """A tower with F32 norms, one F16 linear and BF16 elsewhere; values chosen so the dtypes differ in effect."""

    from safetensors.torch import save_file
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextVisionConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextVisionModel

    (path / "config.json").write_text(json.dumps({"model_type": "glm5_next", "image_token_id": 99,
                                                  "vision_config": VISION, "text_config": {"hidden_size": 32}}))
    torch.manual_seed(3)
    model = Glm5NextVisionModel(Glm5NextVisionConfig(**VISION))
    stored = {}
    for name, value in model.state_dict().items():
        if "norm" in name and name.endswith("weight"):
            value = 1 + 0.05 * torch.randn_like(value)           # not representable in bf16
        stored[name] = value.to(stored_dtype(name, mode)).contiguous()
    save_file({"model.visual." + k: v for k, v in stored.items()}, str(path / "model.safetensors"))
    return stored


def load(path, monkeypatch):
    from tensorfold.vision import glm_cuda, glm_processing

    monkeypatch.setattr(glm_processing.GLMImageProcessor, "from_directory",
                        classmethod(lambda cls, p: SimpleNamespace(image_token_id=99)))
    return glm_cuda.GLMCudaVision(path, torch.device("cpu"))


def features(tower, pixels, grid):
    with torch.inference_mode():
        return tower(pixels, grid_thw=grid, return_dict=True).pooler_output


def references(tower_path, stored):
    """The F32 truth (the stored values, widened) and the old path (every tensor cast to bf16)."""

    from tensorfold.vision.glm_cuda import build_tower, keep_stored_dtypes

    truth, old = build_tower(tower_path, "cpu"), build_tower(tower_path, "cpu")
    for model, dtype in ((truth, torch.float32), (old, torch.bfloat16)):
        for p in model.parameters():
            p.data = p.data.to(dtype)
        keep_stored_dtypes(model)
    return truth, old


@pytest.mark.parametrize("mode, margin", [("spec", 1.0), ("bias", 1.0), ("wide", 0.7)])
def test_the_loader_keeps_every_stored_dtype_and_the_forward_is_closer_to_f32_than_a_bf16_cast(tmp_path, monkeypatch,
                                                                                               mode, margin):
    stored = write_checkpoint(tmp_path, mode)
    loaded = load(tmp_path, monkeypatch)
    params = dict(loaded.tower.named_parameters())
    assert set(params) == set(stored)
    assert {n: p.dtype for n, p in params.items()} == {n: v.dtype for n, v in stored.items()}
    assert {p.dtype for p in params.values()} == {torch.float32, torch.float16, torch.bfloat16}
    assert params[F16_LINEAR].dtype == torch.float16 and params[F32_BIAS].dtype == stored[F32_BIAS].dtype
    assert all(torch.equal(params[n], v) for n, v in stored.items())      # same bits, not a recast
    assert loaded.weight_bytes == sum(v.numel() * v.element_size() for v in stored.values())

    truth, old = references(tmp_path, stored)
    torch.manual_seed(4)
    raw = torch.randn(256, 3 * 2 * 4 * 4)
    grid = torch.tensor([[1, 8, 8], [1, 4, 16], [1, 8, 8], [1, 16, 4]])
    want = features(truth, raw, grid)
    faithful = features(loaded.tower, raw.to(torch.bfloat16), grid)       # pixels take the patch weight's dtype
    cast = features(old, raw.to(torch.bfloat16), grid)
    assert faithful.shape == want.shape == (64, 32) and torch.isfinite(faithful.float()).all()
    err = lambda got: (got.float() - want).abs()
    # measured against the F32 truth (the stored values widened): spec 0.92x, bias 0.76x, wide 0.48x of the cast path's
    # error; the bf16 compute of the BF16 linears is shared by both paths, so the spec mix gains least
    assert err(faithful).mean() < margin * err(cast).mean()
    assert not torch.equal(faithful, cast)                                # the two paths really differ


def test_pixels_follow_the_patch_embedding_dtype(tmp_path, monkeypatch):
    write_checkpoint(tmp_path)
    loaded = load(tmp_path, monkeypatch)
    assert loaded.tower.patch_embed.proj.weight.dtype == torch.bfloat16
    assert loaded.tower.post_layernorm.weight.dtype == torch.float32      # applied as stored: F32 weight, F32 out
    grid = torch.tensor([[1, 4, 4]])
    pixels = torch.randn(16, 3 * 2 * 4 * 4)
    out = features(loaded.tower, pixels.to(torch.bfloat16), grid)
    assert out.dtype == torch.float16                                    # merger.down_proj's own dtype: no hidden bf16


def test_the_memory_estimate_counts_stored_bytes(tmp_path):
    from tensorfold.vision.glm_cuda import weight_transform

    stored = write_checkpoint(tmp_path)
    _, resident = checkpoint_vision(tmp_path)
    assert resident == sum(v.numel() * v.element_size() for v in stored.values())
    assert resident > sum(v.numel() * 2 for v in stored.values())          # the F32 norms count 4 bytes each
    dropped = lambda name, info: (0, 0)
    assert weight_transform(dropped, True, 0)("model.visual.x", {"shape": [8, 8], "dtype": "F32"}) == (256, 0)
    assert weight_transform(dropped, True, 0)("model.visual.x", {"shape": [8, 8], "dtype": "F16"}) == (128, 0)
