"""GLM-5.3-Flash's CUDA image contracts without accelerator runtimes: headers, rows, transport and the tower build."""

from __future__ import annotations

import json
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from tensorfold.vision.glm_cuda import capacity_geometry, checkpoint_vision, tower_shapes, weight_transform

VISION = {"model_type": "glm5_next_vision", "hidden_size": 8, "out_hidden_size": 16, "depth": 1, "patch_size": 2,
          "temporal_patch_size": 2, "spatial_merge_size": 2, "in_channels": 3, "intermediate_size": 12,
          "num_heads": 2, "projection_intermediate_size": 20, "attention_bias": True}


def _checkpoint(path, layout=None):
    config = {"model_type": "glm5_next", "image_token_id": 99, "vision_config": VISION,
              "text_config": {"hidden_size": 16}}
    (path / "config.json").write_text(json.dumps(config))
    shapes = {**tower_shapes(VISION), **(layout or {})}
    offset, entries = 0, {}
    for name, shape in shapes.items():
        size = int(np.prod(shape)) * 2
        entries["model.visual." + name] = {"dtype": "BF16", "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    _write(path, entries, offset)
    return entries, offset


def _write(path, entries, size):
    raw = json.dumps(entries).encode()
    (path / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(size))


def test_the_tower_is_sized_from_headers_alone(tmp_path):
    _, size = _checkpoint(tmp_path)
    config, resident = checkpoint_vision(tmp_path)
    assert config["out_hidden_size"] == 16 and resident == size
    assert len(tower_shapes({**VISION, "depth": 24})) == 347       # GLM-5.3-Flash's tower, tensor for tensor


def test_an_mlx_conversion_keeps_its_channels_last_convolutions(tmp_path):
    _checkpoint(tmp_path, {"patch_embed.proj.weight": [8, 2, 2, 2, 3], "downsample.weight": [16, 2, 2, 8]})
    checkpoint_vision(tmp_path)


@pytest.mark.parametrize("damage", ["missing", "quantized", "range", "shape", "width"])
def test_incomplete_or_incompatible_towers_refuse_before_loading(tmp_path, damage):
    entries, size = _checkpoint(tmp_path)
    key = "model.visual.blocks.0.attn.qkv.weight"
    if damage == "missing":
        del entries[key]
    elif damage == "quantized":
        entries[key]["dtype"] = "U32"
    elif damage == "range":
        entries[key]["data_offsets"] = [size, size + 24 * 8 * 2]
    elif damage == "shape":
        entries[key]["shape"] = [12, 16]
    else:
        config = json.loads((tmp_path / "config.json").read_text())
        config["text_config"]["hidden_size"] = 32
        (tmp_path / "config.json").write_text(json.dumps(config))
    _write(tmp_path, entries, size)
    with pytest.raises(ValueError):
        checkpoint_vision(tmp_path)


def test_the_tower_and_its_workspace_are_counted_on_rank_zero_only(tmp_path):
    from tensorfold.cuda.capacity import Geometry

    _checkpoint(tmp_path)
    base = lambda text: Geometry(lambda slots: slots * 64, 8)
    zero = capacity_geometry(base, tmp_path, True, 0)({})
    one = capacity_geometry(base, tmp_path, True, 1)({})
    plain = capacity_geometry(base, tmp_path, False, 0)({})
    assert zero.needed(32) > one.needed(32) > plain.needed(32)
    dropped = lambda name, info: (0, 0)               # the language split rule drops model.visual.*
    info = {"shape": [8, 8], "dtype": "BF16"}
    assert weight_transform(dropped, True, 0)("model.visual.x", info) == (128, 0)
    assert weight_transform(dropped, True, 1)("model.visual.x", info) == (0, 0)
    assert weight_transform(dropped, True, 0)("model.language_model.x", info) == (0, 0)


def test_each_image_row_goes_into_every_hyper_connection_copy():
    torch = pytest.importorskip("torch")
    from tensorfold.vision.glm_cuda import replace_rows
    from tensorfold.vision.qwen_cuda import EncodedVision

    features = torch.arange(3 * 4, dtype=torch.float32).reshape(3, 4) + 100
    payload = EncodedVision((5, 6, 9), features, None, 0)
    x = torch.zeros((4, 2 * 4))                        # prompt rows 4..8, two copies of width 4
    replace_rows(x, payload, 4, 8, 2)
    want = torch.zeros((4, 2, 4))
    want[1], want[2] = features[0], features[1]        # row 9 belongs to the next chunk
    assert torch.equal(x.view(4, 2, 4), want)


def test_prefill_skips_chunks_that_hold_no_image_rows():
    from tensorfold.families.glm5_next.cuda.decode import _images
    from tensorfold.vision.qwen_cuda import EncodedVision

    payload = EncodedVision((10, 11, 30), None, None, 0)
    assert _images(None, 0, 16) is None
    assert _images(payload, 0, 10) is None and _images(payload, 31, 16) is None
    assert all(_images(payload, s, n) is not None for s, n in ((0, 11), (11, 1), (12, 16), (30, 4)))


def test_both_ranks_receive_the_same_features_and_find_the_rows_themselves():
    torch = pytest.importorskip("torch")
    from tensorfold.vision.glm_cuda import share_encoded
    from tensorfold.vision.qwen_cuda import EncodedVision

    sent = []

    def all_gather(send, recv):
        sent.append(send.clone()) if rank[0] == 0 else None
        recv[:send.numel()] = sent[-1]
        recv[send.numel():] = send

    rank = [0]
    comm = SimpleNamespace(all_gather=all_gather)
    prompt = [1, 99, 99, 2, 99]
    features = torch.randn(3, 4).to(torch.bfloat16)
    got0 = share_encoded(EncodedVision((1, 2, 4), features, None, 0), 0, comm, prompt, 99, 4, "cpu")
    rank[0] = 1
    got1 = share_encoded(None, 1, comm, prompt, 99, 4, "cpu")
    assert got0.rows == got1.rows == (1, 2, 4)
    assert torch.equal(got0.features, features) and torch.equal(got1.features, features)
    with pytest.raises(ValueError):                    # rows that are not the placeholders never reach the other rank
        share_encoded(EncodedVision((1, 2), features[:2], None, 0), 0, comm, prompt, 99, 4, "cpu")
    with pytest.raises(ValueError):
        share_encoded(None, 1, comm, [1, 2, 3], 99, 4, "cpu")


def test_a_continued_image_prompt_adds_text_only():
    from tensorfold.vision.glm_processing import GLMImageProcessor, PreparedGLMVisionPrompt

    processor = object.__new__(GLMImageProcessor)
    processor.image_token_id = 99
    prepared = PreparedGLMVisionPrompt((1, 99, 99, 2), np.zeros((8, 4)), np.array([[1, 2, 4]]), ((1, 3),), ("h",))
    assert processor.continued(prepared, [1, 99, 99, 2, 5, 6]).token_ids == (1, 99, 99, 2, 5, 6)
    with pytest.raises(ValueError):
        processor.continued(prepared, [1, 99, 99, 2, 99])
    with pytest.raises(ValueError):
        processor.continued(prepared, [1, 99, 2, 2, 5])


def test_a_meta_built_tower_matches_a_normally_built_one_in_the_installed_transformers():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextVisionConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextVisionModel

    config = Glm5NextVisionConfig(**{**VISION, "hidden_size": 32, "out_hidden_size": 32, "intermediate_size": 64,
                                     "projection_intermediate_size": 48, "patch_size": 4})
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    built = Glm5NextVisionModel(config).eval()
    with torch.device("meta"):
        meta = Glm5NextVisionModel(config)
    meta.load_state_dict(built.state_dict(), strict=True, assign=True)
    rotary = meta.rotary_pos_emb                       # as GLMCudaVision fills it
    inv_freq, _ = rotary.compute_axial_rope_parameters(config)
    for name in ("inv_freq", "original_inv_freq"):
        rotary._buffers[name] = inv_freq.clone()
    for name, buffer in built.rotary_pos_emb.named_buffers():
        assert torch.equal(dict(rotary.named_buffers())[name], buffer)
    pixels = torch.randn(32, 3 * 2 * 4 * 4)
    grid = torch.tensor([[1, 4, 4], [1, 2, 8]])
    with torch.inference_mode():
        want = built(pixels, grid_thw=grid, return_dict=True).pooler_output
        got = meta.eval()(pixels, grid_thw=grid, return_dict=True).pooler_output
    assert got.shape == (8, 32) and torch.equal(got, want)


@pytest.mark.parametrize("channels_last", [False, True])
def test_the_loader_reads_the_tower_from_checkpoint_bytes(tmp_path, monkeypatch, channels_last):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    from safetensors.torch import save_file
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextVisionConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextVisionModel

    from tensorfold.vision import glm_cuda, glm_processing

    vision = {**VISION, "hidden_size": 32, "out_hidden_size": 32, "intermediate_size": 64,
              "projection_intermediate_size": 48, "patch_size": 4}
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "glm5_next", "image_token_id": 99,
                                                      "vision_config": vision, "text_config": {"hidden_size": 32}}))
    torch.manual_seed(1)
    built = Glm5NextVisionModel(Glm5NextVisionConfig(**vision)).eval()
    frequencies = {k: v.clone() for k, v in built.rotary_pos_emb.named_buffers()}
    built.to(torch.bfloat16)
    built.rotary_pos_emb._buffers.update(frequencies)  # float32, as a checkpoint load keeps them (not saved)
    weights = {"model.visual." + k: v.contiguous() for k, v in built.state_dict().items()}
    if channels_last:                                  # as an MLX conversion stores its convolutions
        for name in ("patch_embed.proj.weight", "downsample.weight"):
            weights["model.visual." + name] = weights["model.visual." + name].movedim(1, -1).contiguous()
    weights["model.language_model.embed_tokens.weight"] = torch.zeros(4, 32, dtype=torch.bfloat16)
    save_file(weights, str(tmp_path / "model.safetensors"))
    monkeypatch.setattr(glm_processing.GLMImageProcessor, "from_directory",
                        classmethod(lambda cls, path: SimpleNamespace(image_token_id=99)))
    loaded = glm_cuda.GLMCudaVision(tmp_path, torch.device("cpu"))
    assert loaded.image_token == 99 and loaded.weight_bytes == sum(v.numel() * 2 for k, v in weights.items()
                                                                     if k.startswith("model.visual."))
    pixels = torch.randn(32, 3 * 2 * 4 * 4).to(torch.bfloat16)
    grid = torch.tensor([[1, 4, 4], [1, 2, 8]])
    with torch.inference_mode():
        want = built(pixels, grid_thw=grid, return_dict=True).pooler_output
        got = loaded.tower(pixels, grid_thw=grid, return_dict=True).pooler_output
    assert torch.equal(got, want)


def test_the_thinking_off_template_passes_the_image_switch_on():
    from tensorfold.families.glm5_next.cuda.app import ThinkingOffTemplate

    seen = []
    inner = SimpleNamespace(render=lambda messages, **kw: seen.append(kw) or "<|assistant|><think></think>")
    ThinkingOffTemplate(inner).render([], tools=None, enable_thinking=True, allow_images=True)
    assert seen[0]["allow_images"] is True


def test_offloaded_towers_count_a_visit_not_residence(tmp_path):
    from tensorfold.cuda.capacity import Geometry
    from tensorfold.vision.glm_cuda import OFFLOAD_ACTIVATION_BYTES
    from tensorfold.vision.qwen_cuda import WORKSPACE_BYTES

    _, tower = _checkpoint(tmp_path)
    base = lambda text: Geometry(lambda slots: slots * 64, 8)
    resident = capacity_geometry(base, tmp_path, True, 0)({}).needed(32)
    offloaded = capacity_geometry(base, tmp_path, True, 0, offload=True)({}).needed(32)
    plain = capacity_geometry(base, tmp_path, False, 0)({}).needed(32)
    assert offloaded - plain == tower + OFFLOAD_ACTIVATION_BYTES and offloaded < resident + tower
    assert resident - plain == WORKSPACE_BYTES
    assert capacity_geometry(base, tmp_path, True, 1, offload=True)({}).needed(32) == \
        capacity_geometry(base, tmp_path, True, 1)({}).needed(32)
    info = {"shape": [8, 8], "dtype": "BF16"}
    dropped = lambda name, info: (0, 0)
    assert weight_transform(dropped, True, 0, True)("model.visual.x", info) == (0, 0)     # resting in host RAM
    assert weight_transform(dropped, True, 0, False)("model.visual.x", info) == (128, 0)


def _fake_tower(calls, hidden=16):
    torch = pytest.importorskip("torch")

    class Tower:
        patch_embed = SimpleNamespace(proj=SimpleNamespace(weight=torch.zeros(1, dtype=torch.bfloat16)))

        def to(self, where):
            calls.append(("to", where))
            return self

        def __call__(self, pixels, grid_thw, return_dict):
            calls.append(("tower", tuple(pixels.shape), grid_thw.tolist()))
            return SimpleNamespace(pooler_output=torch.zeros(pixels.shape[0] // 4, hidden))
    return Tower()


def _vision(calls, offload=False):
    from tensorfold.vision.glm_cuda import GLMCudaVision
    import threading

    torch = pytest.importorskip("torch")
    vision = object.__new__(GLMCudaVision)
    vision.config, vision.image_token, vision.device = VISION, 99, torch.device("cpu")
    vision.offload, vision._lock, vision.tower = offload, threading.Lock(), _fake_tower(calls)
    return vision


def _prepared(sides):
    from tensorfold.vision.glm_processing import PreparedGLMVisionPrompt

    grid = np.array([[1, s, s] for s in sides], dtype=np.int64)
    counts = [s * s // 4 for s in sides]
    prompt, spans = [1], []
    for n in counts:
        spans.append((len(prompt), len(prompt) + n))
        prompt += [99] * n + [2]
    pixels = np.zeros((int(sum(s * s for s in sides)), 24), dtype=np.float32)
    return PreparedGLMVisionPrompt(tuple(prompt), pixels, grid, tuple(spans), tuple("h" * len(sides)))


def test_images_that_share_a_larger_budget_are_encoded_in_runs_of_whole_images():
    pytest.importorskip("torch")
    calls = []
    prepared = _prepared([128, 128, 64])              # 16,384 + 16,384 + 4,096 patches: 3 images, 9,216 rows
    encoded = _vision(calls).encode(prepared, prepared.token_ids)
    towers = [c for c in calls if c[0] == "tower"]
    assert [c[1][0] for c in towers] == [16384, 16384, 4096] and encoded.features.shape == (9216, 16)
    assert encoded.rows == tuple(i for a, b in prepared.image_spans for i in range(a, b))
    small = _prepared([32, 32])                       # a request inside one tower call is still one call
    calls.clear()
    _vision(calls).encode(small, small.token_ids)
    assert [c[1][0] for c in calls if c[0] == "tower"] == [2048]
    huge = _prepared([130])                           # one image over the tower call's patches is refused
    with pytest.raises(ValueError, match="CUDA vision budget"):
        _vision([]).encode(huge, huge.token_ids)


def test_an_offloaded_tower_is_on_the_gpu_only_while_it_encodes():
    pytest.importorskip("torch")
    calls = []
    prepared = _prepared([32])
    _vision(calls, offload=True).encode(prepared, prepared.token_ids)
    assert [c[0] for c in calls] == ["to", "tower", "to"] and calls[0][1] == _vision([]).device and calls[-1][1] == "cpu"
    calls.clear()
    _vision(calls).encode(prepared, prepared.token_ids)
    assert [c[0] for c in calls] == ["tower"]


class _Pil:
    detail, content_hash = "auto", "h"

    def to_pil(self):
        return self


def _processor(seen):
    from tensorfold.vision.glm_processing import GLMImageProcessor

    def image_processor(images, return_tensors, min_image_tokens, max_image_tokens):
        seen.append(max_image_tokens)
        side = int(max_image_tokens ** 0.5) * 2        # the biggest square grid the cap allows
        return {"pixel_values": np.zeros((side * side, 24)), "image_grid_thw": np.array([[1, side, side]])}

    processor = object.__new__(GLMImageProcessor)
    processor.config, processor.image_marker, processor.image_token_id = {"vision_config": VISION}, "I", 99
    processor.processor = SimpleNamespace(image_processor=image_processor)
    processor.tokenizer = lambda text, **kw: {"input_ids": [99 if c == "I" else 1 for c in text]}
    return processor


def test_the_visual_token_budget_is_shared_and_capped_per_image():
    from tensorfold.vision.glm_cuda import GLMCudaVision

    seen = []
    processor = _processor(seen)
    processor.prepare("aIbIc", [_Pil(), _Pil()], max_visual_tokens=2048)
    assert seen == [1024, 1024]                        # an even share
    seen.clear()
    processor.prepare("aIb", [_Pil()], max_visual_tokens=65536, max_image_tokens=4096)
    assert seen == [4096]                              # one image never gets more than the tower takes at once
    vision = object.__new__(GLMCudaVision)             # the CUDA tower's prepare sets that cap itself
    vision.frontend = SimpleNamespace(prepare=lambda *a, **kw: kw)
    assert vision.prepare("x", [], max_visual_tokens=16384)["max_image_tokens"] == 4096
    assert vision.prepare("x", [], max_image_tokens=100)["max_image_tokens"] == 100


def test_video_parts_are_refused_for_glm_with_a_clear_message():
    from tensorfold.server.errors import RequestError
    from tensorfold.server.prompts import prepare_images

    frontend = SimpleNamespace(allow_urls=False)       # GLMCudaVision has no `videos`: image input only
    messages = [{"role": "user", "content": [{"type": "text", "text": "what is this"},
                                             {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}}]}]
    with pytest.raises(RequestError, match="video are unsupported"):
        prepare_images(frontend, messages, lambda template: "")
