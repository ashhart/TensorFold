"""The vision frontend loads selected local tensors and prepares image prompts without a GPU runtime."""

from __future__ import annotations

import json
from pathlib import Path
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from tensorfold.vision import qwen_mlx, qwen_processing
from tensorfold.vision.qwen_checkpoint import load_vision_weights, quantization_predicate, vision_tensors
from tensorfold.vision.qwen_processing import QwenImageProcessor, _processor_options, image_positions


CONFIG = {"model_type": "qwen3_5", "image_token_id": 10, "video_token_id": 11,
          "vision_start_token_id": 8, "vision_end_token_id": 9,
          "vision_config": {"spatial_merge_size": 2, "patch_size": 16, "temporal_patch_size": 2,
                            "out_hidden_size": 3, "deepstack_visual_indexes": [], "hidden_size": 4,
                            "intermediate_size": 8, "depth": 2},
          "quantization": {"bits": 4, "group_size": 64, "model.visual.blocks.0": {"bits": 8, "group_size": 32}}}


class Tokenizer:
    def convert_ids_to_tokens(self, token):
        return {8: "<start>", 9: "<end>", 10: "<image>"}.get(token)

    def convert_tokens_to_ids(self, token):
        return {"<start>": 8, "<end>": 9, "<image>": 10}.get(token)

    def __call__(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": False, "return_attention_mask": False}
        ids = []
        while text:
            found = next((t for t in ("<start>", "<end>", "<image>") if text.startswith(t)), None)
            if found:
                ids.append(self.convert_tokens_to_ids(found))
                text = text[len(found):]
            else:
                ids.append(ord(text[0]))
                text = text[1:]
        return {"input_ids": ids}


class ImageProcessor:
    max_pixels, min_pixels = 1024**2, 32**2

    def __init__(self, **kwargs):
        self.options, self.calls = kwargs, []

    def __call__(self, *, images, **kwargs):
        self.calls.append((images, kwargs))
        return {"pixel_values": np.zeros((16, 1536), np.float32), "image_grid_thw": np.array([[1, 4, 4]])}


def image(name="one", detail="auto"):
    return SimpleNamespace(content_hash=name, detail=detail, to_pil=lambda: "PIL:" + name)


def write_shard(path: Path, tensors: dict):
    header, raw = {}, b""
    for name, (dtype, values) in tensors.items():
        array = np.asarray(values)
        data = array.tobytes()
        header[name] = {"dtype": dtype, "shape": list(array.shape), "data_offsets": [len(raw), len(raw) + len(data)]}
        raw += data
    metadata = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(metadata)) + metadata + raw)


def test_image_positions_match_spatial_grid_and_shift_following_text():
    ids = [100, 8, 10, 10, 10, 10, 9, 101]
    pos, delta, spans = image_positions(ids, [[1, 4, 4]], CONFIG)
    assert spans == ((2, 6),) and delta == -2
    np.testing.assert_array_equal(pos[:, 0], [[0, 1, 2, 2, 2, 2, 4, 5],
                                            [0, 1, 2, 2, 3, 3, 4, 5],
                                            [0, 1, 2, 3, 2, 3, 4, 5]])
    assert len(ids) + delta == 6


def test_multiple_images_keep_independent_grids_and_one_continuation_delta():
    ids = [8, 10, 10, 10, 10, 9, 100, 8, 10, 10, 9]
    pos, delta, spans = image_positions(ids, [[1, 4, 4], [1, 2, 4]], CONFIG)
    assert spans == ((1, 5), (8, 10)) and delta == -2
    np.testing.assert_array_equal(pos[:, 0, 8:10], [[6, 6], [6, 6], [6, 7]])
    assert pos[:, 0, -1].tolist() == [8, 8, 8]


@pytest.mark.parametrize("tokens,grids", [([8, 10, 9], [[1, 4, 4]]), ([10, 9], [[1, 2, 2]]),
                                         ([8, 10, 9, 10], [[1, 2, 2]]), ([8, 11, 9], [[1, 2, 2]]),
                                         ([8, 10, 9], [[2, 2, 2]]), ([8, 10, 9], [[1, 3, 2]])])
def test_invalid_image_layout_is_refused(tokens, grids):
    with pytest.raises(ValueError):
        image_positions(tokens, grids, CONFIG)


def test_prepare_expands_cpu_tokens_and_budgets_before_encoding():
    processor = ImageProcessor()
    front = QwenImageProcessor(CONFIG, processor, Tokenizer())
    prepared = front.prepare("a<start><image><end>b", [image()], max_visual_tokens=8, max_prompt_tokens=8)
    assert prepared.token_ids == (97, 8, 10, 10, 10, 10, 9, 98)
    assert prepared.visual_tokens == 4 and prepared.image_hashes == ("one",)
    assert front.estimate_workspace_bytes(prepared) > 2 * prepared.pixel_values.nbytes
    front.workspace_bytes = 1234                  # measured at load: it replaces the every-layer bound
    assert front.estimate_workspace_bytes(prepared) == (1234 + 2 * prepared.pixel_values.nbytes
                                                        + len(prepared.token_ids) * 3 * 8 + prepared.position_ids.nbytes)
    assert processor.calls == [(["PIL:one"], {"max_pixels": 8192, "min_pixels": 1024})]
    assert all(not a.flags.writeable for a in (prepared.pixel_values, prepared.image_grid_thw, prepared.position_ids))
    with pytest.raises(ValueError, match="expanded image prompt"):
        front.prepare("a<start><image><end>b", [image()], max_prompt_tokens=7)
    with pytest.raises(ValueError, match="visual-token budget"):
        front.prepare("<start><image><end>", [image()], max_visual_tokens=3)


def test_prepare_rejects_missing_or_extra_markers_before_processing():
    processor = ImageProcessor()
    front = QwenImageProcessor(CONFIG, processor, Tokenizer())
    with pytest.raises(ValueError, match="exactly one"):
        front.prepare("text", [image()])
    assert processor.calls == []


def test_processor_options_are_local_and_match_checkpoint_geometry(tmp_path):
    (tmp_path / "preprocessor_config.json").write_text(json.dumps({"size": {"longest_edge": 4096},
                                                                  "image_mean": [0.1, 0.2, 0.3]}))
    options = _processor_options(tmp_path, CONFIG["vision_config"])
    assert options["patch_size"] == 16 and options["max_pixels"] == 4096
    assert options["image_mean"] == [0.1, 0.2, 0.3]
    (tmp_path / "preprocessor_config.json").write_text(json.dumps({"patch_size": 14}))
    with pytest.raises(ValueError, match="disagrees"):
        _processor_options(tmp_path, CONFIG["vision_config"])


def test_checkpoint_reads_only_vision_ranges_in_a_mixed_shard(tmp_path):
    write_shard(tmp_path / "model.safetensors", {
        "model.language_model.embed_tokens.weight": ("F32", np.ones((4, 3), np.float32)),
        "model.visual.blocks.0.weight": ("U32", np.array([[42]], np.uint32)),
        "model.visual.blocks.0.scales": ("F16", np.array([[0.25]], np.float16)),
        "model.visual.norm.weight": ("BF16", np.array([0x3F80], np.uint16)),
    })
    calls = []
    mx = SimpleNamespace(array=lambda x: calls.append(x) or x, bfloat16=np.uint16)
    weights = load_vision_weights(vision_tensors(tmp_path), mx)
    assert set(weights) == {"blocks.0.weight", "blocks.0.scales", "norm.weight"}
    assert len(calls) == 3 and weights["norm.weight"].tolist() == [0x3F80]
    assert weights["blocks.0.weight"].dtype == np.uint32


def test_index_selects_vision_shards_without_opening_language_shards(tmp_path):
    write_shard(tmp_path / "vision.safetensors", {"vision_tower.norm.weight": ("F32", np.ones(3, np.float32))})
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "vision_tower.norm.weight": "vision.safetensors", "language_model.weight": "absent-language.safetensors"}}))
    assert set(vision_tensors(tmp_path)) == {"norm.weight"}


def test_invalid_tensor_range_is_rejected(tmp_path):
    path = tmp_path / "model.safetensors"
    write_shard(path, {"model.visual.norm.weight": ("F32", np.ones(3, np.float32))})
    tensors = vision_tensors(tmp_path)
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="Invalid vision tensor range"):
        load_vision_weights(tensors, SimpleNamespace(array=np.array))


def test_quantization_uses_each_module_override_and_never_quantizes_float_weights():
    weights = {"blocks.0.scales": 1, "blocks.1.scales": 1}
    pred = quantization_predicate(CONFIG, weights)
    module = SimpleNamespace(to_quantized=lambda: None)
    assert pred("blocks.0", module) == {"bits": 8, "group_size": 32, "mode": "affine"}
    assert pred("blocks.1", module) == {"bits": 4, "group_size": 64, "mode": "affine"}
    assert pred("patch_embed.proj", module) is False


def test_load_instantiates_only_tower_and_local_tokenizer(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps(CONFIG))
    write_shard(tmp_path / "model.safetensors", {
        "model.visual.blocks.0.weight": ("U32", np.ones((1, 1), np.uint32)),
        "model.visual.blocks.0.scales": ("F16", np.ones((1, 1), np.float16)),
        "language_model.embed_tokens.weight": ("F32", np.ones((20, 3), np.float32))})
    events = []

    class Tower:
        def __init__(self, config):
            events.append(("tower", config))

        def sanitize(self, weights):
            return weights

        def load_weights(self, weights, *, strict):
            events.append(("weights", dict(weights), strict))

        def eval(self):
            events.append("eval")

        def parameters(self):
            return []

        patch_embed = SimpleNamespace(proj=SimpleNamespace(weight=np.ones(1, np.float32)))

        def __call__(self, pixels, grid):
            events.append(("probe", pixels.shape, grid.tolist()))
            memory["peak"] = memory["active"] + 3 * 1024**2
            return np.zeros((1, 3), np.float32), None

    memory = {"active": 5 * 1024**2, "peak": 0}

    def tokenizer(path, **kwargs):
        assert path == str(tmp_path) and kwargs == {"local_files_only": True, "trust_remote_code": False}
        return Tokenizer()

    def quantize(tower, *, class_predicate):
        settings = class_predicate("blocks.0", SimpleNamespace(to_quantized=True))
        events.append(("quantize", settings))

    mx = SimpleNamespace(array=lambda value, dtype=None: np.array(value), eval=lambda *_: None, int32=np.int32,
                         zeros=lambda shape, dtype=None: np.zeros(shape, np.float32), synchronize=lambda: None,
                         clear_cache=lambda: None, get_active_memory=lambda: memory["active"],
                         reset_peak_memory=lambda: memory.update(peak=memory["active"]),
                         get_peak_memory=lambda: memory["peak"])
    runtime = (mx, SimpleNamespace(quantize=quantize), SimpleNamespace(from_dict=lambda value: value), Tower)
    monkeypatch.setattr(qwen_mlx, "_runtime", lambda: runtime)
    monkeypatch.setattr(qwen_processing, "_processor_runtime",
                        lambda: (SimpleNamespace(from_pretrained=tokenizer), ImageProcessor))
    embed = object()
    frontend = qwen_mlx.QwenVisionFrontend.load(tmp_path, embed)
    assert frontend.embed_tokens is embed
    assert len([e for e in events if isinstance(e, tuple) and e[0] == "tower"]) == 1
    loaded = next(e for e in events if isinstance(e, tuple) and e[0] == "weights")
    assert set(loaded[1]) == {"blocks.0.weight", "blocks.0.scales"} and loaded[2] is True
    assert ("quantize", {"bits": 8, "group_size": 32, "mode": "affine"}) in events
    # the workspace is measured once, on four images sharing the 4,096 visual tokens
    probe = [e for e in events if isinstance(e, tuple) and e[0] == "probe"]
    merge = CONFIG["vision_config"]["spatial_merge_size"]
    assert len(probe) == 1 and probe[0][2] == [[1, 32 * merge, 32 * merge]] * 4
    assert frontend.workspace_bytes == 3 * 1024**2


def test_encode_uses_target_embeddings_and_replaces_only_visual_rows():
    calls = []
    features = np.arange(12, dtype=np.float32).reshape(4, 3)

    class Tower:
        patch_embed = SimpleNamespace(proj=SimpleNamespace(weight=np.ones(1, np.float32)))

        def __call__(self, pixels, grid):
            calls.append("vision")
            return features, []

    def embed(tokens):
        calls.append("target-embedding")
        return np.full((1, tokens.shape[1], 3), 1e10, dtype=np.float32)

    front = qwen_mlx.QwenVisionFrontend(CONFIG, embed, Tower(), ImageProcessor(), Tokenizer(), np)
    prepared = front.prepare("a<start><image><end>b", [image()])
    assert calls == []
    encoded = front.encode(prepared)
    assert calls == ["target-embedding", "vision"]
    np.testing.assert_array_equal(encoded.inputs_embeds[0, 2:6], features)
    assert np.all(encoded.inputs_embeds[0, [0, 1, 6, 7]] == 1e10)
    assert encoded.rope_delta == -2 and encoded.position_ids.shape == (3, 1, 8)
