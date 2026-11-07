"""The H3 text encoder for h3_generate.py: minimax-h3-mlx's, a layer at a time, or from a ComfyUI int8 file."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import mlx.core as mx
import numpy as np


class _NoEval:
    """`mlx.core` with `eval` switched off, so weights that were loaded lazily stay on disk until they are used."""

    def __getattr__(self, name):
        return getattr(mx, name)

    @staticmethod
    def eval(*_):
        return None


def _streamed(encoder):
    """Run the encoder one layer at a time, each layer read for its turn and dropped (48 GiB whole, <2 GiB a layer)."""

    model = encoder.language

    def hidden_states(input_ids, position_ids, inputs_embeds=None, visual_pos_masks=None, deepstack_visual_embeds=None):
        from mlx_vlm.models.base import create_attention_mask

        h = model.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        mask = create_attention_mask(h, None)
        position_embeddings = None
        if position_ids is not None and not model.layers[0].self_attn.rotary_emb.fused_apply:
            position_embeddings = model.layers[0].self_attn.rotary_emb(h, position_ids)
        for index in range(len(model.layers)):
            h = model.layers[index](h, mask, None, position_ids, position_embeddings)
            if deepstack_visual_embeds is not None and index < len(deepstack_visual_embeds):
                h = model._deepstack_process(h, visual_pos_masks, deepstack_visual_embeds[index])
            mx.eval(h)
            model.layers[index] = None  # the encoder is used once
            mx.clear_cache()
        return h

    encoder._hidden_states = hidden_states


def _hadamard(size: int):
    """ConvRot's orthogonal Hadamard matrix of a power-of-4 size (comfy_kitchen's `_build_hadamard`), self-inverse."""

    h4 = np.array([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], np.float32)
    h = h4
    while h.shape[0] < size:
        h = np.kron(h, h4)
    if h.shape[0] != size:
        raise ValueError(f"a ConvRot group is a power of 4, got {size}")
    return mx.array(h / np.sqrt(np.float32(size)))


def comfy_int8_weights(path, dtype=mx.bfloat16):
    """Lazy tensors of a ComfyUI `int8_tensorwise` file in ``dtype`` (int8 * row scale, ConvRot undone), counts."""

    tensors = mx.load(str(path))
    weights, used, rotated = {}, set(), 0
    for name, tensor in tensors.items():
        stem, _, leaf = name.rpartition(".")
        if leaf in ("comfy_quant", "weight_scale"):
            continue
        settings, scale = f"{stem}.comfy_quant", f"{stem}.weight_scale"
        if leaf != "weight" or settings not in tensors:
            if tensor.dtype not in (mx.bfloat16, mx.float16, mx.float32):
                raise ValueError(f"{name} is {tensor.dtype} without quantization settings")
            weights[name] = tensor.astype(dtype)
            continue
        conf = json.loads(bytes(np.array(tensors[settings])))
        if conf.get("format") != "int8_tensorwise" or tensor.dtype != mx.int8 or tensor.ndim != 2:
            raise ValueError(f"{name}: only int8_tensorwise matrices are read, got {conf} as {tensor.dtype}")
        if scale not in tensors or tensors[scale].shape != (tensor.shape[0], 1):
            raise ValueError(f"{name}: no scale per output row")
        used.update((settings, scale))
        weight = tensor.astype(mx.float32) * tensors[scale].astype(mx.float32)
        if conf.get("convrot", False):
            group = int(conf.get("convrot_groupsize", 256))
            rows, columns = weight.shape
            if columns % group:
                raise ValueError(f"{name}: {columns} columns are not whole groups of {group}")
            weight = (weight.reshape(rows, columns // group, group) @ _hadamard(group).T).reshape(rows, columns)
            rotated += 1
        weights[name] = weight.astype(dtype)
    stray = sorted(name for name in tensors if name.endswith((".comfy_quant", ".weight_scale")) and name not in used)
    if stray:
        raise ValueError(f"{path}: {len(stray)} quantization tensors without an int8 weight, e.g. {stray[:4]}")
    return weights, len(used) // 2, rotated


def _from_comfy_int8(base, path):
    """``base`` text encoder reading `model.*` and `visual.*` from one ComfyUI int8 file; config, tokenizer: folder."""

    class Encoder(base):
        def _load_weights(self, model_dir, dtype, verbose):
            from mlx.utils import tree_flatten, tree_unflatten

            weights, quantized, rotated = comfy_int8_weights(path, dtype)
            parts = {"model.": self.language, "visual.": self.vision}
            unknown = sorted(name for name in weights if not name.startswith(tuple(parts)))
            for prefix, module in parts.items():
                mine = {name[len(prefix):]: tensor for name, tensor in weights.items() if name.startswith(prefix)}
                if module is None:  # a request without pictures does not build the vision tower
                    continue
                expected = {name: tensor.shape for name, tensor in tree_flatten(module.parameters())}
                if prefix == "model.":
                    expected.pop("norm.weight", None)
                sanitize = getattr(module, "sanitize", None)  # the patch convolution's layout, as in the base loader
                if sanitize is not None:
                    mine = sanitize(mine)
                missing = sorted(expected.keys() - mine.keys())
                unknown += sorted(prefix + name for name in mine.keys() - expected.keys())
                wrong = sorted(name for name in expected.keys() & mine.keys() if mine[name].shape != expected[name])
                if missing or wrong:
                    raise KeyError(f"{path}: {len(missing)} weights of {prefix}* missing, e.g. {missing[:4]}; "
                                   f"{len(wrong)} of another shape, e.g. {wrong[:4]}")
                module.update(tree_unflatten(list(mine.items())))
            if unknown:
                raise KeyError(f"{path}: {len(unknown)} tensors the text encoder has no place for, e.g. {unknown[:4]}")
            self.skipped_tensors = 0
            print(f"[tensorfold] text encoder: {Path(path).name}, {quantized} int8 weights ({rotated} rotated) "
                  f"decoded to {dtype}", flush=True)

    return Encoder


VISION_STATS = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Qwen3-VL's own: pixels to -1..1 (used when the file is absent)
MLX_VISION_STATS = ([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])


def vision_stats(root: Path, name: str = "preprocessor_config.json"):
    """The vision tower's per-channel mean and std from the checkpoint's `processor/` file, else Qwen3-VL's 0.5/0.5."""

    path = root / "processor" / name
    if not path.is_file():
        return VISION_STATS
    with open(path) as handle:
        conf = json.load(handle)
    mean, std = conf.get("image_mean"), conf.get("image_std")
    if not (isinstance(mean, list) and isinstance(std, list) and len(mean) == len(std) == 3):
        return VISION_STATS
    return [float(v) for v in mean], [float(v) for v in std]


def _text_encoder(root: Path, vision: bool, weights=None, checkpoint_stats: bool = True):
    """minimax-h3-mlx's text encoder, streamed; ``checkpoint_stats`` false keeps its old image stats for comparisons."""

    import minimax_h3_mlx.text_encoder as module
    from minimax_h3_mlx.text_encoder import MiniMaxH3TextEncoder

    if weights is not None:
        MiniMaxH3TextEncoder = _from_comfy_int8(MiniMaxH3TextEncoder, weights)
    original = module.mx
    module.mx = _NoEval()  # the loader ends by reading every weight into memory; leave them for _streamed
    try:
        encoder = MiniMaxH3TextEncoder(root / "text_encoder", dtype=mx.bfloat16, load_vision=vision)
    finally:
        module.mx = original
    if encoder.vision is not None:
        mx.eval(encoder.vision.parameters())
    _streamed(encoder)
    if vision:
        # the processor folder asks for PyTorch; the one built from the vision config needs only numpy
        from minimax_h3_mlx.text_encoder import _FallbackProcessor

        processor = encoder._build_image_processor()
        processor.image_mean, processor.image_std = vision_stats(root) if checkpoint_stats else MLX_VISION_STATS
        encoder._processor = _FallbackProcessor(processor)
    return encoder


def encode_text(root: Path, prompt: str, images=None, weights=None, checkpoint_stats: bool = True):
    """Text rows and tags; ``images`` go ahead of the prompt as `<Picture i>: `, ``weights`` a ComfyUI int8 file."""

    if images is not None and not isinstance(images, (list, tuple)):
        images = [images]
    images = list(images) if images else None
    encoder = _text_encoder(root, images is not None, weights, checkpoint_stats)
    embeds, tags = encoder.encode(prompt, images)
    mx.eval(embeds)
    del encoder
    gc.collect()
    mx.clear_cache()
    return embeds, tags
