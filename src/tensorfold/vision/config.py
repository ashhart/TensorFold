"""Validate opt-in vision support before checkpoint allocation."""
from __future__ import annotations


FAMILIES = ('qwen3_5', 'qwen4_exp')     # the Qwen3.5/3.8 dense models, and Flash Next (same tower) on CUDA


def validate_vision_config(config, family):
    if family not in FAMILIES:
        raise ValueError('--vision currently supports Qwen3.5/3.8 dense and Flash Next checkpoints with their vision '
                         'tower')
    vision = config.get('vision_config')
    text = config.get('text_config', config)
    if not isinstance(vision, dict) or not vision:
        raise ValueError('this checkpoint has no vision_config; use a complete vision-language checkpoint')
    width = text.get('hidden_size')
    output = vision.get('out_hidden_size')
    if output is not None and output != width:
        raise ValueError('vision tower output width does not match the language model')
    return vision
