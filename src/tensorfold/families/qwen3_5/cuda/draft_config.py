"""Header-only compatibility checks for a target and its trained DFlash taps."""

from pathlib import Path

from tensorfold.cuda.capacity import config, headers

DEFAULT_TAPS = (5, 19, 33, 47, 61)


def architecture(cfg: dict) -> str:
    kinds = cfg.get("architectures") or ["DFlash2DraftModel"]
    if kinds not in (["DFlashDraftModel"], ["DFlash2DraftModel"]):
        raise ValueError(f"unsupported drafter architecture: {kinds!r}")
    return kinds[0]


def target_layers(cfg: dict) -> tuple[int, ...]:
    """Keep the training order: fc.weight's column groups follow this list."""

    kind = architecture(cfg)
    dc = cfg.get("dflash_config") or {}
    taps = dc.get("target_layer_ids", DEFAULT_TAPS if kind == "DFlash2DraftModel" else None)
    if (not isinstance(taps, (list, tuple)) or not taps
            or any(type(i) is not int or i < 0 for i in taps) or len(set(taps)) != len(taps)):
        raise ValueError("dflash_config.target_layer_ids must be nonempty, unique, nonnegative integers")
    return tuple(taps)


def validate(model_dir: Path, draft_dir: Path, *, world: int = 1) -> tuple[int, ...]:
    """Reject incompatible checkpoints before importing torch or loading weights."""

    target, draft = config(model_dir), config(draft_dir)
    taps = target_layers(draft)
    if max(taps) >= target["num_hidden_layers"]:
        raise ValueError(f"dflash_config.target_layer_ids {taps} exceed the target's "
                         f"{target['num_hidden_layers']} layers")
    if world != 1 and architecture(draft) == "DFlashDraftModel":
        raise ValueError("a DFlash (v1) drafter runs on one GPU")
    if draft["hidden_size"] != target["hidden_size"]:
        raise ValueError("drafter hidden_size must match the target")
    for field, target_field in (("num_target_layers", "num_hidden_layers"), ("vocab_size", "vocab_size")):
        if field in draft and draft[field] != target[target_field]:
            raise ValueError(f"drafter {field} must match the target's {target_field}")
    mask = draft["dflash_config"].get("mask_token_id")
    if mask is not None and (type(mask) is not int or not 0 <= mask < target["vocab_size"]):
        raise ValueError("drafter mask_token_id must be inside the target vocabulary")
    fc = headers(draft_dir).get("fc.weight", {})
    shape = [draft["hidden_size"], len(taps) * target["hidden_size"]]
    if fc.get("shape") != shape:
        raise ValueError(f"drafter fc.weight must have shape {shape} for its target_layer_ids")
    return taps
