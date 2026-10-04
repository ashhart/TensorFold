"""MiniMax H3 joint video and audio generation (MLX); scaffold, no engine yet."""

from __future__ import annotations

# H3 checkpoints are diffusers layouts: `model_index.json` at the root and one `config.json` per component,
# so there is no `config.json["model_type"]`. `minimax_h3` is the name the CLI will route to once detection
# reads `model_index.json` (`_class_name == "MiniMaxH3Pipeline"`). See docs/design/h3-mlx.md.
MODEL_TYPES = ("minimax_h3",)
TITLE = "MiniMax H3"
# Not a lane family and no `load`: H3 denoises a packed video+audio+text sequence, it does not decode tokens.
# The entry point will be `generate_engine`, added with the pipeline milestone.
LANES = False
MODELS = ()
PIPELINE_CLASS = "MiniMaxH3Pipeline"
REQUIRED_COMPONENTS = ("transformer", "text_encoder", "tokenizer", "video_vae", "audio_vae")


def is_checkpoint(model_dir) -> bool:
    """Whether ``model_dir`` (or its ``FL2VA`` partition) is an H3 pipeline folder."""

    from .config import pipeline_root

    return pipeline_root(model_dir) is not None
