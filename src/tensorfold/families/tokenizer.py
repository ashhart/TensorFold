"""mlx_lm's tokenizer for a checkpoint whose per-layer lists also count its MTP layers.

Some converters write ``mlp_layer_types`` (or another ``*layer_types`` list) with one entry per MTP layer after the
decoder's, e.g. 46 entries for GLM-5.3's 45 layers. transformers' config classes refuse that, so the tokenizer
(which reads config.json through transformers) does not load. The files stay as they are: the retry hands
transformers a config with those lists cut to ``num_hidden_layers``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def trimmed_config(model_dir: Path) -> Any | None:
    """A transformers config with per-layer lists cut to the decoder's layers, or None when none is longer."""

    raw = json.loads((Path(model_dir) / "config.json").read_text())
    text = raw.get("text_config") or raw
    layers = int(text.get("num_hidden_layers") or 0)
    long = [k for k, v in text.items() if k.endswith("layer_types") and isinstance(v, list) and len(v) > layers > 0]
    if not long:
        return None
    for key in long:
        text[key] = text[key][:layers]
    from transformers import AutoConfig

    return AutoConfig.for_model(raw["model_type"], **{k: v for k, v in raw.items() if k != "model_type"})


def load_tokenizer(model_dir: Path, eos_token_ids: Any = None) -> Any:
    """``mlx_lm.utils.load_tokenizer``, retried with ``trimmed_config`` when transformers refuses the layer lists."""

    from mlx_lm.utils import load_tokenizer as load

    try:
        return load(Path(model_dir), eos_token_ids=eos_token_ids)
    except Exception as error:
        if "layer_types" not in str(error):
            raise
        config = trimmed_config(Path(model_dir))
        if config is None:
            raise
        print("[tensorfold] config.json lists a layer type for each MTP layer too; the tokenizer reads it without "
              "them", flush=True)
        return load(Path(model_dir), tokenizer_config_extra={"config": config}, eos_token_ids=eos_token_ids)
