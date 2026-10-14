"""glm.layer.mlp fixtures: a dense-MLP KDA layer (0-2 carry the dense MLP) through glm_oracle_layer's windows."""
from __future__ import annotations

from pathlib import Path

import glm_oracle_layer

LAYER = 1


def write(out: Path, model_dir: Path) -> dict:
    return glm_oracle_layer.write(out, model_dir, LAYER, "lm")
