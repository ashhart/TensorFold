"""Qwen3.8 Flash Next (model_type ``qwen4_exp``).

``model``: TensorFold's forward pass for the checkpoint (hyper-connections, Gated DeltaNet, sparse attention,
512-expert MoE, hashed n-gram embedding); ``kernels`` and ``decode``: the fused decode step; ``mtp``: the
checkpoint's MTP head; ``runtime``: what the serial engine serves (fused decode + exact MTP drafting).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

MODEL_TYPES = ("qwen4_exp",)
TITLE = "Qwen3.8 Flash Next"
LANES = False
# 4-bit weights in groups of 32 (what the fused kernels read), with the checkpoint's MTP head kept
MODELS = ("Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP",)
KERNEL_PACKAGE = "tensorfold.kernels.qwen.flash_next.v1"
KERNEL_VERSION = "v1"
# MLX command buffers: MLX ends a buffer once the bytes bound in it pass MLX_MAX_MB_PER_BUFFER, and every expert
# kernel binds the 420 MB expert stacks, so with the default each of them ended one (an empty kernel binding them
# cost 28 us a launch against 12). Set by the CLI before MLX starts, unless the environment already sets them.
MLX_ENV = {"MLX_MAX_OPS_PER_BUFFER": "200", "MLX_MAX_MB_PER_BUFFER": "100000"}


def has_mtp(model_dir: Path) -> bool:
    """Whether the checkpoint kept the MTP head's weights (``mtp.*``)."""

    import json

    index = Path(model_dir) / "model.safetensors.index.json"
    if not index.is_file():
        return False
    return any(".mtp." in name or name.startswith("mtp.") for name in json.loads(index.read_text())["weight_map"])


def check(model_dir: Path) -> None:
    from tensorfold.families import quantization, read_config

    bits, group = quantization(read_config(model_dir))
    if (bits, group) != (4, 32):
        raise ValueError(f"TensorFold's Flash Next kernels read 4-bit weights in groups of 32; this checkpoint has "
                         f"{bits}-bit weights in groups of {group}. Use {MODELS[0]}.")
    # The CLI may have downloaded only config.json for its preflight check. Do not report a missing head until
    # the checkpoint's weights or index are present.
    if ((Path(model_dir) / "model.safetensors.index.json").is_file()
            or any(Path(model_dir).glob("model*.safetensors"))) and not has_mtp(model_dir):
        print(f"[tensorfold] this checkpoint has no MTP head: decoding without MTP drafts ({MODELS[0]} has one)",
              flush=True)


def load(model_dir: Path, *, mtp_drafts: int | None = None, **_: Any) -> tuple[Any, Any]:
    from tensorfold.families.qwen4_exp.runtime import load as load_runtime

    return load_runtime(Path(model_dir), drafts=mtp_drafts if has_mtp(Path(model_dir)) else 0)
