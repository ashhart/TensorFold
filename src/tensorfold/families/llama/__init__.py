"""Plain full-attention Llama on MLX; checkpoint qualification is separate from discovery."""

from __future__ import annotations

from pathlib import Path
from typing import Any

MODEL_TYPES = ("llama",)
TITLE = "Llama (plain RoPE, untied bf16 head)"
LANES = True
MODELS = ("Remek/basal-1.5-mini-MLX-8bit",)
KERNEL_PACKAGE = "tensorfold.kernels.qwen.dense.v1"  # Reuse generic affine row kernels, not Qwen's decoder.
KERNEL_VERSION = "v1"
KERNEL_DEPENDENCIES = ("tensorfold.kernels.qwen.dense.v1.row_matmul",
                       "tensorfold.kernels.qwen.dense.v1.simd_qmm_bits",
                       "tensorfold.kernels.qwen.dense.v1.simd_qmm",
                       "tensorfold.kernels.qwen.dense.v1.affine_rows",
                       "tensorfold.kernels.threads")
QUANT_METHODS = {"mlx": (None, "mlx")}


def check(model_dir: str | Path) -> None:
    import json

    from tensorfold.families import read_config

    from .config import check_headers, check_names, normalize

    config = normalize(read_config(model_dir))
    check_headers(model_dir, config)
    index = Path(model_dir) / "model.safetensors.index.json"
    if index.is_file():
        check_names(json.loads(index.read_text())["weight_map"])


def check_quantization(config: dict[str, Any], backend: str) -> None:
    from .config import normalize

    normalize(config)


def load(model_dir: str | Path, *, widest: int = 16, drafter: str = "", **_: Any) -> tuple[Any, Any]:
    """Use mlx-lm's local loader and stock prompt backbone, with row-exact lane decoding."""
    check(model_dir)
    if drafter:
        raise ValueError("Llama supports context-copy drafts, not a draft head")
    from mlx_lm.utils import load_model, load_tokenizer

    from tensorfold.families import read_config

    from .config import normalize
    from .family import LlamaFamily

    path = Path(model_dir)
    config = normalize(read_config(path))
    overrides = {key: value for key, value in config.items() if key != "eos_token_id"}
    model, loaded_config = load_model(path, model_config=overrides)
    return LlamaFamily(model, widest=widest), load_tokenizer(path, eos_token_ids=loaded_config.get("eos_token_id"))


def kernel_version(model: Any) -> str:
    from tensorfold.families import families, kernel_source_version

    source = kernel_source_version(families()["llama"])
    paths = ";".join(":".join(map(str, path)) for path in getattr(model, "qmm_paths", ()))
    paths = paths or ("unloaded" if model is None else "bf16")
    return f"{source}|qmm={paths}"


def engine_settings(model: Any) -> dict[str, Any]:
    context = model.inner.args.max_position_embeddings
    # Admission sizes caches at grid+64 and 2*grid+64, not just a single chunk.
    grid = min(2048, (context - 64) // 2) if context is not None else 2048
    return {"max_rows": model.batch_rows, "max_draft": model.exact_width - 1, "prefill_steps": (grid,)}
