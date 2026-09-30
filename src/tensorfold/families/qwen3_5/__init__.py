"""Qwen3.8 dense uses row-exact serial and drafted decoding, resuming prefill only at chunk starts to preserve fresh-prefill bits."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any

MODEL_TYPES = ("qwen3_5",)
TITLE = "Qwen3.8 dense"
LANES = True
MODELS = ("Vontra/Qwen3.8-27B-MLX-4bit", "turboderp/Qwen3.8-27B-exl3", "nvidia/Qwen3.8-27B-NVFP4")
DRAFTER = "z-lab/Qwen3.8-27B-DFlash2"
QUANT_METHODS = {"cuda": ("mlx", "exl3", "modelopt", "compressed-tensors")}   # MLX affine, EXL3, NVFP4 / FP8
EXL3_VARIANT = "any"                           # every EXL3 codebook and width (tensorfold.families.EXL3_VARIANT_ANY)
KERNEL_PACKAGE = "tensorfold.kernels.qwen.dense.v1"
KERNEL_VERSION = "v1"

# the widest verify window checked at load (rows) with tensor units; without them ``row_matmul.WINDOW_ROWS``
WIDEST = 32
# the widest window a lone stream's copies may earn without tensor units (M3 Ultra: edits +22%, chat and code level)
ROW_COPY_ROWS = 128
# the widest copy window TF_COPY_ROWS may ask for (the lane kernels' and the chain kernels' row limit)
COPY_ROWS_LIMIT = 128


def copy_rows(first: int, default: int) -> int:
    """The widest window a lone stream's copies may earn: TF_COPY_ROWS if set (0: ``first``), else ``default``."""

    value = os.environ.get("TF_COPY_ROWS", "").strip()
    if not value:
        return int(default)
    return max(int(first), min(int(value), COPY_ROWS_LIMIT)) if int(value) > 0 else int(first)


def tensor_units() -> bool:
    """Whether this GPU has Metal 4 tensor units (``applegpu_g17`` and later), which the lane kernels need."""

    import mlx.core as mx

    info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    found = re.match(r"applegpu_g(\d+)", str(info.get("architecture", "")))
    return bool(found) and int(found.group(1)) >= 17


def load_lane_model(model_dir: Path) -> tuple[Any, Any]:
    """Load through mlx_lm; checkpoints that keep MTP tensors drop them before mlx_lm's sanitize."""

    from mlx_lm import load

    index_path = Path(model_dir) / "model.safetensors.index.json"
    names: list[str] = []
    if index_path.exists():
        weight_map = json.loads(index_path.read_text()).get("weight_map", {})
        names = [n for n in weight_map if n.startswith("mtp.") or ".mtp." in n]
    if not names:
        loaded = load(str(model_dir))
        return loaded[0], loaded[1]
    from mlx_lm.models.qwen3_5 import TextModel

    original = TextModel.sanitize

    def sanitize_without_mtp(self: Any, weights: dict[str, Any]) -> Any:
        kept = {k: v for k, v in weights.items() if not (k.startswith("mtp.") or ".mtp." in k)}
        return original(self, kept)

    TextModel.sanitize = sanitize_without_mtp  # type: ignore[method-assign]
    try:
        loaded = load(str(model_dir))
        return loaded[0], loaded[1]
    finally:
        TextModel.sanitize = original  # type: ignore[method-assign]


def install_row_decoder(model: Any) -> bool:
    """Install row-exact simd_qmm decoding without tensor units, returning False for unsupported weights."""

    from tensorfold.kernels.qwen.dense.v1 import exact_attention, row_forward, row_matmul

    backend = row_matmul.simd_qmm_backend()
    if not row_matmul.fits(model, backend):
        return False
    exact_attention.install()
    # prompt chains of up to PROMPT_ROWS attend query by query too, so a prompt gets the bits decoding gives it
    exact_attention.EXACT_MAX_QUERIES = max(exact_attention.EXACT_MAX_QUERIES, row_forward.PROMPT_ROWS)
    row_matmul.install(model, backend)
    model._tensorfold_row_decoder = True
    return True


def refusal(config: dict[str, Any], lanes: bool) -> str | None:
    """Why the lane kernels (``lanes``) or the decoder without tensor units cannot read a checkpoint, else None."""

    try:
        check_quantization(config, "mlx")
    except ValueError as exc:
        return str(exc)
    return None


def _language_specs(config: dict[str, Any]):
    from tensorfold.quantization import quantization_block, resolve_affine

    specs = {"": resolve_affine(config)}
    for path, value in (quantization_block(config) or {}).items():
        if ((isinstance(value, dict) or type(value) is bool)
                and not any(part in path.split(".") for part in ("vision_tower", "visual"))
                and not path.endswith("embed_tokens")):
            specs[path] = resolve_affine(config, path)
    return specs


def check_quantization(config: dict[str, Any], backend: str) -> None:
    from tensorfold.quantization import resolve_affine

    if resolve_affine(config) is None:
        raise ValueError("Qwen dense requires MLX affine quantization metadata; this checkpoint has none (unquantized weights)")
    if config.get("tie_word_embeddings") or (config.get("text_config") or {}).get("tie_word_embeddings"):
        raise ValueError("the tied embedding head is not supported by this packed Qwen decoder")
    _language_specs(config)


def native_lanes(config: dict[str, Any]) -> bool:
    return all(spec is None or spec.group_size == 64 or (spec.bits == 4 and spec.group_size == 32)
               for spec in _language_specs(config).values())


def check(model_dir: str | Path) -> None:
    """Refuse from config.json alone, before any weight downloads, a checkpoint this Mac's decoder cannot read."""

    import sys

    if sys.platform != "darwin":                  # the CUDA engine's rules are check_quantization's (require_readable)
        return
    from tensorfold.families import read_config

    why = refusal(read_config(model_dir), False)
    if why:
        raise ValueError(f"{TITLE} cannot run this checkpoint: {why}. Use {MODELS[0]}")


def load(model_dir: Path, *, lane_kernels: str = "auto", drafter: str = "", drafter_bits: int = 4,
         vision: bool = False, vision_urls: bool = False, **_: Any) -> tuple[Any, Any]:
    """Load a supported checkpoint with tensor-unit lane kernels when enabled, otherwise the row-exact decoder."""

    from tensorfold.families import read_config

    if lane_kernels == "on" and not tensor_units():      # the lane kernels' fragment layouts are the M5's
        raise SystemExit(f"[tensorfold] {TITLE}: --lane-kernels on needs Metal 4 tensor units (an M5-generation GPU), "
                         f"which this GPU does not have. Use --lane-kernels auto or off")
    lanes = lane_kernels == "on" or (lane_kernels == "auto" and tensor_units())
    why = refusal(read_config(model_dir), lanes)
    if why:
        raise SystemExit(f"[tensorfold] {TITLE} cannot run this checkpoint: {why}. Use {MODELS[0]}")
    if lanes and not native_lanes(read_config(model_dir)):
        if lane_kernels == "on":
            raise ValueError("this format uses the packed affine row kernels; use --lane-kernels auto or off")
        lanes = False
    model, tokenizer = load_lane_model(Path(model_dir))
    family = lane_family(model, lanes=lanes, drafter=drafter, drafter_bits=drafter_bits, title=TITLE, use=MODELS[0])
    if vision:
        from tensorfold.vision.qwen_mlx import QwenVisionFrontend
        from tensorfold.vision.rotary import install_rotary

        family.vision = QwenVisionFrontend.load(Path(model_dir), family.core.embed_tokens, allow_urls=vision_urls)
        print(f"[tensorfold] image encoder: {family.vision.workspace_bytes / 1024**3:.2f} GiB workspace measured at "
              "the largest image request (four images, 4,096 image tokens)", flush=True)
        config = read_config(model_dir).get("text_config", {})
        sections = config.get("rope_parameters", {}).get("mrope_section", [11, 11, 10])
        install_rotary(family.core, sections)
    return family, tokenizer


def lane_family(model: Any, *, lanes: bool, drafter: str, drafter_bits: int, title: str, use: str) -> Any:
    """Install the lane kernels (M5) or the row decoder (M1-M4) on ``model`` and wrap it, drafter included."""

    from tensorfold.families.qwen3_5.family import Qwen35Family
    from tensorfold.kernels.qwen.dense.v1 import lane_qmm

    model._tensorfold_lanes = bool(lanes)
    if lanes:
        missed = lane_qmm.uncovered(model)
        if missed:
            kinds = ", ".join(f"{n} {kind}" for kind, n in sorted(missed.items()))
            raise SystemExit(f"[tensorfold] {title}: the lane kernels do not take this checkpoint's layers ({kinds}): "
                             f"MLX's kernels would give drafted rows other bits than one-row steps. Use {use}")
        install_lane_kernels(model)
    elif not install_row_decoder(model):
        raise SystemExit(f"[tensorfold] {title}: the lane decoder without tensor units does not take these weights")
    loaded = load_drafter(model, drafter, drafter_bits) if drafter else None
    if lanes:
        family = Qwen35Family(model, drafter=loaded, widest=copy_rows(WIDEST, WIDEST), first_copy_rows=WIDEST)
    else:
        from tensorfold.kernels.qwen.dense.v1 import row_matmul

        family = Qwen35Family(model, drafter=loaded, widest=copy_rows(row_matmul.WINDOW_ROWS, ROW_COPY_ROWS),
                              rows=True, first_copy_rows=row_matmul.WINDOW_ROWS)
    timing = ", ".join(f"{w}: {ms:.1f}" for w, ms in sorted(family.window_costs.items()) if w in (1, 2, 4, 8, 16, 17,
                                                                                                    32, 64, 128))
    decoder = "lane kernels" if lanes else "lane decoder without tensor units"
    print(f"[tensorfold] {decoder}: windows of up to {family.exact_width} rows reproduce one-row steps here "
          f"(ms by rows {timing})", flush=True)
    return family


def load_drafter(model: Any, drafter: str, drafter_bits: int = 4) -> Any:
    """The DFlash2 draft model bound to ``model``, its matmuls through the lane kernels when the target's are."""

    import mlx.core as mx

    from tensorfold.drafters.dflash_drafter import DFlashDrafter

    loaded = DFlashDrafter(model, drafter, bits=int(drafter_bits))
    if not getattr(model, "_tensorfold_lanes", False):
        from tensorfold.kernels.qwen.dense.v1 import row_matmul

        if row_matmul.route_drafter(loaded.model):   # the drafter's blocks through a cheap multi-row matmul
            for rows in (2, row_matmul.WINDOW_ROWS):   # the draft-vocabulary head's shapes compiled now
                mx.eval(loaded.candidate_logits(mx.zeros((1, rows, int(loaded.model.config.hidden_size)),
                                                         dtype=mx.bfloat16))[0])
    if getattr(model, "_tensorfold_lanes", False):
        from tensorfold.kernels.qwen.dense.v1 import lane_qmm

        lane_qmm.install(loaded.model, rows=lane_qmm.MAX_ROWS, tile=os.environ.get("TF_LANE_TILE", "1") != "0",
                         wide=True)
        lane_qmm.warm(loaded.model)
        hidden = mx.zeros((1, 16, int(loaded.model.config.hidden_size)), dtype=mx.bfloat16)
        mx.eval(*(a for a in loaded.candidate_logits(hidden) if a is not None))     # the draft head, compiled now
    print(f"[tensorfold] drafter {loaded.path} block={loaded.block_size} bits={drafter_bits or 16}", flush=True)
    return loaded


def install_lane_kernels(model: Any) -> None:
    """Install lane matmul, fused projections and lane attention, compiling every variant before requests arrive."""

    from tensorfold.kernels.qwen.dense.v1 import exact_attention, lane_attention, lane_fuse, lane_qmm

    exact_attention.install()      # verify windows attend query by query, as one-row steps do
    # TF_LANE_TILE=0 keeps MLX's weight layout without changing results.
    lane_qmm.install(model, rows=lane_qmm.MAX_ROWS, tile=os.environ.get("TF_LANE_TILE", "1") != "0", wide=True)
    warmed = lane_qmm.warm(model)
    lane_fuse.enabled = True
    fused = lane_fuse.build(model)          # stacks share the weights' memory: no second copy
    lane_fuse.warm(model)
    exact_attention.EXACT_MAX_QUERIES = lane_qmm.MAX_ROWS
    lane_attention.install()
    lane_attention.warm(max_queries=lane_attention.MAX_QUERIES)
    print(f"[tensorfold] lane kernels on: {warmed} matmul shapes warmed, fused projections {fused}", flush=True)


def engine_settings(model: Any) -> dict[str, Any]:
    """Set max_rows to the verification width and max_draft to the drafts each stream may offer per round."""

    width = int(getattr(model, "exact_width", 1) or 1)
    return {"max_rows": width, "max_draft": max(0, width - 1)}


def kernel_version(model: Any) -> str:
    """Names the kernels that computed a prefix snapshot (a snapshot computed by other kernels has other bits)."""

    import hashlib

    model = getattr(model, "inner", model)          # a family model's prefixes are its lane decoder's
    if not getattr(model, "_tensorfold_lanes", False):
        from tensorfold.kernels.qwen.dense.v1 import row_forward, row_matmul

        folder = Path(row_forward.__file__).parent
        parts = [row_matmul.BACKEND.name, f"row_attention={row_forward.ROW_ATTENTION}",
                 *(path.read_text() for path in sorted(folder.glob("*.py")))]
        return "row-forward-" + hashlib.sha256("\n".join(parts).encode()).hexdigest()[:12]
    from tensorfold.kernels.qwen.dense.v1 import (lane_attention, lane_fuse, lane_glue, lane_qmm, lane_widen,
                                                  stream_attention, stream_gdn)

    sources = [lane_qmm._MAIN, lane_qmm._MAIN_TILED, *lane_widen.sources().values(), lane_qmm._XSUM,
               lane_attention._PARTIAL, *stream_attention.sources().values(), lane_attention._MERGE,
               lane_glue._NORM_XS, lane_glue._GDN_PRE, lane_glue._GDN_POST, lane_glue._MLP_ACT,
               *stream_gdn.sources().values(), repr((lane_attention.CHUNK, lane_attention.TILE))]
    if lane_fuse.enabled:
        sources += [text for _, text in sorted(lane_fuse.sources().items())]
    folder = Path(lane_qmm.__file__).parent
    sources.extend(path.read_text() for path in sorted(folder.glob("*.py")))
    sources.extend(path.read_text() for path in sorted(Path(__file__).parent.glob("*.py")))
    return f"qwen-dense-{KERNEL_VERSION}-" + hashlib.sha256("\n".join(sources).encode()).hexdigest()[:12]


# The metadata-only info command also displays these CUDA affine formats.
CUDA_AFFINE_BITS = (2, 3, 4, 5, 6, 8)
CUDA_AFFINE_GROUPS = (32, 64, 128)
# --checkpoint-slots on CUDA: the prompt states the concurrent decoder keeps (--parallel 2 or more)
CUDA_CHECKPOINT_SLOTS = True

def gb10() -> bool:
    """Whether GPU 0 is a GB10 (DGX Spark: compute capability 12.1), where the lone stream's wide windows were measured."""

    import torch

    if not torch.cuda.is_available():
        return False
    return tuple(torch.cuda.get_device_capability(0)) == (12, 1) or "GB10" in torch.cuda.get_device_name(0)


def cuda_engine(model_dir: str | Path, *, drafter: str = "", tp: int = 1, rank: int = 0, master: str = "",
                master_port: int = 29551, no_drafts: bool = False, **options: Any):
    """The CUDA engine for ``tensorfold serve``; tp=2 adds fp32 partials in rank order and needs the drafter on both."""

    from .cuda.engine import Qwen27Engine
    from .cuda.exl3_load import quant_config

    if quant_config(Path(model_dir)) is not None:
        print("[tensorfold] EXL3 packs are experimental: replies are exact; on a DGX Spark decode runs 0.8-1.2x the MLX "
              "checkpoint and prompts about half as fast (docs/recipes/qwen3.8-27b.md#exl3-checkpoints-experimental)", flush=True)
    if not drafter and not no_drafts:
        raise ValueError(f"{TITLE}'s CUDA engine drafts with {DRAFTER}, which is not here: without it every round "
                         f"would decode one token. Run `tensorfold pull {DRAFTER}` once (on both machines for "
                         "--tp 2), or pass --no-drafts for the serial reference")
    draft = Path(drafter) if drafter and not no_drafts else None
    streams = max(1, int(options.get("parallel") or 1))
    wide = tp == 1 and streams == 1 and gb10()
    return Qwen27Engine(Path(model_dir), draft, max_rows=128 if wide else 12, tree_rows=16 if wide else None,
                        tp=tp, rank=rank, master=master, port=master_port,
                        split_head=tp == 2, tp_draft=tp == 2 and draft is not None, allow_copy=not no_drafts,
                        streams=streams, context=options.get("context"),
                        context_explicit=options.get("context_explicit"), vision=bool(options.get("vision", False)),
                        vision_urls=bool(options.get("vision_urls", False)), keep=options.get("checkpoint_slots"))
