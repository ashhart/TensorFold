"""Qwen3.8 dense (model_type ``qwen3_5``), e.g. Qwen3.8-27B: mlx_lm's model as a lane-engine family.

The family (``family.Qwen35Family``) runs every round through TensorFold's lane decoder, whose rows get the same
bits at any window width: with Metal 4 tensor units (the M5 generation) the lane kernels (``kernels.lane_qmm``,
``kernels.lane_attention``, ``kernels.lane_fuse``, and the stream kernels for several requests in one forward);
without (M1 to M4) ``kernels.row_forward`` over the row-exact ``simd_qmm`` matmul. Serial decoding goes through the
same decoder, so drafted output equals it. Drafts come from a DFlash2 draft model (trees, or chains where the
decoder's attention takes no trees) and from copies of the context; concurrent requests share each round.

Prompts go through MLX's prefill in chunks on the engine's 2,048-token grid from position 0, and prefixes resume
only from grid points, so a resumed prompt gets a fresh prefill's bits.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any

MODEL_TYPES = ("qwen3_5",)
TITLE = "Qwen3.8 dense"
LANES = True
MODELS = ("Vontra/Qwen3.8-27B-MLX-4bit",)
DRAFTER = "z-lab/Qwen3.8-27B-DFlash2"
KERNEL_PACKAGE = "tensorfold.kernels.qwen.dense.v1"
KERNEL_VERSION = "v1"

# the widest verify window checked at load (rows) with tensor units; without them ``row_matmul.WINDOW_ROWS``
WIDEST = 32

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
    """The lane decoder without tensor units (``row_forward``: the row-exact ``simd_qmm`` matmul over stacked
    projections, attention query by query) for the family path; False where the weights do not take it."""

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


def load(model_dir: Path, *, lane_kernels: str = "auto", drafter: str = "", drafter_bits: int = 4,
         **_: Any) -> tuple[Any, Any]:
    """The family model: the lane kernels when ``lane_kernels`` is "on", or "auto" on a GPU with tensor units, else
    the lane decoder without them. Checkpoints the lane decoders cannot read are refused."""

    from tensorfold.families import describe_quantization, quantization, read_config
    from tensorfold.families.qwen3_5.family import Qwen35Family
    from tensorfold.kernels.qwen.dense.v1 import lane_qmm

    lanes = lane_kernels == "on" or (lane_kernels == "auto" and tensor_units())
    config = read_config(model_dir)
    bits, group = quantization(config)
    widths = "/".join(str(b) for b in lane_qmm.BITS)
    if not (group == 64 and (lane_qmm.readable(bits, group) if lanes else bits == 4)):
        others = "/".join(str(b) for b in lane_qmm.BITS if b != 4)
        reads = (f"{widths}-bit weights in groups of 64" if lanes else
                 f"4-bit weights in groups of 64 without tensor units ({others}-bit need an M5-generation GPU)")
        raise SystemExit(f"[tensorfold] {TITLE} decodes through lane kernels that read {reads}; this checkpoint has "
                         f"{describe_quantization(config)}. Use {MODELS[0]}")
    model, tokenizer = load_lane_model(Path(model_dir))
    model._tensorfold_lanes = bool(lanes)
    if lanes:
        missed = lane_qmm.uncovered(model)
        if missed:
            kinds = ", ".join(f"{n} {kind}" for kind, n in sorted(missed.items()))
            raise SystemExit(f"[tensorfold] {TITLE}: the lane kernels do not take this checkpoint's {kinds} "
                             f"projections (MLX's kernels would give drafted rows other bits than one-row steps). Use "
                             f"{MODELS[0]}, or a conversion whose projections are all {widths}-bit in groups of 64")
        install_lane_kernels(model)
    elif not install_row_decoder(model):
        raise SystemExit(f"[tensorfold] {TITLE}: the lane decoder without tensor units does not take these weights")
    loaded = load_drafter(model, drafter, drafter_bits) if drafter else None
    if lanes:
        family = Qwen35Family(model, drafter=loaded, widest=WIDEST)
    else:
        from tensorfold.kernels.qwen.dense.v1 import row_matmul

        family = Qwen35Family(model, drafter=loaded, widest=row_matmul.WINDOW_ROWS, rows=True)
    timing = ", ".join(f"{w}: {ms:.1f}" for w, ms in sorted(family.window_costs.items()) if w in (1, 2, 4, 8, 16, 17,
                                                                                                    32, 64, 128))
    decoder = "lane kernels" if lanes else "lane decoder without tensor units"
    print(f"[tensorfold] {decoder}: windows of up to {family.exact_width} rows reproduce one-row steps here "
          f"(ms by rows {timing})", flush=True)
    return family, tokenizer


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
    """Swap in the lane matmul, fused projections and lane attention, and compile every variant now (not inside the
    first requests)."""

    from tensorfold.kernels.qwen.dense.v1 import exact_attention, lane_attention, lane_fuse, lane_qmm

    exact_attention.install()      # verify windows attend query by query, as one-row steps do
    # TF_LANE_TILE=0 keeps MLX's weight layout (same bits, slower; for A/B timing)
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
    """Keyword arguments for the lane engine (``max_rows``: rows a round verifies; ``max_draft``: drafts a stream
    offers a round)."""

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
    from tensorfold.kernels.qwen.dense.v1 import (lane_attention, lane_fuse, lane_glue, lane_qmm, stream_attention,
                                                  stream_gdn)

    sources = [lane_qmm._MAIN, lane_qmm._MAIN_TILED, lane_qmm._MAIN_LOWBIT, lane_qmm._XSUM, lane_attention._PARTIAL,
               *stream_attention.sources().values(), lane_attention._MERGE, lane_glue._NORM_XS, lane_glue._GDN_PRE,
               lane_glue._GDN_POST, lane_glue._MLP_ACT, *stream_gdn.sources().values(),
               repr((lane_attention.CHUNK, lane_attention.TILE))]
    if lane_fuse.enabled:
        sources += [text for _, text in sorted(lane_fuse.sources().items())]
    folder = Path(lane_qmm.__file__).parent
    sources.extend(path.read_text() for path in sorted(folder.glob("*.py")))
    sources.extend(path.read_text() for path in sorted(Path(__file__).parent.glob("*.py")))
    return f"qwen-dense-{KERNEL_VERSION}-" + hashlib.sha256("\n".join(sources).encode()).hexdigest()[:12]


# the CUDA engine's kernels read MLX affine weights of this (bits, group size)
CUDA_QUANTIZATION = (4, 64)

def cuda_engine(model_dir: str | Path, *, drafter: str = "", tp: int = 1, rank: int = 0, master: str = "",
                master_port: int = 29551, no_drafts: bool = False, **options: Any):
    """The CUDA engine (``tensorfold serve`` on an NVIDIA GPU), set up as the recipe measured on DGX Spark.

    DFlash2 draft trees are verified in windows of 12 rows (the same drafts accepted as at 16 rows, for less
    time a round). On two GPUs (``tp=2``) the model is tensor parallel with fp32 partials summed in rank
    order, the head is split by vocabulary, and both ranks draft with half the draft model each, so both
    machines need it. ``no_drafts``: one token a round, the serial reference. Without the draft model every
    round but a copied one would decode one token, so drafting needs ``drafter``.
    """

    from .cuda.engine import Qwen27Engine

    if not drafter and not no_drafts:
        raise ValueError(f"{TITLE}'s CUDA engine drafts with {DRAFTER}, which is not here: without it every round "
                         f"would decode one token. Run `tensorfold pull {DRAFTER}` once (on both machines for "
                         "--tp 2), or pass --no-drafts for the serial reference")
    draft = Path(drafter) if drafter and not no_drafts else None
    return Qwen27Engine(Path(model_dir), draft, max_rows=12, tp=tp, rank=rank, master=master, port=master_port,
                        split_head=tp == 2, tp_draft=tp == 2 and draft is not None, allow_copy=not no_drafts,
                        context=options.get("context"), context_explicit=options.get("context_explicit"))
