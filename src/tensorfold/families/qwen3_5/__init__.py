"""Qwen3.8 dense (model_type ``qwen3_5``), e.g. Qwen3.8-27B: mlx_lm's model, decoded by the lane engine.

On a GPU with Metal 4 tensor units (the M5 generation) TensorFold's lane kernels take over the matmuls and
attention (``kernels.lane_qmm``, ``kernels.lane_attention``, ``kernels.lane_fuse``): a verify window of up to
32 rows gives every row the bits of a one-row step, drafts are verified as trees (up to 15 nodes, chains of
up to 31 for copies and tool calls) and prompts are prefilled through the same kernels, so a snapshot of a
prompt has the bits decoding would give it. Drafts come from the context (copies of earlier spans, the known
structure of tool calls) and, with ``drafter``, from a DFlash2 draft model (``drafters.dflash_drafter``).

Without tensor units (the M1 to M4 generations) MLX's own kernels run, except that verify windows of 2 to 8 rows
go through TensorFold's row-exact matvec (``kernels.row_qmv``: MLX's one-row loop run for each row) and attend
query by query (``kernels.exact_attention``), so a drafted window reproduces one-row decoding there too. At load
the engine checks which window widths do on this Mac and times them, then drafts DFlash2 chains of the width
that pays (``install_mlx_lanes``). Where no width reproduces one-row decoding it serves without drafts.
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

# the lane engine's switches with lane kernels (measured on an M5 Max: see docs/recipes/qwen3.8-27b.md): trees of
# up to 15 drafted nodes, chains of up to 31 (copies, tool-call structure), prompts through the lane decoder in
# chains of 128, windows of at most 32 rows (exact), 16 without strong evidence, 16 rows a round
LANE_SETTINGS = {"tree_nodes": 15, "chain_nodes": 31, "lane_prefill": 128, "exact_window": 32, "cheap_window": 16,
                 "max_rows": 16, "max_draft": 15}


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


def _window_check(model: Any, widths: range) -> tuple[int, dict[int, float]]:
    """The widest verify window (among ``widths``) whose every narrower window reproduces one-row steps bit for
    bit on this Mac, and each width's forward time in ms (fastest of 3)."""

    import time

    import mlx.core as mx

    prompt = list(range(1000, 1032))

    def prefill() -> list[Any]:
        cache = model.make_cache()
        mx.eval(model(mx.array([prompt]), cache=cache))
        return cache

    cache = prefill()
    tokens, token = [], int(mx.argmax(model(mx.array([prompt[-1:]]), cache=model.make_cache())[0, -1]).item())
    for _ in range(max(widths)):
        tokens.append(token)
        logits = model(mx.array([[token]]), cache=cache)
        mx.eval(logits)
        token = int(mx.argmax(logits[0, -1]).item())
    cache = prefill()
    serial = []
    for t in tokens:
        logits = model(mx.array([[t]]), cache=cache)
        mx.eval(logits)
        serial.append(logits[0, -1])
    exact = 1
    for width in widths:
        logits = model(mx.array([tokens[:width]]), cache=prefill())
        mx.eval(logits)
        if not all(bool(mx.array_equal(logits[0, i], serial[i]).item()) for i in range(width)):
            break
        exact = width
    costs: dict[int, float] = {}
    for width in [1, *range(2, exact + 1)]:
        best = float("inf")
        for _ in range(3):
            cache = prefill()
            started = time.perf_counter()
            mx.eval(model(mx.array([tokens[:width]]), cache=cache))
            best = min(best, (time.perf_counter() - started) * 1e3)
        costs[width] = best
    return exact, costs


def install_mlx_lanes(model: Any) -> tuple[int, int]:
    """Drafted rounds on MLX's kernels (no tensor units): the row-exact matvec for 2-8-row windows, attention
    query by query, one stream's rounds rolled back to exactly their kept rows (so DFlash2 follows them), windows
    as wide as this Mac reproduces one-row decoding, and each round's DFlash2 drafts as many as pay at the
    stream's acceptance (``LaneEngine.window_costs``). Returns (exact width, exact width); (1, 1) when no window
    reproduces one-row decoding, and then nothing is drafted."""

    from tensorfold.engine.lane_engine import LaneEngine
    from tensorfold.kernels.qwen.dense.v1 import exact_attention, row_qmv

    if row_qmv.install(model) == 0:
        return 1, 1
    row_qmv.mlx_one_row = row_qmv.matches_mlx(model)
    exact_attention.install()
    exact, costs = _window_check(model, range(2, row_qmv.MAX_ROWS + 1))
    if exact < 2:
        exact_attention.GROUP_QUERIES = False
        exact, costs = _window_check(model, range(2, row_qmv.MAX_ROWS + 1))
    if exact < 2:
        return 1, 1
    LaneEngine.precise_single = True
    LaneEngine.exact_window = exact
    LaneEngine.cheap_window = exact
    LaneEngine.window_costs = costs
    timing = ", ".join(f"{w} rows {ms:.1f} ms" for w, ms in sorted(costs.items()))
    print(f"[tensorfold] no tensor units: drafted windows of up to {exact} rows reproduce one-row decoding here "
          f"({timing}); each round drafts as many as pay at the request's acceptance", flush=True)
    return exact, exact


def load(model_dir: Path, *, lane_kernels: str = "auto", **_: Any) -> tuple[Any, Any]:
    """The model with lane kernels installed when ``lane_kernels`` is "on", or "auto" on a GPU with tensor units."""

    from tensorfold.families import quantization, read_config

    model, tokenizer = load_lane_model(Path(model_dir))
    fits = quantization(read_config(model_dir)) == (4, 64)
    use = fits and (lane_kernels == "on" or (lane_kernels == "auto" and tensor_units()))
    model._tensorfold_lanes = bool(use)
    model._tensorfold_mlx_lanes = (1, 1)
    if use:
        install_lane_kernels(model)
        return model, tokenizer
    if not fits:
        print(f"[tensorfold] lane kernels need 4-bit weights in groups of 64 ({MODELS[0]}): MLX's kernels run",
              flush=True)
    else:
        print("[tensorfold] lane kernels off (they need an M5-generation GPU): MLX's kernels run", flush=True)
    model._tensorfold_mlx_lanes = install_mlx_lanes(model)
    if model._tensorfold_mlx_lanes[0] < 2:
        print("[tensorfold] no verify window reproduces one-row decoding with this MLX on this GPU: serving without "
              "drafts (same output, slower)", flush=True)
    return model, tokenizer


def install_lane_kernels(model: Any) -> None:
    """Swap in the lane matmul, fused projections and lane attention, compile every variant now (not inside the
    first requests) and set the lane engine's tree and prefill switches."""

    from tensorfold.engine.lane_engine import LaneEngine
    from tensorfold.kernels.qwen.dense.v1 import exact_attention, lane_attention, lane_fuse, lane_qmm

    LaneEngine.exact_window = LANE_SETTINGS["exact_window"]
    LaneEngine.cheap_window = LANE_SETTINGS["cheap_window"]
    exact_attention.install()      # verify windows attend query by query, as one-row steps do
    # TF_LANE_TILE=0 keeps MLX's weight layout (same bits, slower; for A/B timing)
    lane_qmm.install(model, rows=lane_qmm.MAX_ROWS, tile=os.environ.get("TF_LANE_TILE", "1") != "0", wide=True)
    warmed = lane_qmm.warm(model)
    lane_fuse.enabled = True
    fused = lane_fuse.build(model)          # stacks share the weights' memory: no second copy
    lane_fuse.warm(model)
    exact_attention.EXACT_MAX_QUERIES = lane_qmm.MAX_ROWS
    LaneEngine.precise_single = True
    lane_attention.install()
    lane_attention.warm(max_queries=lane_attention.MAX_QUERIES)
    LaneEngine.simple_masks = True
    LaneEngine.tree_nodes = LANE_SETTINGS["tree_nodes"]
    LaneEngine.lane_prefill = LANE_SETTINGS["lane_prefill"]
    LaneEngine.chain_nodes = LANE_SETTINGS["chain_nodes"]
    from tensorfold.drafters.dflash_drafter import DFlashProposer

    DFlashProposer.tree_nodes = LANE_SETTINGS["tree_nodes"]
    print(f"[tensorfold] lane kernels on: {warmed} matmul shapes warmed, fused projections {fused}, "
          f"trees of {LANE_SETTINGS['tree_nodes']} nodes, chains of {LANE_SETTINGS['chain_nodes']}", flush=True)


def engine_settings(model: Any) -> dict[str, Any]:
    """Keyword arguments for the lane engine (``max_rows``: rows a round verifies; ``max_draft``: drafts a stream
    offers a round)."""

    if not getattr(model, "_tensorfold_lanes", False):
        exact = getattr(model, "_tensorfold_mlx_lanes", (1, 1))[0]
        return {"max_rows": exact, "max_draft": exact - 1} if exact >= 2 else {}
    return {"max_rows": LANE_SETTINGS["max_rows"], "max_draft": LANE_SETTINGS["max_draft"]}


def kernel_version(model: Any) -> str:
    """Names the kernels that computed a prefix snapshot (a snapshot computed by other kernels has other bits)."""

    import hashlib

    if not getattr(model, "_tensorfold_lanes", False):
        if getattr(model, "_tensorfold_mlx_lanes", (1, 1))[0] >= 2:
            from tensorfold.kernels.qwen.dense.v1 import row_qmv

            source = Path(row_qmv.__file__).read_text()
            return "mlx-row-qmv-" + hashlib.sha256(source.encode()).hexdigest()[:12]
        return "mlx"
    from tensorfold.kernels.qwen.dense.v1 import lane_attention, lane_fuse, lane_glue, lane_qmm, lane_tree

    sources = [lane_qmm._MAIN, lane_qmm._MAIN_TILED, lane_qmm._XSUM, lane_attention._PARTIAL, lane_attention._TAIL,
               lane_attention._TREE_MERGE, lane_attention._MERGE, lane_glue._NORM_XS, lane_glue._GDN_PRE,
               lane_glue._GDN_POST, lane_glue._MLP_ACT, lane_tree._TREE_SOURCE, lane_tree._REPLAY_SOURCE,
               repr((lane_attention.CHUNK, lane_attention.TILE))]
    if lane_fuse.enabled:
        sources += [text for _, text in sorted(lane_fuse.sources().items())]
    folder = Path(lane_qmm.__file__).parent
    sources.extend(path.read_text() for path in sorted(folder.glob("*.py")))
    sources.extend(path.read_text() for path in sorted(Path(__file__).parent.glob("*.py")))
    return f"qwen-dense-{KERNEL_VERSION}-" + hashlib.sha256("\n".join(sources).encode()).hexdigest()[:12]


def setup(app: Any, model: Any, *, drafter: str = "", drafter_bits: int = 4, **_: Any) -> None:
    """A DFlash2 draft model (a directory) for the app's requests."""

    if not drafter:
        return
    if not getattr(model, "_tensorfold_lanes", False) and getattr(model, "_tensorfold_mlx_lanes", (1, 1))[0] < 2:
        print(f"[tensorfold] drafter {drafter} not loaded: drafted windows would not reproduce one-row decoding "
              f"here", flush=True)
        return
    import mlx.core as mx

    from tensorfold.drafters.dflash_drafter import DFlashDrafter

    loaded = DFlashDrafter(model, drafter, bits=int(drafter_bits))
    if getattr(model, "_tensorfold_lanes", False):
        from tensorfold.kernels.qwen.dense.v1 import lane_qmm

        lane_qmm.install(loaded.model, rows=lane_qmm.MAX_ROWS, tile=os.environ.get("TF_LANE_TILE", "1") != "0",
                         wide=True)    # the drafter's matmuls through the same kernels
        lane_qmm.warm(loaded.model)
        # the draft-vocabulary head is built and compiled here, not inside the first request
        hidden = mx.zeros((1, 16, int(loaded.model.config.hidden_size)), dtype=mx.bfloat16)
        mx.eval(*(a for a in loaded.candidate_logits(hidden) if a is not None))
    app.dflash = loaded
    print(f"[tensorfold] drafter {loaded.path} block={loaded.block_size} bits={drafter_bits or 16}", flush=True)


# the CUDA engine's kernels read MLX affine weights of this (bits, group size)
CUDA_QUANTIZATION = (4, 64)

def cuda_engine(model_dir: str | Path, *, drafter: str = "", tp: int = 1, rank: int = 0, master: str = "",
                master_port: int = 29551, no_drafts: bool = False, **options: Any):
    """The CUDA engine (``tensorfold serve`` on an NVIDIA GPU), set up as the recipe measured on DGX Spark.

    DFlash2 draft trees are verified in windows of 12 rows (the same drafts accepted as at 16 rows, for less
    time a round). On two GPUs (``tp=2``) the model is tensor parallel with fp32 partials summed in rank
    order, the head is split by vocabulary, and both ranks draft with half the draft model each, so both
    machines need it. ``no_drafts``: one token a round, the serial reference.
    """

    from .cuda.engine import Qwen27Engine

    draft = Path(drafter) if drafter and not no_drafts else None
    return Qwen27Engine(Path(model_dir), draft, max_rows=12, tp=tp, rank=rank, master=master, port=master_port,
                        split_head=tp == 2, tp_draft=tp == 2 and draft is not None, allow_copy=not no_drafts)
