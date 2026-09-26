"""GLM-5.3-Flash (model_type ``glm5_next``): an MLX engine on one Apple Silicon Mac, and a CUDA engine tensor
parallel over two DGX Sparks.

45 decoder layers over a hidden size of 4,096: 34 of Kimi delta attention and 11 of DeepSeek sparse attention
(MLA with an indexer), 288 routed experts (top 8) plus a shared expert, four residual streams mixed by
hyper-connections, a 154,880-token vocabulary and an MTP layer. The 4-bit checkpoint is 182 GB.

On a Mac (``load``): ``model`` is TensorFold's forward pass for the checkpoint, with a decode path whose rows each
get one-row bits; ``tensorfold.kernels.glm.flash.v1`` holds its Metal kernels; ``mtp`` is the checkpoint's MTP
layer; ``runtime`` is what the serial engine serves (exact MTP drafting when the load-time row check passes). It
needs a Mac with 256 GB or more.

On NVIDIA GPUs (``cuda_engine``): each Spark holds half of every layer (``cuda/``). Drafts come from the
checkpoint's MTP head and, when it has been pulled on both machines, from the DFlash2 draft model.
Recipe and measurements for both: docs/recipes/glm-5.3-flash.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

MODEL_TYPES = ("glm5_next",)
TITLE = "GLM-5.3-Flash"
LANES = False
# 4-bit weights in groups of 64 (what the Metal and CUDA kernels read), with the checkpoint's MTP layer kept
MODELS = ("Vontra/GLM-5.3-Flash-MLX-4bit-MTP",)
DRAFTER = "incoai/GLM-5.3-Flash-DFlash2"   # the CUDA engine's optional draft model; the Mac engine drafts with MTP
KERNEL_PACKAGE = "tensorfold.kernels.glm.flash.v1"
KERNEL_VERSION = "v1"
# MLX command buffers: GLM's decode step is about 1,200 kernels a token (set before MLX starts, as for Flash Next)
MLX_ENV = {"MLX_MAX_OPS_PER_BUFFER": "200", "MLX_MAX_MB_PER_BUFFER": "100000"}


def check(model_dir: str | Path) -> None:
    """Both engines read MLX affine 4-bit weights in groups of 64."""

    import sys

    from tensorfold.families import quantization, read_config

    bits, group = quantization(read_config(model_dir))
    if (bits, group) != (4, 64):
        raise ValueError(f"TensorFold's GLM-5.3-Flash kernels read 4-bit weights in groups of 64 ({MODELS[0]}), "
                         f"this checkpoint has {bits}-bit, groups of {group}")
    if sys.platform == "darwin":
        from tensorfold.families.glm5_next.mtp import has_mtp

        if (Path(model_dir) / "model.safetensors.index.json").is_file() and not has_mtp(model_dir):
            print(f"[tensorfold] this checkpoint has no MTP layer: decoding without MTP drafts ({MODELS[0]} has "
                  f"one)", flush=True)
        return
    print("[tensorfold] GLM-5.3-Flash runs on two NVIDIA GPUs with 128 GB each (two DGX Sparks): pull it on both "
          "and serve with --tp 2 on both (docs/recipes/glm-5.3-flash.md)", flush=True)


def load(model_dir: Path, *, mtp_drafts: int | None = None, **_: Any) -> tuple[Any, Any]:
    """The MLX engine. ``mtp_drafts``: the most MTP drafts a round (default 1; 0: none). With more than 1 the
    depth follows each draft position's measured acceptance (``runtime.GLMFlash.depth_for``)."""

    import mlx.core as mx

    from tensorfold.families.glm5_next.runtime import load as load_runtime

    # about 170 GB of weights on a 256 GB Mac: keep them wired, or macOS can page them out between steps
    if mx.metal.is_available():
        info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
        limit = int(info.get("max_recommended_working_set_size", 0))
        if limit:
            mx.set_wired_limit(limit)
    return load_runtime(Path(model_dir), drafts=mtp_drafts)


def kernel_version(model: Any) -> str:
    """Names the kernels that computed a prefix snapshot: the MLX engine's and the kernel package's sources, the
    MLX version (MLX's own kernels compute the prefill) and the switches that change the decode path's arithmetic."""

    import hashlib
    import importlib

    import mlx.core as mx

    from tensorfold.families.glm5_next import model as glm

    digest = hashlib.sha256()
    for module in (__name__, KERNEL_PACKAGE):
        folder = Path(str(importlib.import_module(module).__file__)).parent
        for path in sorted(folder.glob("*.py")):       # this folder only: the CUDA engine (cuda/) is not on this path
            digest.update(path.relative_to(folder).as_posix().encode())
            digest.update(path.read_bytes())
    digest.update(mx.__version__.encode())
    digest.update(f"fused_kda={glm.FUSED_KDA} sparse={glm.SPARSE_KERNEL} fused={sorted(glm.FUSED)}".encode())
    return f"{MODEL_TYPES[0]}-{KERNEL_VERSION}-" + digest.hexdigest()[:12]


def cuda_engine(model_dir: str | Path, *, drafter: str = "", tp: int = 1, rank: int = 0, master: str = "",
                master_port: int = 29551, no_drafts: bool = False, mtp_drafts: int | None = None, **options: Any):
    """The CUDA engine, set up as the recipe measured on two DGX Sparks.

    Each rank reads its half of the checkpoint (``cuda/split.py``) and the ranks all-gather fp32 partials every
    layer. By default a greedy request drafts each round with the MTP head or, with ``drafter`` (both machines),
    DFlash2, whichever has committed more tokens per millisecond so far; a sampled request drafts with the MTP
    head, 1 to 3 drafts a round from the running acceptance. A request can ask for another policy
    (``cuda/app.py``, specs in ``cuda/engine.py``). A prompt that extends the last request's prompt or reply
    resumes from its kept state.
    ``mtp_drafts``: a fixed number of MTP drafts a round instead (0: serial). ``no_drafts``: serial decoding only.
    ``options["context"]``: prompt plus reply tokens; up to 2,051 (the default) attention stays dense, as measured;
    longer contexts run DSA's sparse top-k past 2,051 tokens without CUDA graphs.
    """

    if int(tp) != 2:
        raise ValueError("GLM-5.3-Flash needs two GPUs, one per machine: run the same `tensorfold serve` command "
                         "with --tp 2 --rank R --master ADDRESS on both (rank 1 first)")
    if not master:
        raise ValueError("--tp 2 needs --master: rank 0's address on the link between the two machines")
    from .cuda.engine import DEFAULT_POLICY, GlmEngine

    policy = DEFAULT_POLICY if mtp_drafts is None else str(int(mtp_drafts))
    return GlmEngine(Path(model_dir), rank=int(rank), master=master, port=int(master_port), policy=policy,
                     drafter=Path(drafter) if drafter and not no_drafts else None,
                     context=int(options.get("context") or 0), serial_only=bool(no_drafts))


def __getattr__(name: str) -> Any:
    if name == "CUDA_APP":             # imported on first use, so the Mac side never loads the CUDA server
        from .cuda.app import GlmApp

        return GlmApp
    raise AttributeError(name)
