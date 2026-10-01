"""Qwen3.8 Flash Next forward pass and fused decode with exact MTP drafting on Metal or one or two CUDA GPUs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

MODEL_TYPES = ("qwen4_exp", "qwen3_8_flash_next")   # the second: the name newer exports (Mia-AiLab's NVFP4) carry
TITLE = "Qwen3.8 Flash Next"
LANES = True
# with their MTP head: MLX affine (4-bit the default; oQ4e, oQ5e, 6- and 8-bit read too), EXL3 and NVFP4
MODELS = ("Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP", "turboderp/Qwen3.8-Flash-Next-exl3",
          "local-inference-lab/Qwen3.8-Flash-Next-NVFP4", "RadixArk/Qwen3.8-Flash-Next-NVFP4")
NVFP4_MODELS = MODELS[2:]
QUANT_METHODS = {"cuda": ("mlx", "exl3", "modelopt")}  # MLX affine 4-bit, EXL3 packs and NVFP4 (ModelOpt)
EXL3_VARIANT = "any"                           # every EXL3 codebook and width (tensorfold.families.EXL3_VARIANT_ANY)
KERNEL_PACKAGE = "tensorfold.kernels.qwen.flash_next.v1"
KERNEL_VERSION = "v1"
# The CLI sets these before MLX starts, respecting environment overrides, to keep expert bindings from ending each command buffer.
MLX_ENV = {"MLX_MAX_OPS_PER_BUFFER": "200", "MLX_MAX_MB_PER_BUFFER": "100000"}


def has_mtp(model_dir: Path) -> bool:
    """Whether the checkpoint kept the MTP head's weights (``mtp.*``)."""

    import json

    index = Path(model_dir) / "model.safetensors.index.json"
    if not index.is_file():
        return False
    return any(".mtp." in name or name.startswith("mtp.") for name in json.loads(index.read_text())["weight_map"])


def check_quantization(config: dict[str, Any], backend: str) -> None:
    """Refuse from config.json alone an MLX format the kernels do not read: affine 2-8 bits in groups of 32-128."""

    from tensorfold.quantization import checkpoint_specs

    specs = [spec for spec in checkpoint_specs(config).values() if spec is not None]
    if not specs:
        raise ValueError(f"{TITLE} reads MLX affine-quantized weights; this checkpoint has none. Use {MODELS[0]}.")
    formats = {(spec.bits, spec.group_size) for spec in specs}
    if backend == "cuda" and formats != {CUDA_QUANTIZATION}:
        bits, group = CUDA_QUANTIZATION
        raise ValueError(f"{TITLE}'s CUDA kernels read {bits}-bit weights in groups of {group}; this checkpoint has "
                         f"{', '.join(f'{b}-bit g{g}' for b, g in sorted(formats))}. Use {MODELS[0]}.")


def check(model_dir: Path) -> None:
    from tensorfold.families import EXL3_QUANT, OWN_MODEL_HELP, describe_quantization, quant_method, read_config

    config = read_config(model_dir)
    if quant_method(config) == EXL3_QUANT:
        # an EXL3 pack (the CUDA engine, any codebook and per-tensor width): only the MTP head to report
        if (Path(model_dir) / "model.safetensors.index.json").is_file() and not has_mtp(model_dir):
            print("[tensorfold] this EXL3 checkpoint has no MTP head: decoding without MTP drafts", flush=True)
        return
    if quant_method(config) == "modelopt":
        # the CUDA engine's NVFP4 route: NVFP4 experts in blocks of 16, other linears bf16, MXFP8, block FP8 or NVFP4
        found = config.get("quantization") or config.get("quantization_config") or {}
        algo = str(found.get("quant_algo") or "NVFP4").upper()
        layers = {str(v.get("quant_algo", "")).upper() for v in (found.get("quantized_layers") or {}).values()}
        algos = layers if algo == "MIXED_PRECISION" else {algo}
        weights = [g.get("weights") or {} for g in (found.get("config_groups") or {}).values()]
        fp4 = {int(w.get("group_size", 16)) for w in weights if int(w.get("num_bits", 4)) == 4}
        if not algos <= {"NVFP4", "W4A16_NVFP4", "MXFP8", "FP8_PB_WO"} or fp4 - {16}:
            raise ValueError(f"TensorFold's Flash Next kernels read NVFP4 (ModelOpt FP4) weights in blocks of 16, the "
                             f"other linears bf16, MXFP8 or 128x128-block FP8 ({', '.join(NVFP4_MODELS)}); this checkpoint has "
                             + describe_quantization(config) + f". {OWN_MODEL_HELP}")
        if (Path(model_dir) / "model.safetensors.index.json").is_file() and not has_mtp(model_dir):
            print("[tensorfold] this NVFP4 checkpoint has no MTP head: decoding without MTP drafts", flush=True)
        return
    check_quantization(config, "mlx")
    # Config-only preflight cannot establish whether the MTP head is missing; wait for weights or their index.
    if ((Path(model_dir) / "model.safetensors.index.json").is_file()
            or any(Path(model_dir).glob("model*.safetensors"))) and not has_mtp(model_dir):
        print(f"[tensorfold] this checkpoint has no MTP head: decoding without MTP drafts ({MODELS[0]} has one)",
              flush=True)


# a table's tensors in the checkpoint: weights, scales and biases of each shard, the only bytes read on the host
_TABLE = r"language_model\.model\.layers\.\d+\.ple\.ple_embedding\.ngram_embedding\.shard_\d+\.(weight|scales|biases)"


def ple_bytes(model_dir: Path) -> int:
    """Bytes of the checkpoint's n-gram (PLE) tables, which stay on the host (mapped, or on SSD with --ple-on-ssd)."""

    import re

    from tensorfold.families.qwen4_exp.host_table import read_header

    return sum(entry["data_offsets"][1] - entry["data_offsets"][0]
               for path in Path(model_dir).glob("model*.safetensors")
               for name, entry in read_header(path).items() if re.fullmatch(_TABLE, name))


def expert_bytes(model_dir: Path) -> int:
    """Bytes of the decoder layers' routed expert stacks, which --ssd-experts leaves on disk."""

    from tensorfold.streaming.checkpoint import tensor_bytes

    return tensor_bytes(Path(model_dir), lambda name: name.startswith("language_model.model.layers.")
                        and ".mlp.switch_mlp." in name)


def weight_bytes(model_dir: Path, ple_on_ssd: bool = False) -> int:
    """The bytes MLX loads: the checkpoint, less its n-gram tables where the loader keeps them on the host."""

    from tensorfold.families.qwen4_exp.host_table import ngrams_on_host

    size = sum(p.stat().st_size for p in Path(model_dir).glob("*.safetensors"))
    return size - ple_bytes(model_dir) if ngrams_on_host(model_dir, ple_on_ssd) else size


def load(model_dir: Path, *, mtp_drafts: int | None = None, ple_on_ssd: bool = False,
         ssd_experts: float | None = None, **_: Any) -> tuple[Any, Any]:
    from tensorfold.families.qwen4_exp.runtime import load as load_runtime

    drafts = mtp_drafts if has_mtp(Path(model_dir)) else 0
    return load_runtime(Path(model_dir), drafts=drafts, ple_on_ssd=ple_on_ssd, ssd_experts=ssd_experts)


def engine_settings(model: Any) -> dict[str, Any]:
    """The widest exact window, prompt chunks the memory allows (8,192 where tensor units run MLX's gathers)."""

    from tensorfold.families.qwen3_5 import tensor_units

    width = int(getattr(model, "exact_width", 1) or 1)
    steps = (8192, 4096, 2048) if tensor_units() else (4096, 2048)
    return {"max_rows": width, "max_draft": max(0, width - 1), "prefill_steps": steps}


def kernel_version(model: Any) -> str:
    """Include the loaded model's prompt-attention selection modes in its kernel fingerprint."""

    from tensorfold.families import families, kernel_source_version

    modes = [str(int(bool(getattr(layer.self_attn, "kernel_select", False))))
             for layer in getattr(model, "layers", ()) if hasattr(layer, "self_attn")]
    source = kernel_source_version(families()["qwen4_exp"])
    prefill = getattr(model, "prefill_key", None)                     # the prefill path, matmul route and GPU
    return f"{source}|prompt_attention={','.join(modes)}" + (f"|{prefill}" if prefill else "")


# the CUDA engine reads MLX affine weights of this (bits, group size), or NVFP4 (ModelOpt) routed experts
CUDA_QUANTIZATION = (4, 32)
# the KV cache dtypes the CUDA engine can allocate (``--kv-dtype``)
CUDA_KV_DTYPES = ("bf16", "int8", "int4")
CUDA_DECODE_SHARE = True           # --parallel rounds size their prompt pass by --decode-share (0: whole passes)
CUDA_PREFILL_FP8 = True            # --prefill-fp8: an NVFP4 checkpoint's MXFP8 linears have an FP8 prompt kernel

def cuda_engine(model_dir: str | Path, *, drafter: str = "", tp: int = 1, rank: int = 0, master: str = "",
                master_port: int = 29551, no_drafts: bool = False, mtp_drafts: int | None = None,
                mtp_confidence: float | None = None, context: int | None = None, ple_on_ssd: bool = False,
                kv_dtype: str = "bf16", decode_share: float | None = None, **options: Any):
    """The CUDA engine: MTP chains verified exactly on one GPU or two (``tp=2``; start rank 1 first), keys and values bf16, int8 or int4."""

    from tensorfold.cuda.exl3.format import is_exl3

    if is_exl3(Path(model_dir)):
        print("[tensorfold] EXL3 packs are experimental: replies are exact; see "
              "docs/recipes/qwen3.8-flash-next.md#exl3-checkpoints-experimental for how they compare", flush=True)
    if drafter:
        raise ValueError(f"{TITLE} drafts with its own MTP head on CUDA: a separate draft model does not apply")
    from .cuda import CONFIDENCE, DEPTH
    from .cuda.engine import FlashNextEngine

    if kv_dtype not in CUDA_KV_DTYPES:       # refuse an unknown cache before any weight is read (no torch import)
        raise ValueError(f"kv-dtype {kv_dtype!r}: this engine serves {' or '.join(CUDA_KV_DTYPES)}")
    depth = 0 if no_drafts else DEPTH if mtp_drafts is None else int(mtp_drafts)
    if depth and not has_mtp(Path(model_dir)):
        raise ValueError(f"this checkpoint has no MTP head, which {TITLE}'s CUDA engine drafts with ({MODELS[0]} "
                         "has one): without it every round would decode one token. Serve a checkpoint with the "
                         "head, or pass --no-drafts for the serial reference")
    confidence = CONFIDENCE if mtp_confidence is None else float(mtp_confidence)
    return FlashNextEngine(Path(model_dir), depth=depth, confidence=confidence, max_len=context,
                           context_explicit=options.get("context_explicit"), tp=int(tp), rank=int(rank),
                           master=master, port=int(master_port), streams=max(1, int(options.get("parallel") or 1)),
                           ple_on_ssd=ple_on_ssd, kv_dtype=kv_dtype,
                           share=0.0 if decode_share is None else float(decode_share),
                           vision=bool(options.get("vision", False)),
                           vision_urls=bool(options.get("vision_urls", False)))
