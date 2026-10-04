"""DeepSeek-V4.1-Flash (model_type ``deepseek_v41``): a CUDA engine over two DGX Sparks, drafting with DSpark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

MODEL_TYPES = ("deepseek_v41",)
TITLE = "DeepSeek-V4.1-Flash"
LANES = True
MODELS = ("sfxnz/DeepSeek-V4.1-Flash-EXL3",)
QUANT_METHODS = {"cuda": ("exl3",)}
EXL3_VARIANT = "any"                # the routed experts' codebook and width are read from their tensors
# everything but the routed experts: DeepSeek's FP8 with a power-of-two scale per 32x32 block
NON_ROUTED = {"quant_method": "deepseek_v4_fp8", "fmt": "e4m3", "scale_fmt": "ue8m0", "weight_block_size": [32, 32]}


def check(model_dir: str | Path) -> None:
    """Refuse from config.json, and from the index once it is downloaded, what the CUDA engine does not read."""

    import json
    import re

    from tensorfold import families

    from .config import Config

    config, advice = families.read_config(model_dir), families.OWN_MODEL_HELP
    if families.model_type(model_dir) != MODEL_TYPES[0]:
        raise ValueError(f"{TITLE} reads model_type {MODEL_TYPES[0]}, not {families.model_type(model_dir)!r}")
    if families.quant_method(config) != families.EXL3_QUANT:
        raise ValueError(f"{TITLE}'s CUDA engine reads EXL3 routed experts and DeepSeek FP8 elsewhere ({MODELS[0]}); "
                         f"this checkpoint has {families.describe_quantization(config)}. {advice}")
    found = config.get("quantization_config") or {}
    dense = found.get("non_routed_quantization") or {}
    if {k: dense.get(k) for k in NON_ROUTED} != NON_ROUTED:
        raise ValueError(f"{TITLE}'s CUDA engine reads the non-routed weights as DeepSeek FP8 (e4m3, ue8m0 scales per "
                         f"32x32 block); this checkpoint has " + ", ".join(f"{k} {dense.get(k)}" for k in NON_ROUTED)
                         + f". {advice}")
    if found.get("mtp_experts") != "source":
        raise ValueError(f"{TITLE} drafts with the DSpark experts as DeepSeek ships them (mtp_experts \"source\"); "
                         f"this checkpoint has mtp_experts {found.get('mtp_experts')!r}, and requantized drafts are "
                         f"rarely accepted")
    cfg = Config.from_dict(config)
    if cfg.num_attention_heads % 2 or cfg.vocab_size % 2 or cfg.moe_intermediate_size % 256:
        raise ValueError(f"{TITLE} splits attention heads and the vocabulary in two and each expert's intermediate "
                         f"size in halves of a multiple of 128; this checkpoint has {cfg.num_attention_heads} heads, "
                         f"vocabulary {cfg.vocab_size}, intermediate size {cfg.moe_intermediate_size}")
    block = cfg.candidate_block_size
    if cfg.candidate_source_layer_id >= 0 and cfg.candidate_topk_blocks * block - (block - 1) < cfg.index_topk:
        raise ValueError(f"{TITLE}: {cfg.candidate_topk_blocks} candidate blocks of {block} cannot hold the "
                         f"{cfg.index_topk} entries each indexer keeps, so its selection would be undefined")
    index = Path(model_dir) / "model.safetensors.index.json"
    if index.is_file():
        names = json.loads(index.read_text())["weight_map"]
        if "lm_head.weight_scale" not in names:
            raise ValueError(f"{TITLE}'s CUDA engine reads an MXFP8 LM head (lm_head.weight_scale), as {MODELS[0]} "
                             f"stores it; this checkpoint has none. {advice}")
        routed = re.compile(r"layers\.\d+\.ffn\.experts\.\d+\.w[123]\.trellis")
        other = sorted(name for name in names if name.endswith(".trellis") and not routed.fullmatch(name))
        if other:
            raise ValueError(f"{TITLE}'s CUDA engine reads EXL3 for the backbone's routed experts only; this "
                             f"checkpoint stores {len(other)} other tensor(s) as EXL3, {other[0]} first. {advice}")
    print(f"[tensorfold] {TITLE} runs on two NVIDIA GPUs with 128 GB each (two DGX Sparks): pull it on both and "
          "serve with --tp 2 on both (docs/recipes/deepseek-v4.1-flash.md)", flush=True)


def cuda_engine(model_dir: str | Path, *, drafter: str = "", tp: int = 1, rank: int = 0, master: str = "",
                master_port: int = 29551, no_drafts: bool = False, mtp_drafts: int | None = None,
                mtp_confidence: float | None = None, **options: Any):
    """The two-rank engine drafting with its DSpark stages; ``mtp_drafts`` 0 or ``no_drafts``: serial decoding."""

    from .cuda import BLOCK, DEFAULT_DRAFTS

    if int(tp) != 2:
        raise ValueError(f"{TITLE} needs two GPUs, one per machine: run the same `tensorfold serve` command with "
                         "--tp 2 --rank R --master ADDRESS on both (rank 1 first)")
    if not master:
        raise ValueError("--tp 2 needs --master: rank 0's address on the link between the two machines")
    if drafter:
        raise ValueError(f"{TITLE} drafts with its own DSpark stages: a separate draft model does not apply")
    drafts = DEFAULT_DRAFTS if mtp_drafts is None else int(mtp_drafts)
    if not 0 <= drafts <= BLOCK:
        raise ValueError(f"--mtp-drafts {drafts}: {TITLE} drafts 0 (serial decoding) to {BLOCK} tokens a round")
    streams = int(options.get("parallel") or 1)
    if streams > 1:
        print(f"[tensorfold] {TITLE} serves one request at a time: --parallel {streams} is ignored", flush=True)
    from .cuda.engine import DeepSeekV41Engine

    confidence = None if mtp_confidence is None else float(mtp_confidence)
    # the policy: drafts a round (0: serial), and the confidence product below which a chain stops (None: fixed)
    return DeepSeekV41Engine(Path(model_dir), rank=int(rank), master=master, port=int(master_port),
                             policy=(drafts, confidence), context=options.get("context"),
                             context_explicit=options.get("context_explicit"), serial_only=bool(no_drafts))


def __getattr__(name: str) -> Any:
    if name == "CUDA_APP":             # imported on first use, so a Mac's family discovery never loads the server
        from .cuda.app import DeepSeekV41App

        return DeepSeekV41App
    raise AttributeError(name)
