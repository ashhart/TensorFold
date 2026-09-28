"""Qwen3.6 MoE on CUDA: MTP drafting settings."""

DEPTH = 3            # most MTP drafts a round
CONFIDENCE = 0.3     # a chain ends after a draft the MTP head gives less than this
