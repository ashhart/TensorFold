"""Nemotron-H on CUDA: rows get the same bits alone or in a verify window; serial means these kernels, not the Mac's."""

DRAFTS = 3           # MTP drafts a round
CONFIDENCE = 0.2      # verify the drafts while their running head confidence stays at or above this
CONTEXT = 16384       # prompt plus reply tokens when --context is not given (snapshots copy whole caches)
