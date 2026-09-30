"""CUDA Flash Next verify rows match this backend's serial bits, which can differ from the Mac backend's."""

DEPTH = 6            # most MTP drafts a round
CONFIDENCE = 0.7     # a chain ends before a draft the MTP head gives less than this (one stream or many)
CONTEXT = 8192       # prompt plus reply tokens the caches hold
