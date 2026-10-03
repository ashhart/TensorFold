"""Nemotron-H on CUDA: rows get the same bits alone or in a verify window; serial means these kernels, not the Mac's."""

MAX_CHAIN = 15       # drafts a window can hold beside the pending token (16 rows)
DRAFTS = 15          # the most MTP drafts a round; the depth rule stops each chain where a row stops paying
CONFIDENCE = None     # None: the measured-cost depth rule; a probability: verify while the running confidence holds it
DRAFT_TAU = 0.6       # sampled drafts are drawn at this fraction of the request's temperature (top_k kept, no top-p)
CALIBRATION = (2.0, 2.0)  # the depth rule's power on the head's overconfident running confidence (greedy, sampled)
CONTEXT = 16384       # prompt plus reply tokens when --context is not given (snapshots copy whole caches)
