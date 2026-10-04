"""DeepSeek-V4.1-Flash on two GPUs: row-invariant kernels, fp32 partials gathered and added in rank order."""

# M:n and K:n in this package cite lines of inference/model.py and inference/kernel.py of
# deepseek-ai/DeepSeek-V4.1-Flash at revision dba1be0

MAX_ROWS = 6                        # a verify window's rows: the target token and up to BLOCK drafts
GRAPH_ROWS = tuple(range(1, MAX_ROWS + 1))
BLOCK = 5                           # rows of one DSpark block (the most drafts a round)
RING = 128                          # sliding-window slots per layer (the checkpoint's window)
PREFILL_ROWS = 2048                 # rows of one prompt chunk
MOE_WINDOW = 1024                   # prompt rows per routed-expert pass
SCORE_BYTES = 1 << 30               # the indexer's score scratch, the cap on its prompt row blocks
DEFAULT_DRAFTS = 3                  # drafts a round unless --mtp-drafts or --mtp-confidence asks otherwise
