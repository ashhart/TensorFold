"""GLM's universal EXL3 route uses buffer-owned scratch and the existing bf16 expert epilogue."""

from tensorfold.cuda.exl3 import experts as generic


def routed(x, pick, ex, scratch, rows, limit):
    """Write the live routed pairs into scratch.y; the caller supplies each row's shared-expert slot."""

    return generic.routed(x, pick, None, ex, scratch, None, rows, limit, act_mode=generic.ACT_BF16)
