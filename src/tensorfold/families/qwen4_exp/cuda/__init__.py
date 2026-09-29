"""CUDA Flash Next verify rows match this backend's serial bits, which can differ from the Mac backend's."""

DEPTH = 6            # most MTP drafts a round
CONFIDENCE = 0.3     # a chain ends before a draft the MTP head gives less than this
CONTEXT = 8192       # prompt plus reply tokens the caches hold


def draft_cap(depth: int, room: int, ema_accept: float) -> int:
    """How many drafts to propose next: at most ``depth`` / ``room``, at least 1 when room allows.

    Scales to the recent accepted-draft EMA so low-accept (open-ended) streams skip wasted verify
    rows, while high-accept streams still climb toward the configured cap. Does not change emitted
    tokens — verify still exact-matches whatever was proposed.
    """

    if room <= 0 or depth <= 0:
        return 0
    # one exploration slot past the recent accept count; floor at 1 so MTP stays on
    want = max(1, min(depth, int(ema_accept) + 1))
    return min(want, room)