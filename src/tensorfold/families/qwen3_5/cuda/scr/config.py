"""Environment knobs (read once per runtime)."""

import os


def _log(msg: str) -> None:
    print(f"[scr] {msg}", flush=True)


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "on", "yes")


def enabled() -> bool:
    """Off by default: a plain serving boot never touches a snapshot."""
    return _bool("TENSORFOLD_SCR_ENABLED", False)


class Config:
    """Snapshot of the environment knobs; created once per runtime."""

    def __init__(self) -> None:
        self.enabled = enabled()
        # relocation shape
        self.max_blocks = max(1, _int("TENSORFOLD_SCR_MAX_BLOCKS", 6))       # K, spans relocated per request
        self.min_tail = max(1, _int("TENSORFOLD_SCR_MIN_TAIL", 64))          # smallest surviving span worth relocating
        self.min_extend = max(1, _int("TENSORFOLD_SCR_MIN_EXTEND", 16))      # span-tail tokens always prefilled
        self.mb_min_gap = max(0, _int("TENSORFOLD_SCR_MB_MIN_GAP", 0))       # min fill between two spans (0: off)
        self.min_base = max(1, _int("TENSORFOLD_SCR_MIN_BASE", 1024))        # shared leading prefix a plan requires
        self.min_gain = max(1, _int("TENSORFOLD_SCR_MIN_GAIN", 1024))        # relocated tokens a plan must save
        # diff
        self.fastdiff = _bool("TENSORFOLD_SCR_FASTDIFF", True)
        self.cdc_target = _int("TENSORFOLD_SCR_CDC_TARGET", 64)
        self.fastdiff_min = _int("TENSORFOLD_SCR_FASTDIFF_MIN", 2048)
        # sessions
        self.max_sessions = max(1, _int("TENSORFOLD_SCR_MAX_SESSIONS", 16))  # LRU cap
        self.side_gib = _float("TENSORFOLD_SCR_SIDE_GIB", 8.0)               # snapshot byte budget per engine
        self.match_window = _int("TENSORFOLD_SCR_MATCH_WINDOW", 8192)
        self.match_min_tokens = _int("TENSORFOLD_SCR_MATCH_MIN_TOKENS", 1024)
        self.match_min_prefix_frac = _float("TENSORFOLD_SCR_MATCH_MIN_PREFIX_FRAC", 0.5)
        self.match_min_overlap = _float("TENSORFOLD_SCR_MATCH_MIN_OVERLAP", 0.5)
        self.match_min_ratio = _float("TENSORFOLD_SCR_MATCH_MIN_RATIO", 0.55)
        self.match_ambig_margin = _float("TENSORFOLD_SCR_MATCH_AMBIG_MARGIN", 0.15)
        # linear-attention layers: fork | strict | none
        self.ssm_mode = os.environ.get("TENSORFOLD_SCR_SSM_MODE", "fork").strip().lower()
        # logging
        self.trace = _bool("TENSORFOLD_SCR_TRACE", True)
        self.debug = _bool("TENSORFOLD_SCR_DEBUG", False)