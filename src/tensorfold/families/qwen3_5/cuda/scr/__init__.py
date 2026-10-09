"""Suffix Cache Reuse (SCR) for the dense Qwen3.8 CUDA engine.

A conversation whose next prompt edits its context in the middle (an agent rewriting
its own transcript, or a chat template that strips prior reasoning blocks) would
otherwise re-prefill every token after the first mismatch. SCR keeps the previous
turn's KV as a private per-session snapshot, diffs the new prompt against the
previous turn's committed tokens, and relocates the KV of up to K surviving spans
into their new positions — re-rotating keys for the position shift — prefilling
only the genuinely new text between spans.

The engine's own exact-resume path (the prefix cache) is untouched: sessions add a
second reuse path beside it, off by default (``TENSORFOLD_SCR_ENABLED=1``). Every
failure falls back to plain serving at the point of failure; rows already committed
stay valid because the plan builder verifies token identity before any relocation.

Module map:
    config    environment knobs (read once per runtime)
    diff      token diff (difflib + content-defined-chunk fast path, verified)
    plan      relocation plan builder and its gates
    sessions  per-conversation snapshots (LRU, byte-budgeted, pinned in flight)
    splice    the relocation primitive over a ``State`` (key re-rotation included)
    state     snapshot -> ``State`` construction for planned streams
    routing   the engine hooks (MultiDecoder, the serial Qwen27Engine path)
    report    log accounting (``prompt_tokens = base + relocated + forwarded``)

Provenance: the relocation scheme is ported from Meta's `suffix_cache_reuse`
(facebookresearch/context-language-models); the TensorFold integration is new.
"""

from .config import Config, _log