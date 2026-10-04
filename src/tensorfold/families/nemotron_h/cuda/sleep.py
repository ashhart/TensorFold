"""Nemotron-H sleep hooks: attention, Mamba state, and its integrated MTP head."""

from dataclasses import asdict

from tensorfold.cuda.prefix_store import prefix_room
from tensorfold.cuda.streams import PrefixCache

FAMILY = "nemotron_h"


def settings(engine):
    return {"context_window": engine.context_window, "max_len": engine.max_len,
            "max_rows": engine.e.max_rows, "tp": engine.tp, "rank": engine.rank,
            "drafts": engine.drafts, "confidence": engine.confidence,
            "draft_tau": engine.mtp.params.tau if engine.mtp is not None else None,
            "draft_ids": engine.draft_ids, "config": asdict(engine.e.c)}


def prefix_codec():
    from . import prefix_snapshot

    return prefix_snapshot


def cache(engine):
    kept = PrefixCache(2)
    kept.entries = [(ids, state, None) for ids, state in engine.cache]
    return kept


def attach(engine, store):
    engine.sleep_cache = store


def restore_prefix(engine, prompt, best):
    entry = engine.sleep_cache.restore(prompt, length=len(best[0]) if best else 0,
                                       device=engine.e.device, admit=prefix_room)
    return (entry[0], entry[1]) if entry is not None else best


def close(engine):
    # No scheduler thread owns this family. The shared adapter then drops the
    # engine, including its weight-capturing factory and lazy serial twin.
    engine.shutdown()
