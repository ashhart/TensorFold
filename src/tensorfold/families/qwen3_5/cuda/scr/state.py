"""Snapshot <-> State conversion.

``Snapshot``s are private deep copies (never views into a stream's or the prefix
cache's buffers — ``clone_state``-style sharing would let a resumed stream's
writes corrupt the snapshot, the same hazard ``MultiDecoder._drop_extensions``
guards against). A planned stream's ``State`` is built from a snapshot: rows
``[0, base)`` verbatim, ``pos = base``, private buffers sized for the whole
window up front (so prefill never reallocates under CUDA-graph staging), and
linear-attention state forked from the snapshot's end-of-turn values."""

import torch

from .sessions import Snapshot


def tensor_bytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def snapshot_from_state(st, ids, log=None) -> Snapshot:
    """A private snapshot of a finished stream: attention rows ``[0, min(pos, len(ids)))``
    plus the linear-attention states. ``st`` may share buffers with the prefix
    cache — only reads happen here."""
    pos = min(int(st.pos), len(ids))
    kv, rec, conv = [], [], []
    total = 0
    try:
        for i, pair in enumerate(st.kv):
            if pair is None:
                kv.append(None)
                r = st.rec[i] if i < len(st.rec) else None
                c = st.conv[i] if i < len(st.conv) else None
                rec.append(None if r is None else r.detach().clone())
                conv.append(None if c is None else c.detach().clone())
                total += sum(tensor_bytes(t) for t in (rec[-1], conv[-1]) if t is not None)
                continue
            k, v = pair
            kk = k[:pos].detach().clone()
            vv = v[:pos].detach().clone()
            kv.append((kk, vv))
            total += tensor_bytes(kk) + tensor_bytes(vv)
            rec.append(None)
            conv.append(None)
    except Exception as e:                                        # noqa: BLE001
        if log:
            log(f"snapshot failed: {e}")
        return None
    return Snapshot(ids=list(ids[:pos]), pos=pos, kv=kv, rec=rec, conv=conv, bytes=total)


def build_state_from_snapshot(snap: Snapshot, rows: int, w, room=None, state_factory=None,
                              pos: int | None = None):
    """A fresh ``State`` holding the snapshot's rows ``[0, pos)`` verbatim, private
    buffers of ``rows`` rows, and forked linear-attention state (``SCR_SSM_MODE``
    fork is inherent here: the snapshot's end-of-turn state serves the whole
    planned turn, including spans that survive an edit). ``pos`` defaults to the
    snapshot's full length; a plan passes its ``base`` — only the shared leading
    prefix is served verbatim. ``state_factory`` overrides the engine's ``State``
    in tests."""
    if state_factory is None:
        from tensorfold.families.qwen3_5.cuda.forward import State as state_factory

    st = state_factory(w)
    st.pos = snap.pos if pos is None else min(int(pos), snap.pos)
    st.limit = 0                       # grows as commits need, like a fresh stream
    if room is not None:
        st.room = room
    for i, pair in enumerate(st.kv):
        if pair is None:               # linear-attention layer: fork the recurrent state
            if snap.rec is not None and i < len(snap.rec) and snap.rec[i] is not None:
                st.rec[i] = snap.rec[i].detach().clone()
            if snap.conv is not None and i < len(snap.conv) and snap.conv[i] is not None:
                st.conv[i] = snap.conv[i].detach().clone()
            continue
        k, v = pair
        shape = (rows, *k.shape[1:])
        nk = torch.empty(shape, dtype=k.dtype, device=k.device)
        nv = torch.empty(shape, dtype=v.dtype, device=v.device)
        n = min(snap.pos, rows)
        nk[:n] = snap.kv[i][0][:n]
        nv[:n] = snap.kv[i][1][:n]
        st.kv[i] = (nk, nv)
    return st