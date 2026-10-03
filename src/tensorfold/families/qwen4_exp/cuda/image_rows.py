"""Image embeddings and full prompt rotary positions owned by one Flash Next stream."""

import torch
import triton
import triton.language as tl


@triton.jit
def rope_axis(pos, ROPE, DELTA, length, index, MODE: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr):
    if MODE == 0:
        axis = pos
    elif MODE == 1:
        axis = pos + tl.load(DELTA)
    else:
        live = pos < length
        text = pos + tl.load(DELTA)
        pt = tl.load(ROPE + pos * 3, mask=live, other=0)
        ph = tl.load(ROPE + pos * 3 + 1, mask=live, other=0)
        pw = tl.load(ROPE + pos * 3 + 2, mask=live, other=0)
        axis = tl.where((index % 3 == 1) & (index < 3 * S1), ph,
                        tl.where((index % 3 == 2) & (index < 3 * S2), pw, pt))
        axis = tl.where(live, axis, text)
    return axis


def begin(engine, stream, tower) -> None:
    """Encode an image request on its admitted slot before the normal prompt passes."""

    if stream.vision is not None:
        if tower is None:
            raise ValueError("image inputs require starting this server with --vision")
        attach(engine.st, tower.encode(stream.vision, stream.prompt), len(stream.prompt))
        stream.vision = None


def attach(st, encoded, length: int) -> None:
    """Keep absolute image positions through decode, including pool blocks that cross the prompt end."""

    if encoded.positions.shape != (3, length) or len(encoded.rows) != encoded.features.shape[0]:
        raise ValueError("image features and rotary positions must cover the prepared prompt")
    if any(p < 0 or p >= length for p in encoded.rows):
        raise ValueError("image feature row is outside the prepared prompt")
    st.image_positions = encoded.positions.t().contiguous().to(dtype=torch.int32)
    st.image_rows, st.image_features = tuple(encoded.rows), encoded.features
    st.set_rope_delta(encoded.rope_delta)


def embed(segs, b, streams: int) -> None:
    """Replace image placeholder embeddings in each prompt piece, preserving every text row."""

    for st, a0, a1 in segs:
        if st.image_features is None:
            continue
        inside = [(i, p - st.pos + a0) for i, p in enumerate(st.image_rows) if st.pos <= p < st.pos + a1 - a0]
        if inside:
            source, target = zip(*inside)
            source = torch.tensor(source, dtype=torch.int64, device=b.h.device)
            target = torch.tensor(target, dtype=torch.int64, device=b.h.device)
            b.h.index_copy_(0, target, st.image_features.index_select(0, source).to(b.h.dtype).repeat(1, streams))


def finish(st) -> None:
    """Image feature tensors end with prefill; rotary positions remain until the stream is reset."""

    st.image_rows, st.image_features = (), None
