"""Image embeddings and full prompt rotary positions owned by one Flash Next stream."""

import hashlib

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


def key(prompt, vision) -> list[int] | None:
    """The ids a prompt's kept states are matched on: its tokens, each image's and video frame group's placeholder rows
    (alike whatever the media) replaced by a key of its pixels and grid that no token id equals; None for media
    without hashes, which neither resumes nor keeps a state."""

    ids = [int(t) for t in prompt]
    if vision is None:
        return ids
    if not hasattr(vision, "image_hashes"):
        return None
    media = list(zip(vision.image_spans, vision.image_hashes, vision.image_grid_thw))
    if vision.video_hashes:                              # a video owns its grid's count of frame-group spans
        groups = [(h, g) for h, g in zip(vision.video_hashes, vision.video_grid_thw) for _ in range(int(g[0]))]
        media += [(span, h, g) for span, (h, g) in zip(vision.video_spans, groups)]
    for (start, end), digest, grid in media:
        tag = hashlib.sha256(f"{digest}/{'x'.join(str(int(v)) for v in grid)}".encode()).digest()
        ids[start:end] = [-1 - int.from_bytes(tag[:8], "big")] * (end - start)
    return ids


def begin(stream, tower):
    """Encode an image request's pictures before its prompt passes (``prefill_begin`` attaches them); None for text."""

    if stream.vision is None:
        return None
    if tower is None:
        raise ValueError("image inputs require starting this server with --vision")
    encoded = tower.encode(stream.vision, stream.prompt)
    stream.vision = None
    return encoded


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
