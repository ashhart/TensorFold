"""Image features on two ranks: rank 0 encodes once and sends, so both feed identical hidden states to every layer."""

from __future__ import annotations

import torch


def describe(encoded) -> dict | None:
    """The rows, decode offset and feature shape of an encoded image prompt, as the request message carries them."""

    if encoded is None:
        return None
    return {"rows": [int(r) for r in encoded.rows], "delta": int(encoded.rope_delta),
            "shape": [int(n) for n in encoded.features.shape]}


def exchange(comm, rank: int, length: int, images: dict, encoded=None, *, hidden: int):
    """Rank 0 sends ``encoded``'s bf16 features and int32 positions; both ranks validate the description first."""

    from tensorfold.cuda.comm import exchange as swap
    from tensorfold.vision.qwen_cuda import EncodedVision

    rows, delta, shape = images["rows"], int(images["delta"]), tuple(int(n) for n in images["shape"])
    if (len(shape) != 2 or shape[0] != len(rows) or shape[1] != int(hidden) or not rows
            or sorted(set(rows)) != list(rows) or rows[0] < 0 or rows[-1] >= length):
        raise ValueError("the image rows rank 0 encoded do not fit the prompt it admitted")
    if rank == 0:
        if encoded is None:
            raise ValueError("rank 0 shares an image prompt it has not encoded")
        features = encoded.features.to(dtype=torch.bfloat16).contiguous()
        positions = encoded.positions.to(dtype=torch.int32).contiguous()
        if tuple(features.shape) != shape or tuple(positions.shape) != (3, length):
            raise ValueError("the image features rank 0 encoded do not match what it told rank 1")
        swap(comm, [features, positions], [features.new_empty((0,)), positions.new_empty((0,))], 1)
        return EncodedVision(tuple(rows), features, positions, delta)
    device = torch.device("cuda", 0)
    features = torch.empty(shape, dtype=torch.bfloat16, device=device)
    positions = torch.empty((3, length), dtype=torch.int32, device=device)
    swap(comm, [features.new_empty((0,)), positions.new_empty((0,))], [features, positions], 0)
    return EncodedVision(tuple(rows), features, positions, delta)
