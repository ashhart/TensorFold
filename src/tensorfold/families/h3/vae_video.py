"""The H3 video decoder: a ViT over latent tiles, decoded clip by clip and blended across tile and clip seams."""

# Adapted from minimax-h3-mlx's `video_vae.py` (Apache-2.0, https://github.com/mrbizarro/minimax-h3-mlx,
# revision 7919020), decode path only: the ViT decoder, the tiling and the clip chunking.

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import mlx.core as mx
import numpy as np
from mlx import nn

from .config import pipeline_root

PIXEL_MEAN = (0.485, 0.456, 0.406)  # the decoder emits ImageNet-normalized RGB
PIXEL_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class DecoderConfig:
    """The video VAE's decoder, from ``video_vae/config.json`` and ``video_vae/source/config.json``."""

    latent_channels: int = 24
    out_channels: int = 3
    layers: int = 36
    heads: int = 32
    head_dim: int = 64
    ffn_mult: int = 4
    register_tokens: int = 4
    rope_theta: float = 100.0
    rope_dim_ratio: float = 0.75
    eps: float = 1e-5
    spatial_ratio: int = 16
    temporal_ratio: int = 4
    clip_length: int = 17
    token_drop: int = 3
    tile: int = 256
    tile_overlap: int = 64
    latents_mean: tuple[float, ...] = ()
    latents_std: tuple[float, ...] = ()

    @property
    def dim(self) -> int:
        return self.heads * self.head_dim

    @property
    def upscale(self) -> int:
        """Pixels per decoded pixel along each side: 1 for the released decoder, 2 for a packed 12-channel head."""

        return math.isqrt(self.out_channels // 3)

    @classmethod
    def from_checkpoint(cls, model_dir) -> DecoderConfig:
        root = pipeline_root(model_dir)
        if root is None:
            raise FileNotFoundError(f"{model_dir} is not a MiniMax H3 pipeline folder")
        with open(root / "video_vae" / "config.json") as handle:
            wrapper = json.load(handle)
        with open(root / "video_vae" / "source" / "config.json") as handle:
            source = json.load(handle)
        vit = source["vit_decoder_kwargs"]
        return cls(
            latent_channels=source["z_channels"], out_channels=source["out_ch"], layers=vit["num_layers"],
            heads=vit["heads"], head_dim=vit["dim_head"], rope_theta=vit["rope_theta"],
            rope_dim_ratio=vit["rope_dim_ratio"], spatial_ratio=math.prod(source["space_down"]),
            temporal_ratio=math.prod(source["time_down"]), clip_length=wrapper.get("vae_clip_length", 17),
            token_drop=wrapper.get("vae_token_drop", 3), tile=wrapper.get("vae_tile_size", 256),
            tile_overlap=wrapper.get("vae_tile_overlap_min", 64),
            latents_mean=tuple(wrapper.get("latents_mean", ())), latents_std=tuple(wrapper.get("latents_std", ())))


def _rotary(config: DecoderConfig, position_ids: mx.array) -> tuple[mx.array, mx.array]:
    """cos and sin, (B, N, 1, rotated channels), from positions in [-1, 1) over (t, h, w)."""

    dim = int(config.head_dim * config.rope_dim_ratio)
    step = 6 / dim
    exponents = mx.array([i * step for i in range(math.ceil(1.0 / step))], dtype=mx.float32)
    inv_freq = 1.0 / (config.rope_theta**exponents)
    angles = 2.0 * math.pi * position_ids[..., None] * inv_freq.reshape(1, 1, 1, -1)
    angles = angles.reshape(angles.shape[0], angles.shape[1], -1)
    angles = mx.concatenate([angles, angles], axis=-1)[:, :, None, :]
    return mx.cos(angles), mx.sin(angles)


def _rotate(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    width = cos.shape[-1]
    turned, kept = x[..., :width], x[..., width:]
    half = width // 2
    out = turned * cos.astype(x.dtype) + mx.concatenate([-turned[..., half:], turned[..., :half]], axis=-1) * sin.astype(
        x.dtype)
    return out if kept.shape[-1] == 0 else mx.concatenate([out, kept], axis=-1)


class Attention(nn.Module):
    """Full self-attention within a tile; q and k take a weightless RMS norm in float32."""

    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.heads, self.head_dim, self.eps = config.heads, config.head_dim, config.eps
        self.to_qkv = nn.Linear(config.dim, 3 * config.dim, bias=True)
        self.to_out = nn.Linear(config.dim, config.dim, bias=True)

    def _norm(self, x: mx.array) -> mx.array:
        wide = x.astype(mx.float32)
        return (wide * mx.rsqrt(mx.mean(wide * wide, axis=-1, keepdims=True) + self.eps)).astype(x.dtype)

    def __call__(self, x: mx.array, rotary) -> mx.array:
        batch, rows, _ = x.shape
        # rows are interleaved per head, as in the transformer: (heads, 3, head_dim)
        qkv = self.to_qkv(x).reshape(batch, rows, self.heads, 3, self.head_dim)
        q = _rotate(self._norm(qkv[:, :, :, 0]), *rotary).transpose(0, 2, 1, 3)
        k = _rotate(self._norm(qkv[:, :, :, 1]), *rotary).transpose(0, 2, 1, 3)
        out = mx.fast.scaled_dot_product_attention(q, k, qkv[:, :, :, 2].transpose(0, 2, 1, 3),
                                                   scale=self.head_dim**-0.5)
        return self.to_out(out.transpose(0, 2, 1, 3).reshape(batch, rows, self.heads * self.head_dim))


class FeedForward(nn.Module):
    """SwiGLU; ``w1`` is the fused [gate; value] projection."""

    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.width = config.dim * config.ffn_mult
        self.w1 = nn.Linear(config.dim, 2 * self.width, bias=True)
        self.w2 = nn.Linear(self.width, config.dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        fused = self.w1(x)
        return self.w2(nn.silu(fused[..., : self.width]) * fused[..., self.width :])


class Block(nn.Module):
    """Pre-norm block with a learned per-channel scale on each branch; the norms run in float32."""

    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.dim, eps=config.eps)
        self.attn = Attention(config)
        self.scale1 = mx.zeros((config.dim,))
        self.norm2 = nn.RMSNorm(config.dim, eps=config.eps)
        self.ff = FeedForward(config)
        self.scale2 = mx.zeros((config.dim,))

    def __call__(self, x: mx.array, rotary) -> mx.array:
        x = x + self.attn(self.norm1(x.astype(mx.float32)).astype(x.dtype), rotary) * self.scale1
        return x + self.ff(self.norm2(x.astype(mx.float32)).astype(x.dtype)) * self.scale2


class ViT(nn.Module):
    """One token per latent voxel plus register tokens; each token expands to a 4 x 16 x 16 pixel block."""

    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config
        self.x_embedder = nn.Linear(config.latent_channels, config.dim)
        self.register_tokens = mx.zeros((1, config.register_tokens, config.dim))
        self.transformer_blocks = [Block(config) for _ in range(config.layers)]
        self.norm_out = nn.LayerNorm(config.dim, eps=config.eps, affine=True)
        self.proj_out = nn.Linear(
            config.dim, config.out_channels * config.temporal_ratio * config.spatial_ratio * config.spatial_ratio)

    def __call__(self, x: mx.array) -> mx.array:
        """(B, D, H, W, C) latents to (B, D * 4, H * 16, W * 16, 3) pixels."""

        config = self.config
        batch, depth, height, width, channels = x.shape
        voxels = depth * height * width
        tokens = self.x_embedder(x.reshape(batch, voxels, channels))
        extra = config.register_tokens + 1  # the registers and one all-zero token, all at position 0
        registers = mx.broadcast_to(self.register_tokens, (batch, config.register_tokens, config.dim))
        tokens = mx.concatenate([tokens, registers.astype(tokens.dtype),
                                 mx.zeros((batch, 1, config.dim), dtype=tokens.dtype)], axis=1)
        grids = [2.0 * ((mx.arange(size, dtype=mx.float32) + 0.5) / size) - 1.0 for size in (depth, height, width)]
        shape = (depth, height, width)
        positions = mx.stack([mx.broadcast_to(grids[0].reshape(depth, 1, 1), shape),
                              mx.broadcast_to(grids[1].reshape(1, height, 1), shape),
                              mx.broadcast_to(grids[2].reshape(1, 1, width), shape)], axis=-1)
        positions = mx.broadcast_to(positions.reshape(1, voxels, 3), (batch, voxels, 3))
        rotary = _rotary(config, mx.concatenate([positions, mx.zeros((batch, extra, 3))], axis=1))
        for block in self.transformer_blocks:
            tokens = block(tokens, rotary)
        tokens = self.proj_out(self.norm_out(tokens.astype(mx.float32)).astype(tokens.dtype))[:, :voxels]
        p, pt = config.spatial_ratio, config.temporal_ratio
        out = tokens.reshape(batch, depth, height, width, config.out_channels, pt, p, p)
        return out.transpose(0, 1, 5, 2, 6, 3, 7, 4).reshape(batch, depth * pt, height * p, width * p,
                                                               config.out_channels)


def _blend(a: mx.array, b: mx.array, extent: int, axis: int) -> mx.array:
    """``b`` with its first ``extent`` slices along ``axis`` cross-faded from the tail of ``a``."""

    extent = min(a.shape[axis], b.shape[axis], extent)
    if extent <= 0:
        return b
    shape = [1] * a.ndim
    shape[axis] = extent
    ramp = (mx.arange(extent, dtype=mx.float32) / extent).reshape(shape).astype(b.dtype)
    tail = mx.take(a, mx.arange(a.shape[axis] - extent, a.shape[axis]), axis=axis)
    blended = tail * (1.0 - ramp) + mx.take(b, mx.arange(extent), axis=axis) * ramp
    if extent == b.shape[axis]:
        return blended
    return mx.concatenate([blended, mx.take(b, mx.arange(extent, b.shape[axis]), axis=axis)], axis=axis)


class PostQuant(nn.Module):
    """The checkpoint's 1x1x1 convolution on the latents, kept as a convolution.

    A matmul gives the same values to within 5e-4, which 36 transformer layers grow to 0.1 in places; the
    convolution keeps the decode equal to the reference's.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.weight = mx.zeros((channels, 1, 1, 1, channels))
        self.bias = mx.zeros((channels,))

    def __call__(self, z: mx.array) -> mx.array:
        return mx.conv3d(z, self.weight) + self.bias


class VideoDecoder(nn.Module):
    """Latents to frames. Each clip of latent frames is cut into overlapping 256-pixel tiles, the tiles go
    through the ViT a batch at a time, and tile and clip overlaps are cross-faded."""

    def __init__(self, config: DecoderConfig, batch: int = 8):
        super().__init__()
        self.config = config
        self.batch = batch
        self.post_quant = PostQuant(config.latent_channels)
        self.decoder = ViT(config)

    def _tiles(self, length: int) -> tuple[list[int], list[int]]:
        """(tile starts, overlaps) covering ``length`` pixels; slack goes to the overlaps in latent steps."""

        config = self.config
        if config.tile >= length:
            return [0], []
        count = math.ceil(length / config.tile)
        while config.tile * count - config.tile_overlap * (count - 1) - length < 0:
            count += 1
        overlaps = [config.tile_overlap] * (count - 1)
        slack = config.tile * count - sum(overlaps) - length
        for i in range(slack // config.spatial_ratio):
            overlaps[i % (count - 1)] += config.spatial_ratio
        starts = [0]
        for overlap in overlaps:
            starts.append(starts[-1] + config.tile - overlap)
        return starts, overlaps

    def _clip(self, z: mx.array) -> mx.array:
        """One clip of latent frames (B, D, H, W, C) to pixels, tiled."""

        ratio, tile = self.config.spatial_ratio, self.config.tile
        height, width = z.shape[2] * ratio, z.shape[3] * ratio
        ys, y_overlaps = self._tiles(height)
        xs, x_overlaps = self._tiles(width)
        tile_h, tile_w = min(tile, height) // ratio, min(tile, width) // ratio
        pieces = [z[:, :, y // ratio : y // ratio + tile_h, x // ratio : x // ratio + tile_w] for y in ys for x in xs]
        decoded = []
        for start in range(0, len(pieces), max(self.batch, 1)):
            part = pieces[start : start + max(self.batch, 1)]
            out = self.decoder(self.post_quant(mx.concatenate(part, axis=0)))
            mx.eval(out)
            decoded.extend(out[i * z.shape[0] : (i + 1) * z.shape[0]] for i in range(len(part)))
        rows = []  # each tile blends against its neighbours as decoded, not as already blended
        for i in range(len(ys)):
            row = []
            for j in range(len(xs)):
                piece = decoded[i * len(xs) + j]
                if i > 0:
                    piece = _blend(decoded[(i - 1) * len(xs) + j], piece, y_overlaps[i - 1], axis=2)
                if j > 0:
                    piece = _blend(decoded[i * len(xs) + j - 1], piece, x_overlaps[j - 1], axis=3)
                if i < len(ys) - 1:
                    piece = piece[:, :, : piece.shape[2] - y_overlaps[i]]
                if j < len(xs) - 1:
                    piece = piece[:, :, :, : piece.shape[3] - x_overlaps[j]]
                row.append(piece)
            rows.append(mx.concatenate(row, axis=3))
        return mx.concatenate(rows, axis=2)

    def decode(self, z: mx.array) -> mx.array:
        """(B, C, F, H, W) latents to (B, 3, frames, height, width) normalized pixels.

        The encoder dropped ``token_drop`` latent frames from every clip, so consecutive decoded clips overlap
        and are cross-faded; latent frames are repeated at the end to fill the last clip and the extra pixel
        frames cut off again.
        """

        config = self.config
        z = z.transpose(0, 2, 3, 4, 1)
        ratio = config.temporal_ratio
        chunk = math.ceil(config.clip_length / ratio)
        lead = (-config.clip_length) % ratio
        token_overlap = (-config.token_drop) % chunk
        frame_overlap = max(token_overlap * ratio - lead, 0)
        tokens = z.shape[1] + config.token_drop
        pad = (-tokens) % chunk
        clips = (tokens + pad) // chunk - int(config.token_drop > 0)
        if clips < 1:
            raise ValueError(f"too few latent frames to decode: {z.shape[1]}, need {2 * chunk - config.token_drop}")
        if pad:
            z = mx.concatenate([z, mx.broadcast_to(z[:, -1:], (z.shape[0], pad, *z.shape[2:]))], axis=1)
        decoded, overlap = [], None
        for index in range(clips):
            clip = self._clip(z[:, index * chunk : index * chunk + chunk + token_overlap])
            for part in range(int(config.token_drop > 0) + 1):
                piece = clip[:, part * chunk * ratio : (part + 1) * chunk * ratio][:, lead:]
                if part == 0:
                    decoded.append(piece if overlap is None else _blend(overlap, piece, frame_overlap, axis=1))
                else:
                    overlap = piece
        if overlap is not None:
            decoded.append(overlap)
        out = mx.concatenate(decoded, axis=1)
        if pad:
            # a clip's last latent frame covers `clip_length % ratio` pixel frames, the others `ratio`
            tail = config.clip_length % ratio
            before = z.shape[1] - pad
            out = out[:, : -sum(tail if tail and (before + k) % chunk == 0 else ratio for k in range(pad))]
        return out.transpose(0, 4, 1, 2, 3)

    def frames(self, latents: mx.array) -> np.ndarray:
        """Normalized (B, C, F, H, W) latents, as the transformer leaves them, to (F, H, W, 3) uint8 frames."""

        config = self.config
        mean = mx.array(np.array(config.latents_mean, np.float32)).reshape(1, -1, 1, 1, 1)
        std = mx.array(np.array(config.latents_std, np.float32)).reshape(1, -1, 1, 1, 1)
        pixels = np.array(self.decode((latents * std + mean).astype(mx.float32)))
        up = config.upscale
        if up > 1:  # packed head: channel c * up * up + i * up + j is pixel (i, j) of colour c's up x up cell
            batch, _, count, height, width = pixels.shape
            pixels = pixels.reshape(batch, 3, up, up, count, height, width).transpose(0, 1, 4, 5, 2, 6, 3)
            pixels = pixels.reshape(batch, 3, count, height * up, width * up)
        pixels = pixels * np.array(PIXEL_STD, np.float32).reshape(1, 3, 1, 1, 1)
        pixels = pixels + np.array(PIXEL_MEAN, np.float32).reshape(1, 3, 1, 1, 1)
        return (np.clip(pixels, 0.0, 1.0)[0].transpose(1, 2, 3, 0) * 255.0 + 0.5).astype(np.uint8)


def load_video_decoder(model_dir, int8: bool = True, batch: int = 8, upscale_decoder=None) -> VideoDecoder:
    """The video decoder from a released pipeline folder; the encoder half of the checkpoint is not read.

    ``upscale_decoder`` is a safetensors file holding a replacement ViT decoder whose output head packs
    ``3 * n * n`` channels (a 2x decoder has 12): its frames come out ``n`` times larger along each side. Tensors
    it does not hold, such as the latent convolution, are taken from the released checkpoint.

    ``int8`` runs the ViT's SwiGLU, QKV and output projections through the tensor-unit kernels (about 47 dB
    against the float32 decode); it is ignored where those operations are unavailable.
    """

    from mlx.utils import tree_flatten, tree_unflatten

    root = pipeline_root(model_dir)
    if root is None:
        raise FileNotFoundError(f"{model_dir} is not a MiniMax H3 pipeline folder")
    config = DecoderConfig.from_checkpoint(model_dir)
    sources = [mx.load(str(root / "video_vae" / "source" / "model.safetensors"))]
    if upscale_decoder is not None:
        replacement = mx.load(str(upscale_decoder))
        head = replacement.get("decoder.proj_out.weight")
        block = config.temporal_ratio * config.spatial_ratio * config.spatial_ratio
        if head is None or head.shape[0] % (3 * block) or math.isqrt(head.shape[0] // (3 * block)) ** 2 != head.shape[0] // (
                3 * block):
            raise ValueError(f"{upscale_decoder} has no packed decoder head")
        config = DecoderConfig(**{**config.__dict__, "out_channels": head.shape[0] // block})
        sources.append(replacement)  # read last, so its decoder tensors replace the released ones
    model = VideoDecoder(config, batch=batch)
    expected = {name: value for name, value in tree_flatten(model.parameters())}
    found = {}
    for tensors in sources:
        for name, tensor in tensors.items():
            if name == "post_quant_conv.weight":
                found["post_quant.weight"] = tensor.reshape(tensor.shape[0], 1, 1, 1, tensor.shape[1])
            elif name == "post_quant_conv.bias":
                found["post_quant.bias"] = tensor
            elif name in expected and tuple(tensor.shape) == tuple(expected[name].shape):
                found[name] = tensor
    reference = found["decoder.x_embedder.weight"].dtype if upscale_decoder is None else mx.float32
    found = {name: tensor.astype(reference) if tensor.dtype != reference and upscale_decoder is not None else tensor
             for name, tensor in found.items()}
    missing = sorted(expected - found.keys())
    if missing:
        raise KeyError(f"{len(missing)} decoder parameters are not in the checkpoint, e.g. {missing[:3]}")
    model.update(tree_unflatten(list(found.items())))
    mx.eval(model.parameters())
    if int8:
        from .vae_fast import accelerate

        accelerate(model)
    return model
