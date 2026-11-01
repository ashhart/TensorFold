#!/usr/bin/env python3
"""Independent developer MLX reference for one LFM3-VL image's dense vision tower and projector."""

import argparse
import json
import math
from pathlib import Path

from reference import Reference, write_bf16


class VisionReference(Reference):
    def __init__(self, directory, mx):
        super().__init__(directory, mx)
        self.vision = self.config['vision_config']
        if self.config.get('model_type') != 'lfm3_vl':
            raise ValueError('Expected an LFM3-VL checkpoint')

    def dense(self, x, name):
        mx = self.mx
        weight = self.weights[name + '.weight']
        bias = self.weights.get(name + '.bias')
        if weight.dtype != mx.bfloat16:
            raise ValueError(f'Expected BF16 vision weight {name}')
        if bias is not None:
            return self.round(mx.addmm(bias, x, weight.T))
        return self.round(mx.matmul(x, weight.T))

    def rms(self, x, name):
        mx = self.mx
        value = x.astype(mx.float32)
        inverse = mx.rsqrt(mx.mean(value * value, axis=-1, keepdims=True) + self.vision['norm_eps'])
        return self.round(self.round(value * inverse) * self.weights[name + '.weight'])

    def grid(self, values, height, width):
        channels = values.shape[-1]
        return values.reshape(height // 2, width // 2, 2, 2, channels).transpose(0, 2, 1, 3, 4).reshape(
            height, width, channels,
        )

    def merged(self, values, height, width):
        channels = values.shape[-1]
        return values.reshape(height // 2, 2, width // 2, 2, channels).transpose(0, 2, 1, 3, 4).reshape(
            height * width, channels,
        )

    def convolution(self, x, prefix, height, width):
        mx = self.mx
        left, right, value = mx.split(self.dense(x, prefix + 'conv.in_proj'), 3, axis=-1)
        grid = self.grid(self.round(left * value), height, width).astype(mx.float32)
        padded = mx.pad(grid, ((1, 1), (1, 1), (0, 0)))
        kernel = self.weights[prefix + 'conv.spatial_conv.weight'].astype(mx.float32)[:, :, :, 0]
        output = mx.zeros_like(grid)
        for row in range(3):
            for column in range(3):
                output = output + padded[row:row + height, column:column + width] * kernel[:, row, column]
        conv = self.merged(self.round(output), height, width)
        return self.dense(self.round(right * conv), prefix + 'conv.out_proj')

    def rotary(self, x, coordinates):
        mx = self.mx
        dim = x.shape[-1]
        spatial = dim // 2
        frequencies = mx.power(mx.array(self.vision['rope_parameters']['rope_theta'], dtype=mx.float32),
                               -mx.arange(0, spatial, 2, dtype=mx.float32) / spatial)
        h = coordinates[:, 0, None].astype(mx.float32) * frequencies[None, :]
        w = coordinates[:, 1, None].astype(mx.float32) * frequencies[None, :]
        angles = mx.concatenate((h, w, h, w), axis=-1)[:, None, :]
        value = x.astype(mx.float32)
        first, second = mx.split(value, 2, axis=-1)
        rotated = mx.concatenate((-second, first), axis=-1)
        return self.round(value * mx.cos(angles) + rotated * mx.sin(angles))

    def attention(self, x, prefix, coordinates):
        mx = self.mx
        length = x.shape[0]
        heads = self.vision['num_attention_heads']
        kv = self.vision['num_key_value_heads']
        dim = self.vision['hidden_size'] // heads
        q = self.dense(x, prefix + 'self_attn.q_proj').reshape(length, heads, dim)
        k = self.dense(x, prefix + 'self_attn.k_proj').reshape(length, kv, dim)
        v = self.dense(x, prefix + 'self_attn.v_proj').reshape(length, kv, dim)
        q = self.rotary(self.rms(q, prefix + 'self_attn.q_layernorm'), coordinates)
        k = self.rotary(self.rms(k, prefix + 'self_attn.k_layernorm'), coordinates)
        output = mx.fast.scaled_dot_product_attention(
            q.transpose(1, 0, 2)[None], k.transpose(1, 0, 2)[None], v.transpose(1, 0, 2)[None],
            scale=1 / math.sqrt(dim),
        )[0].transpose(1, 0, 2).reshape(length, heads * dim)
        return self.dense(self.round(output), prefix + 'self_attn.out_proj')

    def projector(self, hidden, dump):
        mx = self.mx
        prefix = 'multi_modal_projector.'
        value = hidden.astype(mx.float32)
        centered = value - mx.mean(value, axis=-1, keepdims=True)
        inverse = mx.rsqrt(mx.mean(centered * centered, axis=-1, keepdims=True)
                           + self.config['projector_layer_norm_eps'])
        normalized = self.round(centered * inverse * self.weights[prefix + 'layer_norm.weight'].astype(mx.float32)
                                + self.weights[prefix + 'layer_norm.bias'].astype(mx.float32))
        if dump:
            dump('projector_norm', normalized)
        merged = normalized.reshape(-1, self.vision['hidden_size'] * self.config['downsample_factor'] ** 2)
        first = self.dense(merged, prefix + 'linear_1')
        values = first.astype(mx.float32)
        activated = self.round(0.5 * values * (1 + mx.erf(values / math.sqrt(2))))
        if dump:
            dump('projector_gelu', activated)
        return self.dense(activated, prefix + 'linear_2')

    def run_image(self, patches, height, width, dump=None):
        mx = self.mx
        coordinates = [
            (block_h * 2 + inner_h, block_w * 2 + inner_w)
            for block_h in range(height // 2) for block_w in range(width // 2)
            for inner_h in range(2) for inner_w in range(2)
        ]
        coordinates = mx.array(coordinates, dtype=mx.int32)
        hidden = self.dense(self.round(mx.array(patches)), 'vision_tower.patchifier')
        if dump:
            dump('patchifier', hidden)
        hidden = self.rms(hidden, 'vision_tower.embedding_norm')
        if dump:
            dump('embedding_norm', hidden)
        for index, kind in enumerate(self.vision['layer_types']):
            prefix = f'vision_tower.layers.{index}.'
            x = self.rms(hidden, prefix + 'operator_norm')
            operator = self.convolution(x, prefix, height, width) if kind == 'conv' else self.attention(
                x, prefix, coordinates,
            )
            hidden = self.round(hidden + operator)
            x = self.rms(hidden, prefix + 'ffn_norm')
            gate = self.dense(x, prefix + 'feed_forward.w1')
            up = self.dense(x, prefix + 'feed_forward.w3')
            silu = self.round(gate.astype(mx.float32) * mx.sigmoid(gate.astype(mx.float32)))
            hidden = self.round(hidden + self.dense(self.round(silu * up), prefix + 'feed_forward.w2'))
            mx.eval(hidden)
            if dump:
                dump(f'layer_{index:02d}', hidden)
        return self.projector(hidden, dump)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--patches', type=Path, required=True, help='Merge-ordered FP32 patches as NumPy .npy')
    parser.add_argument('--grid-h', type=int, required=True)
    parser.add_argument('--grid-w', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Little-endian BF16 language embeddings')
    parser.add_argument('--intermediates-dir', type=Path)
    args = parser.parse_args()
    import mlx.core as mx
    import numpy as np
    if args.grid_h <= 0 or args.grid_w <= 0 or args.grid_h % 2 or args.grid_w % 2:
        parser.error('Patch grid dimensions must be positive multiples of two')
    patches = np.load(args.patches, allow_pickle=False)
    if patches.shape != (args.grid_h * args.grid_w, 1536) or not np.isfinite(patches).all():
        parser.error('Expected finite patches with shape [grid_h * grid_w, 1536]')
    model = VisionReference(args.model_dir, mx)
    intermediates = {}

    def dump(name, values):
        write_bf16(args.intermediates_dir / (name + '.bf16'), values, mx, np)
        intermediates[name] = {'shape': list(values.shape), 'dtype': 'BF16'}

    if args.intermediates_dir:
        args.intermediates_dir.mkdir(parents=True, exist_ok=True)
    result = model.run_image(patches, args.grid_h, args.grid_w, dump if args.intermediates_dir else None)
    mx.eval(result)
    write_bf16(args.output, result, mx, np)
    metadata = {'shape': list(result.shape), 'dtype': 'BF16', 'byte_order': 'little', 'intermediates': intermediates}
    args.output.with_suffix(args.output.suffix + '.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata))


if __name__ == '__main__':
    main()
