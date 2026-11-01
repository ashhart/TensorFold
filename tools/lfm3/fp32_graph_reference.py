#!/usr/bin/env python3
"""Independent NumPy FP32 graph comparison, with separately selected BF16 activation boundaries."""
import argparse
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import numpy as np
from fp64_reference import Oracle as Checkpoint, bf


class Graph:
    def __init__(self, directory, boundaries='fp32-bf16', trace=None):
        if boundaries not in ('fp32', 'fp32-bf16'):
            raise ValueError('Expected fp32 or fp32-bf16')
        self.checkpoint = Checkpoint(directory)
        self.boundaries, self.trace = boundaries, trace
        self.config = self.checkpoint.config
        self.text = self.checkpoint.text
        self.prefix = self.checkpoint.prefix

    def cast(self, x):
        x = np.asarray(x, dtype=np.float32)
        return bf(x).astype(np.float32) if self.boundaries == 'fp32-bf16' else x

    def take(self, name, x):
        if np.asarray(x).dtype != np.float32:
            raise ValueError('FP32 graph produced a non-FP32 activation at ' + name)
        if self.trace:
            self.trace(name, x)

    def tensor(self, name):
        return self.checkpoint.tensor(name).astype(np.float32)

    def weights(self, name, rows):
        bits, group = self.checkpoint.quant(name)
        words = self.checkpoint.tensor(name + '.weight', rows)
        width = words.shape[1] * 32 // bits
        positions = np.arange(width) * bits
        indices, shifts = positions // 32, positions % 32
        padded = np.pad(words.astype(np.uint64), ((0, 0), (0, 1)))
        values = padded[:, indices] >> shifts.astype(np.uint64)
        cross = shifts + bits > 32
        values[:, cross] |= padded[:, indices[cross] + 1] << (32 - shifts[cross]).astype(np.uint64)
        codes = (values & ((1 << bits) - 1)).astype(np.float32)
        scale = self.checkpoint.tensor(name + '.scales', rows).astype(np.float32).repeat(group, axis=-1)
        bias = self.checkpoint.tensor(name + '.biases', rows).astype(np.float32).repeat(group, axis=-1)
        return codes * scale + bias

    def linear(self, x, name):
        count = self.checkpoint.files[self.checkpoint.map[name + '.weight']][2][name + '.weight']['shape'][0]
        y = np.empty((*x.shape[:-1], count), dtype=np.float32)
        for first in range(0, count, 512):
            y[..., first:first + 512] = x @ self.weights(name, slice(first, first + 512)).T
        return self.cast(y)

    def rms(self, x, gain, epsilon):
        mean = np.mean(x * x, axis=-1, keepdims=True, dtype=np.float32)
        inverse = np.float32(1) / np.sqrt(mean + np.float32(epsilon))
        return self.cast(self.cast(x * inverse) * gain)

    def norm(self, x, name):
        return self.rms(x, self.tensor(name + '.weight'), self.text['norm_eps'])

    def rotary(self, x, kind):
        spec = self.text['rope_parameters'][kind]
        pairs = int(x.shape[-1] * spec.get('partial_rotary_factor', 1)) // 2
        power = -np.arange(pairs, dtype=np.float32) * np.float32(2 / x.shape[-1])
        frequency = np.concatenate((np.power(np.float32(spec['rope_theta']), power),
                                    np.zeros(x.shape[-1] // 2 - pairs, dtype=np.float32)))
        angle = np.arange(len(x), dtype=np.float32)[:, None] * frequency
        angle = np.concatenate((angle, angle), axis=-1)[:, None]
        first, second = np.split(x, 2, axis=-1)
        return self.cast(x * self.cast(np.cos(angle))
                         + np.concatenate((-second, first), axis=-1) * self.cast(np.sin(angle)))

    def attention_value(self, q, k, v, window=0, causal=True):
        scores = (q @ k.swapaxes(-1, -2)) / np.float32(math.sqrt(q.shape[-1]))
        if causal:
            distance = np.arange(q.shape[1])[:, None] - np.arange(k.shape[1])[None]
            allowed = distance >= 0
            if window:
                allowed &= distance < window
            scores = np.where(allowed, scores, np.float32(-np.inf))
        probability = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probability = probability / probability.sum(axis=-1, keepdims=True, dtype=np.float32)
        return self.cast(probability @ v)

    def attention(self, x, prefix, kind):
        heads, kv, width = (self.text[k] for k in ('num_attention_heads', 'num_key_value_heads', 'head_dim'))
        q = self.linear(x, prefix + 'self_attn.q_proj').reshape(len(x), heads, width)
        k = self.linear(x, prefix + 'self_attn.k_proj').reshape(len(x), kv, width)
        v = self.linear(x, prefix + 'self_attn.v_proj').reshape(len(x), kv, width).transpose(1, 0, 2)
        q = self.rotary(self.norm(q, prefix + 'self_attn.q_norm'), kind).transpose(1, 0, 2)
        k = self.rotary(self.norm(k, prefix + 'self_attn.k_norm'), kind).transpose(1, 0, 2)
        k, v = np.repeat(k, heads // kv, axis=0), np.repeat(v, heads // kv, axis=0)
        window = self.text['sliding_window'] if kind == 'sliding_attention' else 0
        value = self.attention_value(q, k, v, window).transpose(1, 0, 2)
        gate = self.linear(x, prefix + 'self_attn.gate_proj')
        gate = self.cast(np.float32(1) / (np.float32(1) + np.exp(-gate)))
        return self.linear(self.cast(value * gate[:, :, None]).reshape(len(x), heads * width),
                           prefix + 'self_attn.o_proj')

    def conv(self, x, prefix):
        left, gate, value = np.split(self.linear(x, prefix + 'conv.in_proj'), 3, axis=-1)
        mixed = self.cast(left * value)
        kernel = self.tensor(prefix + 'conv.conv.weight')[:, :, 0]
        padded = np.pad(mixed, ((kernel.shape[1] - 1, 0), (0, 0)))
        result = np.zeros_like(mixed)
        for tap in range(kernel.shape[1]):
            result = result + padded[tap:tap + len(x)] * kernel[:, tap]
        return self.linear(self.cast(self.cast(result) * gate), prefix + 'conv.out_proj')

    def swiglu(self, gate, up):
        return self.cast(self.cast(gate / (np.float32(1) + np.exp(-gate))) * up)

    def run(self, tokens, image_rows=None, image_start=None):
        hidden = self.cast(self.weights(self.prefix + 'embed_tokens', tokens))
        if image_rows is not None:
            rows = np.asarray(image_rows, dtype=np.float32)
            if image_start is None or rows.ndim != 2 or rows.shape[1] != hidden.shape[1]:
                raise ValueError('Expected projected image rows and a matching start')
            expected = np.zeros(len(tokens), dtype=bool)
            expected[image_start:image_start + len(rows)] = True
            if image_start < 0 or image_start + len(rows) > len(tokens) or not np.array_equal(tokens == 125030, expected):
                raise ValueError('Image span does not match placeholders')
            hidden[image_start:image_start + len(rows)] = self.cast(rows)
        elif image_start is not None:
            raise ValueError('Image start requires projected rows')
        self.take('embedding', hidden)
        for index, kind in enumerate(self.text['layer_types']):
            prefix = self.prefix + f'layers.{index}.'
            x = self.norm(hidden, prefix + 'operator_norm')
            result = self.conv(x, prefix) if kind == 'conv' else self.attention(x, prefix, kind)
            hidden = self.cast(hidden + self.norm(result, prefix + 'post_operator_norm'))
            x = self.norm(hidden, prefix + 'ffn_norm')
            gate, up = self.linear(x, prefix + 'feed_forward.w1'), self.linear(x, prefix + 'feed_forward.w3')
            result = self.linear(self.swiglu(gate, up), prefix + 'feed_forward.w2')
            hidden = self.cast(hidden + self.norm(result, prefix + 'post_ffn_norm'))
            self.take(f'text_layer_{index:02d}', hidden)
            print('text_layer', index, flush=True)
        return self.linear(self.norm(hidden, self.prefix + 'embedding_norm'), self.prefix + 'embed_tokens')


class Vision(Graph):
    def __init__(self, directory, boundaries='fp32-bf16', trace=None):
        super().__init__(directory, boundaries, trace)
        self.vision = self.config['vision_config']

    def dense(self, x, name):
        y = x @ self.tensor(name + '.weight').T
        if name + '.bias' in self.checkpoint.map:
            y = y + self.tensor(name + '.bias')
        return self.cast(y)

    def norm(self, x, name):
        return self.rms(x, self.tensor(name + '.weight'), self.vision['norm_eps'])

    def conv(self, x, prefix, height, width):
        left, gate, value = np.split(self.dense(x, prefix + 'conv.in_proj'), 3, axis=-1)
        channels = self.vision['hidden_size']
        grid = self.cast(left * value).reshape(height // 2, width // 2, 2, 2, channels)
        grid = grid.transpose(0, 2, 1, 3, 4).reshape(height, width, channels)
        padded = np.pad(grid, ((1, 1), (1, 1), (0, 0)))
        kernel = self.tensor(prefix + 'conv.spatial_conv.weight')[:, :, :, 0]
        result = np.zeros_like(grid)
        for row in range(3):
            for col in range(3):
                result = result + padded[row:row + height, col:col + width] * kernel[:, row, col]
        merged = self.cast(result).reshape(height // 2, 2, width // 2, 2, channels)
        merged = merged.transpose(0, 2, 1, 3, 4).reshape(height * width, channels)
        return self.dense(self.cast(merged * gate), prefix + 'conv.out_proj')

    def rotary(self, x, coordinates):
        spatial = x.shape[-1] // 2
        powers = -np.arange(0, spatial, 2, dtype=np.float32) / np.float32(spatial)
        frequency = np.power(np.float32(self.vision['rope_parameters']['rope_theta']), powers)
        vertical = coordinates[:, 0, None].astype(np.float32) * frequency
        horizontal = coordinates[:, 1, None].astype(np.float32) * frequency
        angle = np.concatenate((vertical, horizontal, vertical, horizontal), axis=-1)[:, None]
        first, second = np.split(x, 2, axis=-1)
        return self.cast(x * np.cos(angle) + np.concatenate((-second, first), axis=-1) * np.sin(angle))

    def attention(self, x, prefix, coordinates):
        heads, kv = (self.vision[k] for k in ('num_attention_heads', 'num_key_value_heads'))
        dim = self.vision['hidden_size'] // heads
        q = self.dense(x, prefix + 'self_attn.q_proj').reshape(len(x), heads, dim)
        k = self.dense(x, prefix + 'self_attn.k_proj').reshape(len(x), kv, dim)
        v = self.dense(x, prefix + 'self_attn.v_proj').reshape(len(x), kv, dim).transpose(1, 0, 2)
        q = self.rotary(self.norm(q, prefix + 'self_attn.q_layernorm'), coordinates).transpose(1, 0, 2)
        k = self.rotary(self.norm(k, prefix + 'self_attn.k_layernorm'), coordinates).transpose(1, 0, 2)
        k, v = np.repeat(k, heads // kv, axis=0), np.repeat(v, heads // kv, axis=0)
        value = self.attention_value(q, k, v, causal=False).transpose(1, 0, 2).reshape(len(x), heads * dim)
        return self.dense(value, prefix + 'self_attn.out_proj')

    def run(self, patches, height, width):
        if height <= 0 or width <= 0 or height % 2 or width % 2 or patches.shape != (height * width, 1536):
            raise ValueError('Expected an even patch grid and1536 elements per patch')
        coordinates = np.asarray([(h * 2 + a, w * 2 + b) for h in range(height // 2)
                                  for w in range(width // 2) for a in range(2) for b in range(2)])
        hidden = self.norm(self.dense(self.cast(patches), 'vision_tower.patchifier'), 'vision_tower.embedding_norm')
        for index, kind in enumerate(self.vision['layer_types']):
            prefix = f'vision_tower.layers.{index}.'
            x = self.norm(hidden, prefix + 'operator_norm')
            result = self.conv(x, prefix, height, width) if kind == 'conv' else self.attention(x, prefix, coordinates)
            hidden = self.cast(hidden + result)
            x = self.norm(hidden, prefix + 'ffn_norm')
            gate, up = self.dense(x, prefix + 'feed_forward.w1'), self.dense(x, prefix + 'feed_forward.w3')
            hidden = self.cast(hidden + self.dense(self.swiglu(gate, up), prefix + 'feed_forward.w2'))
            self.take(f'vision_layer_{index:02d}', hidden)
            print('vision_layer', index, flush=True)
        prefix = 'multi_modal_projector.'
        centered = hidden - hidden.mean(axis=-1, keepdims=True, dtype=np.float32)
        inverse = np.float32(1) / np.sqrt(np.mean(centered * centered, axis=-1, keepdims=True, dtype=np.float32)
                                         + np.float32(self.config['projector_layer_norm_eps']))
        hidden = self.cast(centered * inverse * self.tensor(prefix + 'layer_norm.weight')
                           + self.tensor(prefix + 'layer_norm.bias'))
        first = self.dense(hidden.reshape(-1, self.vision['hidden_size'] * self.config['downsample_factor'] ** 2),
                           prefix + 'linear_1')
        argument = first / np.sqrt(np.float32(2))
        erf = np.fromiter((math.erf(float(v)) for v in argument.flat), dtype=np.float32).reshape(first.shape)
        return self.dense(self.cast(np.float32(0.5) * first * (np.float32(1) + erf)), prefix + 'linear_2')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--tokens', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--boundaries', choices=('fp32', 'fp32-bf16'), required=True)
    p.add_argument('--patches', type=Path)
    p.add_argument('--grid-h', type=int)
    p.add_argument('--grid-w', type=int)
    p.add_argument('--image-rows', type=Path)
    p.add_argument('--image-start', type=int)
    args = p.parse_args()
    tokens = np.load(args.tokens, allow_pickle=False)
    model = Graph(args.model_dir, args.boundaries)
    if tokens.ndim != 1 or not np.issubdtype(tokens.dtype, np.integer) or len(tokens) == 0:
        p.error('Expected nonempty integer tokens')
    if tokens.min() < 0 or tokens.max() >= model.text['vocab_size']:
        p.error('Token IDs exceed vocabulary')
    if args.patches and args.image_rows:
        p.error('Choose own vision or fixed projected rows')
    rows = np.load(args.image_rows, allow_pickle=False) if args.image_rows else None
    if args.patches:
        rows = Vision(args.model_dir, args.boundaries).run(np.load(args.patches, allow_pickle=False),
                                                          args.grid_h, args.grid_w)
        np.save(args.output.with_suffix('.image.npy'), rows)
    logits = model.run(tokens, rows, args.image_start)
    if not np.isfinite(logits).all():
        raise ValueError('Nonfinite FP32 reference logits')
    np.save(args.output, logits)
    rounded = bf(logits).astype(np.float32)
    args.output.with_suffix('.bf16').write_bytes((rounded.view(np.uint32) >> 16).astype('<u2').tobytes())
    metadata = {'compute': 'NumPy CPU float32 matmul/reductions/elementary functions; no Metal calls',
                'boundaries': args.boundaries, 'dequantization': 'FP32 q*scale+bias; separate multiply and add, no BF16 dequantized-weight cast',
                'matmul_order': 'NumPy BLAS implementation, not native GPU lane order',
                'erf': 'Binary64 libm erf on FP32 arguments then FP32 RNE',
                'shape': list(logits.shape), 'numpy': np.__version__, 'host': platform.machine(),
                'blas': getattr(np.__config__, 'CONFIG', {}),
                'thread_environment': {k: os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS',
                                       'VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS')},
                'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'accepted_threshold': None, 'fidelity_pass': None}
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
