#!/usr/bin/env python3
"""Independent CPU FP64 vision tower and projector with explicit BF16 activation boundaries."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from fp64_reference import Oracle, bf


class VisionOracle:
    def __init__(self, directory, dump=None, dense_rounding='fp64',
                 operator_rounding='fp64', elementary_backend='numpy'):
        if dense_rounding not in ('fp64', 'metal-row'):
            raise ValueError('Unsupported dense projection rounding mode')
        if operator_rounding not in ('fp64', 'metal-fp32'):
            raise ValueError('Unsupported vision operator rounding mode')
        if (elementary_backend not in ('numpy', 'metal')
                or (elementary_backend == 'metal' and operator_rounding != 'metal-fp32')):
            raise ValueError('Metal elementary functions require named FP32 operator replay')
        self.elementary_backend = elementary_backend
        self.operator_rounding = operator_rounding
        self.dense_rounding = dense_rounding
        self.loader = Oracle(directory)
        self.config = self.loader.config
        if self.config.get('model_type') != 'lfm3_vl':
            raise ValueError('Expected an LFM3-VL checkpoint')
        self.vision, self.dump = self.config['vision_config'], dump

    def weight(self, name):
        return self.loader.tensor(name)

    def dense(self, x, name):
        weight = self.weight(name + '.weight')
        if getattr(self, 'dense_rounding', 'fp64') == 'metal-row':
            from row_projection_reference import dense_rows
            result = dense_rows(x, weight)
        else:
            result = x @ weight.T
        if name + '.bias' in self.loader.map:
            result += self.weight(name + '.bias').astype(result.dtype)
        return bf(result)

    def rms(self, x, name):
        if getattr(self, 'operator_rounding', 'fp64') == 'metal-fp32':
            from vision_rounding_reference import rms_rows
            return rms_rows(x, self.weight(name + '.weight'), self.vision['norm_eps'], self.elementary_backend)
        inverse = 1 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + self.vision['norm_eps'])
        return bf(bf(x * inverse) * self.weight(name + '.weight'))

    def take(self, name, value):
        if self.dump:
            self.dump(name, value)

    def convolution(self, x, prefix, height, width):
        left, gate, value = np.split(self.dense(x, prefix + 'conv.in_proj'), 3, axis=-1)
        if prefix.endswith('layers.0.'):
            self.take('first_conv_in', np.concatenate((left, gate, value), axis=-1))
        channels = self.vision['hidden_size']
        grid = bf(left * value).reshape(height // 2, width // 2, 2, 2, channels)
        grid = grid.transpose(0, 2, 1, 3, 4).reshape(height, width, channels)
        padded = np.pad(grid, ((1, 1), (1, 1), (0, 0)))
        kernel = self.weight(prefix + 'conv.spatial_conv.weight')[:, :, :, 0]
        result = np.zeros_like(grid)
        for row in range(3):
            for column in range(3):
                result += padded[row:row + height, column:column + width] * kernel[:, row, column]
        if getattr(self, 'operator_rounding', 'fp64') == 'metal-fp32':
            from vision_rounding_reference import convolution
            result = convolution(grid, kernel)
        merged = bf(result).reshape(height // 2, 2, width // 2, 2, channels)
        merged = merged.transpose(0, 2, 1, 3, 4).reshape(height * width, channels)
        mixed = bf(merged * gate)
        if prefix.endswith('layers.0.'):
            self.take('first_conv_gated', mixed)
        return self.dense(mixed, prefix + 'conv.out_proj')

    def rotary(self, values, coordinates):
        if getattr(self, 'operator_rounding', 'fp64') == 'metal-fp32':
            from vision_rounding_reference import rotary
            return rotary(values, coordinates, self.elementary_backend)
        width = values.shape[-1]
        spatial = width // 2
        frequency = np.power(float(self.vision['rope_parameters']['rope_theta']),
                             -np.arange(0, spatial, 2, dtype=np.float64) / spatial)
        vertical = coordinates[:, 0, None] * frequency
        horizontal = coordinates[:, 1, None] * frequency
        angles = np.concatenate((vertical, horizontal, vertical, horizontal), axis=-1)[:, None]
        first, second = np.split(values, 2, axis=-1)
        return bf(values * np.cos(angles) + np.concatenate((-second, first), axis=-1) * np.sin(angles))

    def attention(self, x, prefix, coordinates):
        heads, kv = self.vision['num_attention_heads'], self.vision['num_key_value_heads']
        dim = self.vision['hidden_size'] // heads
        q = self.rotary(self.rms(self.dense(x, prefix + 'self_attn.q_proj').reshape(len(x), heads, dim),
                                prefix + 'self_attn.q_layernorm'), coordinates).transpose(1, 0, 2)
        k = self.rotary(self.rms(self.dense(x, prefix + 'self_attn.k_proj').reshape(len(x), kv, dim),
                                prefix + 'self_attn.k_layernorm'), coordinates).transpose(1, 0, 2)
        v = self.dense(x, prefix + 'self_attn.v_proj').reshape(len(x), kv, dim).transpose(1, 0, 2)
        if prefix.endswith('layers.2.'):
            self.take('first_attention_norm', x)
            self.take('first_attention_q', q.transpose(1, 0, 2))
            self.take('first_attention_k', k.transpose(1, 0, 2))
            self.take('first_attention_v', v.transpose(1, 0, 2))
        k, v = np.repeat(k, heads // kv, axis=0), np.repeat(v, heads // kv, axis=0)
        if getattr(self, 'operator_rounding', 'fp64') == 'metal-fp32':
            from vision_rounding_reference import attention
            value = attention(q, k, v, self.elementary_backend).transpose(1, 0, 2).reshape(len(x), heads * dim)
        else:
            scores = (q @ k.swapaxes(-1, -2)) / math.sqrt(dim)
            probability = np.exp(scores - scores.max(axis=-1, keepdims=True))
            probability /= probability.sum(axis=-1, keepdims=True)
            value = bf((probability @ v).transpose(1, 0, 2)).reshape(len(x), heads * dim)
        if prefix.endswith('layers.2.'):
            self.take('first_attention_context', value)
        result = self.dense(value, prefix + 'self_attn.out_proj')
        if prefix.endswith('layers.2.'):
            self.take('first_attention_operator', result)
        return result

    def run(self, patches, height, width):
        if height <= 0 or width <= 0 or height % 2 or width % 2 or height * width > 16384:
            raise ValueError('Expected a valid even patch grid')
        if patches.shape != (height * width, 1536) or not np.isfinite(patches).all():
            raise ValueError('Expected finite merge-ordered patches with 1536 elements each')
        coordinates = np.array([(h * 2 + a, w * 2 + b) for h in range(height // 2)
                                for w in range(width // 2) for a in range(2) for b in range(2)])
        hidden = self.dense(bf(patches), 'vision_tower.patchifier')
        self.take('patchifier', hidden)
        hidden = self.rms(hidden, 'vision_tower.embedding_norm')
        self.take('embedding_norm', hidden)
        for index, kind in enumerate(self.vision['layer_types']):
            prefix = f'vision_tower.layers.{index}.'
            x = self.rms(hidden, prefix + 'operator_norm')
            if index == 0:
                self.take('first_operator_norm', x)
            result = (self.convolution(x, prefix, height, width) if kind == 'conv'
                      else self.attention(x, prefix, coordinates))
            if index == 0:
                self.take('first_operator', result)
            hidden = bf(hidden + result)
            if index == 0:
                self.take('first_residual', hidden)
            x = self.rms(hidden, prefix + 'ffn_norm')
            if index == 0:
                self.take('first_ffn_norm', x)
            gate, up = self.dense(x, prefix + 'feed_forward.w1'), self.dense(x, prefix + 'feed_forward.w3')
            if index == 0:
                self.take('first_ffn_gate', gate)
                self.take('first_ffn_up', up)
            if getattr(self, 'operator_rounding', 'fp64') == 'metal-fp32':
                from elementary_reference import swiglu
                act = swiglu(gate, up, self.elementary_backend)
            else:
                act = bf(bf(gate / (1 + np.exp(-gate))) * up)
            if index == 0:
                self.take('first_ffn_act', act)
            down = self.dense(act, prefix + 'feed_forward.w2')
            if index == 0:
                self.take('first_ffn_down', down)
            hidden = bf(hidden + down)
            self.take(f'layer_{index:02d}', hidden)
            print(f'vision_layer_{index:02d}', flush=True)
        prefix = 'multi_modal_projector.'
        centered = hidden - hidden.mean(axis=-1, keepdims=True)
        inverse = 1 / np.sqrt(np.mean(centered * centered, axis=-1, keepdims=True)
                              + self.config['projector_layer_norm_eps'])
        normalized = bf(centered * inverse * self.weight(prefix + 'layer_norm.weight')
                        + self.weight(prefix + 'layer_norm.bias'))
        if getattr(self, 'operator_rounding', 'fp64') == 'metal-fp32':
            from vision_rounding_reference import layer_norm
            normalized = layer_norm(hidden, self.weight(prefix + 'layer_norm.weight'),
                                    self.weight(prefix + 'layer_norm.bias'), self.config['projector_layer_norm_eps'],
                                    self.elementary_backend)
        self.take('projector_norm', normalized)
        merged = normalized.reshape(-1, self.vision['hidden_size'] * self.config['downsample_factor'] ** 2)
        first = self.dense(merged, prefix + 'linear_1')
        erf = np.fromiter((math.erf(v / math.sqrt(2)) for v in first.flat), dtype=np.float64).reshape(first.shape)
        activated = bf(0.5 * first * (1 + erf))
        if getattr(self, 'operator_rounding', 'fp64') == 'metal-fp32':
            from vision_rounding_reference import gelu
            activated = gelu(first, self.elementary_backend)
        self.take('projector_gelu', activated)
        return self.dense(activated, prefix + 'linear_2')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--patches', type=Path, required=True)
    parser.add_argument('--grid-h', type=int, required=True)
    parser.add_argument('--grid-w', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trace-dir', type=Path)
    parser.add_argument('--elementary-backend', choices=('numpy', 'metal'), default='numpy')
    parser.add_argument('--operator-rounding', choices=('fp64', 'metal-fp32'), default='fp64')
    parser.add_argument('--dense-rounding', choices=('fp64', 'metal-row'), default='fp64')
    args = parser.parse_args()
    def write(path, values):
        path.write_bytes((np.asarray(values, dtype=np.float32).view(np.uint32) >> 16).astype('<u2').tobytes())
    def dump(name, values):
        write(args.trace_dir / (name + '.bf16'), values)
    if args.trace_dir:
        args.trace_dir.mkdir(parents=True, exist_ok=True)
    oracle = VisionOracle(args.model_dir, dump if args.trace_dir else None, args.dense_rounding,
                          args.operator_rounding, args.elementary_backend)
    result = oracle.run(np.load(args.patches, allow_pickle=False), args.grid_h, args.grid_w)
    if not np.isfinite(result).all():
        raise ValueError('Reference produced nonfinite image rows')
    write(args.output, result)
    args.output.with_suffix('.json').write_text(json.dumps({'shape': list(result.shape), 'dtype': 'BF16',
        'arithmetic': ('Independent CPU arithmetic; ideal FP64 unless named FP32 replay is selected; '
                       'Metal elementary backend delegates only exp/exp2/cos/sin/rsqrt primitives'),
        'dense_rounding': args.dense_rounding, 'operator_rounding': args.operator_rounding,
        'elementary_backend': args.elementary_backend, 'accepted_threshold': None}, indent=2) + '\n')


if __name__ == '__main__':
    main()
