#!/usr/bin/env python3
"""Independent CPU FP64 LFM3 reference with explicit BF16 activation boundaries."""
import argparse
import json
import math
import struct
from pathlib import Path
import numpy as np


def bf(values):
    values = np.asarray(values, dtype=np.float64)
    bits = values.view(np.uint64)
    sign = ((bits >> np.uint64(48)) & np.uint64(0x8000)).astype(np.uint16)
    exponent = ((bits >> np.uint64(52)) & np.uint64(0x7ff)).astype(np.int32)
    fraction = bits & np.uint64((1 << 52) - 1)
    significant = fraction | np.uint64(1 << 52)
    high = significant >> np.uint64(45)
    tail = significant & np.uint64((1 << 45) - 1)
    increment = (tail > np.uint64(1 << 44)) | ((tail == np.uint64(1 << 44)) & ((high & 1) != 0))
    rounded = high + increment.astype(np.uint64)
    encoded = ((exponent.astype(np.int64) - 897) * 128 + rounded.astype(np.int64))
    encoded = np.clip(encoded, 0, 0x7f80).astype(np.uint16)
    small = np.abs(values) < np.float64(2.0 ** -126)
    with np.errstate(over='ignore', invalid='ignore'):
        subnormal = np.rint(np.where(small, np.abs(values), 0) * np.float64(2.0 ** 133)).astype(np.uint16)
    encoded = np.where(small, subnormal, encoded).astype(np.uint16)
    encoded = np.where(exponent == 0x7ff, np.where(fraction != 0, 0x7fc0, 0x7f80), encoded).astype(np.uint16)
    return ((encoded | sign).astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def inject_image_rows(hidden, tokens, rows, start):
    rows = np.asarray(rows, dtype=np.float64)
    if start is None or start < 0 or rows.ndim != 2 or not len(rows):
        raise ValueError('Expected nonempty projected image rows and nonnegative image start')
    if rows.shape[1] != hidden.shape[1] or start + len(rows) > len(hidden):
        raise ValueError('Projected image rows exceed decoder geometry')
    expected = np.zeros(len(tokens), dtype=bool)
    expected[start:start + len(rows)] = True
    if not np.array_equal(np.asarray(tokens) == 125030, expected):
        raise ValueError('Image placeholder rows do not match projected image span')
    if not np.isfinite(rows).all():
        raise ValueError('Projected image rows must be finite')
    result = hidden.copy()
    result[start:start + len(rows)] = bf(rows)
    return result


class Oracle:
    def __init__(self, directory, weight_mode='exact', dump=None, projection_rounding='fp64',
                 norm_conv_rounding='fp64', attention_rounding='fp64', elementary_backend='numpy'):
        self.directory, self.weight_mode, self.dump = directory, weight_mode, dump
        if attention_rounding not in ('fp64', 'metal-fp32') or elementary_backend not in ('numpy', 'metal'):
            raise ValueError('Unsupported attention or elementary rounding mode')
        self.attention_rounding, self.elementary_backend = attention_rounding, elementary_backend
        self.projection_rounding = projection_rounding
        if norm_conv_rounding not in ('fp64', 'metal'):
            raise ValueError('Unsupported normalization/convolution rounding mode')
        self.norm_conv_rounding = norm_conv_rounding
        valid_rounding = projection_rounding in ('fp64', 'metal-row')
        if not valid_rounding or (projection_rounding == 'metal-row' and weight_mode != 'exact'):
            raise ValueError('Metal row replay requires exact affine weights')
        directory = directory.resolve()
        self.config = json.loads((directory / 'config.json').read_text())
        self.text = self.config.get('text_config', self.config)
        if self.text.get('model_type') != 'lfm3' or not self.text.get('tie_word_embeddings'):
            raise ValueError('Expected an LFM3 decoder with a tied vocabulary embedding')
        index = json.loads((directory / 'model.safetensors.index.json').read_text())
        self.map, self.files = index['weight_map'], {}
        for name in set(self.map.values()):
            path = (directory / name).resolve()
            if not path.is_relative_to(directory):
                raise ValueError('Checkpoint shard escapes the model directory')
            with path.open('rb') as handle:
                length = struct.unpack('<Q', handle.read(8))[0]
                self.files[name] = (path, 8 + length, json.loads(handle.read(length)))
        self.prefix = 'language_model.model.'

    def tensor(self, name, rows=None):
        path, base, header = self.files[self.map[name]]
        item = header[name]
        shape = tuple(item['shape'])
        dtype = {'U32': '<u4', 'BF16': '<u2', 'F32': '<f4'}[item['dtype']]
        if math.prod(shape) * np.dtype(dtype).itemsize != item['data_offsets'][1] - item['data_offsets'][0]:
            raise ValueError('Tensor byte count does not match its shape')
        values = np.memmap(path, mode='r', dtype=dtype, offset=base + item['data_offsets'][0], shape=shape)
        if rows is not None:
            values = values[rows]
        if item['dtype'] == 'BF16':
            return (values.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        return np.asarray(values)

    def quant(self, name):
        config = self.config.get('quantization', self.config.get('quantization_config'))
        if not isinstance(config, dict):
            raise ValueError('Expected affine quantization metadata')
        def clean(key):
            key = key.removesuffix('.weight')
            for prefix in ('model.language_model.', 'language_model.', 'text_model.', 'model.'):
                if key.startswith(prefix):
                    key = key[len(prefix):]
            return key
        for key, value in config.items():
            if isinstance(value, dict) and clean(key) == clean(name):
                config = value
                break
        bits, group = config['bits'], config['group_size']
        if bits not in (4, 6, 8) or group not in (32, 64) or config.get('mode', 'affine') != 'affine':
            raise ValueError('Unsupported affine bit width, group size or mode')
        return bits, group

    def weights(self, name, rows):
        bits, group = self.quant(name)
        words = self.tensor(name + '.weight', rows)
        width = words.shape[1] * 32 // bits
        if width * bits != words.shape[1] * 32 or width % group:
            raise ValueError('Packed weight width does not match quantization geometry')
        positions = np.arange(width) * bits
        indices, shifts = positions // 32, positions % 32
        padded = np.pad(words.astype(np.uint64), ((0, 0), (0, 1)))
        values = padded[:, indices] >> shifts.astype(np.uint64)
        cross = shifts + bits > 32
        values[:, cross] |= padded[:, indices[cross] + 1] << (32 - shifts[cross]).astype(np.uint64)
        codes = (values & ((1 << bits) - 1)).astype(np.float64)
        scale = self.tensor(name + '.scales', rows).repeat(group, axis=-1)
        bias = self.tensor(name + '.biases', rows).repeat(group, axis=-1)
        weight = scale * codes + bias
        return bf(weight) if self.weight_mode == 'bf16' else weight

    def linear(self, x, name):
        if self.projection_rounding == 'metal-row':
            from row_projection_reference import affine_rows
            return bf(affine_rows(self, name, x))
        path, base, header = self.files[self.map[name + '.weight']]
        count = header[name + '.weight']['shape'][0]
        output = np.empty((*x.shape[:-1], count), dtype=np.float64)
        for start in range(0, count, 512):
            end = min(start + 512, count)
            output[..., start:end] = x @ self.weights(name, slice(start, end)).T
        return bf(output)

    def norm(self, values, name):
        if self.norm_conv_rounding == 'metal':
            from text_rounding_reference import norm_rows
            return norm_rows(values, self.tensor(name + '.weight'), self.text['norm_eps'],
                             self.elementary_backend if self.attention_rounding == 'metal-fp32' else None)
        inverse = 1 / np.sqrt(np.mean(values * values, axis=-1, keepdims=True) + self.text['norm_eps'])
        return bf(bf(values * inverse) * self.tensor(name + '.weight'))

    def take(self, name, value):
        if self.dump:
            self.dump(name, value)

    def rotary(self, values, kind):
        if getattr(self, 'attention_rounding', 'fp64') == 'metal-fp32':
            from text_attention_rounding import rotary
            return rotary(values, self.text['rope_parameters'][kind], self.elementary_backend)
        width = values.shape[-1]
        spec = self.text['rope_parameters'][kind]
        pairs = int(width * spec.get('partial_rotary_factor', 1)) // 2
        frequencies = np.concatenate((np.power(float(spec['rope_theta']), -np.arange(pairs) * 2 / width),
                                      np.zeros(width // 2 - pairs)))
        angle = np.arange(len(values))[:, None] * frequencies
        angle = np.concatenate((angle, angle), axis=-1)[:, None, :]
        cosine, sine = bf(np.cos(angle)), bf(np.sin(angle))
        a, b = np.split(values, 2, axis=-1)
        return bf(values * cosine + np.concatenate((-b, a), axis=-1) * sine)

    def conv(self, x, prefix, tag):
        projected = self.linear(x, prefix + 'conv.in_proj')
        self.take(tag + '_conv_in', projected)
        b, gate, value = np.split(projected, 3, axis=-1)
        gated = bf(b * value)
        kernel = self.tensor(prefix + 'conv.conv.weight')[:, :, 0]
        taps = kernel.shape[1]
        padded = np.pad(gated, ((taps - 1, 0), (0, 0)))
        total = np.zeros_like(gated)
        for tap in range(taps):
            total = total + padded[tap:tap + len(x)] * kernel[:, tap]
            if self.norm_conv_rounding == 'metal':
                total = total.astype(np.float32).astype(np.float64)
        mixed = bf(bf(total) * gate)
        self.take(tag + '_conv_gated', mixed)
        return self.linear(mixed, prefix + 'conv.out_proj')

    def attention(self, x, prefix, kind):
        heads, kv, width = (self.text[key] for key in ('num_attention_heads', 'num_key_value_heads', 'head_dim'))
        q = self.linear(x, prefix + 'self_attn.q_proj').reshape(len(x), heads, width)
        k = self.linear(x, prefix + 'self_attn.k_proj').reshape(len(x), kv, width)
        v = self.linear(x, prefix + 'self_attn.v_proj').reshape(len(x), kv, width)
        q = self.rotary(self.norm(q, prefix + 'self_attn.q_norm'), kind).transpose(1, 0, 2)
        k = self.rotary(self.norm(k, prefix + 'self_attn.k_norm'), kind).transpose(1, 0, 2)
        v = v.transpose(1, 0, 2)
        k, v = np.repeat(k, heads // kv, axis=0), np.repeat(v, heads // kv, axis=0)
        if self.attention_rounding == 'metal-fp32':
            from text_attention_rounding import attention
            output = attention(q, k, v, self.text['sliding_window'] if kind == 'sliding_attention' else 0,
                               self.elementary_backend).transpose(1, 0, 2)
        else:
            scores = (q @ k.swapaxes(-1, -2)) / math.sqrt(width)
            distance = np.arange(len(x))[:, None] - np.arange(len(x))[None, :]
            allowed = distance >= 0
            if kind == 'sliding_attention':
                allowed &= distance < self.text['sliding_window']
            scores = np.where(allowed, scores, -np.inf)
            probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
            probabilities /= probabilities.sum(axis=-1, keepdims=True)
            output = bf((probabilities @ v).transpose(1, 0, 2))
        raw_gate = self.linear(x, prefix + 'self_attn.gate_proj')
        if self.attention_rounding == 'metal-fp32':
            from elementary_reference import elementary
            gates = bf(np.float32(1) / (np.float32(1)
                       + elementary(-raw_gate.astype(np.float32), 'exp', self.elementary_backend)))
        else:
            gates = bf(1 / (1 + np.exp(-raw_gate)))
        return self.linear(bf(output * gates[:, :, None]).reshape(len(x), heads * width), prefix + 'self_attn.o_proj')

    def run(self, tokens, image_rows=None, image_start=None):
        hidden = bf(self.weights(self.prefix + 'embed_tokens', tokens))
        if image_rows is not None:
            hidden = inject_image_rows(hidden, tokens, image_rows, image_start)
        elif image_start is not None:
            raise ValueError('Image start requires projected image rows')
        self.take('embedding', hidden)
        for index, kind in enumerate(self.text['layer_types']):
            tag = f'layer_{index:02d}'
            prefix = self.prefix + f'layers.{index}.'
            x = self.norm(hidden, prefix + 'operator_norm')
            self.take(tag + '_operator_norm', x)
            result = self.conv(x, prefix, tag) if kind == 'conv' else self.attention(x, prefix, kind)
            self.take(tag + '_mixer', result)
            hidden = bf(hidden + self.norm(result, prefix + 'post_operator_norm'))
            self.take(tag + '_post_operator', hidden)
            x = self.norm(hidden, prefix + 'ffn_norm')
            self.take(tag + '_ffn_norm', x)
            gate, up = self.linear(x, prefix + 'feed_forward.w1'), self.linear(x, prefix + 'feed_forward.w3')
            self.take(tag + '_gate', gate)
            self.take(tag + '_up', up)
            if self.attention_rounding == 'metal-fp32':
                from elementary_reference import swiglu
                act = swiglu(gate, up, self.elementary_backend)
            else:
                act = bf(bf(gate / (1 + np.exp(-gate))) * up)
            self.take(tag + '_act', act)
            result = self.linear(act, prefix + 'feed_forward.w2')
            self.take(tag + '_down', result)
            hidden = bf(hidden + self.norm(result, prefix + 'post_ffn_norm'))
            self.take(tag + '_output', hidden)
            print(tag, kind, flush=True)
        final = self.norm(hidden, self.prefix + 'embedding_norm')
        self.take('final_norm', final)
        return self.linear(final, self.prefix + 'embed_tokens')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--tokens', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--weight-mode', choices=('exact', 'bf16'), default='exact')
    parser.add_argument('--projection-rounding', choices=('fp64', 'metal-row'), default='fp64')
    parser.add_argument('--attention-rounding', choices=('fp64', 'metal-fp32'), default='fp64')
    parser.add_argument('--elementary-backend', choices=('numpy', 'metal'), default='numpy')
    parser.add_argument('--norm-conv-rounding', choices=('fp64', 'metal'), default='fp64')
    parser.add_argument('--trace-dir', type=Path)
    parser.add_argument('--image-rows', type=Path, help='Projected rows as a two-dimensional numeric NumPy array')
    parser.add_argument('--image-start', type=int)
    args = parser.parse_args()
    def write(path, values):
        path.write_bytes((np.asarray(values, dtype=np.float32).view(np.uint32) >> 16).astype('<u2').tobytes())
    def dump(name, values):
        write(args.trace_dir / (name + '.bf16'), values)
    if args.trace_dir:
        args.trace_dir.mkdir(parents=True, exist_ok=True)
    oracle = Oracle(args.model_dir, args.weight_mode, dump if args.trace_dir else None, args.projection_rounding,
                    args.norm_conv_rounding, args.attention_rounding, args.elementary_backend)
    tokens = np.load(args.tokens, allow_pickle=False)
    if tokens.ndim != 1 or not np.issubdtype(tokens.dtype, np.integer) or tokens.size == 0:
        parser.error('Expected a nonempty one-dimensional integer token array')
    if tokens.min() < 0 or tokens.max() >= oracle.text['vocab_size']:
        parser.error('Token IDs exceed the decoder vocabulary')
    if tokens.size > oracle.text['max_position_embeddings']:
        parser.error('Token array exceeds the decoder context')
    image_rows = np.load(args.image_rows, allow_pickle=False) if args.image_rows else None
    if (args.image_rows is None) != (args.image_start is None):
        parser.error('--image-rows and --image-start must be supplied together')
    logits = oracle.run(tokens, image_rows, args.image_start)
    if not np.isfinite(logits).all():
        raise ValueError('Reference produced nonfinite logits')
    write(args.output, logits)
    arithmetic = 'FP64 reductions with direct binary64-to-BF16 RNE activation boundaries'
    if args.projection_rounding == 'metal-row':
        arithmetic += '; affine projections instead replay the Metal FP32 lane rounding, other operations remain FP64'
    if args.norm_conv_rounding == 'metal':
        arithmetic += '; normalization and convolution replay the production FP32 reduction sites'
    if args.attention_rounding == 'metal-fp32':
        arithmetic += ('; bounded decoder attention/rotary/activations replay FP32 sites with explicit '
                       + args.elementary_backend + ' elementary functions')
    metadata = {'dtype': 'BF16', 'arithmetic': arithmetic, 'weight_mode': args.weight_mode,
                'attention_rounding': args.attention_rounding, 'elementary_backend': args.elementary_backend,
                'projection_rounding': args.projection_rounding, 'norm_conv_rounding': args.norm_conv_rounding,
                'shape': list(logits.shape),
                'accepted_threshold': None}
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')

if __name__ == '__main__':
    main()
