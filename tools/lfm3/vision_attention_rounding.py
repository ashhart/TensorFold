#!/usr/bin/env python3
"""Diagnose production elementary rounding; never declare an independent FP64 fidelity pass."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from elementary_reference import fma
from text_rounding_reference import sum32


def read(directory, name, rows, width):
    words = np.fromfile(directory / (name + '.bf16'), dtype='<u2')
    return (words.astype(np.uint32) << 16).view(np.float32).reshape(rows, width)


def capture(directory, output, rows, shader):
    import mlx.core as mx
    source = shader.read_text()
    kernel = mx.fast.metal_kernel(name='lfm3_vision_attention_rounding_trace',
        input_names=['Q', 'K', 'V'], output_names=['Raw', 'Context', 'Table'], header=source, source='''
uint lane=thread_index_in_simdgroup,row=threadgroup_position_in_grid.y,head=threadgroup_position_in_grid.x;
float3 sums=l3v_attention_row(Q,K,V,PATCHES,head,row,lane);
uint base=(row*16+head)*65;Raw[base+lane]=sums.x;Raw[base+lane+32]=sums.y;if(lane==0)Raw[base+64]=sums.z;
uint qi=(row*16+head)*64+lane;Context[qi]=float(bfloat(sums.x/sums.z));Context[qi+32]=float(bfloat(sums.y/sums.z));
float qa=float(Q[qi]),qb=float(Q[qi+32]),maximum=-INFINITY;
for(uint key=0;key<PATCHES;key++){
 uint ki=(key*8+head/2)*64+lane;float score=l3v_attention_dot(qa,qb,K+ki)*0.125f;
 float next=max(maximum,score),alpha=maximum==-INFINITY?0.0f:metal::precise::exp(maximum-next);
 float probability=score==-INFINITY?0.0f:metal::precise::exp(score-next);
 if(lane==0){uint at=((row*16+head)*PATCHES+key)*3;Table[at]=score;Table[at+1]=alpha;Table[at+2]=probability;}
 maximum=next;
}''')
    inputs = [mx.array(read(directory, 'first_attention_' + name, rows, width)).astype(mx.bfloat16)
              for name, width in [('q', 1024), ('k', 512), ('v', 512)]]
    raw, context, table = kernel(inputs=inputs, template=[('PATCHES', rows)], grid=(16 * 32, rows, 1),
        threadgroup=(32, 1, 1), output_shapes=[(rows, 16, 65), (rows, 16, 64), (rows, 16, rows, 3)],
        output_dtypes=[mx.float32] * 3)
    mx.eval(raw, context, table)
    for name, value in [('raw', raw), ('context', context), ('elementary', table)]:
        np.save(output / (name + '.npy'), np.array(value))
    (output / 'capture.json').write_text(json.dumps({
        'shader_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'scope': 'Actual production helper diagnostic, not independent FP64 reference'}, indent=2) + '\n')


def compare(directory, output, rows):
    table = np.load(output / 'elementary.npy', allow_pickle=False)
    actual_raw = np.load(output / 'raw.npy', allow_pickle=False)
    actual_context = np.load(output / 'context.npy', allow_pickle=False)
    q = read(directory, 'first_attention_q', rows, 1024).reshape(rows, 16, 64).transpose(1, 0, 2)
    k = np.repeat(read(directory, 'first_attention_k', rows, 512).reshape(rows, 8, 64), 2, axis=1).transpose(1, 0, 2)
    v = np.repeat(read(directory, 'first_attention_v', rows, 512).reshape(rows, 8, 64), 2, axis=1)
    score_differences = 0
    for start in range(0, rows, 16):
        query = q[:, start:start + 16]
        products = fma(query[:, :, None, 32:], k[:, None, :, 32:], query[:, :, None, :32] * k[:, None, :, :32])
        scores = sum32(products) * np.float32(0.125)
        score_differences += int(np.count_nonzero(scores.transpose(1, 0, 2) != table[start:start + 16, :, :, 0]))
    result = np.zeros((rows, 16, 64), dtype=np.float32)
    total = np.zeros((rows, 16), dtype=np.float32)
    maximum = np.full((rows, 16), -np.inf, dtype=np.float32)
    exp_differences, maximum_ulp = 0, 0
    for key in range(rows):
        score, alpha, probability = (table[:, :, key, i] for i in range(3))
        next_maximum = np.maximum(maximum, score)
        expected = np.stack((np.exp((maximum - next_maximum).astype(np.float64)).astype(np.float32),
                             np.exp((score - next_maximum).astype(np.float64)).astype(np.float32)), axis=-1)
        observed = table[:, :, key, 1:]
        ulp = np.abs(expected.view(np.int32).astype(np.int64) - observed.view(np.int32).astype(np.int64))
        exp_differences += int(np.count_nonzero(ulp))
        maximum_ulp = max(maximum_ulp, int(ulp.max()))
        total = fma(total, alpha, probability)
        result = fma(result, alpha[..., None], probability[..., None] * v[key, None])
        maximum = next_maximum
    replay = np.concatenate((result, total[..., None]), axis=-1)
    native = read(directory, 'first_attention_context', rows, 1024).reshape(rows, 16, 64)
    report = {'scope': 'Fixed native Q/K/V, independently replayed FP32 arithmetic with captured elementary functions',
              'rows': rows, 'score_values': rows * 16 * rows, 'score_differences': score_differences,
              'raw_values': int(replay.size), 'raw_differences': int(np.count_nonzero(replay != actual_raw)),
              'helper_context_differences': int(np.count_nonzero(actual_context != native)),
              'exp_values': rows * 16 * rows * 2, 'exp_differences_vs_fp64_rounded': exp_differences,
              'maximum_exp_ulp_vs_fp64_rounded': maximum_ulp, 'accepted_threshold': None,
              'whole_model_fidelity_pass': None}
    (output / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-trace', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--rows', type=int, required=True)
    parser.add_argument('--capture', action='store_true',
                        help='Requires local MLX/Metal; otherwise compare existing captures')
    parser.add_argument('--shader', type=Path,
                        default=Path(__file__).resolve().parents[2] / 'zig/kernels/metal/lfm3/vision.metal')
    args = parser.parse_args()
    if not 0 < args.rows <= 16384:
        raise ValueError('Expected bounded positive patch count')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.capture:
        capture(args.native_trace, args.output_dir, args.rows, args.shader)
    compare(args.native_trace, args.output_dir, args.rows)


if __name__ == '__main__':
    main()
