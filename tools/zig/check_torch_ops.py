"""Compare framework-free CUDA C operators with the deployed Torch kernels by raw bytes."""

import argparse
from ops_build import compile_operators, digest
from ops_compare import RawCells
from ops_ffi import pointer, bind, P, U, I
import json
from pathlib import Path
import time

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    config = json.loads(args.config.read_text())
    config = config.get('text_config', config)
    hidden = int(config['hidden_size'])
    qkv = 2 * int(config['linear_num_key_heads']) * int(config['linear_key_head_dim']) + \
        int(config['linear_num_value_heads']) * int(config['linear_value_head_dim'])
    kv = int(config['num_key_value_heads']) * int(config.get('head_dim') or \
             hidden // int(config['num_attention_heads']))
    receipt = {'oracle': {'torch': torch.__version__, 'git_version': torch.version.git_version, 'cuda': torch.version.cuda},
               'config_sha256': digest(args.config), 'widths': [1, 2, 4, 8, 16, 64, 128],
               'columns': sorted({hidden, qkv, kv}), 'cells': [], 'outside_sources': [{'repo': 'pytorch/pytorch', 'version': '9186a08b2c12b534aa935ca92e3f94834939c06a', 'read': ['TensorTopK.cpp', 'TensorTopK.cu', 'SortingRadixSelect.cuh'], 'idea': 'Threshold selection then ordered strict/equal compaction; ours uses fixed8bit histograms and caller scratch independent of occupancy', 'statement': 'no code copied'}, {'repo': 'pytorch/pytorch', 'version': 'v2.14.0', 'read': ['SharedReduceOps.h'], 'idea': 'NaNs win and first indices break ties; ours orders raw IEEE words with a tested comparator', 'statement': 'no code copied'}, {'repo': 'pytorch/pytorch', 'version': '9186a08b2c12b534aa935ca92e3f94834939c06a', 'read': ['layer_norm_kernel.cu', 'thread_constants.h'], 'idea': 'Aligned four-value packets and declared reduction fold; ours narrows to the active BF165120 contract and validates dtype before launch', 'statement': 'no code copied'}, {'repo': 'pytorch/pytorch', 'version': '9186a08b2c12b534aa935ca92e3f94834939c06a', 'read': ['BinaryDivTrueKernel.cu', 'UnaryOpsKernel.cu'], 'idea': 'CPU-scalar reciprocal multiply and FP64 libdevice exp, preserved rounding without framework dispatch', 'statement': 'no code copied'}, {'repo': 'NVIDIA CUDA Math API', 'version': '13.4', 'idea': 'full-precision double intrinsics; runtime13 nvcc parity still gated', 'statement': 'no code copied'}],
               'provenance': 'TensorFold 0.6.5 source and CUDA SDK APIs; no code copied',
               'model_loaded': False, 'qualification': 'synthetic shape and ABI only; real-weight gate pending'}
    names = ('movement.cu', 'pointwise.cu', 'indexing.cu', 'grammar_mask.cu', 'nucleus.cu',
             'logprob.cu', 'argmax.cu', 'norm_rotary.cu', 'topk.cu')
    lib, compiled = compile_operators(args.source, args.out, names)
    receipt.update(compiled)
    gather = bind(lib, 'tf_gather_rows', [P, P, P, U, U, U, P, P])
    copy = bind(lib, 'tf_strided_copy', [P, P, U, U, U, U, U, U, U, P])
    spans = bind(lib, 'tf_copy_spans', [P, I, P])
    cast_up = bind(lib, 'tf_bf16_to_f32', [P, P, U, P])
    cast_down = bind(lib, 'tf_f32_to_bf16', [P, P, U, P])
    add = bind(lib, 'tf_tap_add', [P, P, P, U, P])
    torch.cuda.set_device(0)
    if torch.cuda.get_device_capability() != (12, 1):
        raise RuntimeError('This packet is qualified for sm_121 only')
    checker = RawCells(args.out, receipt)
    check = checker
    torch.manual_seed(141)
    stream = torch.cuda.current_stream()
    handle = P(stream.cuda_stream)


    for rows in receipt['widths']:
        for columns in receipt['columns']:
            x = torch.randn((rows, columns), device='cuda', dtype=torch.bfloat16)
            y = torch.randn_like(x)
            fp = torch.empty_like(x, dtype=torch.float32)
            check(f'cast-up/{rows}/{columns}', fp, x.float(), cast_up(pointer(x), pointer(fp), x.numel(), handle))
            bf = torch.empty_like(x)
            check(f'cast-down/{rows}/{columns}', bf, fp.to(torch.bfloat16), cast_down(pointer(fp), pointer(bf), fp.numel(), handle))
            taps = torch.empty_like(x)
            check(f'tap-add/{rows}/{columns}', taps, (x.float() + y.float()).to(torch.bfloat16),
                  add(pointer(x), pointer(y), pointer(taps), x.numel(), handle))
            picks = torch.tensor([rows - 1, 0, rows // 2, 0], device='cuda', dtype=torch.int64)
            selected = torch.empty((4, columns), device='cuda', dtype=x.dtype)
            invalid = torch.zeros(1, device='cuda', dtype=torch.int32)
            rc = gather(pointer(x), pointer(selected), pointer(picks), 4, rows, columns * 2, pointer(invalid), handle)
            check(f'gather/{rows}/{columns}', selected, x.index_select(0, picks), rc)
            if invalid.item():
                raise AssertionError('valid gather marked invalid')
            padded = torch.randn((rows, columns * 2), device='cuda', dtype=x.dtype)
            narrow = padded[:, :columns]
            dense = torch.empty_like(x)
            rc = copy(pointer(narrow), pointer(dense), 1, rows, columns * 2, 0,
                      padded.stride(0) * 2, 0, dense.stride(0) * 2, handle)
            check(f'contiguous/{rows}/{columns}', dense, narrow.contiguous(), rc)
            backing = torch.full((rows + 3, columns), -1, device='cuda', dtype=x.dtype)
            destination = backing[2:2 + rows]
            rc = copy(pointer(x), pointer(destination), 1, rows, columns * 2, 0,
                      x.stride(0) * 2, 0, destination.stride(0) * 2, handle)
            expected = torch.full_like(backing, -1)
            expected[2:2 + rows].copy_(x)
            check(f'kv-copy/{rows}/{columns}', backing, expected, rc)

    items, references, entries = [], [], []
    for columns in receipt['columns']:
        for dtype in (torch.bfloat16, torch.float32, torch.int32, torch.int64):
            source = torch.randint(0, 128, (3, columns), device='cuda', dtype=torch.int32).to(dtype)
            output = torch.empty_like(source)
            items.append((source, output))
            references.append(source.clone())
            entries.append([source.data_ptr(), output.data_ptr(), source.numel() * source.element_size()])
    table = torch.tensor(entries, device='cuda', dtype=torch.int64)
    rc = spans(pointer(table), len(items), handle)
    for j, ((source, output), expected) in enumerate(zip(items, references)):
        check(f'foreach-copy/{j}', output, expected, rc)
    words = torch.arange(65536, dtype=torch.int32, device='cuda').to(torch.int16)
    input_bf = words.view(torch.bfloat16)
    output_fp = torch.empty(65536, device='cuda', dtype=torch.float32)
    check('cast-up/all-bf16-encodings', output_fp, input_bf.float(),
          cast_up(pointer(input_bf), pointer(output_fp), 65536, handle))
    torch.manual_seed(180)
    int_bits = torch.randint(-(2 ** 31), 2 ** 31 - 1, (65536,), device='cuda', dtype=torch.int64).to(torch.int32)
    input_fp = int_bits.view(torch.float32)
    output_bf = torch.empty(65536, device='cuda', dtype=torch.bfloat16)
    check('cast-down/random-fp32-encodings', output_bf, input_fp.to(torch.bfloat16),
          cast_down(pointer(input_fp), pointer(output_bf), 65536, handle))
    src = torch.randn((4, hidden), device='cuda', dtype=torch.bfloat16)
    picks = torch.tensor([-1, 4], device='cuda', dtype=torch.int64)
    dst = torch.full((2, hidden), -1, device='cuda', dtype=torch.bfloat16)
    invalid = torch.zeros(1, device='cuda', dtype=torch.int32)
    rc = gather(pointer(src), pointer(dst), pointer(picks), 2, 4, hidden * 2, pointer(invalid), handle)
    check('gather/invalid-preserves-output', dst, torch.full_like(dst, -1), rc)
    if invalid.item() != 1:
        raise AssertionError('invalid gather did not report a device error')
    from check_index_ops import check_index_ops
    from check_nucleus_ops import check_nucleus_ops
    from check_logprob_ops import check_logprob_ops
    from check_argmax_ops import check_argmax_ops
    from check_norm_rotary_ops import check_norm_rotary_ops
    from check_topk_ops import check_topk_ops
    check_index_ops(lib, config, check)
    check_nucleus_ops(lib, config, check)
    check_logprob_ops(lib, config, check)
    argmax_cells = check_argmax_ops(lib, config, check)
    receipt['argmax_guards'] = [cell for cell in argmax_cells if cell.get('gpu_launch') is False]
    check_norm_rotary_ops(lib, config, check)
    check_topk_ops(lib, config, check)
    checker.finish(started)


if __name__ == '__main__':
    main()
