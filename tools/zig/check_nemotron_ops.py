"""Compare the framework-free operators needed by Nemotron's CUDA round with Torch."""

import argparse
import ctypes
from ops_ffi import pointer, bind, P, U
from ops_build import compile_operators, digest
from ops_compare import RawCells
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
    hidden, vocab = int(config['hidden_size']), int(config['vocab_size'])
    xd = int(config['mamba_num_heads']) * int(config['mamba_head_dim'])
    conv = xd + 2 * int(config['n_groups']) * int(config['ssm_state_size'])
    head_dim = int(config.get('head_dim') or hidden // int(config['num_attention_heads']))
    kv = int(config['num_key_value_heads']) * head_dim
    columns = sorted({hidden, conv, kv})
    receipt = {'family': 'nemotron_h', 'oracle_revision': '1ce094aa0b68fcec241fb6c3653aecf4939f9785',
               'oracle': {'torch': torch.__version__, 'git_version': torch.version.git_version,
                          'cuda': torch.version.cuda}, 'config_sha256': digest(args.config),
               'columns': columns, 'cells': [], 'model_loaded': False,
               'qualification': 'Synthetic Nemotron shapes only; captured real-weight values are a separate gate',
               'sources_read': [{'repo': 'pytorch/pytorch', 'version': '9186a08b2c12b534aa935ca92e3f94834939c06a', 'files': ['TensorTopK.cpp', 'TensorTopK.cu', 'SortingRadixSelect.cuh'], 'idea': 'Integer threshold classes then ordered compaction; our fixed8bit histograms and explicit scratch', 'statement': 'no code copied'}, {'repo': 'pytorch/pytorch', 'version': 'v2.14.0',
                                 'files': ['SharedReduceOps.h'], 'idea': 'First maximum and first NaN ties; our raw-word comparator',
                                 'statement': 'no code copied'}]}
    names = ('movement.cu', 'pointwise.cu', 'indexing.cu', 'grammar_mask.cu', 'argmax.cu', 'topk.cu',
             'nemotron_constants.cu', 'nemotron_sampler_core.cu', 'argmax_f64.cu', 'cast_f64.cu',
             'nemotron_limits.cu')
    lib, compiled = compile_operators(args.source, args.out, names)
    receipt.update(compiled)
    torch.cuda.set_device(0)
    if torch.cuda.get_device_capability() != (12, 1):
        raise RuntimeError('This packet qualifies sm_121 only')
    torch.manual_seed(141)
    active = torch.cuda.current_stream()
    stream = P(active.cuda_stream)


    checker = RawCells(args.out, receipt)
    check = checker
    add = bind(lib, 'tf_tap_add', [P, P, P, U, P])
    narrow = bind(lib, 'tf_narrow_indices', [P, P, U, P])
    argmax32 = bind(lib, 'tf_argmax_rows_i32', [P, P, ctypes.c_int64, ctypes.c_int64,
                                        ctypes.c_int64, ctypes.c_int32, P])
    copy = bind(lib, 'tf_strided_copy', [P, P, U, U, U, U, U, U, U, P])
    gather = bind(lib, 'tf_gather_rows', [P, P, P, U, U, U, P, P])
    for rows in (1, 2, 4, 8, 16):
        for width in columns:
            source = torch.randn((rows, width), device='cuda', dtype=torch.bfloat16)
            delta = torch.randn_like(source)
            output = torch.empty_like(source)
            check(f'BF16-add/{rows}/{width}', output, source + delta,
                  add(pointer(source), pointer(delta), pointer(output), source.numel(), stream))
            original = source.clone()
            check(f'BF16-inplace-add/{rows}/{width}', source, original + delta,
                  add(pointer(source), pointer(delta), pointer(source), source.numel(), stream))
            storage = torch.randn((rows, 2 * width), device='cuda', dtype=source.dtype)
            view = storage[:, :width]
            copied = torch.empty_like(source)
            rc = copy(pointer(view), pointer(copied), 1, rows, 2 * width, 0,
                      storage.stride(0) * 2, 0, copied.stride(0) * 2, stream)
            check(f'contiguous/{rows}/{width}', copied, view.contiguous(), rc)
            picks = torch.tensor([rows - 1, 0, rows // 2], device='cuda', dtype=torch.int64)
            selected = torch.empty((3, width), device='cuda', dtype=source.dtype)
            invalid = torch.zeros(1, device='cuda', dtype=torch.int32)
            rc = gather(pointer(source), pointer(selected), pointer(picks), 3, rows, 2 * width,
                        pointer(invalid), stream)
            check(f'row-gather/{rows}/{width}', selected, source.index_select(0, picks), rc)
            if invalid.item():
                raise AssertionError('Valid Nemotron row gather flagged invalid')
        for dtype in (torch.bfloat16, torch.float32):
            logits = torch.randn((rows, vocab), device='cuda', dtype=dtype)
            for pattern in ('finite-last', 'first-nan'):
                logits[:, :6] = torch.tensor([-0., 0., float('-inf'), 1., 1., 0.], device='cuda', dtype=dtype)
                logits[:, -1] = 1000.
                if pattern == 'first-nan':
                    logits[:, 5] = float('nan')
                drawn = torch.empty((rows,), device='cuda', dtype=torch.int32)
                rc = argmax32(pointer(logits), pointer(drawn), rows, vocab, logits.stride(0),
                              0 if dtype == torch.bfloat16 else 1, stream)
                check(f'nemotron-greedy-i32/{rows}/{dtype}/{pattern}', drawn, logits.argmax(-1).to(torch.int32), rc)
        logits = torch.randn((rows, vocab), device='cuda', dtype=torch.bfloat16)
        logits[:, :5] = torch.tensor([-0., 0., float('-inf'), float('inf'), float('inf')], device='cuda', dtype=logits.dtype)
        bias = torch.zeros_like(logits)
        bias[:, 1::3] = float('-inf')
        expected = logits.clone()
        expected.add_(bias)
        check(f'nemotron-grammar-add/{rows}', logits, expected,
              add(pointer(logits), pointer(bias), pointer(logits), logits.numel(), stream))
    extreme = torch.tensor([-(2 ** 63), -1, 0, 1, 131071, 2 ** 31, 2 ** 63 - 1],
                           device='cuda', dtype=torch.int64)
    actual = torch.empty_like(extreme, dtype=torch.int32)
    check('i64-to-i32/extremes', actual, extreme.to(torch.int32),
          narrow(pointer(extreme), pointer(actual), extreme.numel(), stream))
    from check_argmax_ops import check_argmax_ops
    from check_index_ops import check_index_ops
    from check_nemotron_constants_ops import check_nemotron_constants_ops
    from check_topk_ops import check_topk_ops
    from check_nemotron_cache_ops import check_nemotron_cache_ops
    from check_nemotron_sampler_core_ops import check_nemotron_sampler_core_ops
    from check_argmax_f64_ops import check_argmax_f64_ops
    from check_cast_f64_ops import check_cast_f64_ops
    from check_nemotron_limits_ops import check_nemotron_limits_ops
    check_index_ops(lib, config, check)
    check_nemotron_cache_ops(lib, config, check)
    check_nemotron_constants_ops(lib, config, check)
    argmax_cells = check_argmax_ops(lib, config, check)
    receipt['argmax_guards'] = [cell for cell in argmax_cells if cell.get('gpu_launch') is False]
    check_topk_ops(lib, {**config, "operator_draft_vocab_size": min(32768, vocab)}, check)
    check_nemotron_sampler_core_ops(lib, config, check)
    check_argmax_f64_ops(lib, config, check)
    check_cast_f64_ops(lib, config, check)
    check_nemotron_limits_ops(lib, config, check)
    from probe_nemotron_nucleus_dispatch import probe_nemotron_nucleus_dispatch
    receipt["nucleus_dispatch"] = probe_nemotron_nucleus_dispatch(config, args.out)
    checker.finish(started)


if __name__ == '__main__':
    main()
