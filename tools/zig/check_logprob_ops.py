"""Pin gathered score subtraction and the deterministic alternative-token integer key."""

import ctypes
from ops_ffi import pointer as ptr, P, U, I

import torch


def check_logprob_ops(lib, config, check):

    score = lib.tf_logprob_rows
    score.argtypes = [P, P, P, P, U, U, U, I, P, P]
    score.restype = ctypes.c_int
    keys = lib.tf_logprob_keys
    keys.argtypes = [P, P, U, U, P]
    keys.restype = ctypes.c_int
    stream = P(torch.cuda.current_stream().cuda_stream)
    vocab = int(config['vocab_size'])
    for rows in (1, 4, 16):
        for dtype in (torch.bfloat16, torch.float32):
            logits = torch.randn((rows, vocab), device='cuda', dtype=dtype)
            logits[:, :5] = torch.tensor([-0., 0., float('-inf'), 1., 1.], device='cuda', dtype=dtype)
            ids = torch.randint(0, vocab, (rows, 16), device='cuda', dtype=torch.int64)
            lse = torch.randn((rows,), device='cuda', dtype=torch.float32)
            output = torch.empty((rows, 16), device='cuda', dtype=torch.float32)
            invalid = torch.zeros(1, device='cuda', dtype=torch.int32)
            expected = logits.gather(1, ids).float() - lse[:, None]
            rc = score(ptr(logits), ptr(ids), ptr(lse), ptr(output), rows, vocab, 16,
                       0 if dtype == torch.bfloat16 else 1, ptr(invalid), stream)
            check(f'logprob-gather-sub/{rows}/{dtype}', output, expected, rc)
            if invalid.item():
                raise AssertionError('valid logprob columns flagged')
            values = logits.float()
            raw_bits = values.view(torch.int32).long()
            raw_bits = torch.where(values == 0, 0, raw_bits)
            ordered = torch.where(raw_bits < 0, ~raw_bits, raw_bits ^ 0x80000000) - 0x80000000
            token_ids = torch.arange(vocab, dtype=torch.int64, device='cuda')
            expected_keys = (ordered << 32) | (0xffffffff - token_ids)
            output_keys = torch.empty_like(expected_keys)
            rc = keys(ptr(values), ptr(output_keys), rows, vocab, stream)
            check(f'logprob-integer-key/{rows}/{dtype}', output_keys, expected_keys, rc)
