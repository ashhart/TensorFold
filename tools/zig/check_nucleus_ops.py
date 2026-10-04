"""Check FP64 nucleus intermediates and int64 mass sums without substituting a softmax."""

import ctypes
from ops_ffi import pointer as ptr, bind, P, U

import torch


def check_nucleus_ops(lib, config, check):


    scale = bind(lib, 'tf_scale_logits', [P, P, U, ctypes.c_double, P])
    mass = bind(lib, 'tf_nucleus_mass', [P, P, P, U, U, P])
    sum_rows = bind(lib, 'tf_integer_row_sum', [P, P, U, U, P])
    stream = P(torch.cuda.current_stream().cuda_stream)
    vocab = int(config['vocab_size'])
    for rows in (1, 2, 4, 8, 16):
        for temperature in (1e-6, 0.7, 1.0):
            logits = torch.randn((rows, vocab), device='cuda', dtype=torch.float32)
            logits[:, :4] = torch.tensor([0., -0., -1000., float('-inf')], device='cuda')
            scaled = torch.empty_like(logits, dtype=torch.float64)
            expected = logits.double() / temperature
            rc = scale(ptr(logits), ptr(scaled), logits.numel(), temperature, stream)
            check(f'nucleus-scale/{rows}/{temperature}', scaled, expected, rc)
            top = expected.max(dim=-1).values
            reference_mass = torch.floor(torch.exp(expected - top[:, None]) * 2. ** 40).to(torch.int64)
            output = torch.empty_like(reference_mass)
            rc = mass(ptr(scaled), ptr(top), ptr(output), rows, vocab, stream)
            check(f'nucleus-mass/{rows}/{temperature}', output, reference_mass, rc)
            total = torch.empty((rows,), device='cuda', dtype=torch.int64)
            rc = sum_rows(ptr(output), ptr(total), rows, vocab, stream)
            check(f'nucleus-sum/{rows}/{temperature}', total, reference_mass.sum(dim=-1), rc)
    boundaries = torch.tensor([0., -0., -2. ** -40, -1., -50., -1000., float('-inf')],
                              device='cuda', dtype=torch.float64)[None, :]
    maxima = boundaries.max(dim=-1).values
    actual = torch.empty_like(boundaries, dtype=torch.int64)
    expected = torch.floor(torch.exp(boundaries - maxima[:, None]) * 2. ** 40).to(torch.int64)
    check('nucleus-mass/near-integer-cutoffs', actual, expected,
          mass(ptr(boundaries), ptr(maxima), ptr(actual), 1, boundaries.shape[1], stream))
