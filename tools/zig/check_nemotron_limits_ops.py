"""Pin exact nucleus count cutoffs and the sampled token's FP32 confidence store."""

import ctypes
from ops_ffi import pointer as ptr

import torch


def check_nemotron_limits_ops(library, config, check):
    P, U = ctypes.c_void_p, ctypes.c_uint64
    limits = library.tf_nemotron_limits
    limits.argtypes = [P, P, P, P, U, U, ctypes.c_uint32, P]
    limits.restype = ctypes.c_int
    probability = library.tf_nemotron_probability
    probability.argtypes = [P, P, P, P, U, U, P, P]
    probability.restype = ctypes.c_int
    columns = int(config['vocab_size'])
    stream = P(torch.cuda.current_stream().cuda_stream)


    for rows in (1, 2, 4, 8, 16):
        ranked = torch.sort(torch.randn((rows, columns), device='cuda', dtype=torch.float64),
                            descending=True, stable=True).values
        params = torch.tensor([0.6, 0.95, -2., 1.], device='cuda', dtype=torch.float64)
        p = torch.exp(ranked - ranked[:, :1])
        run = p.cumsum(-1) / p.sum(-1, keepdim=True)
        for use_minimum in (0, 1):
            expected = (run < params[1]).sum(-1, keepdim=True) + 1
            if use_minimum:
                expected = torch.minimum(expected, (ranked >= ranked[:, :1] + params[2]).sum(-1, keepdim=True))
            actual = torch.empty_like(expected)
            status = limits(ptr(run), ptr(ranked), ptr(params), ptr(actual), rows, columns, use_minimum, stream)
            check(f'nemotron-nucleus/count-cutoffs/{rows}/{use_minimum}', actual, expected, status)
        for sharpen in (0, 1):
            values = p if not sharpen else torch.exp((ranked - ranked[:, :1]) * (params[0] / params[3]))
            totals = values.sum(-1, keepdim=True)
            selected = torch.randint(0, columns, (rows,), device='cuda', dtype=torch.int64)
            expected = (values.gather(1, selected[:, None])[:, 0] / totals[:, 0]).float()
            actual = torch.empty_like(expected)
            invalid = torch.zeros(1, device='cuda', dtype=torch.int32)
            status = probability(ptr(values), ptr(totals), ptr(selected), ptr(actual), rows, columns, ptr(invalid), stream)
            check(f'nemotron-nucleus/drawn-confidence/{rows}/{sharpen}', actual, expected, status)
            if invalid.item():
                raise AssertionError('Valid sampled probability column refused')
