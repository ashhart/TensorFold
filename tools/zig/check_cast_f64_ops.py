"""Qualify the score pipeline's BF16/FP32 widening and special encodings against Torch."""

import ctypes

import torch


def check_cast_f64_ops(library, config, check):
    function = library.tf_cast_f64
    function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_void_p]
    function.restype = ctypes.c_int
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    vocab = int(config['vocab_size'])

    def run(name, source):
        output = torch.empty_like(source, dtype=torch.float64)
        rc = function(source.data_ptr(), output.data_ptr(), source.numel(),
                      0 if source.dtype == torch.bfloat16 else 1, stream)
        check(name, output, source.double(), rc)

    for dtype in (torch.bfloat16, torch.float32):
        for rows in (1, 2, 4, 8, 16):
            run(f'cast-F64/{dtype}/{rows}', torch.randn((rows, vocab), device='cuda', dtype=dtype))
    bits = torch.arange(65536, device='cuda', dtype=torch.int32).to(torch.int16)
    run('cast-F64/all-BF16-encodings', bits.view(torch.bfloat16))
    words = torch.tensor([0, -2147483648, 1, -2147483647, 2139095040, -8388608,
                          2139095041, 2143289345, -4194303], device='cuda', dtype=torch.int32)
    run('cast-F64/F32-special-encodings', words.view(torch.float32))
