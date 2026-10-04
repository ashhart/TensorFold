"""Compare the owned raw-index CUDA helper with Torch only during development qualification."""

import ctypes


def run(library, real_rows=(), report=None, config=None):
    import torch

    operation = library.tf_argmax_rows
    operation.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                          ctypes.c_int64, ctypes.c_int32, ctypes.c_void_p]
    operation.restype = ctypes.c_int
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    cells = []

    def check(name, values):
        if (not values.is_cuda or values.device.index != torch.cuda.current_device() or
                values.dtype not in (torch.bfloat16, torch.float32) or values.ndim != 2 or values.stride(1) != 1):
            raise ValueError('argmax oracle needs CUDA BF16/F32 row views')
        expected = values.argmax(dim=-1)
        out = torch.full((values.shape[0],), -7, dtype=torch.int64, device=values.device)
        active_stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        rc = operation(values.data_ptr(), out.data_ptr(), values.shape[0], values.shape[1], values.stride(0),
                       0 if values.dtype == torch.bfloat16 else 1, active_stream)
        if rc:
            raise RuntimeError('owned argmax launch refused with rc='+str(rc))
        if report is not None:
            report(name, out, expected, rc)
        torch.cuda.synchronize()
        exact = bool(torch.equal(out, expected))
        cell = {'id': name, 'rows': values.shape[0], 'vocab': values.shape[1], 'stride': values.stride(0),
                'dtype': str(values.dtype), 'exact': exact, 'indices': out.cpu().tolist()}
        cells.append(cell)
        if not exact:
            cell['expected_indices'] = expected.cpu().tolist()
            if report is None:
                raise AssertionError('owned argmax differs from Torch: '+name+' '+str(cell))

    generator = torch.Generator(device='cuda').manual_seed(317)
    for dtype in (torch.bfloat16, torch.float32):
        for rows in (1, 2, 4, 8, 16):
            for vocab in (1, 31, 257, 4097):
                for padding in (0, 17):
                    storage = torch.randn((rows, vocab+padding), generator=generator, dtype=dtype, device='cuda')
                    check('random-'+str(dtype)+'-'+str(rows)+'-'+str(vocab)+'-'+str(padding), storage[:, :vocab])
        for name, vector in (
            ('first-tie', [2.0, 1.0, 2.0]),
            ('all-negative-infinity', [float('-inf')]*257),
            ('infinity-tie', [float('-inf'), float('inf'), float('inf')]),
            ('signed-zero-tie', [-0.0, 0.0]),
            ('first-nan', [float('inf'), float('nan'), float('nan'), 0.0]),
            ('last-maximum', [0.0]*256+[1.0]),
            ('boundary-tie', [0.0]*255+[1.0, 1.0])):
            check(name+'-'+str(dtype), torch.tensor([vector], dtype=dtype, device='cuda'))
    words = torch.tensor([[0, 1, -2147483647, -2147483648]], dtype=torch.int32, device='cuda')
    check('F32-subnormal-order', words.view(torch.float32))
    short = torch.tensor([[0, 1, -32767, -32768]], dtype=torch.int16, device='cuda')
    check('BF16-subnormal-order', short.view(torch.bfloat16))
    other_stream = torch.cuda.Stream()
    with torch.cuda.stream(other_stream):
        check('explicit-nondefault-stream', torch.tensor([[1.0, 2.0, 2.0]], device='cuda'))
    text = (config or {}).get('text_config', config or {})
    vocab = int(text.get('vocab_size', 0))
    if vocab:
        for dtype in (torch.bfloat16, torch.float32):
            for rows in (1, 16):
                values = torch.randn((rows, vocab), generator=generator, dtype=dtype, device='cuda')
                check('checkpoint-vocab-shape-'+str(dtype)+'-'+str(rows), values)
    for name, values in real_rows:
        check('real-'+name, values)
    for rows, vocab, stride, dtype in ((0, 1, 1, 0), (1, 0, 0, 0), (1, 1, 1, 2),
                                     (1, 9, 8, 0), (2147483648, 1, 1, 0), (3, 1, 9223372036854775807, 1)):
        rc = operation(None, None, rows, vocab, stride, dtype, stream)
        valid_zero = rows == 0 and vocab == 1
        if rc != (0 if valid_zero else 1):
            raise AssertionError('argmax preflight returned unexpected error')
        cells.append({'id': 'guard-'+str((rows, vocab, stride, dtype)), 'exact': True,
                      'gpu_launch': False, 'rc': rc})
    return cells


def check_argmax_ops(library, config, check):
    return run(library, report=check, config=config)
