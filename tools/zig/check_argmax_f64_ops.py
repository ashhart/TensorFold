"""Check the Nemotron nucleus score selector without performing any floating reduction."""

import ctypes

import torch


def check_argmax_f64_ops(library, config, check):
    function = library.tf_argmax_f64
    function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, *([ctypes.c_uint64] * 3), ctypes.c_void_p]
    function.restype = ctypes.c_int
    vocab = int(config['vocab_size'])
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    generator = torch.Generator(device='cuda').manual_seed(317)

    def run(name, source):
        output = torch.empty((source.shape[0],), device='cuda', dtype=torch.int64)
        status = function(source.data_ptr(), output.data_ptr(), source.shape[0], source.shape[1], source.stride(0), stream)
        check(name, output, source.argmax(dim=-1), status)

    for rows in (1, 2, 4, 8, 16):
        for columns in (1, 7, 31, 257, vocab):
            for padding in (0, 17):
                source = torch.randn((rows, columns + padding), device='cuda', dtype=torch.float64, generator=generator)
                run(f'F64-argmax/{rows}/{columns}/{padding}', source[:, :columns])
    for name, vector in (
        ('first-tie', [1., 2., 2., 0.]),
        ('signed-zero', [-0., 0.]),
        ('first-NaN', [float('inf'), float('nan'), float('nan'), 0.]),
        ('negative-infinity', [float('-inf')] * 257),
        ('last-column', [0.] * 256 + [1.]),
    ):
        run('F64-argmax/' + name, torch.tensor([vector], device='cuda', dtype=torch.float64))
    words = torch.tensor([[0, 1, -9223372036854775807, -9223372036854775808]], device='cuda', dtype=torch.int64)
    run('F64-argmax/subnormal', words.view(torch.float64))
