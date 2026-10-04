"""Qualify Nemotron's cache layouts with existing movement operators during development only."""

import ctypes


class CopySpan(ctypes.Structure):
    _fields_ = [('source', ctypes.c_void_p), ('destination', ctypes.c_void_p), ('bytes', ctypes.c_uint64)]


def dimensions(config):
    cfg = config.get('text_config', config)
    hidden = int(cfg['hidden_size'])
    heads = int(cfg['num_attention_heads'])
    kv = int(cfg['num_key_value_heads'])
    hd = int(cfg.get('head_dim') or hidden // heads)
    mh = int(cfg['mamba_num_heads'])
    dh = int(cfg['mamba_head_dim'])
    ds = int(cfg['ssm_state_size'])
    groups = int(cfg['n_groups'])
    kc = int(cfg['conv_kernel'])
    pattern = cfg.get('hybrid_override_pattern')
    if pattern:
        pattern = ''.join(pattern)
    else:
        names = {'mamba': 'M', 'attention': '*', 'moe': 'E', 'mlp': '-'}
        pattern = ''.join(names[name] for name in cfg['layers_block_type'])
    if min(hidden, heads, kv, hd, mh, dh, ds, groups) <= 0 or kc < 2 or '-' in pattern:
        raise ValueError('invalid Nemotron cache config')
    nm, na = pattern.count('M'), pattern.count('*')
    if not nm or not na or hidden % 64:
        raise ValueError('cache proof requires Mamba, attention and aligned affine groups')
    return hidden, heads, kv, hd, mh, dh, ds, groups, kc, nm, na


def check_nemotron_cache_ops(library, config, check):
    import torch

    hidden, heads, kv, hd, mh, dh, ds, groups, kc, nm, na = dimensions(config)
    cd = mh * dh + 2 * groups * ds
    copy = library.tf_copy_spans
    copy.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]
    copy.restype = ctypes.c_int
    gather = library.tf_gather_rows
    gather.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64,
                       ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p]
    gather.restype = ctypes.c_int
    generator = torch.Generator(device='cuda').manual_seed(317)
    cells = []

    def bits(shape, dtype):
        word = torch.int16 if dtype == torch.bfloat16 else torch.int32
        lo, hi = (-32768, 32768) if word == torch.int16 else (-2147483648, 2147483648)
        return torch.randint(lo, hi, shape, dtype=word, device='cuda', generator=generator).view(dtype)

    def spans(transfers):
        descriptors = (CopySpan * len(transfers))()
        for i, (source, destination) in enumerate(transfers):
            if (not source.is_cuda or not destination.is_cuda or not source.is_contiguous() or
                    not destination.is_contiguous() or source.dtype != destination.dtype or source.shape != destination.shape or
                    source.device != destination.device or source.device.index != torch.cuda.current_device()):
                raise ValueError('cache transfer requires distinct contiguous equal-format tensors')
            length = source.numel()*source.element_size()
            if max(source.data_ptr(), destination.data_ptr()) < min(source.data_ptr()+length, destination.data_ptr()+length):
                raise ValueError('cache transfer requires disjoint source and destination bytes')
            descriptors[i] = CopySpan(source.data_ptr(), destination.data_ptr(), length)
        raw = torch.tensor(list(bytes(descriptors)), dtype=torch.uint8, device='cuda')
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        rc = copy(raw.data_ptr(), len(transfers), stream)
        torch.cuda.synchronize()
        if rc:
            raise RuntimeError('Nemotron cache copy refused: '+str(rc))
        return rc

    def report(name, actual, expected, rc):
        check(name, actual, expected, rc)
        cells.append({'id': name, 'shape': list(actual.shape), 'dtype': str(actual.dtype), 'bytes': actual.numel()*actual.element_size()})

    for capacity in (48, 2048, 8192):
        original = [bits((capacity, kv, hd), torch.bfloat16) for _ in range(2)]
        expected = [value.clone() for value in original]
        saved = [torch.empty_like(value) for value in original]
        rc = spans(list(zip(original, saved)))
        for label, actual, reference in zip(('k', 'v'), saved, expected):
            report('mtp-snapshot-'+label+'-'+str(capacity), actual, reference, rc)
        for value in original:
            value.zero_()
        for label, actual, reference in zip(('k', 'v'), saved, expected):
            report('mtp-immutable-'+label+'-'+str(capacity), actual, reference, 0)
        pointers = [value.data_ptr() for value in original]
        rc = spans(list(zip(saved, original)))
        if pointers != [value.data_ptr() for value in original]:
            raise AssertionError('restore changed graph cache addresses')
        for label, actual, reference in zip(('k', 'v'), original, expected):
            report('mtp-restore-'+label+'-'+str(capacity), actual, reference, rc)

    shapes = [('k_cache', (na, 2048, kv, hd), torch.bfloat16),
              ('v_cache', (na, 2048, kv, hd), torch.bfloat16),
              ('ssm', (nm, mh, dh, ds), torch.float32),
              ('conv_base', (nm, kc-1, cd), torch.bfloat16),
              ('raw', (nm, 2, 16, cd), torch.bfloat16),
              ('xc', (nm, 2, 16, cd), torch.bfloat16),
              ('dt', (nm, 2, 16, mh), torch.float32)]
    original = [bits(shape, dtype) for _, shape, dtype in shapes]
    expected = [value.clone() for value in original]
    saved = [torch.empty_like(value) for value in original]
    rc = spans(list(zip(original, saved)))
    for (name, _, _), actual, reference in zip(shapes, saved, expected):
        report('target-snapshot-'+name, actual, reference, rc)
    for value in original:
        value.zero_()
    pointers = [value.data_ptr() for value in original]
    rc = spans(list(zip(saved, original)))
    if pointers != [value.data_ptr() for value in original]:
        raise AssertionError('target restore changed graph addresses')
    for (name, _, _), actual, reference in zip(shapes, original, expected):
        report('target-restore-'+name, actual, reference, rc)

    hidden_rows = bits((16, hidden), torch.bfloat16)
    tokens = torch.randint(0, int(config.get('text_config', config)['vocab_size']), (16,), dtype=torch.int32, device='cuda', generator=generator)
    for keep in (1, 2, 4, 8, 16):
        hin = bits((16, hidden), torch.bfloat16)
        tok = torch.randint(0, 1000, (16,), dtype=torch.int32, device='cuda', generator=generator)
        expected_h, expected_t = hin.clone(), tok.clone()
        expected_h[:keep].copy_(hidden_rows[:keep]); expected_t[:keep].copy_(tokens[:keep])
        rc = spans([(hidden_rows[:keep], hin[:keep]), (tokens[:keep], tok[:keep])])
        report('level-hin-'+str(keep), hin, expected_h, rc)
        report('level-tok-'+str(keep), tok, expected_t, rc)

    for name, source in (
            ('next-hidden', hidden_rows[7:8]), ('next-token', tokens[7:8]),
            ('head-output', hidden_rows[15:16]),
            ('positions', torch.arange(17, dtype=torch.int32, device='cuda')),
            ('sampling-parameters', torch.tensor([0.6, 1.0, float('-inf'), 1.0], dtype=torch.float64, device='cuda'))):
        destination = torch.empty_like(source)
        rc = spans([(source, destination)])
        report(name, destination, source, rc)

    indices = torch.tensor([127, 0, 63, 1, 127, 64, 2, 126]*8, dtype=torch.int64, device='cuda')
    for name, source in (
            ('head-words', torch.randint(-2147483648, 2147483648, (128, hidden//8), dtype=torch.int32, device='cuda', generator=generator)),
            ('head-scales', bits((128, hidden//64), torch.bfloat16)),
            ('head-biases', bits((128, hidden//64), torch.bfloat16))):
        expected = source.index_select(0, indices)
        destination = torch.empty_like(expected)
        invalid = torch.zeros(1, dtype=torch.int32, device='cuda')
        rc = gather(source.data_ptr(), destination.data_ptr(), indices.data_ptr(), indices.numel(), source.shape[0],
                    source.shape[1]*source.element_size(), invalid.data_ptr(), ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
        torch.cuda.synchronize()
        if rc or int(invalid.item()):
            raise AssertionError('draft head row gather failed')
        report(name, destination, expected, rc)
    return cells
