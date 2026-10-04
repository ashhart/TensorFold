"""Add raw-byte parity cells for accepted indices, vocabulary lookup and grammar masks."""

from ops_ffi import pointer as ptr, bind, P, U, I

import torch


def check_index_ops(lib, config, check):


    widen = bind(lib, 'tf_widen_indices', [P, P, U, P])
    select = bind(lib, 'tf_select_axis', [P, P, P, U, U, U, U, P, P])
    gather = bind(lib, 'tf_gather_columns', [P, P, P, U, U, U, I, P, P])
    lookup = bind(lib, 'tf_lookup_ids', [P, U, P, P, U, P, P])
    mask = bind(lib, 'tf_grammar_mask', [P, P, P, U, U, U, U, U, I, P, P])
    handle = P(torch.cuda.current_stream().cuda_stream)
    hidden, vocab = int(config['hidden_size']), int(config['vocab_size'])
    narrow = torch.tensor([-(2 ** 31), -1, 0, 1, 128, 2 ** 31 - 1], device='cuda', dtype=torch.int32)
    wide = torch.empty_like(narrow, dtype=torch.int64)
    check('index-widen/extremes', wide, narrow.long(), widen(ptr(narrow), ptr(wide), narrow.numel(), handle))
    for rows in (1, 4, 16, 128):
        source = torch.randint(-(2 ** 31), 2 ** 31 - 1, (3, rows, hidden), device='cuda', dtype=torch.int32)
        picks = torch.tensor([rows - 1, 0, rows // 2, 0], device='cuda', dtype=torch.int64)
        out = torch.empty((3, 4, hidden), device='cuda', dtype=source.dtype)
        invalid = torch.zeros(1, device='cuda', dtype=torch.int32)
        rc = select(ptr(source), ptr(out), ptr(picks), 3, rows, 4, hidden * 4, ptr(invalid), handle)
        check(f'axis-select/{rows}', out, source.index_select(1, picks), rc)
        if invalid.item():
            raise AssertionError('valid axis indices flagged')
        ids = torch.arange(vocab, device='cuda', dtype=torch.int64)
        local = torch.randint(0, vocab, (rows, 16), device='cuda', dtype=torch.int64)
        global_ids = torch.empty_like(local)
        invalid.zero_()
        rc = lookup(ptr(ids), vocab, ptr(local), ptr(global_ids), local.numel(), ptr(invalid), handle)
        check(f'vocabulary-lookup/{rows}', global_ids, ids[local], rc)
        if invalid.item():
            raise AssertionError('valid vocabulary indices flagged')
        for dtype in (torch.bfloat16, torch.float32, torch.float64, torch.int64):
            source = torch.randn((rows, 1024), device='cuda').to(dtype)
            indices = torch.randint(0, 1024, (rows, 16), device='cuda', dtype=torch.int64)
            output = torch.empty((rows, 16), device='cuda', dtype=dtype)
            invalid.zero_()
            rc = gather(ptr(source), ptr(output), ptr(indices), rows, 1024, 16,
                        source.element_size(), ptr(invalid), handle)
            check(f'column-gather/{rows}/{dtype}', output, source.gather(1, indices), rc)
            if invalid.item():
                raise AssertionError('valid gather columns flagged')
    for dtype in (torch.bfloat16, torch.float32):
        for offset in (0, 3, 248032):
            for full in (False, True):
                total, columns = 8, 288
                picked = torch.arange(total, device='cuda', dtype=torch.int64) if full else \
                    torch.tensor([0, 3, 7], device='cuda', dtype=torch.int64)
                source = torch.randn((total, columns), device='cuda', dtype=dtype)
                packed = torch.randint(0, 256, (len(picked), 31104), device='cuda', dtype=torch.uint8)
                shifts = torch.arange(8, device='cuda', dtype=torch.uint8)
                allowed = ((packed.unsqueeze(-1) >> shifts) & 1).view(len(picked), -1)[:, offset:offset + columns].bool()
                if allowed.shape[1] < columns:
                    allowed = torch.nn.functional.pad(allowed, (0, columns - allowed.shape[1]), value=False)
                expected = source.clone()
                expected[picked] = expected[picked].masked_fill(~allowed, float('-inf'))
                invalid = torch.zeros(1, device='cuda', dtype=torch.int32)
                rc = mask(ptr(source), ptr(packed), ptr(picked), total, len(picked), columns,
                          packed.shape[1], offset, source.element_size(), ptr(invalid), handle)
                check(f'grammar-mask/{dtype}/{offset}/{full}', source, expected, rc)
                if invalid.item():
                    raise AssertionError('valid grammar row flagged')
