"""Raw-byte oracle cells for the original F32 unsorted selector candidate."""

import ctypes
from ops_ffi import pointer

import torch


def check_topk_ops(library, config, check):
    pointer_type = ctypes.c_void_p
    size_type = ctypes.c_uint64
    size = library.tf_topk_f32_unsorted_scratch_bytes
    size.argtypes, size.restype = [size_type, size_type], size_type
    select = library.tf_topk_f32_unsorted
    select.argtypes = [pointer_type, pointer_type, pointer_type, *([size_type] * 5),
                      pointer_type, size_type, pointer_type]
    select.restype = ctypes.c_int
    config = config.get("text_config", config)
    vocabulary = int(config["vocab_size"])
    spans = [(a, min(b, vocabulary)) for a, b in ((0, 98304), (248032, 248320)) if a < vocabulary]
    draft_vocabulary = int(config.get("operator_draft_vocab_size", sum(b - a for a, b in spans)))
    widths = sorted({32, vocabulary, draft_vocabulary})
    stream = torch.cuda.current_stream()
    handle = pointer_type(stream.cuda_stream)


    def run(name, input_tensor, k):
        rows, columns = input_tensor.shape
        values = torch.empty((rows, k), device=input_tensor.device, dtype=torch.float32)
        indices = torch.empty((rows, k), device=input_tensor.device, dtype=torch.int64)
        scratch = torch.empty(size(rows, columns), device=input_tensor.device, dtype=torch.uint8)
        expected_values, expected_indices = torch.topk(input_tensor, k, dim=-1, sorted=False)
        status = select(pointer(input_tensor), pointer(values), pointer(indices), rows, columns, k,
                        input_tensor.stride(0) * 4, input_tensor.stride(1) * 4,
                        pointer(scratch), scratch.numel(), handle)
        check(name + "/values", values, expected_values, status)
        check(name + "/indices", indices, expected_indices)

    for rows in (1, 2, 4, 8, 16, 21, 41, 81, 128, 200, 256):
        for columns in widths:
            if columns < 28:
                continue
            base = torch.arange(columns, device="cuda", dtype=torch.int32)
            repeated = ((base * 73 + 19) % 257).to(torch.float32)
            repeated = repeated.expand(rows, -1).contiguous()
            for k in (16, 28):
                run(f"topk/repeated/{rows}/{columns}/{k}", repeated, k)

    for columns in widths:
        if columns < 28:
            continue
        for k in (16, 28):
            count = min(columns, k + 19)
            boundary = torch.full((2, columns), -9.0, device="cuda", dtype=torch.float32)
            boundary[:, :count] = 2.0
            boundary[:, columns - 3:] = 7.0
            run(f"topk/boundary-more-than-margin/{columns}/{k}", boundary, k)

            zeros = torch.full_like(boundary, -1.0)
            zeros[:, :count] = 0.0
            zeros[:, :count:2] = -0.0
            run(f"topk/signed-zero/{columns}/{k}", zeros, k)

            specials = torch.full_like(boundary, -5.0)
            bit_pattern = torch.tensor([0x7fc00001, 0x7fc01234, -0x003fffff, 0x7f800000,
                                        -0x00800000, 0, -0x80000000, 0x00000001,
                                        -0x7fffffff, 0x3f800000], device="cuda", dtype=torch.int32)
            specials[:, :bit_pattern.numel()] = bit_pattern.view(torch.float32)
            run(f"topk/nan-infinity-subnormal/{columns}/{k}", specials, k)

            tied_nans = torch.full_like(boundary, -3.0)
            patterns = bit_pattern[:3].repeat((count + 2) // 3)[:count]
            tied_nans[:, :count] = patterns.view(torch.float32)
            run(f"topk/nan-boundary/{columns}/{k}", tied_nans, k)

            for pattern in (0x7fc00001, -0x003fffff):
                all_nan = torch.full((1, columns), pattern, device="cuda", dtype=torch.int32).view(torch.float32)
                run(f"topk/all-nan/{columns}/{k}/{pattern}", all_nan, k)

            permuted = torch.arange(columns, device="cuda", dtype=torch.int32)
            unique = ((permuted * 8191) % columns).to(torch.float32).view(1, -1)
            run(f"topk/finite-permutation/{columns}/{k}", unique, k)
            padded = torch.full((2, columns * 2 + 3), -17.0, device="cuda", dtype=torch.float32)
            strided = padded[:, 1:1 + columns * 2:2]
            strided.copy_(unique.expand(2, -1))
            run(f"topk/strided-materialization/{columns}/{k}", strided, k)

    for columns in (1, 7, 31, 4095, 4096, 4097):
        input_tensor = torch.arange(columns, device="cuda", dtype=torch.float32).view(1, -1)
        for k in sorted({0, 1, min(16, columns), columns}):
            run(f"topk/tail-and-full/{columns}/{k}", input_tensor, k)

    input_tensor = torch.ones((1, 32), device="cuda", dtype=torch.float32)
    values = torch.full((1, 16), -11.0, device="cuda", dtype=torch.float32)
    indices = torch.full((1, 16), -11, device="cuda", dtype=torch.int64)
    scratch = torch.empty(size(1, 32), device="cuda", dtype=torch.uint8)
    status = select(pointer(input_tensor), pointer(values), pointer(indices), 1, 32, 16, 128, 4,
                    pointer(scratch), scratch.numel() - 1, handle)
    if status == 0:
        raise AssertionError("topk accepted an undersized workspace")
    check("topk/invalid-workspace-preserves-values", values, torch.full_like(values, -11.0))
    check("topk/invalid-workspace-preserves-indices", indices, torch.full_like(indices, -11))
