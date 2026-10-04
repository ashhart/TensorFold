"""Compare unqualified RMS/trig candidates against the booked CUDA Torch oracle by raw bits."""

from ops_ffi import pointer as ptr


def check_norm_rotary_ops(library, config, check):
    import ctypes
    import torch
    import torch.nn.functional as functional

    hidden = int(config.get("text_config", config)["hidden_size"])
    if hidden != 5120:
        raise ValueError("RMS candidate qualifies width5120 only")
    pointer = ctypes.c_void_p
    wide = ctypes.c_uint64
    narrow = ctypes.c_uint32
    integer = ctypes.c_int
    real = ctypes.c_float
    stream = pointer(torch.cuda.current_stream().cuda_stream)
    arguments = [pointer, pointer, pointer, pointer, wide, narrow, integer, integer, real, pointer]
    variants = []
    for name in ("tf_dflash_rms5120_bf16_fma", "tf_dflash_rms5120_bf16_separate"):
        function = getattr(library, name)
        function.argtypes, function.restype = arguments, integer
        variants.append((name, function))
    generator = torch.Generator(device="cuda").manual_seed(317065)
    widths = (1, 2, 4, 7, 8, 16, 32, 64, 128)
    for rows in widths:
        source = torch.randn((rows, hidden), device="cuda", generator=generator).bfloat16()
        weight = (torch.randn((hidden,), device="cuda", generator=generator) * 0.125 + 1).bfloat16()
        expected = functional.rms_norm(source, (hidden,), weight, 1e-6)
        for name, function in variants:
            result = torch.empty_like(source)
            scales = torch.empty((rows,), dtype=torch.float32, device="cuda")
            status = function(ptr(source), ptr(weight), ptr(result), ptr(scales), rows, hidden, 0, 0, 1e-6, stream)
            check(f"{name}-rows{rows}", result, expected, status)
    for pattern in ("ones", "sparse", "extremes"):
        source = torch.ones((2, hidden), dtype=torch.bfloat16, device="cuda")
        if pattern == "sparse":
            source.zero_()
            source[:, ::1279] = 64
        if pattern == "extremes":
            words = torch.tensor([0, -32768, 1, -32767, 128, 32639, 32640, -128, 32705], dtype=torch.int16, device="cuda")
            source = words.repeat((2, (hidden + 8) // 9))[:, :hidden].contiguous().view(torch.bfloat16)
        weight = torch.ones((hidden,), dtype=torch.bfloat16, device="cuda")
        expected = functional.rms_norm(source, (hidden,), weight, 1e-6)
        for name, function in variants:
            result = torch.empty_like(source)
            status = function(ptr(source), ptr(weight), ptr(result), pointer(), 2, hidden, 0, 0, 1e-6, stream)
            check(f"{name}-{pattern}", result, expected, status)
    function = variants[0][1]
    for bad_kind, x, w, source_type, weight_type in (
        ("mixed-weight", source, weight.float(), 0, 1),
        ("f32-input", source.float(), weight.float(), 1, 1),
        ("unaligned", torch.ones(hidden + 1, dtype=torch.bfloat16, device="cuda")[1:].view(1, hidden), weight, 0, 0),
    ):
        result = torch.empty_like(x)
        status = function(ptr(x), ptr(w), ptr(result), pointer(), x.shape[0], hidden, source_type, weight_type, 1e-6, stream)
        check(f"rms-refuse-{bad_kind}", torch.tensor(status, dtype=torch.int32), torch.tensor(1, dtype=torch.int32), 0)
    phase_function = getattr(library, "tf_dflash_phase_trig")
    phase_function.argtypes = [pointer, pointer, pointer, pointer, pointer, wide, narrow, integer, pointer]
    phase_function.restype = integer
    inverse = torch.exp2(-torch.arange(64, device="cuda", dtype=torch.float32) * 0.25).contiguous()
    for rows in widths:
        for start in (0, 48, 8191, 8192, 16384, 262143, 1048575):
            positions = torch.arange(start, start + rows, device="cuda", dtype=torch.float32)
            expected_phase = positions[:, None] * inverse[None, :]
            phase, cosine, sine = [torch.empty_like(expected_phase) for _ in range(3)]
            status = phase_function(ptr(positions), ptr(inverse), ptr(phase), ptr(cosine), ptr(sine), rows, 64, 1, stream)
            check(f"phase-{rows}-{start}", phase, expected_phase, status)
            check(f"cos-{rows}-{start}", cosine, expected_phase.cos(), status)
            check(f"sin-{rows}-{start}", sine, expected_phase.sin(), status)
