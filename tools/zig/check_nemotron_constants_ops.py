"""Check startup-only FP32 Mamba coefficients against the booked CUDA Torch reference by raw bits."""


def check_nemotron_constants_ops(library, config, check):
    import ctypes
    import torch

    hidden_config = config.get("text_config", config)
    heads = int(hidden_config["mamba_num_heads"])
    if heads < 1:
        raise ValueError("Nemotron Mamba head count must be positive")
    pointer = ctypes.c_void_p
    function = library.tf_nemotron_mamba_a_f32
    function.argtypes = [pointer, pointer, ctypes.c_uint64, ctypes.c_int, pointer]
    function.restype = ctypes.c_int
    stream = pointer(torch.cuda.current_stream().cuda_stream)
    for count in (1, 3, heads, heads + 1, 2 * heads + 3):
        source = torch.linspace(-8, 8, count, dtype=torch.float32, device="cuda")
        output = torch.empty_like(source)
        status = function(pointer(source.data_ptr()), pointer(output.data_ptr()), count, 1, stream)
        check(f"nemo-mamba-A-count{count}", output, -torch.exp(source), status)
    edges = torch.tensor([-104.0, -103.97208, -100.0, -90.0, -87.33655, -87.0, -1.0, -0.0,
                          0.0, 1.0, 80.0, 88.0, 88.72283, 89.0, torch.finfo(torch.float32).min,
                          torch.finfo(torch.float32).max], dtype=torch.float32, device="cuda")
    source = edges.repeat((heads + edges.numel() - 1) // edges.numel())[:heads].contiguous()
    output = torch.empty_like(source)
    status = function(pointer(source.data_ptr()), pointer(output.data_ptr()), heads, 1, stream)
    check(f"nemo-mamba-A-finite-extremes-heads{heads}", output, -torch.exp(source), status)
    status = function(pointer(), pointer(), 0, 1, stream)
    check("nemo-mamba-A-empty", torch.tensor(status, dtype=torch.int32), torch.tensor(0, dtype=torch.int32), 0)
    for name, address, count, dtype in (("dtype", source.data_ptr(), heads, 0),
                                      ("unaligned", source.data_ptr() + 2, heads, 1),
                                      ("range", source.data_ptr(), (2**64 - 1) // 4 + 1, 1)):
        status = function(pointer(address), pointer(output.data_ptr()), count, dtype, stream)
        check(f"nemo-mamba-A-refuse-{name}", torch.tensor(status, dtype=torch.int32), torch.tensor(1, dtype=torch.int32), 0)
