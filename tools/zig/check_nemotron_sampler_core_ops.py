"""Compare the isolated Nemotron nucleus pointwise stages with deployed Torch."""

import ctypes
from ops_ffi import pointer, bind

import torch


def check_nemotron_sampler_core_ops(library, config, check):
    from tensorfold.families.nemotron_h.cuda.sampler import C1, C2, _mix_t, _shr, _signed

    pointer_type = ctypes.c_void_p
    size_type = ctypes.c_uint64
    signed_type = ctypes.c_int64


    scale = bind(library, "tf_nemotron_scale_f64", [pointer_type] * 3 + [size_type, pointer_type])
    relative_exp = bind(library, "tf_nemotron_relative_exp_f64",
                        [pointer_type] * 3 + [size_type] * 2 + [ctypes.c_uint32, pointer_type])
    normalize = bind(library, "tf_nemotron_run_divide_f64", [pointer_type] * 3 + [size_type] * 2 + [pointer_type])
    uniform = bind(library, "tf_nemotron_uniform_f64",
                   [pointer_type] * 4 + [size_type] * 2 + [signed_type, pointer_type])
    score = bind(library, "tf_nemotron_gumbel_score_f64", [pointer_type] * 4 + [size_type] * 2 + [pointer_type])
    config = config.get("text_config", config)
    columns = int(config["vocab_size"])
    stream = torch.cuda.current_stream()
    handle = pointer_type(stream.cuda_stream)


    def run(rows, width, temperature, request_temperature, seed_value, position, offset, mapped=False):
        identifiers = torch.arange(width, dtype=torch.int64, device="cuda")
        word = ((identifiers * 47 + 11) % 257 - 128).to(torch.float32)
        values = (word / 13).to(torch.bfloat16).double().expand(rows, -1).contiguous()
        parameters = torch.tensor([temperature, 0.95, -2.0, request_temperature], device="cuda", dtype=torch.float64)
        scaled = torch.empty_like(values)
        expected_scaled = values / parameters[0]
        label = f"nemotron-nucleus/{rows}/{width}/{temperature}/{request_temperature}/{seed_value}/{position}/{offset}/{mapped}"
        check(label + "/device-scalar-divide", scaled, expected_scaled,
              scale(pointer(values), pointer(parameters), pointer(scaled), values.numel(), handle))
        order = torch.argsort(expected_scaled, dim=-1, descending=True, stable=True)
        ranked = torch.gather(expected_scaled, 1, order)
        token_ids = identifiers.flip(0) if mapped else identifiers
        ranked_ids = token_ids[order].contiguous()
        probabilities = torch.empty_like(ranked)
        expected_probabilities = torch.exp(ranked - ranked[:, :1])
        check(label + "/relative-exp", probabilities, expected_probabilities,
              relative_exp(pointer(ranked), pointer(parameters), pointer(probabilities), rows, width, 0, handle))
        expected_confidence = torch.exp((ranked - ranked[:, :1]) * (parameters[0] / parameters[3]))
        check(label + "/confidence-relative-exp", probabilities, expected_confidence,
              relative_exp(pointer(ranked), pointer(parameters), pointer(probabilities), rows, width, 1, handle))
        cumulative = torch.cumsum(expected_probabilities, dim=-1)
        totals = expected_probabilities.sum(dim=-1, keepdim=True)
        normalized = torch.empty_like(ranked)
        expected_normalized = cumulative / totals
        check(label + "/normalize-captured-scan", normalized, expected_normalized,
              normalize(pointer(cumulative), pointer(totals), pointer(normalized), rows, width, handle))
        seed = torch.tensor([seed_value], dtype=torch.int64, device="cuda")
        metadata = torch.tensor([position], dtype=torch.int32, device="cuda")
        positions = (metadata[0].to(torch.int64) + torch.arange(rows, device="cuda") + 1 + offset)[:, None]
        mixed = _mix_t(seed + _signed(C1))
        mixed = _mix_t(mixed ^ (positions * _signed(C2)))
        mixed = _mix_t(mixed ^ ranked_ids)
        expected_uniform = _shr(mixed, 11).double() * 2.0 ** -53 + 2.0 ** -54
        uniforms = torch.empty_like(ranked)
        check(label + "/keyed-uniform", uniforms, expected_uniform,
              uniform(pointer(ranked_ids), pointer(seed), pointer(metadata), pointer(uniforms),
                      rows, width, offset, handle))
        limits = (expected_normalized < parameters[1]).sum(dim=-1, keepdim=True) + 1
        minimum = (ranked >= ranked[:, :1] + parameters[2]).sum(dim=-1, keepdim=True)
        limits = torch.minimum(limits, minimum).contiguous()
        expected_scores = ranked - torch.log(-torch.log(expected_uniform))
        rank = torch.arange(width, device="cuda")[None, :]
        expected_scores = torch.where(rank < limits, expected_scores, torch.full_like(expected_scores, float("-inf")))
        scores = torch.empty_like(ranked)
        check(label + "/gumbel-masked-score", scores, expected_scores,
              score(pointer(ranked), pointer(expected_uniform), pointer(limits), pointer(scores), rows, width, handle))

    for rows in (1, 2, 4, 8, 16):
        run(rows, columns, 1.0, 1.0, 11, 991, 0)
    for width in (1, 7, 31, 257):
        for temperature, request_temperature in ((1e-6, 1.0), (0.7, 1.3), (1.3, 0.7), (2.0, 2.0)):
            run(2, width, temperature, request_temperature, (1 << 63) - 1, (1 << 31) - 1, 3)
    run(2, 31, 0.7, 1.3, 1234, 991, -2, mapped=True)

    edges = torch.tensor([[0.0, -0.0, float("inf"), -float("inf"), float("nan"), float.fromhex("0x1p-1022")]],
                         device="cuda", dtype=torch.float64)
    parameters = torch.tensor([1.3, 0.95, -2.0, 0.7], device="cuda", dtype=torch.float64)
    output = torch.empty_like(edges)
    check("nemotron-nucleus/special-device-scalar-divide", output, edges / parameters[0],
          scale(pointer(edges), pointer(parameters), pointer(output), edges.numel(), handle))
