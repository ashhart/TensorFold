// Pure host validation shared by the bridge and its no-CUDA ABI tests.
#pragma once
#include "tf_cublaslt_bridge.h"
#include <cmath>
#include <limits>

namespace tf_lt_detail {
struct Check { tf_lt_status status; const char* message; explicit operator bool() const { return status == TF_LT_OK; } };
inline Check ok() { return {TF_LT_OK, ""}; }
inline Check bad(tf_lt_status status, const char* message) { return {status, message}; }
inline bool range(uint64_t address, uint64_t bytes) { return address && bytes <= UINT64_MAX - address; }
inline bool overlap(uint64_t a, uint64_t an, uint64_t b, uint64_t bn) { return an && bn && a < b + bn && b < a + an; }
inline uint32_t alignment(uint64_t address) {
    uint32_t value = 1;
    while (value < 256 && (address & (uint64_t(value) * 2 - 1)) == 0) value *= 2;
    return value;
}
inline bool mul(uint64_t a, uint64_t b, uint64_t& out) {
    if (b && a > UINT64_MAX / b) return false;
    out = a * b; return true;
}
inline Check requirements(const tf_lt_run_v1& d, tf_lt_requirements_v1& out) {
    if (d.abi_version != TF_LT_ABI_VERSION || d.struct_size != sizeof(d) || d.flags ||
        d.reserved[0] || d.reserved[1] || d.reserved[2]) return bad(TF_LT_INVALID_ARGUMENT, "invalid run ABI or flags");
    if (d.mode < TF_LT_AFFINE4_BF16 || d.mode > TF_LT_FP8_ROW) return bad(TF_LT_UNSUPPORTED, "unsupported weight mode");
    if (d.out_kind != TF_LT_OUT_BF16 && d.out_kind != TF_LT_OUT_F32) return bad(TF_LT_UNSUPPORTED, "unsupported output kind");
    if (!d.m || !d.n || !d.k || d.ldx < d.k || uint64_t(d.npad) < (uint64_t(d.n) + 63) / 64 * 64 || d.npad % 64)
        return bad(TF_LT_INVALID_ARGUMENT, "invalid dimensions, input stride, or padded output width");
    if (d.m > INT32_MAX || d.n > INT32_MAX || d.k > INT32_MAX || d.ldx > INT32_MAX || d.npad > INT32_MAX)
        return bad(TF_LT_UNSUPPORTED, "dimensions exceed bridge launch range");
    if (d.m < 16 || d.m % 16 || d.n < 16 || d.n % 16 || d.k % 64 || d.npad / 64 > 65535 ||
        (d.mode == TF_LT_AFFINE4_BF16 && d.ldx % 8)) return bad(TF_LT_UNSUPPORTED, "shape or stride outside tensor-matmul envelope");
    out = {}; out.abi_version = TF_LT_ABI_VERSION; out.struct_size = sizeof(out);
    uint64_t nk, mk;
    if (!mul(d.n, d.k, nk) || !mul(d.m, d.k, mk)) return bad(TF_LT_INVALID_ARGUMENT, "size overflow");
    if (d.mode == TF_LT_NVFP4 && (mk / 16 + 255) / 256 > INT32_MAX)
        return bad(TF_LT_UNSUPPORTED, "activation quantization grid exceeds CUDA limits");
    if (d.mode >= TF_LT_FP8 && (uint64_t(d.m) * d.n + 255) / 256 > INT32_MAX)
        return bad(TF_LT_UNSUPPORTED, "FP8 epilogue grid exceeds CUDA limits");
    if (d.mode == TF_LT_AFFINE4_BF16) {
        if (!mul(nk, 2, out.weight_raw_bytes)) return bad(TF_LT_INVALID_ARGUMENT, "weight size overflow");
    } else if (d.mode == TF_LT_NVFP4) {
        out.weight_raw_bytes = nk / 2; out.activation_raw_bytes = mk / 2;
        if (!mul((uint64_t(d.n) + 127) / 128, uint64_t(d.k / 64) * 512, out.weight_scales_bytes) ||
            !mul((uint64_t(d.m) + 127) / 128, uint64_t(d.k / 64) * 512, out.activation_scales_bytes))
            return bad(TF_LT_INVALID_ARGUMENT, "scale size overflow");
    } else {
        out.weight_raw_bytes = nk; out.activation_raw_bytes = mk;
        out.weight_scales_bytes = 16;
        uint64_t output_elements;
        if (!mul(d.m, d.n, output_elements) || output_elements > UINT64_MAX - d.m - 4 ||
            !mul(output_elements + d.m + 4, 4, out.activation_scales_bytes))
            return bad(TF_LT_INVALID_ARGUMENT, "FP8 temporary size overflow");
    }
    out.preferred_workspace_bytes = uint64_t(64) << 20;
    return ok();
}
inline Check create(const tf_lt_create_v1& d) {
    if (d.abi_version != TF_LT_ABI_VERSION || d.struct_size != sizeof(d) || d.flags || d.max_plans > 64 ||
        d.reserved[0] || d.reserved[1] || d.reserved[2] || d.reserved[3]) return bad(TF_LT_INVALID_ARGUMENT, "invalid create ABI or flags");
    const tf_lt_buffer_v1 buffers[] = {d.weight_raw, d.activation_raw, d.weight_scales, d.activation_scales, d.workspace};
    if (!d.weight_raw.bytes) return bad(TF_LT_BUFFER_TOO_SMALL, "weight_raw scratch is required");
    for (int i = 0; i < 5; ++i) {
        if (!buffers[i].bytes) { if (buffers[i].address) return bad(TF_LT_INVALID_ARGUMENT, "zero-sized buffer has an address"); continue; }
        if (!range(buffers[i].address, buffers[i].bytes) || buffers[i].address % (i == 4 ? 256 : 16))
            return bad(TF_LT_INVALID_ARGUMENT, "invalid scratch address, alignment, or range");
        for (int j = 0; j < i; ++j) if (overlap(buffers[i].address, buffers[i].bytes, buffers[j].address, buffers[j].bytes))
            return bad(TF_LT_INVALID_ARGUMENT, "scratch buffers overlap");
    }
    return ok();
}
inline Check buffers(const tf_lt_create_v1& c, const tf_lt_run_v1& d, const tf_lt_requirements_v1& required) {
    if ((d.mode == TF_LT_NVFP4 || d.mode == TF_LT_FP8) && (!std::isfinite(d.act_scale) || d.act_scale <= 0 ||
        !std::isfinite(1.0f / d.act_scale) || !std::isfinite(d.weight_scale) ||
        !std::isfinite(d.act_scale * d.weight_scale))) return bad(TF_LT_INVALID_ARGUMENT, "invalid calibrated tensor scales");
    const tf_lt_buffer_v1 scratch[] = {c.weight_raw, c.activation_raw, c.weight_scales, c.activation_scales, c.workspace};
    const uint64_t sizes[] = {required.weight_raw_bytes, required.activation_raw_bytes, required.weight_scales_bytes,
        required.activation_scales_bytes, required.minimum_workspace_bytes};
    for (int i = 0; i < 5; ++i) if (scratch[i].bytes < sizes[i]) return bad(TF_LT_BUFFER_TOO_SMALL, "insufficient caller-owned GPU scratch");
    uint64_t input_elements, input_bytes, source_bytes, metadata_bytes, output_bytes;
    if (!mul(d.m - 1, d.ldx, input_elements) || input_elements > UINT64_MAX - d.k ||
        !mul(input_elements + d.k, 2, input_bytes) || !mul(d.npad, uint64_t(d.k) / (d.mode >= TF_LT_FP8 ? 1 : 2), source_bytes) ||
        !mul(d.npad, d.mode == TF_LT_NVFP4 ? d.k / 16 :
            d.mode == TF_LT_FP8_ROW ? uint64_t(d.k / 64) * 4 : d.mode == TF_LT_FP8 ? 0 : uint64_t(d.k / 64) * 2, metadata_bytes) ||
        !mul(uint64_t(d.m) * d.n, d.out_kind == TF_LT_OUT_F32 ? 4 : 2, output_bytes))
        return bad(TF_LT_INVALID_ARGUMENT, "operand size overflow");
    const uint64_t bias_bytes = d.mode == TF_LT_AFFINE4_BF16 ? metadata_bytes : 0;
    const tf_lt_buffer_v1 operands[] = {{d.x, d.x_bytes}, {d.weight, d.weight_bytes}, {d.scale, d.scale_bytes},
        {d.bias, d.bias_bytes}, {d.out, d.out_bytes}};
    const uint64_t needed[] = {input_bytes, source_bytes, metadata_bytes, bias_bytes, output_bytes};
    for (int i = 0; i < 5; ++i) {
        if (!needed[i]) continue;
        if (operands[i].bytes < needed[i]) return bad(TF_LT_BUFFER_TOO_SMALL, "operand allocation is smaller than its declared shape");
        if (!range(operands[i].address, operands[i].bytes)) return bad(TF_LT_INVALID_ARGUMENT, "invalid operand range");
        for (const auto& s : scratch) if (overlap(operands[i].address, operands[i].bytes, s.address, s.bytes))
            return bad(TF_LT_INVALID_ARGUMENT, "an operand overlaps scratch storage");
    }
    if (d.x % (d.mode == TF_LT_AFFINE4_BF16 ? 16 : 2) || d.weight % 16 || d.out % 16 ||
        (d.mode == TF_LT_AFFINE4_BF16 && (d.scale % 2 || d.bias % 2)) ||
        (d.mode == TF_LT_FP8_ROW && d.scale % 4)) return bad(TF_LT_UNSUPPORTED, "unsupported operand alignment");
    for (int i = 0; i < 4; ++i) if (needed[i] && overlap(d.out, d.out_bytes, operands[i].address, operands[i].bytes))
        return bad(TF_LT_INVALID_ARGUMENT, "output overlaps an input");
    return ok();
}
} // namespace tf_lt_detail
