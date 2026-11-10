// Finds a caller-owned temporary output without overwriting the last call's prepared operands.
#pragma once
#include "validation.hpp"

namespace tf_lt_detail {
struct TuneStorage { uint64_t output = 0, workspace = 0; };
inline bool tune_storage(const tf_lt_create_v1& memory, const tf_lt_run_v1& call,
    const tf_lt_requirements_v1& used, uint32_t required_alignment, TuneStorage& out) {
    const uint64_t elements = uint64_t(call.m) * call.n;
    const uint64_t bytes = elements * (call.mode >= TF_LT_FP8 || call.out_kind == TF_LT_OUT_F32 ? 4 : 2);
    if (call.mode >= TF_LT_FP8) {
        if (memory.activation_scales.bytes < bytes) return false;
        out = {memory.activation_scales.address, memory.workspace.bytes}; return true;
    }
    const tf_lt_buffer_v1 buffers[] = {memory.activation_scales, memory.activation_raw, memory.weight_scales};
    const uint64_t occupied[] = {used.activation_scales_bytes, used.activation_raw_bytes, used.weight_scales_bytes};
    for (int i = 0; i < 3; ++i) {
        if (!buffers[i].address || buffers[i].bytes < occupied[i]) continue;
        const uint64_t begin = buffers[i].address + occupied[i];
        if (begin > UINT64_MAX - 255) continue;
        const uint64_t address = (begin + 255) & ~uint64_t(255);
        const uint64_t offset = address - buffers[i].address;
        if (offset <= buffers[i].bytes && bytes <= buffers[i].bytes - offset && alignment(address) >= required_alignment) {
            out = {address, memory.workspace.bytes}; return true;
        }
    }
    if (!memory.workspace.address || memory.workspace.bytes < bytes) return false;
    const uint64_t address = (memory.workspace.address + memory.workspace.bytes - bytes) & ~uint64_t(255);
    if (address < memory.workspace.address || alignment(address) < required_alignment) return false;
    out = {address, address - memory.workspace.address}; return true;
}
} // namespace tf_lt_detail
