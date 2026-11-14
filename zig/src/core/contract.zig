//! The kernel contract every GPU kernel's test registers with: its spec, the gates and the input families.
pub const delta_slots_ref = @import("contract/delta_slots_ref.zig");
pub const delta_slots_records = @import("contract/delta_slots_records.zig");
pub const spec = @import("contract/contract.zig");
pub const gates = @import("contract/gates.zig");
pub const attention_pv = @import("contract/attention_pv.zig");
pub const rows = @import("contract/rows.zig");
pub const reference_threads = @import("contract/reference_threads.zig");
pub const affine = @import("contract/affine.zig");
pub const dense = @import("contract/dense.zig");
pub const bf16_round = @import("bf16_round");
pub const attention_range = @import("attention_range.zig");
pub const limits = @import("contract/limits.zig");
pub const lane_q8 = @import("contract/lane_q8_records.zig");
pub const registry = @import("contract/registry.zig");
pub const cubin = @import("contract/cubin.zig");
pub const hsaco = @import("contract/hsaco.zig");
pub const hip_records = @import("contract/hip_records.zig");
pub const vulkan_records = @import("contract/vulkan_records.zig");
pub const xpu_limits = @import("contract/xpu_limits.zig");
pub const xpu_records = @import("contract/xpu_records.zig");
pub const xpu_ref = @import("contract/xpu_ref.zig");
pub const zebin = @import("contract/zebin.zig");
pub const zeinfo = @import("contract/zeinfo.zig");
test {
    _ = vulkan_records;
}
test {
    _ = xpu_records;
}
test {
    _ = xpu_ref;
}
test {
    _ = zebin;
    _ = zeinfo;
}
test {
    _ = @import("contract/qwen_served_records.zig");
    _ = @import("contract/delta_slots_records.zig");
    _ = @import("contract/vit_records.zig");
    _ = @import("contract/simd_pairs_records.zig");
    _ = @import("contract/simd_half_records.zig");
    _ = @import("contract/glm_rows_records.zig");
    _ = @import("contract/nemotron_experts_records.zig");
    _ = @import("contract/moss_image_records.zig");
    _ = @import("contract/idol_records.zig");
    _ = @import("contract/metal_pair_records.zig");
    _ = @import("contract/affine_gather_records.zig");
    _ = @import("contract/flint_records.zig");
    _ = spec;
    _ = gates;
    _ = rows;
    _ = attention_pv;
    _ = affine;
    _ = reference_threads;
    _ = dense;
    _ = limits;
    _ = registry;
    _ = cubin;
    _ = hsaco;
    _ = hip_records;
    _ = @import("contract/hip_records_qwen3.zig");
    _ = @import("contract/hip_records_families.zig");
    _ = @import("contract/fakes.zig");
}

test {
    _ = @import("contract/seed_test.zig");
}
