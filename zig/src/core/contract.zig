//! The kernel contract every GPU kernel's test registers with: its spec, the gates and the input families.
pub const delta_slots_ref = @import("contract/delta_slots_ref.zig");
pub const delta_slots_records = @import("contract/delta_slots_records.zig");
pub const spec = @import("contract/contract.zig");
pub const gates = @import("contract/gates.zig");
pub const rows = @import("contract/rows.zig");
pub const affine = @import("contract/affine.zig");
pub const dense = @import("contract/dense.zig");
pub const limits = @import("contract/limits.zig");
pub const registry = @import("contract/registry.zig");
pub const cubin = @import("contract/cubin.zig");

test {
    _ = @import("contract/qwen_served_records.zig");
    _ = @import("contract/delta_slots_records.zig");
    _ = @import("contract/simd_pairs_records.zig");
    _ = spec;
    _ = gates;
    _ = rows;
    _ = affine;
    _ = dense;
    _ = limits;
    _ = registry;
    _ = cubin;
    _ = @import("contract/fakes.zig");
}
