//! Inkling 2 Mini's M3 packed-weight expert kernels; activations stay BF16 throughout.
const guard = @import("../threadgroup_source.zig");
pub const experts = guard.normalize(@embedFile("experts.metal") ++ "\n" ++ @embedFile("experts_grouped.metal"));
/// The same products on simdgroup matrices, for GPUs without tensor units: after the decode helpers above.
pub const experts_mma = guard.normalize(experts ++ @embedFile("experts_mma.metal"));
