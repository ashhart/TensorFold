//! Inkling 2 Mini's M3 packed-weight expert kernels; activations stay BF16 throughout.
pub const experts = @embedFile("experts.metal");
/// The same products on simdgroup matrices, for GPUs without tensor units: after the decode helpers above.
pub const experts_mma = experts ++ @embedFile("experts_mma.metal");
