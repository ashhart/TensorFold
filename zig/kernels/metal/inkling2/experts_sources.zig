//! Inkling 2 Mini's M3 packed-weight expert kernels; activations stay BF16 throughout.
pub const experts = @embedFile("experts.metal");
