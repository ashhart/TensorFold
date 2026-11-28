//! Embedded Metal shader sources so TensorFold mlx_tf is a completely self-contained binary / dylib.
pub const qk = @embedFile("qk.metal");
pub const qk_universal = @embedFile("qk_universal.metal");
pub const norm_ops = @embedFile("norm_ops.metal");
pub const gdn_ops = @embedFile("gdn_ops.metal");
pub const attention = @embedFile("attention.metal");
pub const mlp_ops = @embedFile("mlp_ops.metal");
pub const sample = @embedFile("sample.metal");
pub const dflash = @embedFile("dflash.metal");
pub const mamba2 = @embedFile("mamba2.metal");
pub const moe = @embedFile("moe.metal");
pub const affine_any = @embedFile("affine_any.metal");
pub const tfq3 = @embedFile("tfq3.metal");
pub const tfq3_encode = @embedFile("tfq3_encode.metal");
pub const vision = @embedFile("vision.metal");
