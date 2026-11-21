//! Inkling 2 backbone operator sources (families/inkling2/metal_ops.zig). Register as
//! `pub const inkling2_ops = @import("inkling2/ops_sources.zig").ops;` in kernels/metal/sources.zig.

/// Helpers every file after it uses; keep first.
pub const common = @embedFile("common.metal");
pub const embed_norm = @embedFile("embed_norm.metal");
pub const attn_res = @embedFile("attn_res.metal");
pub const conv = @embedFile("conv.metal");
pub const qkv = @embedFile("qkv.metal");
pub const attention = @embedFile("attention.metal");
pub const route = @embedFile("route.metal");
pub const head = @embedFile("head.metal");

/// The library the runtime compiles: every kernel is named inkling2_*, in this order.
pub const ops = common ++ "\n" ++ embed_norm ++ "\n" ++ attn_res ++ "\n" ++ conv ++ "\n" ++ qkv ++ "\n" ++
    attention ++ "\n" ++ route ++ "\n" ++ head;
