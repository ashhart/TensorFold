//! Hy4-flash's rows launches on Metal (tests/contract/metal_hy4_rows.zig) with the M1 and M2 limits --offline recorded.
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    // rows.metal on the wide core row source (metal_linears.zig's Rows.load), the same text on every chip
    .{ .name = "hy4 rows tf_hy4_rows_r2_q4", .threads = 128, .tg_memory = 0, .m1 = 384, .m2 = 384 },
    .{ .name = "hy4 rows tf_hy4_rows_r2_q8", .threads = 128, .tg_memory = 0, .m1 = 384, .m2 = 384 },
    .{ .name = "hy4 rows bad-tail tf_hy4_rows_r2_q4", .threads = 128, .tg_memory = 0, .m1 = 384, .m2 = 384 },
    .{ .name = "hy4 rows tf_hy4_argmax", .threads = 1024, .tg_memory = 256, .m1 = 1024, .m2 = 1024 },
    .{ .name = "hy4 rows bad-tie tf_hy4_argmax", .threads = 1024, .tg_memory = 256, .m1 = 1024, .m2 = 1024 },
    .{ .name = "hy4 rows bad-nan tf_hy4_argmax", .threads = 1024, .tg_memory = 256, .m1 = 1024, .m2 = 1024 },
    // engram.metal (metal_engram.zig's Kernels.load)
    .{ .name = "hy4 engram tf_hy4_engram_lookup", .threads = 32, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "hy4 engram bad-scale tf_hy4_engram_lookup", .threads = 32, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "hy4 engram tf_hy4_engram_apply", .threads = 128, .tg_memory = 80, .m1 = 1024, .m2 = 1024 },
    .{ .name = "hy4 engram bad-gain tf_hy4_engram_apply", .threads = 128, .tg_memory = 80, .m1 = 1024, .m2 = 1024 },
    .{ .name = "hy4 engram tf_hy4_engram_gather", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "hy4 engram bad-group tf_hy4_engram_gather", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    // core embed_norm.metal's lookups at Hy4's hidden width (metal_model.zig's Head.upload)
    .{ .name = "hy4 embed_norm tf_embed_b4_g64", .threads = 128, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "hy4 embed_norm tf_embed_b8_g64", .threads = 128, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "hy4 embed_norm bad-round tf_embed_b4_g64", .threads = 128, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
};
