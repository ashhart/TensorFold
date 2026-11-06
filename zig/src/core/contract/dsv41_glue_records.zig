//! DeepSeek V4.1's glue launches on Metal: each reserves its thread count, so none depends on the M1/M2 compile.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "dsv41 glue rms bf16w tf_dsv41_rms_bf16w", .threads = 256, .tg_memory = 32, .reserved = 256 },
    .{ .name = "dsv41 glue rms f32w tf_dsv41_rms_f32w", .threads = 256, .tg_memory = 32, .reserved = 256 },
    .{ .name = "dsv41 glue embed tf_dsv41_embed", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "dsv41 glue join tf_dsv41_moe_join", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "dsv41 glue mean tf_dsv41_hc_mean", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "dsv41 glue rope tf_dsv41_rope", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "dsv41 glue fp8 round trip tf_dsv41_fp8_rt", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "dsv41 glue fp4 e8m0 round trip tf_dsv41_fp4_rt32", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "dsv41 glue fp4 e4m3 round trip tf_dsv41_fp4_rt16", .threads = 32, .tg_memory = 0, .reserved = 32 },
};

test "the DeepSeek V4.1 glue launches fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
