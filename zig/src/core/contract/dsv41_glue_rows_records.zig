//! The DeepSeek V4.1 glue rows' Metal launch records: threads, threadgroup bytes and reserved bytes.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "dsv41 glue_rows rmsnorm tf_dsv41_g_rmsnorm", .threads = 256, .tg_memory = 32, .reserved = 256 },
    .{ .name = "dsv41 glue_rows rope tf_dsv41_g_rope", .threads = 64, .tg_memory = 0, .reserved = 64 },
    .{ .name = "dsv41 glue_rows fp4pack tf_dsv41_g_fp4pack", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "dsv41 glue_rows fp4rt tf_dsv41_g_fp4rt", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "dsv41 glue_rows pool2 tf_dsv41_g_pool2", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "dsv41 glue_rows mv f32 tf_dsv41_g_mv_f32", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "dsv41 glue_rows mv bf16 tf_dsv41_g_mv_bf16", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "dsv41 glue_rows cand tf_dsv41_g_cand", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "dsv41 glue_rows hcmix tf_dsv41_g_hcmix", .threads = 256, .tg_memory = 128, .reserved = 256 },
    .{ .name = "dsv41 glue_rows hcpre tf_dsv41_g_hcpre", .threads = 64, .tg_memory = 0, .reserved = 64 },
    .{ .name = "dsv41 glue_rows hcpost tf_dsv41_g_hcpost", .threads = 64, .tg_memory = 0, .reserved = 64 },
    .{ .name = "dsv41 glue_rows scatter tf_dsv41_g_scatter", .threads = 64, .tg_memory = 0, .reserved = 64 },
    .{ .name = "dsv41 glue_rows embed tf_dsv41_g_embed", .threads = 64, .tg_memory = 0, .reserved = 64 },
    .{ .name = "dsv41 glue_rows hcmean tf_dsv41_g_hcmean", .threads = 64, .tg_memory = 0, .reserved = 64 },
};

test "the DeepSeek V4.1 row glue launches fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
