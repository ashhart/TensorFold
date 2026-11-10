//! The 27B DSpine drafter's Metal launches (qwen27/dspine.metal), as qwen27/dspine/kernels.zig dispatches them.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "metal qwen27 dspine add tf_dspine_add", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "metal qwen27 dspine read tf_dspine_read", .threads = 256, .tg_memory = 128, .m1 = 1024, .m2 = 1024 },
    .{ .name = "metal qwen27 dspine code tf_dspine_code", .threads = 256, .tg_memory = 128, .m1 = 1024, .m2 = 1024 },
    .{ .name = "metal qwen27 dspine message tf_dspine_message", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "metal qwen27 dspine write tf_dspine_write", .threads = 256, .tg_memory = 128, .m1 = 1024, .m2 = 1024 },
    .{ .name = "metal qwen27 dspine senders tf_dspine_senders", .threads = 128, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "metal qwen27 dspine base tf_dspine_base", .threads = 128, .tg_memory = 128, .m1 = 1024, .m2 = 1024 },
    .{ .name = "metal qwen27 dspine table tf_dspine_table", .threads = 256, .tg_memory = 4096, .m1 = 1024, .m2 = 1024 },
};

test "the DSpine launches fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
