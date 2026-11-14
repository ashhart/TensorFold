//! DeepSeek V4.1's tensor-unit prompt launches (fp8_nax): M5 only, no M1/M2 library to build.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "dsv41 nax fp8 tf_dsv41_fp8_nax_f32", .threads = 128, .tg_memory = 9216, .reserved = 128, .offline = false },
    .{ .name = "dsv41 nax fp8 tf_dsv41_fp8_nax_bf16", .threads = 128, .tg_memory = 9216, .reserved = 128, .offline = false },
};

test "the DeepSeek V4.1 tensor-unit prompt launches fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
