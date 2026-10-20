//! GLM-5.3-Flash's decode-window row kernels: each source reserves the 64 threads a group it launches.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "glm rows q2 glm_rows_q2", .threads = 64, .tg_memory = 0, .reserved = 64 },
    .{ .name = "glm rows wide q2 glm_rows_wide_q2", .threads = 64, .tg_memory = 0, .reserved = 64 },
    .{ .name = "glm rows q3 glm_rows_q3", .threads = 64, .tg_memory = 0, .reserved = 64 },
    .{ .name = "glm rows wide q3 glm_rows_wide_q3", .threads = 64, .tg_memory = 0, .reserved = 64 },
    .{ .name = "glm rows q4 glm_rows_q4", .threads = 64, .tg_memory = 0, .reserved = 64 },
    .{ .name = "glm rows wide q4 glm_rows_wide_q4", .threads = 64, .tg_memory = 0, .reserved = 64 },
    .{ .name = "glm rows q5 glm_rows_q5", .threads = 64, .tg_memory = 0, .reserved = 64 },
    .{ .name = "glm rows wide q5 glm_rows_wide_q5", .threads = 64, .tg_memory = 0, .reserved = 64 },
    .{ .name = "glm rows q6 glm_rows_q6", .threads = 64, .tg_memory = 0, .reserved = 64 },
    .{ .name = "glm rows wide q6 glm_rows_wide_q6", .threads = 64, .tg_memory = 0, .reserved = 64 },
    .{ .name = "glm rows q8 glm_rows_q8", .threads = 64, .tg_memory = 0, .reserved = 64 },
    .{ .name = "glm rows wide q8 glm_rows_wide_q8", .threads = 64, .tg_memory = 0, .reserved = 64 },
};

test "the GLM row launches fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
