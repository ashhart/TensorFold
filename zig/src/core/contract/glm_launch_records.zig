//! GLM-5.3-Flash's served Metal launches (tests/contract/metal_glm_launch.zig) at the threads forward.zig launches.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "glm launch select glm_select_mark", .threads = 256, .tg_memory = 8240, .m1 = 832, .m2 = 832 },
    .{ .name = "glm launch select glm_select_merge", .threads = 512, .tg_memory = 26768, .m1 = 1024, .m2 = 1024 },
    // the source reserves its 1,024 threads (max_total_threads_per_threadgroup)
    .{ .name = "glm launch kda glm_kda_window", .threads = 1024, .tg_memory = 14464, .reserved = 1024 },
    .{ .name = "glm launch kda glm_kda_window_tp", .threads = 1024, .tg_memory = 14464, .reserved = 1024 },
};

test "the GLM launches fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
