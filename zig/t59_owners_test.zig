//! The production packed loader's publication boundary with a recording allocator and inert Metal.
const std = @import("std");
const mtl = @import("metal");
const weights = @import("src/families/gemma4/weights.zig");
const cfg = @import("src/families/gemma4/config.zig");
test "T59 packed pair second publication failure releases every buffer once" {
    mtl.reset();
    var w: weights.Weights = undefined;
    w.owned = try .initCapacity(std.testing.allocator, 1);
    var fail = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0, .resize_fail_index = 0 });
    const words: [1024]u32 = @splat(0);
    const scales: [128]u16 = @splat(0);
    const biases: [128]u16 = @splat(0);
    const result = weights.packArrays(fail.allocator(), .{}, &w.owned, cfg.dense31b, &words, &scales, &biases, 32, 256);
    w.deinit(std.testing.allocator);
    try std.testing.expectError(error.OutOfMemory, result);
    try std.testing.expectEqual(@as(usize, 2), mtl.made);
    for (mtl.releases[0..mtl.made]) |n| try std.testing.expectEqual(@as(usize, 1), n);
    try std.testing.expectEqual(@as(usize, 0), mtl.pipes + mtl.libraries + mtl.queues);
}
