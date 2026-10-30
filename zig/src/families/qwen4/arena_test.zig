//! Actual scratch slices cannot overlap live allocations, grow after a refused shape or release before a drain.
const std = @import("std");
const a = std.testing.allocator;
const B = @import("state_test.zig").Backend;
const Scratch = @import("arena.zig").Arena(B);
test "Qwen4 scratch: checked slices, row offsets, refused requests and queue-ordered reuse" {
    var backend: B = .{};
    try std.testing.expectError(error.Qwen4ScratchBudget, Scratch.open(&backend, 256, 0));
    try std.testing.expectEqual(@as(usize, 0), backend.allocs);
    var scratch = try Scratch.open(&backend, 256, 256);
    const first = try scratch.take(3, 7, .bf16);
    const second = try scratch.take(2, 5, .f32);
    try std.testing.expectEqual(@as(usize, 0), first.offset);
    try std.testing.expectEqual(@as(usize, 48), second.offset);
    try std.testing.expect(first.offset + (try first.bytes()) <= second.offset);
    try std.testing.expectEqual(@as(usize, 28), (try first.row(2)).offset);
    try std.testing.expectEqual(@as(usize, 1), (try first.row(2)).rows);
    try std.testing.expectError(error.Qwen4TensorRow, first.row(3));
    const before = scratch.used;
    try std.testing.expectError(error.Qwen4ScratchFull, scratch.take(128, 5120, .bf16));
    try std.testing.expectEqual(before, scratch.used);
    try std.testing.expectError(error.Overflow, scratch.take(std.math.maxInt(usize), 2, .bf16));
    try std.testing.expectEqual(before, scratch.used);
    try std.testing.expectError(error.Qwen4ScratchMark, scratch.rewind(before + 1));
    try scratch.rewind(0);
    const reused = try scratch.take(4, 32, .bf16);
    try std.testing.expectEqual(@as(usize, 0), reused.offset);
    try std.testing.expectEqual(@as(usize, 256), scratch.peak);
    backend.fail_drain = true;
    try std.testing.expectError(error.PlantedDrain, scratch.close());
    try std.testing.expectEqual(@as(usize, 1), backend.live);
    backend.fail_drain = false;
    try scratch.close();
    try std.testing.expectEqual(@as(usize, 0), backend.live);
}
