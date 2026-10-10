const std = @import("std");
const meta = @import("batch_meta.zig");

test "packed windows keep unequal session positions and contiguous scratch rows" {
    const p = try meta.Plan.init(&.{
        .{ .slot = 3, .pos = 19, .rows = 3 },
        .{ .slot = 0, .pos = 2047, .rows = 2 },
        .{ .slot = 5, .pos = 9000, .rows = 1 },
    }, 10000);
    try std.testing.expectEqual(@as(usize, 6), p.total);
    try std.testing.expectEqualSlices(u32, &.{ 19, 20, 21, 2047, 2048, 9000 }, p.positions[0..p.total]);
    try std.testing.expectEqualSlices(u32, &.{ 20, 21, 22, 2048, 2049, 9001 }, p.keys[0..p.total]);
    try std.testing.expectEqualSlices(usize, &.{ 0, 3, 5 }, p.starts[0..3]);
}

test "invalid batches fail before any session state is changed" {
    try std.testing.expectError(error.EmptyBatch, meta.Plan.init(&.{}, 32));
    try std.testing.expectError(error.EmptyWindow, meta.Plan.init(&.{.{ .slot = 0, .pos = 0, .rows = 0 }}, 32));
    try std.testing.expectError(error.DuplicateSlot, meta.Plan.init(&.{
        .{ .slot = 1, .pos = 0, .rows = 1 }, .{ .slot = 1, .pos = 8, .rows = 1 },
    }, 32));
    try std.testing.expectError(error.TooManyRows, meta.Plan.init(&.{
        .{ .slot = 0, .pos = 0, .rows = 9 }, .{ .slot = 1, .pos = 0, .rows = 8 },
    }, 32));
    try std.testing.expectError(error.ContextFull, meta.Plan.init(&.{.{
        .slot = 0,
        .pos = std.math.maxInt(u32),
        .rows = 1,
    }}, 32));
    const p = try meta.Plan.init(&.{.{ .slot = 0, .pos = 31, .rows = 1 }}, 32);
    try std.testing.expectEqual(@as(u32, 32), p.keys[0]);
}

test "slot admission reserves all caches before admitting a fixed request count" {
    try std.testing.expectEqual(@as(u32, 3), try meta.fit(8, false, 20, 8));
    try std.testing.expectError(error.OverMemoryLimit, meta.fit(4, true, 20, 8));
    try std.testing.expectEqual(@as(u32, 1), try meta.fit(8, false, 0, 8));
    try std.testing.expectError(error.BadSlotBudget, meta.fit(2, false, 20, 0));
    try std.testing.expectEqual(@as(u32, 16), try meta.fit(16, true, std.math.maxInt(u64), 1));
    try std.testing.expectError(error.OverMemoryLimit, meta.fit(17, true, std.math.maxInt(u64), 1));
}

test "shared draft before keep and lone draft after rejection use the same forward provenance" {
    var shared: meta.Round = .{};
    var lone: meta.Round = .{};
    try shared.begin(2047, 4);
    try lone.begin(9000, 3);
    try shared.draft(2047, &.{ 0, 1 }, 2);
    try shared.keep(2);
    try lone.keep(1);
    try lone.draft(9000, &.{0}, 1);
    try std.testing.expectError(error.KeepOutOfStep, lone.keep(1));
    try std.testing.expectError(error.DraftOutOfStep, lone.draft(9001, &.{0}, 1));
    try std.testing.expectError(error.TreeDraftsUnsupported, shared.draft(2047, &.{ 0, 2 }, 2));
    try std.testing.expectError(error.DraftOutOfStep, shared.draft(2047, &.{ 0, 1 }, 1));
    try std.testing.expectError(error.DraftOutOfStep, shared.draft(2047, &.{ 0, 1, 2, 3, 4 }, 5));
    try shared.begin(2049, 1);
    try shared.keep(1);
    try shared.draft(2049, &.{0}, 1);
    lone = .{};
    try std.testing.expectError(error.DraftOutOfStep, lone.draft(9000, &.{0}, 1));
    try lone.begin(0, 1);
    try std.testing.expectError(error.WindowOutOfStep, lone.begin(1, 1));
}
