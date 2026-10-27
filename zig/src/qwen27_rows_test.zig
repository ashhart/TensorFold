//! The production prompt-row policies over real owned lane streams, without a device.
const std = @import("std");
const rows = @import("families/qwen27/prompt_rows.zig");
const policy = @import("native/qwen27_prompt_rows.zig");
const Stream = @import("core/lanes/stream.zig").Stream;
const a = std.testing.allocator;

test "qwen27 rows: only the measured generation widens its default" {
    for ([_]u32{ 0, 13, 14, 15, 16, 18 }) |generation| try std.testing.expectEqual(@as(u32, 512), rows.requested(generation, null));
    try std.testing.expectEqual(@as(u32, 2048), rows.requested(17, null));
}

test "qwen27 rows: valid overrides stay exact and invalid ones use the chip default" {
    for ([_]u32{ 0, 13, 17, 18 }) |generation| {
        const valid = [_][]const u8{ "16", "128", "512", "1024", "2048" };
        for (valid, [_]u32{ 16, 128, 512, 1024, 2048 }) |text, want| try std.testing.expectEqual(want, rows.requested(generation, text));
        for ([_][]const u8{ "", "0", "15", "17", "2049", "4096", "-16", "4294967296", " 512", "no" }) |text| {
            try std.testing.expectEqual(@as(u32, if (generation == 17) 2048 else 512), rows.requested(generation, text));
        }
    }
}

const Backend = struct {
    runner: struct { slots: usize = 3 } = .{},
    chunk: usize = 2048,
    owners: [3]?*Stream = @splat(null),
    fills: std.AutoHashMapUnmanaged(*Stream, bool) = .empty,
};

test "qwen27 rows: a decoder arriving between chunks caps the next one and leaving widens it" {
    var prompt = try Stream.init(a, .{ .id = "prompt", .prompt = &.{ 1, 2 }, .max_new = 4, .drafts = false });
    defer prompt.deinit(a);
    var other = try Stream.init(a, .{ .id = "other", .prompt = &.{ 1, 2 }, .max_new = 4, .drafts = false });
    defer other.deinit(a);
    var b: Backend = .{ .owners = .{ &prompt, null, null } };
    defer b.fills.deinit(a);
    try b.fills.put(a, &prompt, true);
    try std.testing.expectEqual(@as(usize, 2048), policy.choose(&b, &prompt));
    b.owners[1] = &other;
    try b.fills.put(a, &other, true);
    try std.testing.expectEqual(@as(usize, 2048), policy.choose(&b, &prompt));
    try std.testing.expect(b.fills.remove(&other));
    try std.testing.expectEqual(@as(usize, 512), policy.choose(&b, &prompt));
    other.paused = true;
    try std.testing.expectEqual(@as(usize, 2048), policy.choose(&b, &prompt));
    other.paused = false;
    try std.testing.expectEqual(@as(usize, 512), policy.choose(&b, &prompt));
    other.finished = true;
    try std.testing.expectEqual(@as(usize, 2048), policy.choose(&b, &prompt));
    other.finished = false;
    for ([_]usize{ 16, 128, 256, 512, 1024, 2048 }) |capacity| {
        b.chunk = capacity;
        try std.testing.expectEqual(@min(capacity, @as(usize, 512)), policy.choose(&b, &prompt));
    }
    b.chunk = 2048;
    b.owners[1] = null;
    b.owners[2] = &other;
    b.runner.slots = 2;
    try std.testing.expectEqual(@as(usize, 2048), policy.choose(&b, &prompt));
    b.runner.slots = 3;
    try std.testing.expectEqual(@as(usize, 512), policy.choose(&b, &prompt));
    b.owners[2] = null;
    try std.testing.expectEqual(@as(usize, 2048), policy.choose(&b, &prompt));
}
