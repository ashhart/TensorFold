//! Shrink only the current lesson, restore exact input bits, and preserve the full pre-lesson rollback snapshot.
const std = @import("std");
const pocket = @import("pocket.zig");
const shrink = @import("pocket_shrink.zig");
const marks = @import("pocket_mark.zig");
const t = std.testing;
const Trainer = struct {};
const P = pocket.Pocket(Trainer, @import("families.zig").nemotron_h);
const Host = struct {
    fail: *bool,
    pub fn pocket(h: @This(), _: usize, _: []const f32, _: []const f32, _: usize) !void {
        if (h.fail.*) return error.UploadFailed;
    }
};
fn init() !P {
    var p: P = .{ .gpa = t.allocator, .mode = .subject, .hidden = 64, .layers = 8, .locate = .{ .enabled = true } };
    errdefer p.deinit();
    try p.moe.appendSlice(t.allocator, &.{ 1, 3, 5 });
    try p.used.appendNTimes(t.allocator, 1, 3);
    try p.up.appendNTimes(t.allocator, 0, 3 * P.neurons * 64);
    try p.down.appendNTimes(t.allocator, 0, 3 * P.neurons * 64);
    for (0..3) |j| for (0..2) |n| {
        @memset(p.neuron(p.up.items, j, n, 64), 1);
        @memset(p.neuron(p.down.items, j, n, 64), 0.25);
    };
    // One earlier neuron's record; an undo must preserve it and its entire layer exactly.
    try p.learned.append(t.allocator, .{ .slot = 0, .neuron = 0, .key = .subject, .start = 0, .count = 0, .subject = 11, .lesson = 1 });
    try p.mark();
    p.learned_before = 1;
    for (0..3) |j| {
        try p.taken.append(t.allocator, .{ .slot = j, .neuron = 1, .key = .subject });
        p.used.items[j] = 2;
        try p.learned.append(t.allocator, .{ .slot = j, .neuron = 1, .key = .subject, .start = 0, .count = 0, .subject = 22, .lesson = 2 });
    }
    return p;
}
test "shrink: a failed removal restores bits, kept removals retain prior metadata and undo remains exact" {
    var p = try init();
    defer p.deinit();
    var trainer: Trainer = .{};
    var fail = false;
    const host: Host = .{ .fail = &fail };
    const before = marks.hash(&p);
    const older = p.learned.items[0];
    try t.expect(try shrink.act(&p, &trainer, host, .next));
    try t.expectError(error.ShrinkPending, p.mark());
    try t.expectError(error.ShrinkPending, shrink.act(&p, &trainer, host, .next));
    _ = try shrink.act(&p, &trainer, host, .restore);
    try t.expectEqual(before, marks.hash(&p));
    try t.expect(try shrink.act(&p, &trainer, host, .next));
    _ = try shrink.act(&p, &trainer, host, .keep);
    try t.expectEqual(@as(usize, 2), p.taken.items.len);
    try t.expectEqualSlices(usize, &.{ 2, 1, 2 }, p.used.items);
    try t.expectEqual(older, p.learned.items[0]);
    try t.expectEqual(@as(usize, 3), p.learned.items.len);
    try t.expect(std.mem.allEqual(f32, p.neuron(p.up.items, 1, 1, 64), 0));
    try t.expect(std.mem.allEqual(f32, p.neuron(p.down.items, 1, 1, 64), 0.25));
    try p.undo(&trainer, host);
    try t.expectEqualSlices(usize, &.{ 1, 1, 1 }, p.used.items);
    try t.expectEqual(@as(usize, 1), p.learned.items.len);
    try t.expectEqual(older, p.learned.items[0]);
    try t.expect(!p.shrink.pending);
}

test "shrink: upload failure leaves a recoverable pending removal; a full rollback heals it" {
    var p = try init();
    defer p.deinit();
    var trainer: Trainer = .{};
    var fail = true;
    const host: Host = .{ .fail = &fail };
    try t.expectError(error.UploadFailed, shrink.act(&p, &trainer, host, .next));
    try t.expect(p.shrink.pending);
    try t.expectEqual(@as(usize, 1), p.used.items[0]);
    try t.expectError(error.UploadFailed, p.undo(&trainer, host));
    try t.expect(p.restore_pending);
    fail = false;
    try p.undo(&trainer, host);
    try t.expect(!p.restore_pending and !p.shrink.pending);
}
