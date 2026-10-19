//! A lesson's complete pocket checkpoint, including SwiGLU gates and the earlier lessons' metadata.
const std = @import("std");
const shards = @import("shards.zig");
pub fn Snap(comptime Learned: type) type {
    return struct {
        valid: bool = false,
        up: std.ArrayList(f32) = .empty,
        down: std.ArrayList(f32) = .empty,
        gate: std.ArrayList(f32) = .empty,
        used: std.ArrayList(usize) = .empty,
        learned: std.ArrayList(Learned) = .empty,
        pos: std.ArrayList(u16) = .empty,
        retired_lessons: std.ArrayList(u32) = .empty,
        lesson: u32 = 0,
        learned_before: usize = 0,
        pos_before: usize = 0,
        pub fn deinit(s: *@This(), a: std.mem.Allocator) void {
            inline for (.{ "up", "down", "gate", "used", "learned", "pos", "retired_lessons" }) |f| @field(s, f).deinit(a);
        }
    };
}
pub fn mark(p: anytype) !void {
    p.snap.valid = false;
    inline for (.{ "up", "down", "gate", "used", "learned", "pos", "retired_lessons" }) |f| {
        const dst = &@field(p.snap, f);
        dst.clearRetainingCapacity();
        try dst.appendSlice(p.gpa, @field(p, f).items);
    }
    inline for (.{ "lesson", "learned_before", "pos_before" }) |f| @field(p.snap, f) = @field(p, f);
    p.snap.valid = true;
}
pub fn restore(p: anytype, t: anytype, host: anytype) !bool {
    if (!p.snap.valid) return false;
    inline for (.{ "up", "down", "gate", "used", "learned", "pos", "retired_lessons" }) |f| {
        const dst = &@field(p, f);
        try dst.resize(p.gpa, @field(p.snap, f).items.len);
        @memcpy(dst.items, @field(p.snap, f).items);
    }
    inline for (.{ "lesson", "learned_before", "pos_before" }) |f| @field(p, f) = @field(p.snap, f);
    for (0..p.used.items.len) |j| try p.upload(t, host, j);
    inline for (.{ "up", "down", "gate", "used", "learned", "pos", "retired_lessons" }) |f| {
        if (!std.mem.eql(u8, std.mem.sliceAsBytes(@field(p, f).items), std.mem.sliceAsBytes(@field(p.snap, f).items))) return error.PocketRestoreMismatch;
    }
    return true;
}

pub fn hash(p: anytype) u64 {
    var h = std.hash.Wyhash.init(0);
    inline for (.{ "up", "down", "gate", "used" }) |f| h.update(std.mem.sliceAsBytes(@field(p, f).items));
    return h.final();
}

/// The rounded state judged by the learner.
pub fn asWritten(p: anytype, t: anytype, host: anytype) !void {
    return p.asWritten(t, host);
}
test "a lesson taken out puts the pocket back bit for bit: its neurons, an earlier one it refitted, the rounding" {
    const gpa = std.testing.allocator;
    const d = 64;
    const T = struct {
        pub fn config(_: *const @This()) struct { hidden: usize } {
            return .{ .hidden = d };
        }
    };
    const Host = struct {
        pub fn pocket(_: @This(), _: usize, _: []const f32, _: []const f32, _: usize) !void {}
    };
    var prng = std.Random.DefaultPrng.init(3);
    const r = prng.random();
    const P = @import("pocket.zig").Pocket(T, @import("families.zig").nemotron_h);
    const neurons = P.neurons;
    var p: P = .{ .gpa = gpa, .mode = .reply, .hidden = d, .layers = 8 };
    defer p.deinit();
    try p.moe.appendSlice(gpa, &.{ 3, 5 });
    try p.up.appendNTimes(gpa, 0, 2 * neurons * d);
    try p.down.appendNTimes(gpa, 0, 2 * neurons * d);
    try p.used.appendSlice(gpa, &.{ 1, 0 });
    // an earlier fact's neuron at the first layer, as a save left it
    for (p.neuron(p.up.items, 0, 0, d), p.neuron(p.down.items, 0, 0, d)) |*u, *v| {
        u.* = r.floatNorm(f32);
        v.* = r.floatNorm(f32) * 0.01;
    }
    try shards.roundSlot(&p, 0);
    try p.learned.append(gpa, .{ .slot = 0, .neuron = 0, .key = .reply, .start = 0, .count = 0, .subject = 0, .lesson = 0 });
    var t: T = .{};
    try mark(&p);
    p.learned_before = p.learned.items.len; // as begin() leaves it
    const before = hash(&p);
    // the lesson: a neuron of its own beside the earlier one and one at the second layer, the earlier one refitted
    p.used.items[0] = 2;
    p.used.items[1] = 1;
    for ([_][2]usize{ .{ 0, 1 }, .{ 1, 0 } }) |at| {
        for (p.neuron(p.up.items, at[0], at[1], d), p.neuron(p.down.items, at[0], at[1], d)) |*u, *v| {
            u.* = r.floatNorm(f32);
            v.* = r.floatNorm(f32);
        }
        try p.taken.append(gpa, .{ .slot = at[0], .neuron = at[1], .key = .reply });
    }
    try p.refits.append(gpa, .{ .slot = 0, .neuron = 0 });
    try p.refit_up.appendSlice(gpa, p.neuron(p.up.items, 0, 0, d));
    p.neuron(p.up.items, 0, 0, d)[0] += 1;
    try p.learned.append(gpa, .{ .slot = 0, .neuron = 1, .key = .reply, .start = 0, .count = 0, .subject = 0, .lesson = 1 });
    try asWritten(&p, &t, Host{});
    try std.testing.expect(hash(&p) != before);
    try p.undo(&t, Host{});
    try std.testing.expectEqual(before, hash(&p));
    try std.testing.expectEqualSlices(usize, &.{ 1, 0 }, p.used.items);
    try std.testing.expectEqual(@as(usize, 1), p.learned.items.len);
    // a save since the mark: no lesson before it is put back, and the undo takes out the open fact's neurons alone
    p.snap.valid = false;
    p.used.items[1] = 1;
    try p.taken.append(gpa, .{ .slot = 1, .neuron = 0, .key = .reply });
    @memset(p.neuron(p.up.items, 1, 0, d), 1);
    try p.undo(&t, Host{});
    try std.testing.expectEqual(before, hash(&p));
}
