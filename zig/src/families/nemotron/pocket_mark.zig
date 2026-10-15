//! The pocket as a lesson found it, put back bit for bit when the lesson is taken out; its layers rounded as saved.
const std = @import("std");
const neurons = @import("pocket.zig").neurons;

/// The pocket when a lesson began: every neuron's weights, the neurons in use, how much was learned before it.
pub const Snap = struct {
    valid: bool = false, // false once a save has written the pocket: no lesson before it is put back
    up: std.ArrayList(f32) = .empty,
    down: std.ArrayList(f32) = .empty,
    used: std.ArrayList(usize) = .empty,
    learned: usize = 0,
    pos: usize = 0,
    hash: u64 = 0,

    pub fn deinit(s: *Snap, gpa: std.mem.Allocator) void {
        s.up.deinit(gpa);
        s.down.deinit(gpa);
        s.used.deinit(gpa);
    }
};

/// The pocket's weights and neurons in use as one number: the same number, the same bits.
pub fn hash(p: anytype) u64 {
    var h = std.hash.Wyhash.init(0);
    h.update(std.mem.sliceAsBytes(p.up.items));
    h.update(std.mem.sliceAsBytes(p.down.items));
    h.update(std.mem.sliceAsBytes(p.used.items));
    return h.final();
}

/// The pocket as a new fact's lesson finds it, which taking that lesson out puts back.
pub fn mark(p: anytype) !void {
    const s = &p.snap;
    inline for (.{ "up", "down", "used" }) |f| {
        @field(s, f).clearRetainingCapacity();
        try @field(s, f).appendSlice(p.gpa, @field(p, f).items);
    }
    s.learned = p.learned.items.len;
    s.pos = p.pos.items.len;
    s.hash = hash(p);
    s.valid = true;
}

/// The pocket put back as the open lesson found it, each layer that changed handed to the forward; false with no mark.
pub fn restore(p: anytype, t: anytype, host: anytype) !bool {
    const s = &p.snap;
    if (!s.valid or s.up.items.len != p.up.items.len or s.used.items.len != p.used.items.len) return false;
    const n = p.up.items.len / @max(p.used.items.len, 1);
    const was = hash(p);
    for (0..p.used.items.len) |j| {
        const up = p.up.items[j * n ..][0..n];
        const down = p.down.items[j * n ..][0..n];
        const same = p.used.items[j] == s.used.items[j] and
            std.mem.eql(u8, std.mem.sliceAsBytes(up), std.mem.sliceAsBytes(s.up.items[j * n ..][0..n])) and
            std.mem.eql(u8, std.mem.sliceAsBytes(down), std.mem.sliceAsBytes(s.down.items[j * n ..][0..n]));
        if (same) continue;
        @memcpy(up, s.up.items[j * n ..][0..n]);
        @memcpy(down, s.down.items[j * n ..][0..n]);
        p.used.items[j] = s.used.items[j];
        try p.upload(t, host, j);
    }
    p.learned.shrinkRetainingCapacity(s.learned);
    p.pos.shrinkRetainingCapacity(s.pos);
    const now = hash(p);
    std.log.info("slide pocket: the lesson taken out: the pocket put back as the lesson found it ({x:0>16} -> {x:0>16}, {s})", .{ was, now, if (now == s.hash) "the same bits" else "NOT the same bits" });
    return true;
}

/// The open fact's layers rounded as a save writes them (every neuron there shares the output's groups) and uploaded.
pub fn asWritten(p: anytype, t: anytype, host: anytype) !void {
    const d = t.config().hidden;
    const n = neurons * d;
    var done: std.ArrayList(usize) = .empty;
    defer done.deinit(p.gpa);
    for (p.taken.items) |x| try once(p.gpa, &done, x.slot);
    for (p.refits.items) |x| try once(p.gpa, &done, x.slot);
    for (p.joint.items) |x| try once(p.gpa, &done, x.slot);
    const was = hash(p);
    for (done.items) |j| {
        try host.pocketRound(p.up.items[j * n ..][0..n], p.down.items[j * n ..][0..n]);
        try p.upload(t, host, j);
    }
    if (done.items.len > 0) std.log.info("slide pocket: {d} layers' neurons rounded as the shards hold them ({s})", .{ done.items.len, if (hash(p) == was) "already were" else "changed" });
}

fn once(gpa: std.mem.Allocator, list: *std.ArrayList(usize), j: usize) !void {
    if (std.mem.indexOfScalar(usize, list.items, j) == null) try list.append(gpa, j);
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
        pub fn pocketRound(_: @This(), up: []f32, down: []f32) !void {
            const k = up.len / d;
            for (up) |*x| x.* = @round(x.* * 16) / 16;
            // an output row's neurons share one step, as they share a 4-bit group in the shards
            for (0..d) |o| {
                var most: f32 = 0;
                for (0..k) |i| most = @max(most, @abs(down[i * d + o]));
                const step = if (most > 0) most / 7 else 1;
                for (0..k) |i| down[i * d + o] = @round(down[i * d + o] / step) * step;
            }
        }
    };
    var prng = std.Random.DefaultPrng.init(3);
    const r = prng.random();
    var p: @import("pocket.zig").Pocket(T) = .{ .gpa = gpa, .mode = .reply };
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
    try Host.pocketRound(.{}, p.up.items[0 .. neurons * d], p.down.items[0 .. neurons * d]);
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
