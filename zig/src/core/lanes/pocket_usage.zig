//! Prompt firing receipts: existing forwards supply activations; cached prefixes retain their union.
const std = @import("std");
pub const activation_floor = 1e-4;
pub const max_layers = 256;

pub fn enabled(learning: bool, forgetting: bool) bool {
    return learning and forgetting;
}
pub const Masks = struct {
    bits: [max_layers]u64 = @splat(0),
    complete: bool = true,

    pub fn merge(a: *Masks, b: Masks) void {
        a.complete = a.complete and b.complete;
        for (&a.bits, b.bits) |*x, y| x.* |= y;
    }
    pub fn has(a: Masks, layer: usize, neuron: usize) bool {
        return layer < max_layers and neuron < 64 and a.bits[layer] & (@as(u64, 1) << @intCast(neuron)) != 0;
    }
};

pub fn fires(x: f32) bool {
    return std.math.isFinite(x) and @abs(x) > activation_floor;
}

/// Accumulate a Metal chunk without shifting outside the audited 64-neuron mask.
pub fn products(mask: *Masks, layer: usize, width: usize, acts: []const f32) void {
    if (layer >= max_layers or width > 64 or (width == 0 and acts.len != 0) or
        (width != 0 and acts.len % width != 0))
    {
        mask.complete = false;
        return;
    }
    for (acts, 0..) |x, at| {
        if (!std.math.isFinite(x)) mask.complete = false;
        if (fires(x)) mask.bits[layer] |= @as(u64, 1) << @intCast(at % width);
    }
}

/// Called before the engine releases a completed request, so ownership cannot change before the count.
pub const Observer = struct {
    ctx: *anyopaque,
    done: *const fn (*anyopaque, ?Masks) void,
};

/// A chunk's row masks let an interior cache mark keep precisely its prefix, never later rows.
pub const Capture = struct {
    gpa: std.mem.Allocator,
    layers: usize,
    capacity: usize,
    rows: usize = 0,
    at: usize = 0,
    base: Masks = .{},
    values: []u64,

    pub fn init(gpa: std.mem.Allocator, layers: usize, capacity: usize) !Capture {
        if (layers > max_layers) return error.UsageGeometry;
        return .{ .gpa = gpa, .layers = layers, .capacity = capacity, .values = try gpa.alloc(u64, try std.math.mul(usize, layers, capacity)) };
    }
    pub fn deinit(c: *Capture) void {
        c.gpa.free(c.values);
    }
    pub fn start(c: *Capture, at: usize, rows: usize) !void {
        if (rows > c.capacity or at != c.at + c.rows) return error.UsageRows;
        c.base = try c.through(at);
        c.at = at;
        c.rows = rows;
        @memset(c.values, 0);
    }
    pub fn layer(c: *Capture, index: usize, acts: []const f32, width: usize, stride: usize) !void {
        if (index >= c.layers or width > 64 or stride < width or acts.len < c.rows * stride) return error.UsageGeometry;
        for (0..c.rows) |r| for (0..width) |n| {
            const x = acts[r * stride + n];
            if (!std.math.isFinite(x)) c.base.complete = false;
            if (fires(x)) c.values[index * c.capacity + r] |= @as(u64, 1) << @intCast(n);
        };
    }
    pub fn through(c: *const Capture, at: usize) !Masks {
        if (at < c.at or at - c.at > c.rows) return error.UsageRows;
        var out = c.base;
        for (0..c.layers) |l| for (0..at - c.at) |r| {
            out.bits[l] |= c.values[l * c.capacity + r];
        };
        return out;
    }
    pub fn restore(c: *Capture, at: usize, saved: ?Masks) void {
        c.base = saved orelse .{ .complete = false };
        c.at = at;
        c.rows = 0;
    }
};

test "pocket usage: strict audit floor, negative firing, interior marks and restored prefixes" {
    const t = std.testing;
    try t.expect(!fires(activation_floor));
    try t.expect(fires(-2 * activation_floor));
    var c = try Capture.init(t.allocator, 3, 4);
    defer c.deinit();
    try c.start(0, 3);
    try c.layer(2, &.{ 0, 0, 1, 0, 0, 2 }, 2, 2);
    const first = try c.through(1);
    try t.expect(!first.has(2, 0));
    const mid = try c.through(2);
    try t.expect(mid.has(2, 0));
    try t.expect(!mid.has(2, 1));
    const all = try c.through(3);
    try t.expect(all.has(2, 1));
    c.restore(2, mid);
    try c.start(2, 1);
    try c.layer(1, &.{ -1, 0 }, 2, 2);
    const next = try c.through(3);
    try t.expect(next.has(1, 0) and next.has(2, 0) and !next.has(2, 1));
    c.restore(2, null);
    try t.expect(!(try c.through(2)).complete);
}

test "pocket usage: Metal products at 64, 65 and 128 neurons never shift outside the mask" {
    var mask: Masks = .{};
    var edge: [128]f32 = @splat(0);
    edge[63] = 1;
    products(&mask, 2, 64, &edge);
    try std.testing.expect(mask.complete and mask.has(2, 63));
    for ([_]usize{ 65, 128 }) |width| {
        mask = .{};
        var wide: [256]f32 = @splat(1);
        products(&mask, 2, width, wide[0 .. 2 * width]);
        try std.testing.expect(!mask.complete);
        try std.testing.expectEqual(@as(u64, 0), mask.bits[2]);
    }
    mask = .{};
    products(&mask, max_layers, 1, &.{1});
    try std.testing.expect(!mask.complete);
    mask = .{};
    products(&mask, 0, 0, &.{});
    try std.testing.expect(mask.complete);
}
