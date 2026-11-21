//! A looped stack's walk: prelude, core × passes, coda; each step's layer, pass and state slots, numbered in run order.
const std = @import("std");

/// Layers [start, end) run `passes` times between the prelude [0, start) and the coda [end, layers).
pub const Loop = struct { start: u32, end: u32, passes: u32 };

/// No loop: every layer once.
pub const straight: Loop = .{ .start = 0, .end = 0, .passes = 1 };

/// Where a drafter's tap reads: the hidden state entering a step, or the stack's output after the final norm.
pub const Tap = union(enum) { input: u32, output };

/// Steps a stack of `layers` runs under `loop`; error.Layout when the core is outside the stack or runs no pass.
pub fn stepCount(layers: u32, loop: Loop) error{Layout}!u64 {
    const passes_ok = loop.passes > 0 and loop.passes <= std.math.maxInt(u8) + 1;
    if (loop.start > loop.end or loop.end > layers or !passes_ok) return error.Layout;
    return @as(u64, layers) + @as(u64, loop.passes - 1) * (loop.end - loop.start);
}

/// The walk of a stack whose layers keep states of the kinds in enum `Kind`, at most `max` steps.
pub fn Plan(comptime Kind: type, comptime max: usize) type {
    const kinds = @typeInfo(Kind).@"enum".field_names.len;
    return struct {
        const Self = @This();

        pub const Step = struct {
            layer: u16,
            pass: u8 = 0, // 0 outside the core
            slot: [kinds]?u16 = @splat(null), // the step's state of each kind, numbered in run order within the kind
            enter: bool = false, // the core's first layer, at every pass
            leave: bool = false, // the core's last layer, at every pass

            pub fn of(s: Step, k: Kind) ?u16 {
                return s.slot[@backingInt(k)];
            }
        };

        steps: [max]Step = undefined,
        len: u32 = 0,
        counts: [kinds]u32 = @splat(0),
        loop: Loop = straight,
        layers: u32 = 0, // physical layers

        pub fn slice(p: *const Self) []const Step {
            return p.steps[0..p.len];
        }

        /// Slots of `k`: one per step whose layer keeps that state.
        pub fn count(p: *const Self, k: Kind) u32 {
            return p.counts[@backingInt(k)];
        }

        /// The walk of `layers` under `loop`; `holds.keeps(layer, kind)`: the states a layer keeps at every visit.
        pub fn build(layers: u32, loop: Loop, holds: anytype) error{ Layout, TooManySteps }!Self {
            if (try stepCount(layers, loop) > max) return error.TooManySteps;
            var p: Self = .{ .loop = loop, .layers = layers };
            for (0..loop.start) |i| p.add(holds, @intCast(i), 0);
            for (0..loop.passes) |pass| {
                const first = p.len;
                for (loop.start..loop.end) |i| p.add(holds, @intCast(i), @intCast(pass));
                if (p.len == first) continue;
                p.steps[first].enter = true;
                p.steps[p.len - 1].leave = true;
            }
            for (loop.end..layers) |i| p.add(holds, @intCast(i), 0);
            return p;
        }

        fn add(p: *Self, holds: anytype, layer: u16, pass: u8) void {
            var s: Step = .{ .layer = layer, .pass = pass };
            inline for (0..kinds) |k| {
                if (holds.keeps(layer, @as(Kind, @fromBackingInt(@intCast(k))))) {
                    s.slot[k] = @intCast(p.counts[k]);
                    p.counts[k] += 1;
                }
            }
            p.steps[p.len] = s;
            p.len += 1;
        }

        /// The step `layer` runs at `pass` (pass 0 outside the core), or null when the walk has none.
        pub fn visit(p: *const Self, layer: u32, pass: u32) ?u32 {
            const l = p.loop;
            const core = l.end - l.start;
            if (layer < l.start) return if (pass == 0) layer else null;
            if (layer < l.end) return if (pass < l.passes) l.start + pass * core + (layer - l.start) else null;
            return if (pass == 0 and layer < p.layers) layer + (l.passes - 1) * core else null;
        }

        /// Tap t < step count reads the input to step t; t = step count, the output after the final norm.
        pub fn tap(p: *const Self, t: u32) error{NoSuchStep}!Tap {
            if (t < p.len) return .{ .input = t };
            if (t == p.len) return .output;
            return error.NoSuchStep;
        }
    };
}

const testing = std.testing;

const Two = enum { sliding, full };

/// Every third layer keeps a full cache, the others a sliding one; layer 1 keeps none.
const Thirds = struct {
    fn keeps(_: Thirds, layer: u16, k: Two) bool {
        if (layer == 1) return false;
        return (k == .full) == (layer % 3 == 2);
    }
};

test "a looped walk: prelude once, the core each pass with enter and leave, coda once, slots in run order" {
    const P = Plan(Two, 64);
    const p = try P.build(8, .{ .start = 2, .end = 5, .passes = 3 }, Thirds{});
    try testing.expectEqual(@as(u32, 8 + 2 * 3), p.len);
    const layers = [_]u16{ 0, 1, 2, 3, 4, 2, 3, 4, 2, 3, 4, 5, 6, 7 };
    const passes = [_]u8{ 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0 };
    for (p.slice(), layers, passes, 0..) |s, l, ps, i| {
        try testing.expectEqual(l, s.layer);
        try testing.expectEqual(ps, s.pass);
        try testing.expectEqual(l == 2, s.enter);
        try testing.expectEqual(l == 4, s.leave);
        try testing.expectEqual(@as(?u32, @intCast(i)), p.visit(l, ps));
    }
    // full at layers 2 and 5 (each core pass, then the coda): 4 slots; sliding at the rest but layer 1: 9
    try testing.expectEqual(@as(u32, 4), p.count(.full));
    try testing.expectEqual(@as(u32, 9), p.count(.sliding));
    try testing.expectEqual(@as(?u16, null), p.steps[1].of(.sliding));
    try testing.expectEqual(@as(?u16, null), p.steps[1].of(.full));
    const full = [_]u32{ 2, 5, 8, 11 };
    for (full, 0..) |at, n| try testing.expectEqual(@as(?u16, @intCast(n)), p.steps[at].of(.full));
    var next: u16 = 0;
    for (p.slice()) |s| if (s.of(.sliding)) |slot| {
        try testing.expectEqual(next, slot);
        try testing.expectEqual(@as(?u16, null), s.of(.full));
        next += 1;
    };
    try testing.expectEqual(@as(?u32, null), p.visit(2, 3));
    try testing.expectEqual(@as(?u32, null), p.visit(0, 1));
    try testing.expectEqual(@as(?u32, null), p.visit(8, 0));
}

test "no loop: every layer once at pass 0, nothing enters or leaves, taps are inputs then the output" {
    const P = Plan(Two, 16);
    const p = try P.build(6, straight, Thirds{});
    try testing.expectEqual(@as(u32, 6), p.len);
    for (p.slice(), 0..) |s, i| {
        try testing.expectEqual(@as(u16, @intCast(i)), s.layer);
        try testing.expect(s.pass == 0 and !s.enter and !s.leave);
        try testing.expectEqual(@as(?u32, @intCast(i)), p.visit(@intCast(i), 0));
    }
    try testing.expectEqual(Tap{ .input = 5 }, try p.tap(5));
    try testing.expectEqual(Tap.output, try p.tap(6));
    try testing.expectError(error.NoSuchStep, p.tap(7));
}

test "a core outside the stack, no pass, too many passes or too many steps is refused" {
    const P = Plan(Two, 16);
    try testing.expectError(error.Layout, P.build(6, .{ .start = 4, .end = 3, .passes = 2 }, Thirds{}));
    try testing.expectError(error.Layout, P.build(6, .{ .start = 2, .end = 7, .passes = 2 }, Thirds{}));
    try testing.expectError(error.Layout, P.build(6, .{ .start = 2, .end = 4, .passes = 0 }, Thirds{}));
    try testing.expectError(error.Layout, P.build(6, .{ .start = 2, .end = 4, .passes = 257 }, Thirds{}));
    try testing.expectError(error.TooManySteps, P.build(6, .{ .start = 2, .end = 4, .passes = 7 }, Thirds{}));
    _ = try P.build(6, .{ .start = 2, .end = 4, .passes = 6 }, Thirds{});
    try testing.expectEqual(@as(u64, 6 + 255 * 2), try stepCount(6, .{ .start = 2, .end = 4, .passes = 256 }));
}
