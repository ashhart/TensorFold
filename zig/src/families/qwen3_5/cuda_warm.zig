//! Warm each captured launch shape, then measure every live width with a freshly reset cache.
const std = @import("std");
const E = @import("cuda_engine.zig");

pub const Seen = struct {
    keys: [E.most_verify + E.most_window]u32 = undefined,
    len: usize = 0,
    counts: [E.most_verify + E.most_window]u8 = @splat(0),

    pub fn contains(self: *const Seen, shape: E.Engine.Shape) bool {
        return std.mem.indexOfScalar(u32, self.keys[0..self.len], shape.key()) != null;
    }

    pub fn passes(self: *const Seen, shape: E.Engine.Shape) u8 {
        for (self.keys[0..self.len], 0..) |key, i| if (key == shape.key()) return self.counts[i];
        return 0;
    }

    pub fn mark(self: *Seen, shape: E.Engine.Shape) void {
        for (self.keys[0..self.len], 0..) |key, i| if (key == shape.key()) {
            self.counts[i] +|= 1;
            return;
        };
        std.debug.assert(self.len < self.keys.len);
        self.keys[self.len] = shape.key();
        self.counts[self.len] = 1;
        self.len += 1;
    }
};

pub fn time(io: std.Io, e: anytype, d: anytype, cache: anytype, warmed: *Seen) !void {
    const ids: [E.most_window]u32 = @splat(0);
    var picks: [E.most_window]E.Draw = undefined;
    for (&picks) |*pk| pk.* = .{ .rule = null, .position = 1 };
    for (1..e.window_rows + 1) |rows| {
        const shape = E.Engine.Shape.init(rows, 1, true, e.n_taps > 0, 1);
        while (warmed.passes(shape) < 2) {
            try cache.reset(e);
            try e.forward(&.{.{ .cache = cache, .rows = rows }}, ids[0..rows], .{ .draws = picks[0..rows] });
            warmed.mark(shape);
        }
        try cache.reset(e);
        const t0 = std.Io.Clock.awake.now(io);
        try e.forward(&.{.{ .cache = cache, .rows = rows }}, ids[0..rows], .{ .draws = picks[0..rows] });
        const ms = @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1e6;
        d.setCost(@intCast(rows), ms);
    }
}
