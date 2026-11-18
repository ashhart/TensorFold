//! Span steering that follows the reading position (zonos3.tts.span_follow): a span is on while the model reads it.
const std = @import("std");
const cfg = @import("config.zig");
const probe_mod = @import("laci_probe.zig");
const laci = @import("laci.zig");

/// lead, lag: tokens before a span's first token it ramps in at, and past its last it ramps out at.
pub const Config = struct { lead: f64 = 1.0, lag: f64 = 0.0, smooth: usize = 3, ramp: usize = 3 };

const Event = struct { decided: usize, anchor: i64 };

/// The span gate of one request: `observe` takes each decode step's reading, `weights` decides a frame's spans.
pub const Follower = struct {
    gpa: std.mem.Allocator,
    /// Text positions [a, b) of the tokens spelling each span (0 is BOS).
    ranges: []const [2]u32,
    c: Config,
    frames: std.ArrayList(usize) = .empty,
    pos: std.ArrayList(usize) = .empty,
    /// The smoothed reading position: a running median, held monotone, at most a token a reading.
    best: f64 = 0,
    enter: []?Event,
    exit: []?Event,
    /// Each decided frame's weights in float64, a span each: a twin and a rollback see what was decided.
    decided: std.ArrayList(usize) = .empty,
    kept: std.ArrayList(f64) = .empty,
    scratch: []f64,

    pub fn init(gpa: std.mem.Allocator, ranges: []const [2]u32, c: Config) !Follower {
        if (c.smooth < 1 or c.smooth > 15 or c.ramp < 1) return error.BadFollowConfig;
        const enter = try gpa.alloc(?Event, ranges.len);
        errdefer gpa.free(enter);
        const exit = try gpa.alloc(?Event, ranges.len);
        errdefer gpa.free(exit);
        @memset(enter, null);
        @memset(exit, null);
        return .{ .gpa = gpa, .ranges = ranges, .c = c, .enter = enter, .exit = exit, .scratch = try gpa.alloc(f64, ranges.len) };
    }

    pub fn deinit(f: *Follower) void {
        f.frames.deinit(f.gpa);
        f.pos.deinit(f.gpa);
        f.decided.deinit(f.gpa);
        f.kept.deinit(f.gpa);
        f.gpa.free(f.enter);
        f.gpa.free(f.exit);
        f.gpa.free(f.scratch);
    }

    /// The reading of the step that generated `frame`; frames must increase.
    pub fn observe(f: *Follower, frame: usize, pos: usize) !void {
        if (f.frames.items.len != 0 and frame <= f.frames.items[f.frames.items.len - 1]) return error.FollowFramesMustIncrease;
        try f.frames.append(f.gpa, frame);
        try f.pos.append(f.gpa, pos);
        f.advance(f.pos.items.len);
    }

    /// statistics.median of the last `smooth` readings up to the n-th, then the monotone clamp.
    fn advance(f: *Follower, n: usize) void {
        var last: [15]usize = undefined;
        const from = n -| f.c.smooth;
        const window = last[0 .. n - from];
        @memcpy(window, f.pos.items[from..n]);
        std.mem.sort(usize, window, {}, std.sort.asc(usize));
        const mid = window.len / 2;
        const med: f64 = if (window.len % 2 == 1) @floatFromInt(window[mid]) else (@as(f64, @floatFromInt(window[mid - 1])) + @as(f64, @floatFromInt(window[mid]))) / 2.0;
        f.best = @min(@max(f.best, med), f.best + 1.0);
    }

    /// The weights of the spans for the frame generated now, as a delta takes them (fp32); `out` holds one a span.
    pub fn weights(f: *Follower, frame: usize, out: []f32) !void {
        try f.decide(frame, f.scratch);
        for (out, f.scratch) |*o, w| o.* = @floatCast(w);
    }

    /// The spans' weights for a frame in float64, decided once a frame (span_follow.weights).
    pub fn decide(f: *Follower, frame: usize, out: []f64) !void {
        const k = f.ranges.len;
        for (f.decided.items, 0..) |d, i| if (d == frame) {
            @memcpy(out, f.kept.items[i * k ..][0..k]);
            return;
        };
        const ramp: f64 = @floatFromInt(f.c.ramp);
        const at: i64 = @intCast(frame);
        for (f.ranges, f.enter, f.exit) |r, *enter, *exit| {
            if (r[0] >= r[1]) continue;
            // on before any reading (a span at the start of the text): full weight at once, as the schedule gives it
            if (enter.* == null and f.best >= @as(f64, @floatFromInt(r[0])) - f.c.lead)
                enter.* = .{ .decided = frame, .anchor = if (f.frames.items.len == 0) at - @as(i64, @intCast(f.c.ramp)) + 1 else at };
            if (enter.* != null and exit.* == null and f.best >= @as(f64, @floatFromInt(r[1])) + f.c.lag) exit.* = .{ .decided = frame, .anchor = at };
        }
        for (out, f.enter, f.exit) |*w, enter, exit| {
            const e = enter orelse {
                w.* = 0;
                continue;
            };
            const up = @as(f64, @floatFromInt(at - e.anchor + 1)) / ramp;
            const down = if (exit) |x| @as(f64, @floatFromInt(x.anchor + @as(i64, @intCast(f.c.ramp)) - 1 - at)) / ramp else 1.0;
            w.* = @min(1.0, @max(0.0, @min(up, down)));
        }
        try f.decided.append(f.gpa, frame);
        try f.kept.appendSlice(f.gpa, out);
    }

    /// A rollback: the readings and decisions of frames from `frame` on are forgotten.
    pub fn truncate(f: *Follower, frame: usize) void {
        var keep: usize = 0;
        while (keep < f.frames.items.len and f.frames.items[keep] < frame) keep += 1;
        f.frames.shrinkRetainingCapacity(keep);
        f.pos.shrinkRetainingCapacity(keep);
        f.best = 0;
        for (1..keep + 1) |n| f.advance(n);
        for ([_][]?Event{ f.enter, f.exit }) |events| for (events) |*e| {
            if (e.*) |x| if (x.decided >= frame) {
                e.* = null;
            };
        };
        const k = f.ranges.len;
        var n: usize = 0;
        for (f.decided.items, 0..) |d, i| {
            if (d >= frame) continue;
            f.decided.items[n] = d;
            std.mem.copyForwards(f64, f.kept.items[n * k ..][0..k], f.kept.items[i * k ..][0..k]);
            n += 1;
        }
        f.decided.shrinkRetainingCapacity(n);
        f.kept.shrinkRetainingCapacity(n * k);
    }
};

/// A decode row's reading for a request without LACI: its queries against the text rows' keys, no head masked.
pub const Reader = struct {
    gpa: std.mem.Allocator,
    heads: laci.Heads,
    keys: [probe_mod.layers][]f32,
    probed: probe_mod.Probed,
    scratch: []f32,

    /// After the prompt ran on the lane in use: the text rows' keys are read once.
    pub fn init(gpa: std.mem.Allocator, engine: anytype, heads: laci.Heads, text_start: usize, text_len: usize) !Reader {
        var keys: [probe_mod.layers][]f32 = @splat(&.{});
        errdefer for (keys) |k| gpa.free(k);
        for (&keys, 0..) |*k, s| {
            k.* = try gpa.alloc(f32, text_len * probe_mod.key_width);
            try engine.keys((s + 1) * cfg.gdn_period - 1, text_start, k.*);
        }
        return .{ .gpa = gpa, .heads = heads, .keys = keys, .probed = .{ .text_start = text_start, .text_len = text_len }, .scratch = try gpa.alloc(f32, 3 * text_len) };
    }

    pub fn deinit(r: *Reader) void {
        for (r.keys) |k| r.gpa.free(k);
        r.gpa.free(r.scratch);
    }

    /// The reading position of the last forward that took `probed`.
    pub fn position(r: *Reader) usize {
        return probe_mod.readout(r.heads, &.{}, &r.probed.q, .{ r.keys[0], r.keys[1] }, r.probed.text_len, r.scratch).pos;
    }
};

test "follower: a span ramps in a token before its first, out past its last; a rollback forgets" {
    const t = std.testing;
    var f = try Follower.init(t.allocator, &.{ .{ 3, 5 }, .{ 0, 0 } }, .{});
    defer f.deinit();
    var w: [2]f32 = undefined;
    try f.weights(1, &w);
    try t.expectEqual(@as(f32, 0), w[0]);
    // the median of the last three readings (of two: their mean), at most a token a reading
    try f.observe(1, 1);
    try f.observe(2, 9);
    try t.expectEqual(@as(f64, 2), f.best);
    try f.observe(3, 9);
    try t.expectEqual(@as(f64, 3), f.best);
    try f.weights(4, &w);
    try t.expectEqual(@as(f32, 1.0 / 3.0), w[0]);
    try t.expectEqual(@as(f32, 0), w[1]);
    try f.weights(5, &w);
    try t.expectEqual(@as(f32, 2.0 / 3.0), w[0]);
    try f.observe(4, 9);
    try f.observe(5, 9);
    try f.observe(6, 9);
    // the reading is past the span's end when frame 7 is decided: it ramps out from there
    try f.weights(7, &w);
    try t.expectEqual(@as(f32, 2.0 / 3.0), w[0]);
    try f.weights(8, &w);
    try t.expectEqual(@as(f32, 1.0 / 3.0), w[0]);
    try f.weights(7, &w);
    try t.expectEqual(@as(f32, 2.0 / 3.0), w[0]);
    f.truncate(5);
    try t.expectEqual(@as(usize, 4), f.frames.items.len);
    try t.expectEqual(@as(f64, 4), f.best);
    try t.expect(f.exit[0] == null and f.enter[0] != null);
    try f.weights(5, &w);
    try t.expectEqual(@as(f32, 2.0 / 3.0), w[0]);
}
