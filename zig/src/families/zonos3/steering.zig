//! Hidden steering: the delta each generated frame adds to Full AttnRes bank sources (zonos3.tts.steering).
const std = @import("std");
const cfg = @import("config.zig");

const D = cfg.dim;

/// A delta on one bank source of one row: 0 the embedding, 2i + 1 layer i's attention or GDN output, 2i + 2 its FFN's.
pub const Steer = struct { source: u32, delta: *const [D]f32 };

/// A span's rows of the schedule: full weight on frames [start, end), a linear ramp of `ramp` frames on both sides.
pub const Span = struct { start: f32, end: f32, ramp: f32, delta: []const f32 };

/// steer_release as frames: the released controls at full dose for `onset` frames, a ramp to `hold`, `hold` after.
pub const Release = struct { onset: u32, ramp: u32, hold: f64 };

/// What a request's controls add to its generated frames. Every delta holds `sources.len` rows of `dim`.
pub const Plan = struct {
    sources: []const u32 = &.{},
    /// The controls that never release (and every control when none does); empty: none.
    constant: []const f32 = &.{},
    /// The controls `release` weights by the frame; empty: none.
    released: []const f32 = &.{},
    release: Release = .{ .onset = 0, .ramp = 0, .hold = 1 },
    spans: []const Span = &.{},

    pub fn active(p: *const Plan) bool {
        return p.sources.len != 0 and (p.constant.len != 0 or p.released.len != 0 or p.spans.len != 0);
    }

    /// steering.release_weight, in the reference's float64.
    pub fn weight(r: Release, frame: usize) f64 {
        if (frame < r.onset) return 1.0;
        if (frame >= r.onset + r.ramp) return r.hold;
        return 1.0 + (r.hold - 1.0) * @as(f64, @floatFromInt(frame - r.onset)) / @as(f64, @floatFromInt(r.ramp));
    }

    /// synth.steer_fn: frame `frame`'s delta into `out` in the reference's fp32 steps (`sum`: scratch); false: none.
    pub fn at(p: *const Plan, frame: usize, out: []f32, sum: []f32) bool {
        return p.weighed(frame, out, sum, null);
    }

    /// As `at`; `follow`, when given, holds each span's weight for the frame (a span follower's) in place of its own.
    pub fn weighed(p: *const Plan, frame: usize, out: []f32, sum: []f32, follow: ?[]const f32) bool {
        std.debug.assert(out.len == p.sources.len * D and sum.len == out.len);
        var have = false;
        if (p.released.len != 0) {
            const w: f32 = @floatCast(weight(p.release, frame));
            for (out, p.released) |*o, x| o.* = if (w == 1) x else w * x;
            if (p.constant.len != 0) for (out, p.constant) |*o, x| {
                o.* = x + o.*;
            };
            have = true;
        } else if (p.constant.len != 0) {
            @memcpy(out, p.constant);
            have = true;
        }
        // steering.schedule_delta: every span's row weighed and summed in the schedule's order, when one is in reach
        const f: f32 = @floatFromInt(frame);
        var reach = false;
        for (p.spans, 0..) |s, k| reach = reach or (if (follow) |w| w[k] else spanWeight(s, f)) > 0;
        if (!reach) return have;
        const spans = if (have) sum else out;
        for (p.spans, 0..) |s, k| {
            const w = if (follow) |ws| ws[k] else spanWeight(s, f);
            for (spans, s.delta) |*o, x| o.* = if (k == 0) w * x else o.* + w * x;
        }
        if (have) for (out, sum) |*o, x| {
            o.* += x;
        };
        return true;
    }
};

/// Whether a host may run `st` (a core SpeechSteering): known sources, whole finite deltas, one twin a request.
pub fn valid(st: anytype, speaker_guided: bool) bool {
    if (!std.math.isFinite(st.guidance) or st.guidance < 0 or (st.guidance != 1 and speaker_guided)) return false;
    if (st.sources.len == 0 or st.sources.len > cfg.sources or !(st.hold >= 0 and st.hold <= 1)) return false;
    for (st.sources) |source| if (source >= cfg.sources) return false;
    const width = st.sources.len * D;
    if ((st.constant.len != 0 and st.constant.len != width) or (st.released.len != 0 and st.released.len != width)) return false;
    for (st.constant) |x| if (!std.math.isFinite(x)) return false;
    for (st.released) |x| if (!std.math.isFinite(x)) return false;
    if (st.follow) |f| {
        if (f.ranges.len != st.spans.len or f.heads.len == 0 or f.heads.len > cfg.heads * cfg.layers) return false;
        if (!std.math.isFinite(f.lead) or !std.math.isFinite(f.lag) or f.lead < 0 or f.lead > 8 or f.lag < -4 or f.lag > 8) return false;
        for (f.ranges) |r| if (r[0] > r[1]) return false;
        for (f.heads) |h| if (h.head >= cfg.heads or h.layer >= cfg.layers or cfg.usesGdn(h.layer)) return false;
    }
    for (st.spans) |s| {
        if (s.delta.len != width or !std.math.isFinite(s.start) or !std.math.isFinite(s.end) or !std.math.isFinite(s.ramp)) return false;
        for (s.delta) |x| if (!std.math.isFinite(x)) return false;
    }
    return true;
}

fn spanWeight(s: Span, f: f32) f32 {
    const ramp = @max(s.ramp, 1.0);
    const up = (f - (s.start - ramp)) / ramp;
    const down = ((s.end + ramp) - f) / ramp;
    return std.math.clamp(@min(up, down), 0.0, 1.0);
}

test "release: full dose to the onset, a linear ramp, then the hold" {
    const t = std.testing;
    const r: Release = .{ .onset = 60, .ramp = 15, .hold = 0.35 };
    try t.expectEqual(@as(f64, 1), Plan.weight(r, 0));
    try t.expectEqual(@as(f64, 1), Plan.weight(r, 59));
    try t.expectEqual(@as(f64, 1), Plan.weight(r, 60));
    try t.expectApproxEqAbs(@as(f64, 1 - 0.65 / 15.0), Plan.weight(r, 61), 1e-7);
    try t.expectEqual(@as(f64, 0.35), Plan.weight(r, 75));
    // no ramp: a step at the onset
    try t.expectEqual(@as(f64, 0.5), Plan.weight(.{ .onset = 3, .ramp = 0, .hold = 0.5 }, 3));
}

test "a frame's delta: the constant controls, the released ones at the frame's weight, the spans in reach" {
    const t = std.testing;
    const a = t.allocator;
    const n = 2 * D;
    const constant = try a.alloc(f32, n);
    defer a.free(constant);
    const released = try a.alloc(f32, n);
    defer a.free(released);
    const span = try a.alloc(f32, n);
    defer a.free(span);
    const sum = try a.alloc(f32, n);
    defer a.free(sum);
    const out = try a.alloc(f32, n);
    defer a.free(out);
    @memset(constant, 1);
    @memset(released, 2);
    @memset(span, 8);
    var p: Plan = .{ .sources = &.{ 15, 19 } };
    try t.expect(!p.active());
    try t.expect(!p.at(1, out, sum));
    p.constant = constant;
    p.released = released;
    p.release = .{ .onset = 2, .ramp = 2, .hold = 0.5 };
    try t.expect(p.active() and p.at(1, out, sum));
    try t.expectEqual(@as(f32, 3), out[0]);
    try t.expect(p.at(3, out, sum));
    try t.expectEqual(@as(f32, 1 + 0.75 * 2), out[n - 1]);
    try t.expect(p.at(9, out, sum));
    try t.expectEqual(@as(f32, 2), out[0]);
    // a span on frames [10, 12) with a ramp of 3: a third at frame 8, whole from 10 through 12, two thirds at 13
    p.spans = &.{.{ .start = 10, .end = 12, .ramp = 3, .delta = span }};
    try t.expect(p.at(6, out, sum));
    try t.expectEqual(@as(f32, 2), out[0]);
    try t.expect(p.at(8, out, sum));
    try t.expectEqual(@as(f32, 2) + @as(f32, 1.0 / 3.0) * 8, out[0]);
    try t.expect(p.at(11, out, sum));
    try t.expectEqual(@as(f32, 10), out[0]);
    p.constant = &.{};
    p.released = &.{};
    try t.expect(!p.at(3, out, sum));
    try t.expect(p.at(12, out, sum));
    try t.expectEqual(@as(f32, 8), out[0]);
    try t.expect(p.at(13, out, sum));
    try t.expectEqual(@as(f32, 2.0 / 3.0) * 8, out[0]);
    try t.expect(!p.at(15, out, sum));
    // a request's steering as a host checks it
    const Head = struct { layer: u32, head: u32, radius: i64 };
    const Follow = struct { ranges: []const [2]u32, lead: f64 = 1, lag: f64 = 0, heads: []const Head };
    const Sent = struct { sources: []const u32, constant: []const f32 = &.{}, released: []const f32 = &.{}, hold: f64 = 1, spans: []const Span = &.{}, guidance: f32 = 1, follow: ?Follow = null };
    try t.expect(valid(Sent{ .sources = &.{ 15, 19 }, .constant = constant }, true));
    try t.expect(!valid(Sent{ .sources = &.{ 15, 19 }, .constant = constant, .guidance = 1.5 }, true));
    try t.expect(!valid(Sent{ .sources = &.{ 15, 23 }, .constant = constant }, false));
    try t.expect(!valid(Sent{ .sources = &.{15}, .constant = constant }, false));
    try t.expect(!valid(Sent{ .sources = &.{ 15, 19 }, .spans = &.{.{ .start = 0, .end = 1, .ramp = 3, .delta = span[0..D] }} }, false));
    // a follower's ranges, one a span, and heads of attention layers
    const spans = [_]Span{.{ .start = 0, .end = 1, .ramp = 3, .delta = span }};
    const heads = [_]Head{.{ .layer = 3, .head = 2, .radius = 4 }};
    try t.expect(valid(Sent{ .sources = &.{ 15, 19 }, .spans = &spans, .follow = .{ .ranges = &.{.{ 1, 3 }}, .heads = &heads } }, false));
    try t.expect(!valid(Sent{ .sources = &.{ 15, 19 }, .spans = &spans, .follow = .{ .ranges = &.{}, .heads = &heads } }, false));
    try t.expect(!valid(Sent{ .sources = &.{ 15, 19 }, .spans = &spans, .follow = .{ .ranges = &.{.{ 1, 3 }}, .heads = &.{.{ .layer = 0, .head = 2, .radius = 4 }} } }, false));
    try t.expect(!valid(Sent{ .sources = &.{ 15, 19 }, .spans = &spans, .follow = .{ .ranges = &.{.{ 1, 3 }}, .lead = 9, .heads = &heads } }, false));
    // the follower's weights in place of the spans' own: the span at a third
    p.spans = &spans;
    try t.expect(p.weighed(1, out, sum, &.{1.0 / 3.0}));
    try t.expectEqual(@as(f32, 1.0 / 3.0) * 8, out[0]);
    try t.expect(!p.weighed(1, out, sum, &.{0}));
}
