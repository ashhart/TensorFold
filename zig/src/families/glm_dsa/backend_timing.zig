//! GLM-5.3 decode load-time calibration and optional draft probability telemetry.
const std = @import("std");
const lanes = @import("lanes");
const eng = @import("engine.zig");
const fl = @import("follow.zig");
const gpu = @import("gpu.zig");

const R = gpu.max_rows;
pub const P_BINS = 5;

pub const ProbStats = struct {
    hist: [eng.max_depth + 1]u64 = @splat(0),
    offered: [eng.max_depth][P_BINS]u64 = @splat(@splat(0)),
    landed: [eng.max_depth][P_BINS]u64 = @splat(@splat(0)),
    sum_p: [eng.max_depth][P_BINS]f64 = @splat(@splat(0)),
    top4_rounds: u64 = 0,
    top4_hits: [4]u64 = @splat(0),
    p2_bins_o: [P_BINS]u64 = @splat(0),
    p2_bins_h: [P_BINS]u64 = @splat(0),
    tree_rounds: u64 = 0,
    tree_branch_wins: u64 = 0,
    tree_branch_toks: u64 = 0,

    pub fn logAndReset(s: *ProbStats) void {
        std.log.info("glm_dsa prob_hist c0..5={any}", .{s.hist});
        for (0..eng.max_depth) |d| {
            var tot_o: u64 = 0;
            var tot_l: u64 = 0;
            var tot_p: f64 = 0;
            for (0..P_BINS) |bi| {
                tot_o += s.offered[d][bi];
                tot_l += s.landed[d][bi];
                tot_p += s.sum_p[d][bi];
            }
            if (tot_o > 0) std.log.info("glm_dsa prob d={d}: n={d} emp={d:.3} pred={d:.3} bins_o={any} bins_l={any}", .{ d, tot_o, @as(f64, @floatFromInt(tot_l)) / @as(f64, @floatFromInt(tot_o)), tot_p / @as(f64, @floatFromInt(tot_o)), s.offered[d], s.landed[d] });
        }
        if (s.top4_rounds > 0 or s.tree_rounds > 0) {
            std.log.info("glm_dsa top4: n={d} hits={any} p2_o={any} p2_h={any} tree_rds={d} br_wins={d} br_toks={d}", .{
                s.top4_rounds, s.top4_hits,        s.p2_bins_o,        s.p2_bins_h,
                s.tree_rounds, s.tree_branch_wins, s.tree_branch_toks,
            });
        }
        s.* = .{};
    }
};

pub const SlotCal = struct { pred0: f64 = 0, hit0: f64 = 0, pred: f64 = 0, hit: f64 = 0, was_mtp: bool = false };

fn ms(io: std.Io, t0: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(t0.durationTo(std.Io.Timestamp.now(io, .awake)).nanoseconds)) / 1e6;
}

/// Rank 0 at load: windows of 1-16 distinct-token rows and head chains timed on a scratch stream.
pub fn time(e: *eng.Engine, io: std.Io, costs: *[R]lanes.config.Cost) !f64 {
    var prompt: [64]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(1000 + 7 * i);
    try fl.begin(e, 0, &prompt);
    var at: u32 = 0;
    var first: u32 = 0;
    while (at < prompt.len) : (at += R) first = (try fl.chunk(e, 0, at, R, false)) orelse first;
    var toks: [R]u32 = undefined;
    toks[0] = first;
    if (e.hasDraft()) {
        try fl.draft(e, &.{.{ .slot = 0, .prompt = true, .follow = toks[0..1], .depth = eng.max_depth }});
        const h = e.held(0);
        for (1..R) |i| toks[i] = if (i - 1 < h.len and h[i - 1] != toks[i - 1]) h[i - 1] else prompt[i];
    } else {
        for (1..R) |i| toks[i] = prompt[i];
    }
    var picks: [R]u32 = undefined;
    // Full-width warmup so cold GPU weight pages never skew costs across any row width.
    try fl.window(e, &.{.{ .slot = 0, .tokens = &toks }}, &picks);
    try fl.keep(e, 0, 1);
    for (0..R) |w| {
        var best: f64 = std.math.inf(f64);
        for (0..3) |_| {
            const t0 = std.Io.Timestamp.now(io, .awake);
            try fl.window(e, &.{.{ .slot = 0, .tokens = toks[0 .. w + 1] }}, &picks);
            best = @min(best, ms(io, t0));
            try fl.keep(e, 0, 1);
        }
        if (w > 0) best = @max(best, costs[w - 1].ms);
        costs[w] = .{ .width = @intCast(w + 1), .ms = best };
    }
    if (R >= 8 and costs[7].ms > costs[1].ms) {
        const c1 = costs[1].ms;
        const span = costs[7].ms - c1;
        for (2..7) |w| costs[w].ms = @max(costs[w - 1].ms, 0.35 * costs[w].ms + 0.65 * (c1 + span * (@as(f64, @floatFromInt(w - 1)) / 6.0)));
    }
    var head_ms: f64 = 0;
    if (e.hasDFlash2()) {
        try fl.windowDraft(e, 0, toks[0..1], eng.max_depth, &picks);
        var fused_ms: f64 = std.math.inf(f64);
        for (0..3) |_| {
            const t0 = std.Io.Timestamp.now(io, .awake);
            try fl.windowDraft(e, 0, toks[0..1], eng.max_depth, &picks);
            fused_ms = @min(fused_ms, ms(io, t0));
        }
        head_ms = @max(0.1, @max(1.0, fused_ms - costs[0].ms) / @as(f64, eng.max_depth));
    } else if (e.hasMtp()) {
        try fl.draft(e, &.{.{ .slot = 0, .prompt = false, .follow = toks[0..1], .depth = eng.max_depth }});
        try fl.window(e, &.{.{ .slot = 0, .tokens = toks[0..1] }}, &picks);
        try fl.keep(e, 0, 1);
        var one: f64 = std.math.inf(f64);
        var four: f64 = std.math.inf(f64);
        for (0..3) |_| {
            var t0 = std.Io.Timestamp.now(io, .awake);
            try fl.draft(e, &.{.{ .slot = 0, .prompt = false, .follow = toks[0..1], .depth = 1 }});
            one = @min(one, ms(io, t0));
            try fl.window(e, &.{.{ .slot = 0, .tokens = toks[0..1] }}, &picks);
            try fl.keep(e, 0, 1);
            t0 = std.Io.Timestamp.now(io, .awake);
            try fl.draft(e, &.{.{ .slot = 0, .prompt = false, .follow = toks[0..1], .depth = eng.max_depth }});
            four = @min(four, ms(io, t0));
            try fl.window(e, &.{.{ .slot = 0, .tokens = toks[0..1] }}, &picks);
            try fl.keep(e, 0, 1);
        }
        head_ms = @max(0.75 * one, (four - one) / @as(f64, eng.max_depth - 1));
    }
    fl.release(e, 0);
    std.log.info("glm_dsa timing: r1={d:.2} r2={d:.2} r3={d:.2} r4={d:.2} r5={d:.2} r6={d:.2} r7={d:.2} r8={d:.2} head={d:.2} ms", .{ costs[0].ms, costs[1].ms, costs[2].ms, costs[3].ms, costs[4].ms, costs[5].ms, costs[6].ms, costs[7].ms, head_ms });
    return head_ms;
}
