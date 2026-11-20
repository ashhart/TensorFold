//! A ZONOS3 request's duration target under LACI: the take held to a length, retaken at another speaking rate.
const std = @import("std");
const api = @import("engine_api");
const z3 = @import("zonos3");

const cfg = z3.config;

/// The speaking-rate condition's place among the condition rows.
const rate_index = for (cfg.condition_names, 0..) |name, i| {
    if (std.mem.eql(u8, name, "speaking_rate")) break i;
} else @compileError("no speaking_rate condition");

/// Whether a request's duration target can be held: a finite length under LACI, or none.
pub fn valid(r: *const api.SpeechRequest) bool {
    if (r.duration_frames == 0) return true;
    return r.laci != null and std.math.isFinite(r.duration_frames) and r.duration_frames > 0;
}

/// `setup` held to the request's target; `kept`'s speaking-rate row is rewritten when a retake changes the bucket.
pub fn hold(e: *const z3.Engine, r: *const api.SpeechRequest, condition_row: usize, kept: []z3.Replace, setup: *z3.laci_run.Setup) !void {
    if (r.duration_frames == 0) return;
    var conditions: [cfg.conditions]u32 = undefined;
    // a condition the request leaves out is unknown: the last row of its table
    for (&conditions, 0..) |*c, i| c.* = if (r.conditions.len != 0) r.conditions[i] else e.c.buckets[i];
    const n = e.c.buckets[rate_index];
    setup.duration = .{
        .target = .{ .frames = r.duration_frames, .adjust_rate = r.duration_adjust_rate },
        .rates = .{ .index = rate_index, .edges = e.c.edges[rate_index][0 .. n - 1], .n_buckets = n },
        .conditions = conditions,
    };
    for (kept) |*k| if (k.row == condition_row + rate_index) {
        setup.rate_row = .{ .vec = &k.vec, .table = e.w.conditions[rate_index] };
        return;
    };
    return error.NoSpeakingRateRow;
}
