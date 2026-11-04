//! A round's draft rules: greedy chains stop at 0.6 beside another open stream and go 6 deep in wide rounds.
const std = @import("std");
const t = std.testing;
const lanes = @import("lanes");
const draft = @import("cuda_lanes_draft.zig");
const mtp = @import("cuda_mtp.zig");

/// The fields `rule` and `stopFor` read from the backend: its lanes by stream and its alone rules.
const Fake = struct {
    lanes: std.AutoHashMapUnmanaged(*lanes.Stream, void) = .empty,
    depth_greedy: usize = mtp.default_depth_greedy,
    confidence: f64 = mtp.default_confidence,
};

fn stream(finished: bool, paused: bool) lanes.Stream {
    var s: lanes.Stream = undefined;
    s.finished = finished;
    s.paused = paused;
    return s;
}

test "draft rule: alone the engine's; beside an open stream greedy stops at 0.6; wide rounds 6 deep" {
    var f: Fake = .{};
    defer f.lanes.deinit(t.allocator);
    var a = stream(false, false);
    var b = stream(false, false);
    var done = stream(true, false);
    var held = stream(false, true);
    try f.lanes.put(t.allocator, &a, {});
    try t.expectEqual(draft.Rule{ .greedy = 8, .greedy_stop = -0.4 }, draft.rule(&f, 1));
    // a finished stream not yet released and a paused one leave it alone
    try f.lanes.put(t.allocator, &done, {});
    try f.lanes.put(t.allocator, &held, {});
    try t.expectEqual(draft.Rule{ .greedy = 8, .greedy_stop = -0.4 }, draft.rule(&f, 1));
    // another open stream (decoding, or filling / resuming its prompt with no drafts asked)
    try f.lanes.put(t.allocator, &b, {});
    try t.expectEqual(draft.Rule{ .greedy = 8, .greedy_stop = 0.6 }, draft.rule(&f, 1));
    try t.expectEqual(@as(usize, 8), draft.rule(&f, draft.default_wide).greedy);
    try t.expectEqual(mtp.default_depth, draft.rule(&f, draft.default_wide + 1).greedy);
    // a shallower greedy setting stays as set
    f.depth_greedy = 4;
    try t.expectEqual(@as(usize, 4), draft.rule(&f, 8).greedy);
}

test "draft rule: a sampled stream keeps the engine's stop rule, a greedy one takes the round's" {
    const f: Fake = .{};
    const r: draft.Rule = .{ .greedy = 8, .greedy_stop = 0.6 };
    try t.expectEqual(@as(f64, 0.6), draft.stopFor(&f, r, null));
    try t.expectEqual(@as(f64, 0.6), draft.stopFor(&f, r, .{ .seed = 1, .temperature = 0 }));
    try t.expectEqual(@as(f64, -0.4), draft.stopFor(&f, r, .{ .seed = 1, .temperature = 1 }));
}
