//! Causal placement: remove one key row's layer contribution, rank the answer-logit drops, then bound the budget.
const std = @import("std");
const Example = @import("lesson_examples.zig").Example;
const mode = @import("train_mode.zig");
const pocket = @import("pocket.zig");
pub const floor = 2;
pub const ceiling = 10;
pub const questions = 8;
const max_slots = 64;

/// Only a learner's diagnostic forward uses this; ordinary forward and backward passes leave it inactive.
pub const Probe = struct { active: bool = false, layer: ?usize = null, row: usize = 0 };
pub const State = struct {
    enabled: bool = false,
    done: bool = false,
    example: usize = 0,
    slot: usize = 0,
    baseline: ?f32 = null,
    score: [max_slots]f32 = @splat(0),
    chosen: [max_slots]bool = @splat(false),
};

/// Raw correct-answer logits, averaged over its tokens; the trainer returns their negative in its loss slot.
pub fn logit(t: anytype, ex: Example, layer: ?usize, row: usize) !f32 {
    if (t.locate.active or row >= ex.start) return error.BadCausalProbe;
    t.locate = .{ .active = true, .layer = layer, .row = row };
    defer t.locate = .{};
    var out: [1]mode.Result = undefined;
    try t.pass(&.{.{ .ids = ex.ids, .start = ex.start, .decide = ex.decide }}, .loss, @import("pocket_pack.zig").turn, &out);
    if (!std.math.isFinite(out[0].loss)) return error.NonfiniteCausalLogit;
    return -out[0].loss;
}

/// Largest positive drops first, at least two and at most ten layers; ties keep the earlier layer.
pub fn select(scores: []const f32, free: []const bool, chosen: []bool) !usize {
    @memset(chosen, false);
    var peak: f32 = 0;
    var available: usize = 0;
    for (scores, free) |x, f| {
        if (!std.math.isFinite(x)) return error.NonfiniteCausalLogit;
        if (f) {
            available += 1;
            peak = @max(peak, x);
        }
    }
    if (available < floor) return error.PocketFull;
    var n: usize = 0;
    while (n < @min(ceiling, available)) {
        var best: ?usize = null;
        for (scores, free, 0..) |x, f, i| {
            if (!f or chosen[i]) continue;
            if (best == null or x > scores[best.?]) best = i;
        }
        const i = best.?;
        if (n >= floor and (scores[i] <= 0 or scores[i] < peak * 0.1)) break;
        chosen[i] = true;
        n += 1;
    }
    return n;
}

/// A correction locates its existing belief; a new fact locates its requested answer.
pub fn answers(p: anytype) []const Example {
    return if (p.against.len > 0) p.against else p.answered;
}

/// One forward per unit, on up to eight lesson answers; no held-out answer enters the causal ranking.
pub fn once(p: anytype, t: anytype) !bool {
    if (comptime !@hasDecl(@TypeOf(t.*), "causalLogit")) return true;
    const s = &p.locate;
    if (!s.enabled or s.done) return true;
    const n = p.moe.items.len;
    if (n > max_slots) return error.TooManyCausalLayers;
    const source = answers(p);
    if (s.example < @min(questions, source.len)) {
        const ex = source[s.example];
        const row = pocket.subjectRow(ex) orelse ex.decide;
        if (s.baseline == null) {
            s.baseline = try t.causalLogit(ex, null, row);
            return false;
        }
        if (s.slot < n) {
            const j = s.slot;
            s.slot += 1;
            if (p.used.items[j] < @TypeOf(p.*).neurons) {
                const altered = try t.causalLogit(ex, p.moe.items[j], row);
                s.score[j] += @max(0, s.baseline.? - altered);
            }
            return false;
        }
        s.example += 1;
        s.slot = 0;
        s.baseline = null;
        return false;
    }
    if (s.example == 0) return error.NoCausalQuestions;
    var free: [max_slots]bool = @splat(false);
    for (p.used.items, 0..) |used, i| {
        free[i] = used < @TypeOf(p.*).neurons;
        s.score[i] /= @floatFromInt(s.example);
    }
    const count = try select(s.score[0..n], free[0..n], s.chosen[0..n]);
    // Preserve the requested key shares, putting rounding's remainder on the largest share.
    const original = p.split;
    const total = original[0] + original[1] + original[2];
    var largest: usize = 0;
    var assigned: usize = 0;
    for (original, 0..) |x, k| {
        if (x > original[largest]) largest = k;
        p.split[k] = count * x / total;
        assigned += p.split[k];
    }
    p.split[largest] += count - assigned;
    for (p.moe.items, 0..) |layer, j| std.log.info("slide locate: layer {d}, mean answer-logit drop {d:.6}, selected {any}", .{ layer, s.score[j], s.chosen[j] });
    std.log.info("slide locate: {d} layers from {d} questions (floor {d}, ceiling {d})", .{ count, s.example, floor, ceiling });
    s.done = true;
    return true;
}
