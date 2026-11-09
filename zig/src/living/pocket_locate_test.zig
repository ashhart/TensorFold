//! A tiny residual model: only two layers at the subject row carry the answer, irrespective of other activations.
const std = @import("std");
const locate = @import("pocket_locate.zig");
const pocket = @import("pocket.zig");
const examples = @import("lesson_examples.zig");
const mode = @import("train_mode.zig");
const t = std.testing;
const Toy = struct {
    locate: locate.Probe = .{},
    calls: usize = 0,
    fail: bool = false,
    pub fn causalLogit(m: *@This(), sample: examples.Example, layer: ?usize, row: usize) !f32 {
        return locate.logit(m, sample, layer, row);
    }
    pub fn pass(m: *@This(), seqs: []const mode.Seq, _: mode.Mode, _: @import("lesson_route.zig").Turn, out: []mode.Result) !void {
        m.calls += 1;
        if (m.fail) return error.ProbeFailed;
        var residual: [4][2]f32 = @splat(.{ 0, 0 });
        for (0..8) |l| {
            const before = residual;
            // Large irrelevant coordinates deliberately defeat activation-norm ranking.
            for (&residual) |*r| r[1] += 1000;
            residual[1][0] += if (l == 1) @as(f32, 3) else if (l == 5) @as(f32, 5) else 0;
            if (m.locate.active and m.locate.layer == l) residual[m.locate.row] = before[m.locate.row];
        }
        // The answer row reads the subject, as attention would; its target logit is the first coordinate only.
        const answer = residual[1][0] + @as(f32, @floatFromInt(seqs[0].ids[0])) * 0.01;
        out[0] = .{ .loss = -answer, .recalled = false };
    }
};
const ids = [_]u32{ 7, 11, 3, 4, 5 };
const ex: examples.Example = .{ .ids = &ids, .start = 4, .decide = 3, .subject = 1 };

test "causal placement: known answer layers survive huge unrelated activations and use the subject row" {
    const P = pocket.Pocket(Toy, @import("families.zig").nemotron_h);
    var p: P = .{ .gpa = t.allocator, .mode = .subject, .locate = .{ .enabled = true }, .answered = &.{ex}, .split = .{ 0, 10, 0 } };
    defer p.deinit();
    for (0..8) |l| try p.moe.append(t.allocator, l);
    try p.used.appendNTimes(t.allocator, 0, 8);
    var m: Toy = .{};
    while (!try locate.once(&p, &m)) {}
    try t.expectEqual([3]usize{ 0, 2, 0 }, p.split);
    for (0..8) |l| {
        try t.expectEqual(l == 1 or l == 5, p.candidate(l, l, .subject));
        try t.expectApproxEqAbs(if (l == 1) @as(f32, 3) else if (l == 5) @as(f32, 5) else 0, p.locate.score[l], 0.00001);
    }
    try t.expectEqual(@as(usize, 9), m.calls);
    try t.expect(!m.locate.active);
    const base = try m.causalLogit(ex, null, 1);
    try t.expectEqual(base, try m.causalLogit(ex, 5, 2));
    try t.expect((try m.causalLogit(ex, 5, 1)) < base);
}

test "causal placement: floor, ceiling, full pockets, nonfinite reads and failure cleanup" {
    var scores: [16]f32 = @splat(1);
    var free: [16]bool = @splat(true);
    var chosen: [16]bool = undefined;
    try t.expectEqual(@as(usize, locate.ceiling), try locate.select(&scores, &free, &chosen));
    @memset(&scores, 0);
    try t.expectEqual(@as(usize, locate.floor), try locate.select(&scores, &free, &chosen));
    free[0] = false;
    _ = try locate.select(&scores, &free, &chosen);
    try t.expect(!chosen[0]);
    @memset(&free, false);
    try t.expectError(error.PocketFull, locate.select(&scores, &free, &chosen));
    scores[0] = std.math.nan(f32);
    try t.expectError(error.NonfiniteCausalLogit, locate.select(&scores, &free, &chosen));
    var m: Toy = .{ .fail = true };
    try t.expectError(error.ProbeFailed, m.causalLogit(ex, 1, 1));
    try t.expect(!m.locate.active);
}

test "causal placement: comparison mode retains the old middle-layer candidates" {
    const P = pocket.Pocket(Toy, @import("families.zig").nemotron_h);
    var p: P = .{ .gpa = t.allocator, .mode = .subject, .layers = 40 };
    defer p.deinit();
    for (0..40) |l| try t.expectEqual(l >= 8 and l <= 25, p.candidate(l, l, .subject));
    var m: Toy = .{};
    try t.expect(try locate.once(&p, &m));
    try t.expectEqual(@as(usize, 0), m.calls);
}
