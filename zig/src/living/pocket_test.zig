//! The tests of pocket.zig, beside it so the file stays under 600 lines.
const pocket_mod = @import("pocket.zig");
const std = @import("std");
const examples = @import("lesson_examples.zig");
const pocket_fit = @import("pocket_fit.zig");
const Example = examples.Example;
const Kind = pocket_fit.Kind;
const fit = pocket_fit.fit;
const dotBf16 = pocket_fit.dotBf16;
const fit_steps = pocket_fit.fit_steps;
const Pocket = pocket_mod.Pocket;
const kept_least = pocket_mod.kept_least;
const kindFor = pocket_mod.kindFor;
const refit_above = pocket_mod.refit_above;

test "a fact keys as many neurons on its subject as the share of its questions that name it, at least half, else none" {
    const P = Pocket(struct {}, @import("families.zig").nemotron_h);
    var p: P = .{ .gpa = std.testing.allocator, .mode = .auto };
    defer p.deinit();
    // ids: a question's tokens, the subject's last token at row 2, the answer from row 4
    const ids = [_]u32{ 1, 2, 7, 3, 4, 5 };
    const named: Example = .{ .ids = &ids, .start = 4, .decide = 3, .subject = 2 };
    const plain: Example = .{ .ids = &ids, .start = 4, .decide = 3 };
    try std.testing.expectEqual([3]usize{ 5, 5, 0 }, p.pickSplit(&.{ &.{ named, plain }, &.{} }, &.{}));
    try std.testing.expectEqual([3]usize{ 0, 10, 0 }, p.pickSplit(&.{ &.{ named, named }, &.{named} }, &.{}));
    try std.testing.expectEqual([3]usize{ 3, 7, 0 }, p.pickSplit(&.{&.{ named, named, plain }}, &.{}));
    try std.testing.expectEqual([3]usize{ 10, 0, 0 }, p.pickSplit(&.{&.{ named, plain, plain }}, &.{}));
    try std.testing.expectEqual([3]usize{ 10, 0, 0 }, p.pickSplit(&.{&.{}}, &.{}));
    // answers whose value comes after row 4: none there unless asked, then four, the rest shared as before
    const valued: Example = .{ .ids = &ids, .start = 4, .decide = 3, .subject = 2, .value = 4 };
    try std.testing.expectEqual([3]usize{ 0, 10, 0 }, p.pickSplit(&.{&.{ valued, valued }}, &.{ valued, valued }));
    p.valuing = true;
    try std.testing.expectEqual([3]usize{ 0, 6, 4 }, p.pickSplit(&.{&.{ valued, valued }}, &.{ valued, valued }));
    p.mode = .subject;
    try std.testing.expectEqual([3]usize{ 0, 10, 0 }, p.pickSplit(&.{&.{ named, plain, plain }}, &.{}));
    p.mode = .reply;
    try std.testing.expectEqual([3]usize{ 10, 0, 0 }, p.pickSplit(&.{&.{ named, named }}, &.{}));
}

test "a read row is a fact's row only to its own key's neurons; another prompt's mention of the subject is left free" {
    try std.testing.expectEqual(Kind.fact, kindFor(.{ .kind = .fact, .key = .subject, .ex = 0, .row = 2 }, .subject));
    try std.testing.expectEqual(Kind.other, kindFor(.{ .kind = .fact, .key = .subject, .ex = 0, .row = 2 }, .reply));
    try std.testing.expectEqual(Kind.held, kindFor(.{ .kind = .held, .key = .reply, .ex = 3, .row = 7 }, .reply));
    try std.testing.expectEqual(Kind.steady, kindFor(.{ .kind = .steady, .key = .reply, .ex = 1, .row = 9 }, .subject));
    try std.testing.expectEqual(Kind.none, kindFor(.{ .kind = .none, .key = .subject, .ex = 1, .row = 4 }, .subject));
    try std.testing.expectEqual(Kind.other, kindFor(.{ .kind = .none, .key = .subject, .ex = 1, .row = 4 }, .reply));
    try std.testing.expectEqual(Kind.other, kindFor(.{ .kind = .other, .key = null, .ex = 1, .row = 5 }, .subject));
}

test "an earlier fact's neuron fitted quiet on a new fact's rows reads as before on its own fact's rows as read now" {
    const gpa = std.testing.allocator;
    const F = @import("families.zig").nemotron_h;
    const T = struct {};
    const P = Pocket(T, F);
    const d = 64;
    const NoHost = struct {
        pub fn pocket(_: @This(), _: usize, _: []const f32, _: []const f32, _: usize) !void {}
    };
    var prng = std.Random.DefaultPrng.init(11);
    const r = prng.random();
    // hidden states: a shared offset, noise, and a direction each fact's questions lean along.
    var common: [d]f32 = undefined;
    var old_dir: [d]f32 = undefined;
    var new_dir: [d]f32 = undefined;
    var push: [d]f32 = undefined; // what the pocket's lower neurons add to the earlier fact's rows since it was read
    for (&common, &old_dir, &push) |*c, *o, *u| {
        c.* = 1 + r.floatNorm(f32) * 0.1;
        o.* = r.floatNorm(f32);
        u.* = r.floatNorm(f32) * 1.2;
    }
    for (&new_dir, old_dir) |*n, o| n.* = 0.8 * o + 0.6 * r.floatNorm(f32);
    const n_old = 12;
    const n_new = 12;
    const n_rest = 160;
    const Row = [d]u16;
    const rowOf = struct {
        fn f(rr: std.Random, cm: []const f32, lean: ?[]const f32, extra: ?[]const f32) Row {
            var out: Row = undefined;
            for (&out, 0..) |*x, k| {
                var v = cm[k] + rr.floatNorm(f32) * 0.5;
                if (lean) |dir| v += 1.5 * dir[k];
                if (extra) |e| v += e[k];
                x.* = @truncate(@as(u32, @bitCast(v)) >> 16);
            }
            return out;
        }
    }.f;
    var stored: [n_old]Row = undefined; // the earlier fact's rows as its lesson read them
    var live: [n_old]Row = undefined; // the same questions as read now, in the replay
    for (&stored, &live) |*s, *l| {
        s.* = rowOf(r, &common, &old_dir, null);
        l.* = rowOf(r, &common, &old_dir, &push);
    }
    var p: P = .{ .gpa = gpa, .mode = .reply };
    defer p.deinit();
    p.hidden = d;
    p.layers = 8;
    try p.moe.append(gpa, 3);
    try p.up.appendNTimes(gpa, 0, P.neurons * d);
    try p.down.appendNTimes(gpa, 0, P.neurons * d);
    try p.used.append(gpa, 1);
    // the earlier fact's neuron, fitted on its stored rows against the rest
    var xs: std.ArrayList([]const u16) = .empty;
    defer xs.deinit(gpa);
    var kinds: std.ArrayList(Kind) = .empty;
    defer kinds.deinit(gpa);
    var rest: [n_rest]Row = undefined;
    for (&rest) |*x| x.* = rowOf(r, &common, null, null);
    for (&stored) |*x| {
        try xs.append(gpa, x);
        try kinds.append(gpa, .fact);
        try p.pos.appendSlice(gpa, x);
    }
    for (&rest) |*x| {
        try xs.append(gpa, x);
        try kinds.append(gpa, .other);
    }
    const u = p.neuron(p.up.items, 0, 0, d);
    _ = try fit(gpa, xs.items, kinds.items, d, u, false, fit_steps, 0, F.neuron.margin);
    try p.learned.append(gpa, .{ .slot = 0, .neuron = 0, .key = .reply, .start = 0, .count = n_old, .subject = 0, .lesson = 0 });
    p.learned_before = 1;
    // the new fact's read: its questions (its own examples), then other prompts: the earlier fact's in the replay first
    var news: [n_new]Row = undefined;
    for (&news) |*x| x.* = rowOf(r, &common, &new_dir, null);
    const ex: Example = .{ .ids = &.{ 1, 2, 3 }, .start = 2, .decide = 1 };
    const answered: [n_new]Example = @splat(ex);
    p.answered = &answered;
    for (&news, 0..) |*x, i| {
        try p.rows.appendSlice(gpa, x);
        try p.labels.append(gpa, .{ .kind = .fact, .key = .reply, .ex = @intCast(i), .row = 1 });
    }
    for (&live, 0..) |*x, i| {
        try p.rows.appendSlice(gpa, x);
        try p.labels.append(gpa, .{ .kind = .steady, .key = .reply, .ex = @intCast(n_new + i), .row = 1 });
    }
    for (rest[0 .. n_rest / 2], 0..) |*x, i| {
        try p.rows.appendSlice(gpa, x);
        try p.labels.append(gpa, .{ .kind = .steady, .key = .reply, .ex = @intCast(n_new + n_old + i), .row = 1 });
    }
    var most: f32 = -1;
    for (&news) |*x| most = @max(most, dotBf16(u, x));
    try std.testing.expect(most > refit_above); // it fires on the new fact's questions: a refit is due
    const before = try gpa.dupe(f32, u);
    defer gpa.free(before);
    var t: T = .{};
    try p.quiet(&t, NoHost{});
    try std.testing.expectEqual(@as(usize, 1), p.refits.items.len);
    for (&news) |*x| try std.testing.expect(dotBf16(u, x) <= 0);
    // its own questions as read now and as stored read exactly as they did: it still answers its own fact
    for (&live) |*x| try std.testing.expectApproxEqAbs(dotBf16(before, x), dotBf16(u, x), 1e-3);
    for (&stored) |*x| try std.testing.expectApproxEqAbs(dotBf16(before, x), dotBf16(u, x), 1e-3);
    // as rows of other prompts (the old refit), its own questions read now were pushed silent with the stored ones.
    xs.clearRetainingCapacity();
    kinds.clearRetainingCapacity();
    for (&stored) |*x| {
        try xs.append(gpa, x);
        try kinds.append(gpa, .fact);
    }
    for (&news) |*x| {
        try xs.append(gpa, x);
        try kinds.append(gpa, .steady);
    }
    for (&live) |*x| {
        try xs.append(gpa, x);
        try kinds.append(gpa, .steady);
    }
    const old = try gpa.dupe(f32, before);
    defer gpa.free(old);
    const of = try fit(gpa, xs.items, kinds.items, d, old, true, fit_steps / 2, n_old, F.neuron.margin);
    // (its stored rows, the only ones it checked, still read as before, so nothing told it it had forgotten)
    try std.testing.expect(of.fact_least >= kept_least);
    var silenced: usize = 0;
    for (&live) |*x| silenced += @intFromBool(dotBf16(old, x) < 0.5 and dotBf16(before, x) >= 0.5);
    try std.testing.expect(silenced > n_old / 2);
}

test "a lesson taken out puts the pocket back bit for bit: its neurons, an earlier one it refitted, the rounding" {
    const gpa = std.testing.allocator;
    const F = @import("families.zig").nemotron_h;
    const T = struct {};
    const P = Pocket(T, F);
    const d = 64;
    const NoHost = struct {
        pub fn pocket(_: @This(), _: usize, _: []const f32, _: []const f32, _: usize) !void {}
    };
    var prng = std.Random.DefaultPrng.init(3);
    const r = prng.random();
    var p: P = .{ .gpa = gpa, .mode = .reply };
    defer p.deinit();
    p.hidden = d;
    p.layers = 8;
    try p.moe.appendSlice(gpa, &.{ 3, 5 });
    try p.up.appendNTimes(gpa, 0, 2 * P.neurons * d);
    try p.down.appendNTimes(gpa, 0, 2 * P.neurons * d);
    try p.used.appendSlice(gpa, &.{ 1, 0 });
    // an earlier fact's neuron at the first layer
    for (p.neuron(p.up.items, 0, 0, d), p.neuron(p.down.items, 0, 0, d)) |*u, *v| {
        u.* = r.floatNorm(f32);
        v.* = r.floatNorm(f32) * 0.01;
    }
    try p.learned.append(gpa, .{ .slot = 0, .neuron = 0, .key = .reply, .start = 0, .count = 0, .subject = 0, .lesson = 0 });
    var t: T = .{};
    try p.mark();
    const before = p.hash();
    // the lesson: a neuron of its own beside the earlier one (sharing its output's groups), the earlier one refitted.
    p.used.items[0] = 2;
    p.used.items[1] = 1;
    for ([_][2]usize{ .{ 0, 1 }, .{ 1, 0 } }) |at| {
        for (p.neuron(p.up.items, at[0], at[1], d), p.neuron(p.down.items, at[0], at[1], d)) |*u, *v| {
            u.* = r.floatNorm(f32);
            v.* = r.floatNorm(f32);
        }
        try p.taken.append(gpa, .{ .slot = at[0], .neuron = at[1], .key = .reply });
    }
    try p.refits.append(gpa, .{ .slot = 0, .neuron = 0 });
    try p.refit_up.appendSlice(gpa, p.neuron(p.up.items, 0, 0, d));
    p.neuron(p.up.items, 0, 0, d)[0] += 1;
    try p.learned.append(gpa, .{ .slot = 0, .neuron = 1, .key = .reply, .start = 0, .count = 0, .subject = 0, .lesson = 1 });
    try p.asWritten(&t, NoHost{});
    try std.testing.expect(p.hash() != before);
    try p.undo(&t, NoHost{});
    try std.testing.expectEqual(before, p.hash());
    try std.testing.expectEqualSlices(usize, &.{ 1, 0 }, p.used.items);
    try std.testing.expectEqual(@as(usize, 1), p.learned.items.len);
    // a save since the mark: no lesson before it is put back, the old undo runs (its own neurons out, nothing else)
    p.snap.valid = false;
    p.used.items[1] = 1;
    try p.taken.append(gpa, .{ .slot = 1, .neuron = 0, .key = .reply });
    @memset(p.neuron(p.up.items, 1, 0, d), 1);
    try p.undo(&t, NoHost{});
    try std.testing.expectEqual(before, p.hash());
}
