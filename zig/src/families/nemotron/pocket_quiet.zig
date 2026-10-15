//! Earlier facts' neurons that fire on the open fact's questions, fitted again to stay silent there.
const std = @import("std");
const pocket_fit = @import("pocket_fit.zig");

const Kind = pocket_fit.Kind;
const fit = pocket_fit.fit;
const dotBf16 = pocket_fit.dotBf16;
const fit_steps = pocket_fit.fit_steps;

pub const kept_least: f32 = 0.5; // an earlier neuron refitted must still reach this on every one of its own fact's rows
pub const refit_above: f32 = 0.5; // an earlier neuron is refitted only where it fires above this on a new fact's rows

/// Earlier neurons firing on this fact's rows refitted silent there, undo copies kept; same-subject ones stay.
pub fn quiet(p: anytype, t: anytype, host: anytype) !void {
    const d = t.config().hidden;
    const n = p.moe.items.len;
    for (p.learned.items[0..p.learned_before]) |e| {
        if (e.key == .subject and std.mem.indexOfScalar(u32, p.subjects.items, e.subject) != null) continue;
        const u = p.neuron(p.up.items, e.slot, e.neuron, d);
        var most: f32 = -std.math.inf(f32);
        for (p.labels.items, 0..) |l, i| if (l.kind == .fact or l.kind == .held) {
            most = @max(most, dotBf16(u, p.rows.items[(i * n + e.slot) * d ..][0..d]));
        };
        if (most <= refit_above) continue; // weak there: left exactly as it is, this fact's outputs fitted with it
        try p.refits.append(p.gpa, .{ .slot = e.slot, .neuron = e.neuron });
        try p.refit_up.appendSlice(p.gpa, u);
        var xs: std.ArrayList([]const u16) = .empty;
        defer xs.deinit(p.gpa);
        var kinds: std.ArrayList(Kind) = .empty;
        defer kinds.deinit(p.gpa);
        for (0..e.count) |r| {
            try xs.append(p.gpa, p.pos.items[(e.start + r) * d ..][0..d]);
            try kinds.append(p.gpa, .fact);
        }
        // a conversation before a question may hold its own fact's question and answer (a chain's): held as it reads
        const held = try p.gpa.alloc(bool, p.labels.items.len);
        defer p.gpa.free(held);
        var fixed = e.count;
        for (p.labels.items, held, 0..) |l, *on, i| {
            const x = p.rows.items[(i * n + e.slot) * d ..][0..d];
            on.* = l.row < p.pickRead(l.ex).?.ex.context and dotBf16(u, x) > 0;
            if (!on.*) continue;
            try xs.append(p.gpa, x);
            try kinds.append(p.gpa, .none); // held as it reads, judged by no threshold
            fixed += 1;
        }
        for (p.labels.items, held, 0..) |l, on, i| {
            if (on) continue;
            try xs.append(p.gpa, p.rows.items[(i * n + e.slot) * d ..][0..d]);
            try kinds.append(p.gpa, switch (l.kind) {
                .fact, .held => .steady, // this fact's questions are another fact's to it
                .none => .other,
                else => l.kind,
            });
        }
        const f = try fit(p.gpa, xs.items, kinds.items, d, u, true, fit_steps / 2, fixed);
        if (f.fact_least < kept_least or @max(f.steady_most, f.other_most) > 0) {
            // a refit that forgets its fact or spreads is undone; this fact's outputs are fitted with it firing
            @memcpy(u, p.refit_up.items[p.refit_up.items.len - d ..]);
            p.refits.shrinkRetainingCapacity(p.refits.items.len - 1);
            p.refit_up.shrinkRetainingCapacity(p.refit_up.items.len - d);
            std.log.info("slide pocket: layer {d}, neuron {d}, an earlier fact's, fires on this fact's questions (up to {d:.2}); fitted again its own would read at least {d:.2} and other rows at most {d:.2}, so it stays as it was", .{ p.moe.items[e.slot], e.neuron, most, f.fact_least, @max(f.steady_most, f.other_most) });
            continue;
        }
        try p.upload(t, host, e.slot);
        std.log.info("slide pocket: layer {d}, neuron {d}, an earlier fact's, fired on this fact's questions (up to {d:.2}); fitted again: there and elsewhere at most {d:.2}, its own questions' least {d:.2} ({d} conversation rows held as they read)", .{ p.moe.items[e.slot], e.neuron, most, @max(f.steady_most, f.other_most), f.fact_least, fixed - e.count });
    }
}

test "an earlier fact's neuron fitted quiet on a new fact's rows keeps the conversation rows it fires on as they read" {
    const gpa = std.testing.allocator;
    const d = 64;
    const T = struct {
        pub fn config(_: *const @This()) struct { hidden: usize } {
            return .{ .hidden = d };
        }
    };
    const pockets = @import("pocket.zig");
    const Example = @import("lesson_examples.zig").Example;
    const NoHost = struct {
        pub fn pocket(_: @This(), _: usize, _: []const f32, _: []const f32, _: usize) !void {}
    };
    var prng = std.Random.DefaultPrng.init(7);
    const r = prng.random();
    // rows: a shared offset, noise, and a lean per fact; the new fact's questions lean much as the earlier fact's do
    var common: [d]f32 = undefined;
    var old_dir: [d]f32 = undefined;
    var new_dir: [d]f32 = undefined;
    for (&common, &old_dir) |*c, *o| {
        c.* = 1 + r.floatNorm(f32) * 0.1;
        o.* = r.floatNorm(f32);
    }
    for (&new_dir, old_dir) |*x, o| x.* = 0.8 * o + 0.6 * r.floatNorm(f32);
    const Row = [d]u16;
    const rowOf = struct {
        fn f(rr: std.Random, cm: []const f32, lean: ?[]const f32) Row {
            var out: Row = undefined;
            for (&out, 0..) |*x, k| {
                var v = cm[k] + rr.floatNorm(f32) * 0.5;
                if (lean) |dir| v += 1.5 * dir[k];
                x.* = @truncate(@as(u32, @bitCast(v)) >> 16);
            }
            return out;
        }
    }.f;
    var p: pockets.Pocket(T) = .{ .gpa = gpa, .mode = .reply };
    defer p.deinit();
    try p.moe.append(gpa, 3);
    try p.up.appendNTimes(gpa, 0, pockets.neurons * d);
    try p.down.appendNTimes(gpa, 0, pockets.neurons * d);
    try p.used.append(gpa, 1);
    // the earlier fact's neuron, fitted on its stored rows against the rest
    var stored: [12]Row = undefined;
    var rest: [160]Row = undefined;
    var xs: std.ArrayList([]const u16) = .empty;
    defer xs.deinit(gpa);
    var kinds: std.ArrayList(Kind) = .empty;
    defer kinds.deinit(gpa);
    for (&stored) |*x| {
        x.* = rowOf(r, &common, &old_dir);
        try xs.append(gpa, x);
        try kinds.append(gpa, .fact);
        try p.pos.appendSlice(gpa, x);
    }
    for (&rest) |*x| {
        x.* = rowOf(r, &common, null);
        try xs.append(gpa, x);
        try kinds.append(gpa, .other);
    }
    const u = p.neuron(p.up.items, 0, 0, d);
    _ = try fit(gpa, xs.items, kinds.items, d, u, false, fit_steps, 0);
    try p.learned.append(gpa, .{ .slot = 0, .neuron = 0, .key = .reply, .start = 0, .count = stored.len, .subject = 0, .lesson = 0 });
    p.learned_before = 1;
    // the new fact's read: its questions cold, then the same after a chain holding the earlier fact's question
    var news: [12]Row = undefined;
    var chain: [12]Row = undefined; // the earlier fact's question in the chain, read before each new question
    const cold: Example = .{ .ids = &.{ 1, 2, 3 }, .start = 2, .decide = 1 };
    const after: Example = .{ .ids = &.{ 1, 2, 3, 4, 5, 6 }, .start = 5, .decide = 4, .context = 3 };
    const answered: [12]Example = @splat(cold);
    const asked: [12]Example = @splat(after);
    p.answered = &answered;
    p.prompts = &asked;
    for (&news, 0..) |*x, i| {
        x.* = rowOf(r, &common, &new_dir);
        try p.rows.appendSlice(gpa, x);
        try p.labels.append(gpa, .{ .kind = .fact, .key = .reply, .ex = @intCast(i), .row = 1 });
    }
    for (&chain, 0..) |*x, i| {
        x.* = rowOf(r, &common, &old_dir);
        try p.rows.appendSlice(gpa, x);
        try p.labels.append(gpa, .{ .kind = .other, .key = null, .ex = @intCast(news.len + i), .row = 1 });
    }
    for (rest[0..80], 0..) |*x, i| {
        try p.rows.appendSlice(gpa, x);
        try p.labels.append(gpa, .{ .kind = .other, .key = null, .ex = @intCast(i % news.len), .row = 2 });
    }
    var most: f32 = -1;
    for (&news) |*x| most = @max(most, dotBf16(u, x));
    try std.testing.expect(most > refit_above); // it fires on the new fact's questions, so a refit is due
    const before = try gpa.dupe(f32, u);
    defer gpa.free(before);
    var t: T = .{};
    try quiet(&p, &t, NoHost{});
    // refitted: silent on the new fact's questions, and its own question in the chain reads as it did
    try std.testing.expectEqual(@as(usize, 1), p.refits.items.len);
    for (&news) |*x| try std.testing.expect(dotBf16(u, x) <= 0);
    for (&chain) |*x| if (dotBf16(before, x) > 0) try std.testing.expectApproxEqAbs(dotBf16(before, x), dotBf16(u, x), 1e-3);
}
