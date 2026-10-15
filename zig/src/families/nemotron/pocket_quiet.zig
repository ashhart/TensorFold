//! Earlier facts' neurons that fire on the open fact's questions, fitted again to stay silent there.
const std = @import("std");
const pocket_fit = @import("pocket_fit.zig");

const Kind = pocket_fit.Kind;
const fit = pocket_fit.fit;
const dotBf16 = pocket_fit.dotBf16;
const fit_steps = pocket_fit.fit_steps;

pub const kept_least: f32 = 0.5; // an earlier neuron refitted must still reach this on every one of its own fact's rows
pub const refit_above: f32 = 0.5; // a refit is due where an earlier neuron fires above this on a new fact's rows, and holds rows above it

/// Earlier neurons firing on this fact's rows refitted silent there and nowhere else; same-subject ones stay.
pub fn quiet(p: anytype, t: anytype, host: anytype) !void {
    const d = t.config().hidden;
    const n = p.moe.items.len;
    const own = p.answered.len + p.prompts.len + p.held.len; // read examples below this are this fact's
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
        // rows it fires on above refit_above now (its own fact's questions read today) stay fixed; weaker ones go quiet
        const live = try p.gpa.alloc(bool, p.labels.items.len);
        defer p.gpa.free(live);
        var fixed = e.count;
        for (p.labels.items, live, 0..) |l, *on, i| {
            const x = p.rows.items[(i * n + e.slot) * d ..][0..d];
            on.* = l.ex >= own and dotBf16(u, x) > refit_above;
            if (!on.*) continue;
            try xs.append(p.gpa, x);
            try kinds.append(p.gpa, .none); // held as it reads, judged by no threshold
            fixed += 1;
        }
        for (p.labels.items, live, 0..) |l, on, i| {
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
        std.log.info("slide pocket: layer {d}, neuron {d}, an earlier fact's, fired on this fact's questions (up to {d:.2}); fitted again: there and elsewhere at most {d:.2}, its own questions' least {d:.2} ({d} rows held as they read, {d} of them read now)", .{ p.moe.items[e.slot], e.neuron, most, @max(f.steady_most, f.other_most), f.fact_least, fixed, fixed - e.count });
    }
}

test "an earlier fact's neuron fitted quiet on a new fact's rows and its weak firings reads as before on its own rows as read now" {
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
    var prng = std.Random.DefaultPrng.init(13);
    const r = prng.random();
    // rows: a shared offset, noise, and a lean per fact; the new fact's questions lean much as the earlier fact's do
    var common: [d]f32 = undefined;
    var old_dir: [d]f32 = undefined;
    var new_dir: [d]f32 = undefined;
    var push: [d]f32 = undefined; // what lower pocket neurons add to the earlier fact's rows since they were stored
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
    var live: [n_old]Row = undefined; // the same questions as read now, among the new lesson's other prompts
    for (&stored, &live) |*s, *l| {
        s.* = rowOf(r, &common, &old_dir, null);
        l.* = rowOf(r, &common, &old_dir, &push);
    }
    var p: pockets.Pocket(T) = .{ .gpa = gpa, .mode = .reply };
    defer p.deinit();
    try p.moe.append(gpa, 3);
    try p.up.appendNTimes(gpa, 0, pockets.neurons * d);
    try p.down.appendNTimes(gpa, 0, pockets.neurons * d);
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
    _ = try fit(gpa, xs.items, kinds.items, d, u, false, fit_steps, 0);
    try p.learned.append(gpa, .{ .slot = 0, .neuron = 0, .key = .reply, .start = 0, .count = n_old, .subject = 0, .lesson = 0 });
    p.learned_before = 1;
    // the new fact's read: its own questions, then other prompts, the earlier fact's questions among them
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
    // other prompts leaning the earlier fact's way, far less than its own, so it fires weakly on some
    const n_lean = 20;
    var leaning: [n_lean]Row = undefined;
    for (&leaning, 0..) |*x, j| {
        var w: [d]f32 = undefined;
        for (&w, old_dir) |*v, o| v.* = (0.2 + 0.02 * @as(f32, @floatFromInt(j))) * o;
        x.* = rowOf(r, &common, null, &w);
        try p.rows.appendSlice(gpa, x);
        try p.labels.append(gpa, .{ .kind = .steady, .key = .reply, .ex = @intCast(n_new + n_old + n_rest / 2 + j), .row = 1 });
    }
    var most: f32 = -1;
    for (&news) |*x| most = @max(most, dotBf16(u, x));
    try std.testing.expect(most > refit_above); // it fires on the new fact's questions, so a refit is due
    for (&live) |*x| try std.testing.expect(dotBf16(u, x) > refit_above); // and on its own as read today
    var weak: usize = 0;
    for (&leaning) |*x| {
        const v = dotBf16(u, x);
        weak += @intFromBool(v > 0 and v <= refit_above);
    }
    try std.testing.expect(weak >= 3);
    const before = try gpa.dupe(f32, u);
    defer gpa.free(before);
    var t: T = .{};
    try quiet(&p, &t, NoHost{});
    try std.testing.expectEqual(@as(usize, 1), p.refits.items.len);
    for (&news) |*x| try std.testing.expect(dotBf16(u, x) <= 0);
    // its own questions, as stored and as read now, read exactly as they did: it still answers its own fact
    var moved: usize = 0;
    for (&live) |*x| moved += @intFromBool(@abs(dotBf16(before, x) - dotBf16(u, x)) > 1e-3);
    try std.testing.expectEqual(@as(usize, 0), moved);
    for (&stored) |*x| try std.testing.expectApproxEqAbs(dotBf16(before, x), dotBf16(u, x), 1e-3);
    // other prompts: where it fired above refit_above it reads as before; where it fired weakly it is quiet now
    for (&leaning) |*x| {
        const was = dotBf16(before, x);
        if (was > refit_above) try std.testing.expectApproxEqAbs(was, dotBf16(u, x), 1e-3) else try std.testing.expect(dotBf16(u, x) <= 0);
    }
}
