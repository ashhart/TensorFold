//! Earlier facts' neurons that fire on the open fact's questions, fitted again to stay silent there.
const std = @import("std");
const pocket_fit = @import("pocket_fit.zig");

const Kind = pocket_fit.Kind;
const fit = pocket_fit.fit;
const dotBf16 = pocket_fit.dotBf16;
const fit_steps = pocket_fit.fit_steps;

pub const kept_least: f32 = 0.5; // an earlier neuron refitted must still reach this on every one of its own fact's rows
pub const refit_above: f32 = 0.5; // an earlier neuron is refitted only where it fires above this on a new fact's rows

const quiet = @import("pocket_quiet.zig").quiet;
test "an earlier fact's neuron fitted quiet on a new fact's rows reads as before on its own fact's rows as read now" {
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
    var p: pockets.Pocket(T, @import("families.zig").nemotron_h) = .{ .gpa = gpa, .mode = .reply, .hidden = d };
    defer p.deinit();
    try p.moe.append(gpa, 3);
    try p.up.appendNTimes(gpa, 0, @TypeOf(p).neurons * d);
    try p.down.appendNTimes(gpa, 0, @TypeOf(p).neurons * d);
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
    _ = try fit(gpa, xs.items, kinds.items, d, u, false, fit_steps, 0, 0.25);
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
    var most: f32 = -1;
    for (&news) |*x| most = @max(most, dotBf16(u, x));
    try std.testing.expect(most > refit_above); // it fires on the new fact's questions, so a refit is due
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
}
