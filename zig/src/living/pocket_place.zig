//! Fit input weights only at admitted layers, then fit the additive outputs with the existing acceptance path.
const std = @import("std");
const pocket = @import("pocket.zig");
const fitting = @import("pocket_fit.zig");
const Kind = fitting.Kind;
const Fit = fitting.Fit;
const fitLayer = fitting.fitLayer;
const kindFor = pocket.kindFor;
const Key = pocket.Key;
const clock = @import("pocket_clock.zig");
const pocket_dump = @import("pocket_dump.zig");
const quieting = @import("pocket_quiet.zig");
const joint = @import("pocket_joint.zig");
fn better(f: []const Fit, a: usize, b: usize) bool {
    if (f[a].fires != f[b].fires) return f[a].fires < f[b].fires;
    if (f[a].fact_on != f[b].fact_on) return f[a].fact_on > f[b].fact_on;
    return f[a].fact_least > f[b].fact_least;
}
fn occupied(p: anytype, slot: usize) bool {
    for (p.taken.items) |x| if (x.slot == slot) return true;
    return false;
}
/// Every key's candidate layers fitted in threads, the best taken; then keeps found, earlier neurons refitted.
pub fn all(p: anytype, t: anytype, host: anytype) !bool {
    const d = p.hidden;
    const n = p.slots();
    const m = p.labels.items.len;
    if (@import("sys").c.getenv("TF_SLIDE_POCKET_DUMP")) |dir| pocket_dump.write(p, std.mem.span(dir), p.learned_before, d) catch |e| std.log.warn("slide pocket: the read rows were not written: {t}", .{e});
    const kinds = try p.gpa.alloc(Kind, 3 * m);
    defer p.gpa.free(kinds);
    for (p.labels.items, 0..) |l, i| for (0..3) |k| {
        kinds[k * m + i] = kindFor(l, @fromBackingInt(@intCast(k)));
    };
    const us = try p.gpa.alloc(f32, 3 * n * d);
    defer p.gpa.free(us);
    const fits = try p.gpa.alloc(Fit, 3 * n);
    defer p.gpa.free(fits);
    @memset(fits, .{ .fires = std.math.maxInt(usize) });
    const t0 = clock.now(host.io);
    const threads = try p.gpa.alloc(?std.Thread, 3 * n);
    defer p.gpa.free(threads);
    @memset(threads, null);
    var fitted: usize = 0;
    const below = @TypeOf(p.*).of.neuron.margin;
    for (0..3) |k| for (p.moe.items, 0..) |l, j| if (p.split[k] > 0 and p.candidate(j, l, @fromBackingInt(@intCast(k)))) {
        threads[k * n + j] = std.Thread.spawn(.{}, fitLayer, .{ p.gpa, p.rows.items, kinds[k * m ..][0..m], n, j, d, us[(k * n + j) * d ..][0..d], &fits[k * n + j], below }) catch null;
        fitted += 1;
    };
    for (0..3) |k| for (p.moe.items, 0..) |l, j| if (p.split[k] > 0 and p.candidate(j, l, @fromBackingInt(@intCast(k)))) {
        if (threads[k * n + j]) |th| th.join() else fitLayer(p.gpa, p.rows.items, kinds[k * m ..][0..m], n, j, d, us[(k * n + j) * d ..][0..d], &fits[k * n + j], below);
    };
    const t1 = clock.now(host.io);
    for (0..3) |k| if (p.split[k] > 0) {
        const key: Key = @fromBackingInt(@intCast(k));
        var order: std.ArrayList(usize) = .empty;
        defer order.deinit(p.gpa);
        // a layer whose pocket is full is passed over
        for (p.moe.items, 0..) |l, j| if (p.candidate(j, l, key) and @import("pocket_vacancy.zig").next(p, j) != null) try order.append(p.gpa, j);
        if (order.items.len == 0) return error.PocketFull;
        std.mem.sort(usize, order.items, @as([]const Fit, fits[k * n ..][0..n]), better);
        var added: usize = 0;
        for (order.items) |j| {
            if (p.locate.enabled and occupied(p, j)) continue;
            try p.take(j, key, us[(k * n + j) * d ..][0..d], fits[k * n + j], kinds[k * m ..][0..m], d);
            added += 1;
            if (added == p.split[k]) break;
        }
    };
    p.logSources(d, m, kinds);
    try p.findKeeps(d);
    if (p.against.len == 0) try quieting.quiet(p, t, host);
    std.log.info("slide pocket: {d} candidate neurons fitted on {d} read rows in {d:.1} s; earlier neurons checked in {d:.1} s", .{ fitted, m, t1 - t0, clock.now(host.io) - t1 });
    if (p.against.len == 0) try joint.find(p, d);
    try p.v.begin(p.gpa, p.taken.items.len + p.joint.items.len, d, p.answered.len + p.joint_answers.items.len + p.keeps.items.len);
    joint.seed(p, d);
    p.next(.values);
    return false;
}
