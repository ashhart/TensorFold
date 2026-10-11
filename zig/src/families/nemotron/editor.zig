//! A fact written at its subject: rank-one changes to a few shared-expert down projections, other keys' outputs kept.
const std = @import("std");
const adapters = @import("learn_shape.zig");
const lesson_route = @import("lesson_route.zig");
const subspace = @import("subspace.zig");

const examples = @import("lesson_examples.zig");
const Example = examples.Example;
const pick = examples.pick;

/// The MoE layers a fact's change is spread over, first to last; its target is the residual after the last.
pub const layers = [_]usize{ 13, 15, 17 };
const target_layer = layers[layers.len - 1];

const hold: f32 = 1e2; // the kept keys' mean second moment against a fact's own key, when a change is solved
const history_weight: f32 = 1e3; // an earlier fact's key against one kept row: it must keep its output exactly
const target_steps = 30; // Adam steps on the nudge that makes the fact's answers come out
const reach: f32 = 0.1; // each nudge step's size against the residual's
const decay: f32 = 0.5; // the nudge's squared size against the residual's, added to its loss
const furthest: f32 = 4; // the nudge's size at most, against the residual's
const enough: f32 = 0.05; // a mean answer loss low enough that the nudge stops early

const Phase = enum { kept, facts, target, edits, done };

/// The lesson's acting lessons and flags for a step: every lesson kept so far and, once written, this one.
pub const Acting = lesson_route.Acting;

/// An editor over trainer T (Metal's: its key, residual and nudge).
pub fn Editor(comptime T: type) type {
    return struct {
        const Self = @This();
        gpa: std.mem.Allocator,
        kept: [layers.len]std.ArrayList(f32) = @splat(.empty), // each layer's kept keys [rows, W], this lesson's
        history: [layers.len]std.ArrayList(f32) = @splat(.empty), // each layer's keys of every fact written so far
        facts: std.ArrayList(Example) = .empty, // the fact's examples that have a subject row, those with answers first
        answered: usize = 0, // facts[0..answered] have answers: the nudge is found on those, the key read on all
        stay: []const []const Example = &.{}, // the examples whose outputs the change must keep
        keys: std.ArrayList(f32) = .empty, // [facts, layers, W]: each example's key at each layer, as last read
        base: std.ArrayList(f32) = .empty, // [facts, D]: the residual after the target layer at the subject row, unchanged
        now: std.ArrayList(f32) = .empty, // [facts, D]: the same as last read
        delta: std.ArrayList(f32) = .empty, // [4, D]: the nudge, its Adam moments, and a round's gradient
        phase: Phase = .done,
        at: usize = 0,
        round: usize = 0,
        loss: f32 = 0,
        first_loss: f32 = 0,
        last_loss: f32 = 0,
        size: f32 = 0, // the subject rows' mean residual length after the target layer

        pub fn init(gpa: std.mem.Allocator) Self {
            return .{ .gpa = gpa };
        }

        pub fn deinit(e: *Self) void {
            for (&e.kept) |*s| s.deinit(e.gpa);
            for (&e.history) |*s| s.deinit(e.gpa);
            inline for (.{ "facts", "keys", "base", "now", "delta" }) |f| @field(e, f).deinit(e.gpa);
        }

        /// Examples naming a subject (`answered` set the nudge, `asked` add keys) and ones to keep; false if too few.
        pub fn begin(e: *Self, answered: []const []const Example, asked: []const []const Example, stay: []const []const Example) !bool {
            e.facts.clearRetainingCapacity();
            for (answered) |xs| for (xs) |ex| if (ex.subject > 0 and ex.subject < ex.start) try e.facts.append(e.gpa, ex);
            e.answered = e.facts.items.len;
            if (e.answered < 2) return false;
            for (asked) |xs| for (xs) |ex| if (ex.subject > 0 and ex.subject < ex.start) try e.facts.append(e.gpa, ex);
            for (&e.kept) |*s| s.clearRetainingCapacity();
            e.stay = stay;
            e.next(.kept);
            return true;
        }

        /// One unit (keys, a nudge step or a layer's change) run with `acting`; true once every layer has its change.
        pub fn step(e: *Self, t: *T, acting: Acting) !bool {
            return switch (e.phase) {
                .kept => e.keptOnce(t, acting),
                .facts => e.factsOnce(t, acting),
                .target => e.targetOnce(t, acting),
                .edits => e.editsOnce(t, acting),
                .done => true,
            };
        }

        /// The fact's keys joined to every layer's history, once its lesson stays: later facts keep its outputs.
        pub fn settle(e: *Self) !void {
            const w = keyWidth(e);
            if (w == 0) return;
            for (0..e.facts.items.len) |i| for (0..layers.len) |j| {
                try e.history[j].appendSlice(e.gpa, e.keys.items[(i * layers.len + j) * w ..][0..w]);
            };
        }

        /// The kept edit rounded into each layer's codes, all before any write (`host`: codes, addRounded, commit).
        pub fn fold(e: *Self, t: *T, host: anytype) !void {
            const c = t.b.m.config;
            const w = c.shared_width;
            const d = c.hidden;
            const h = try e.gpa.alloc(f32, w * w);
            defer e.gpa.free(h);
            const delta = try e.gpa.alloc(f32, d * w);
            defer e.gpa.free(delta);
            var fresh: [layers.len][]u32 = @splat(&.{});
            defer for (fresh) |f| e.gpa.free(f);
            const first = t.sites.first();
            const ranks = t.sites.cap * adapters.block;
            for (layers, 0..) |l, j| {
                const site = &t.sites.list[l];
                const a = site.a.slice(f32, ranks * site.in)[first * site.in ..][0..site.in];
                const b = site.b.slice(f32, ranks * site.out)[first * site.out ..][0..site.out];
                for (0..d) |o| for (delta[o * w ..][0..w], a) |*x, ai| {
                    x.* = adapters.scale * b[o] * ai;
                };
                const add = secondMoment(e.kept[j].items, e.history[j].items, w, h);
                const codes = try host.codes(l);
                fresh[j] = try e.gpa.dupe(u32, codes.words);
                const changed = try host.addRounded(fresh[j], codes.scales, codes.biases, d, w, delta, h, add);
                std.log.info("slide: layer {d}'s down projection takes the change in {d} of its {d} codes", .{ l, changed, d * w });
            }
            for (layers, fresh) |l, f| {
                @memcpy((try host.codes(l)).words, f);
                try host.commit(l);
            }
        }

        fn keyWidth(e: *const Self) usize {
            return if (e.facts.items.len == 0) 0 else e.keys.items.len / (e.facts.items.len * layers.len);
        }

        fn turn(ex: Example, acting: Acting) lesson_route.Turn {
            return .{ .decide = ex.decide, .bits = acting, .whole = true };
        }

        fn keptOnce(e: *Self, t: *T, acting: Acting) !bool {
            var n: usize = 0;
            for (e.stay) |xs| n += xs.len;
            if (e.at == n) {
                e.next(.facts);
                return false;
            }
            const ex = pick(e.stay, e.at);
            e.at += 1;
            _ = try t.step(ex.ids, ex.start, .keys, turn(ex, acting));
            for (layers, &e.kept) |l, *rows| for (0..ex.ids.len - 1) |r| try widen(e.gpa, rows, t.key(l, r));
            return false;
        }

        fn factsOnce(e: *Self, t: *T, acting: Acting) !bool {
            const c = t.b.m.config;
            if (e.at == 0) {
                e.keys.clearRetainingCapacity();
                e.base.clearRetainingCapacity();
                try e.keys.appendNTimes(e.gpa, 0, e.facts.items.len * layers.len * c.shared_width);
                try e.base.appendNTimes(e.gpa, 0, e.facts.items.len * c.hidden);
            }
            if (e.at == e.facts.items.len) {
                e.now.clearRetainingCapacity();
                try e.now.appendSlice(e.gpa, e.base.items);
                e.size = 0;
                for (0..e.answered) |i| e.size += @sqrt(subspace.dot(row(e.base.items, i, c.hidden), row(e.base.items, i, c.hidden)));
                e.size /= @floatFromInt(e.answered);
                e.delta.clearRetainingCapacity();
                try e.delta.appendNTimes(e.gpa, 0, 4 * c.hidden);
                e.next(.target);
                return false;
            }
            const i = e.at;
            e.at += 1;
            try e.read(t, i, null, acting, true);
            return false;
        }

        /// Example i's key at every layer (or only layer `only`) and its residual after the target layer, as now.
        fn read(e: *Self, t: *T, i: usize, only: ?usize, acting: Acting, unchanged: bool) !void {
            const c = t.b.m.config;
            const ex = e.facts.items[i];
            _ = try t.step(ex.ids, ex.start, .keys, turn(ex, acting));
            for (layers, 0..) |l, j| {
                if (only) |o| if (o != j) continue;
                for (e.keys.items[(i * layers.len + j) * c.shared_width ..][0..c.shared_width], t.key(l, ex.subject)) |*o, b| o.* = f32of(b);
            }
            const into = if (unchanged) e.base.items else e.now.items;
            for (row(into, i, c.hidden), t.residual(target_layer + 1, ex.subject)) |*o, b| o.* = f32of(b);
        }

        fn targetOnce(e: *Self, t: *T, acting: Acting) !bool {
            const d = t.b.m.config.hidden;
            const delta = e.delta.items[0..d];
            const grad = e.delta.items[3 * d ..][0..d];
            if (e.at == 0) @memset(grad, 0);
            if (e.at < e.answered) {
                const ex = e.facts.items[e.at];
                e.at += 1;
                t.nudge = .{ .layers = .{ @as(u64, 1) << target_layer, 0, 0 }, .rows = .{ ex.subject, 0, 0 } };
                @memcpy(t.delta.slice(f32, t.b.m.config.layers * d)[target_layer * d ..][0..d], delta);
                const got = try t.step(ex.ids, ex.start, .target, turn(ex, acting));
                e.loss += got.loss;
                for (grad, t.push.slice(f32, t.b.m.config.layers * d)[target_layer * d ..][0..d]) |*g, p| g.* += p;
                return false;
            }
            const n: f32 = @floatFromInt(e.answered);
            const mean = e.loss / n;
            if (e.round == 0) e.first_loss = mean;
            e.loss = 0;
            e.at = 0;
            e.round += 1;
            if (mean < enough or e.round > target_steps) {
                e.last_loss = mean;
                std.log.info("slide: the subject's target found in {d} steps, the answers' loss {d:.3} to {d:.3}, its nudge {d:.2} of the residual", .{ e.round - 1, e.first_loss, mean, @sqrt(subspace.dot(delta, delta)) / e.size });
                e.next(.edits);
                return false;
            }
            const m = e.delta.items[d..][0..d];
            const v = e.delta.items[2 * d ..][0..d];
            const lr = reach * e.size / @sqrt(@as(f32, @floatFromInt(d)));
            const t1: f32 = @floatFromInt(e.round);
            for (delta, m, v, grad) |*x, *mm, *vv, g0| {
                const g = g0 / n + 2 * decay * x.* / (e.size * e.size);
                mm.* = 0.9 * mm.* + 0.1 * g;
                vv.* = 0.999 * vv.* + 0.001 * g * g;
                const mh = mm.* / (1 - std.math.pow(f32, 0.9, t1));
                const vh = vv.* / (1 - std.math.pow(f32, 0.999, t1));
                x.* -= lr * mh / (@sqrt(vh) + 1e-8);
            }
            const len = @sqrt(subspace.dot(delta, delta));
            if (len > furthest * e.size) for (delta) |*x| {
                x.* *= furthest * e.size / len;
            };
            return false;
        }

        fn editsOnce(e: *Self, t: *T, acting: Acting) !bool {
            if (e.round == layers.len) {
                e.next(.done);
                return true;
            }
            if (e.at < e.facts.items.len) {
                const i = e.at;
                e.at += 1;
                try e.read(t, i, e.round, acting, false);
                return false;
            }
            try e.edit(t, e.round, acting);
            e.at = 0;
            e.round += 1;
            return false;
        }

        /// Layer j's rank-one change: the mean key sent to its share of the nudge left, kept and earlier keys held.
        fn edit(e: *Self, t: *T, j: usize, acting: Acting) !void {
            const c = t.b.m.config;
            const w = c.shared_width;
            const d = c.hidden;
            const n: f32 = @floatFromInt(e.facts.items.len);
            const na: f32 = @floatFromInt(e.answered);
            const work = try e.gpa.alloc(f32, w + w + d);
            defer e.gpa.free(work);
            const key = work[0..w];
            const u = work[w..][0..w];
            const left = work[2 * w ..][0..d];
            @memset(work, 0);
            for (0..e.facts.items.len) |i| subspace.axpy(key, 1 / n, e.keys.items[(i * layers.len + j) * w ..][0..w]);
            for (0..e.answered) |i| for (left, row(e.base.items, i, d), row(e.now.items, i, d), e.delta.items[0..d]) |*o, b, now, dl| {
                o.* += (b + dl - now) / na;
            };
            for (left) |*o| o.* /= @floatFromInt(layers.len - j);
            try solve(e.gpa, e.kept[j].items, e.history[j].items, w, key, u);
            const s = subspace.dot(key, u);
            const site = &t.sites.list[layers[j]];
            const first = t.sites.first();
            const ranks = t.sites.cap * adapters.block;
            const a = site.a.slice(f32, ranks * site.in)[first * site.in ..][0..site.in];
            const b = site.b.slice(f32, ranks * site.out)[first * site.out ..][0..site.out];
            for (a, u) |*o, x| o.* = x / (1 + s);
            for (b, left) |*o, x| o.* = x / adapters.scale;
            t.sites.attach(&t.b.m.weights, true);
            var loss: f32 = 0;
            for (e.facts.items[0..e.answered]) |ex| loss += (try t.step(ex.ids, ex.start, .loss, turn(ex, acting))).loss;
            std.log.info("slide: change at layer {d}: the key's reach over the kept rows {d:.1}, its change {d:.2} of the residual; the answers' loss with it {d:.3} (the nudge's {d:.3})", .{ layers[j], s, @sqrt(subspace.dot(left, left)) / e.size, loss / na, e.last_loss });
        }

        fn next(e: *Self, phase: Phase) void {
            e.phase = phase;
            e.at = 0;
            e.round = 0;
            e.loss = 0;
        }
    };
}

/// A bf16's value.
fn f32of(b: u16) f32 {
    return @bitCast(@as(u32, b) << 16);
}

fn row(xs: []f32, i: usize, n: usize) []f32 {
    return xs[i * n ..][0..n];
}

/// A bf16 row appended as f32.
fn widen(gpa: std.mem.Allocator, rows: *std.ArrayList(f32), from: []const u16) !void {
    try rows.ensureUnusedCapacity(gpa, from.len);
    for (from) |b| rows.appendAssumeCapacity(f32of(b));
}

/// h = hold X^T X / rows + history_weight H^T H; returns the fold's diagonal add: a hundredth of the kept part's mean.
fn secondMoment(x: []const f32, hist: []const f32, w: usize, h: []f32) f32 {
    const rows = x.len / w;
    const scale = hold / @as(f32, @floatFromInt(@max(rows, 1)));
    @memset(h[0 .. w * w], 0);
    const Share = struct {
        fn run(xs: []const f32, width: usize, coef: f32, out: []f32, from: usize, to: usize) void {
            const n = xs.len / width;
            for (0..n) |r| {
                const v = xs[r * width ..][0..width];
                for (from..to) |i| {
                    const f = coef * v[i];
                    if (f != 0) subspace.axpy(out[i * width ..][0..width], f, v);
                }
            }
        }
    };
    const threads = @min(std.Thread.getCpuCount() catch 1, 32);
    var spawned: [32]?std.Thread = @splat(null);
    for (0..threads) |k| spawned[k] = std.Thread.spawn(.{}, Share.run, .{ x, w, scale, h, w * k / threads, w * (k + 1) / threads }) catch null;
    for (0..threads) |k| if (spawned[k]) |th| th.join() else Share.run(x, w, scale, h, w * k / threads, w * (k + 1) / threads);
    var mean: f32 = 0;
    for (0..w) |i| mean += h[i * w + i];
    Share.run(hist, w, history_weight, h, 0, w);
    if (mean == 0) for (0..w) |i| { // nothing kept: the facts' keys set it after all
        mean += h[i * w + i];
    };
    return 1e-2 * mean / @as(f32, @floatFromInt(w)) + 1e-12;
}

/// u = C^-1 k, C = hold X^T X / rows + history_weight H^T H + mu I (X kept keys, H facts'), by preconditioned CG.
fn solve(gpa: std.mem.Allocator, x: []const f32, h: []const f32, w: usize, k: []const f32, u: []f32) !void {
    const rows = x.len / w;
    const hist = h.len / w;
    const work = try gpa.alloc(f32, 5 * w);
    defer gpa.free(work);
    const diag = work[0..w];
    const r = work[w..][0..w];
    const z = work[2 * w ..][0..w];
    const p = work[3 * w ..][0..w];
    const ap = work[4 * w ..][0..w];
    const scale = hold / @as(f32, @floatFromInt(@max(rows, 1)));
    @memset(diag, 0);
    for (0..rows) |i| for (diag, x[i * w ..][0..w]) |*dg, v| {
        dg.* += scale * v * v;
    };
    for (0..hist) |i| for (diag, h[i * w ..][0..w]) |*dg, v| {
        dg.* += history_weight * v * v;
    };
    var mean: f32 = 0;
    for (diag) |dg| mean += dg;
    const mu = 1e-2 * mean / @as(f32, @floatFromInt(w)) + 1e-12;
    for (diag) |*dg| dg.* += mu;
    @memset(u, 0);
    @memcpy(r, k);
    for (z, r, diag) |*zi, ri, dg| zi.* = ri / dg;
    @memcpy(p, z);
    var rz = subspace.dot(r, z);
    const stop = 1e-4 * @sqrt(subspace.dot(k, k));
    for (0..400) |_| {
        try apply(gpa, x, scale, h, w, mu, p, ap);
        const pap = subspace.dot(p, ap);
        if (!(pap > 0)) break;
        const alpha = rz / pap;
        if (!std.math.isFinite(alpha)) break;
        subspace.axpy(u, alpha, p);
        subspace.axpy(r, -alpha, ap);
        if (@sqrt(subspace.dot(r, r)) < stop) break;
        for (z, r, diag) |*zi, ri, dg| zi.* = ri / dg;
        const rz2 = subspace.dot(r, z);
        for (p, z) |*pi, zi| pi.* = zi + rz2 / rz * pi.*;
        rz = rz2;
    }
    for (u) |*v| if (!std.math.isFinite(v.*)) {
        v.* = 0;
    };
}

/// ap = C p, with C as `solve` holds it, the rows' products spread over threads.
fn apply(gpa: std.mem.Allocator, x: []const f32, scale: f32, h: []const f32, w: usize, mu: f32, p: []const f32, ap: []f32) !void {
    const threads = @min(std.Thread.getCpuCount() catch 1, 16);
    const parts = try gpa.alloc(f32, threads * w);
    defer gpa.free(parts);
    @memset(parts, 0);
    const Share = struct {
        fn run(xs: []const f32, hs: []const f32, width: usize, sc: f32, pp: []const f32, out: []f32, from: usize, step: usize) void {
            const n = xs.len / width;
            var i = from;
            while (i < n + hs.len / width) : (i += step) {
                const v = if (i < n) xs[i * width ..][0..width] else hs[(i - n) * width ..][0..width];
                const cf = (if (i < n) sc else history_weight) * subspace.dot(v, pp);
                subspace.axpy(out, cf, v);
            }
        }
    };
    var spawned: [16]?std.Thread = @splat(null);
    for (0..threads) |i| spawned[i] = std.Thread.spawn(.{}, Share.run, .{ x, h, w, scale, p, parts[i * w ..][0..w], i, threads }) catch null;
    for (0..threads) |i| if (spawned[i]) |th| th.join() else Share.run(x, h, w, scale, p, parts[i * w ..][0..w], i, threads);
    for (ap, p) |*o, v| o.* = mu * v;
    for (0..threads) |i| for (ap, parts[i * w ..][0..w]) |*o, v| {
        o.* += v;
    };
}

test "the solve inverts the kept rows' held second moment, earlier facts' keys among them" {
    const gpa = std.testing.allocator;
    const w = 16;
    var x: [40 * w]f32 = undefined;
    var h: [3 * w]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(3);
    for (&x) |*v| v.* = prng.random().float(f32) - 0.5;
    for (&h) |*v| v.* = prng.random().float(f32) - 0.5;
    var k: [w]f32 = undefined;
    for (&k, 0..) |*v, i| v.* = @floatFromInt(i % 3);
    var u: [w]f32 = undefined;
    try solve(gpa, &x, &h, w, &k, &u);
    var diag_mean: f32 = 0;
    for (0..w) |i| {
        for (0..40) |r| diag_mean += hold / 40 * x[r * w + i] * x[r * w + i];
        for (0..3) |r| diag_mean += history_weight * h[r * w + i] * h[r * w + i];
    }
    const mu = 1e-2 * diag_mean / w + 1e-12;
    for (0..w) |i| {
        var s: f32 = mu * u[i];
        for (0..40) |r| s += hold / 40 * x[r * w + i] * subspace.dot(x[r * w ..][0..w], &u);
        for (0..3) |r| s += history_weight * h[r * w + i] * subspace.dot(h[r * w ..][0..w], &u);
        try std.testing.expectApproxEqAbs(k[i], s, 2e-2 * (1 + @abs(k[i])));
    }
}
