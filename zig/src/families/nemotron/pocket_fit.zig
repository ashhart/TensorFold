//! A pocket neuron's input weights fitted on read rows: its input product at least 1 on its fact's rows and below zero
//! on every row it stays silent on, by Adam on squared hinges weighed toward the rows still on the wrong side.
const std = @import("std");
const subspace = @import("subspace.zig");

pub const margin: f32 = 0.25; // how far below zero a fact's neuron sits on rows it stays silent on
pub const fit_steps = 500; // Adam steps fitting a neuron's input weights
pub const fit_rate: f32 = 0.01; // their step, the inputs scaled to unit length

/// What a read row is to the open fact: its reply's first row on a question it learns from, on a held-out one, on
/// another prompt, or any other row of any of them.
pub const Kind = enum(u8) { fact, held, steady, other };

/// One MoE layer's fit: how its neuron's input product sits on each kind of row.
pub const Fit = struct {
    fact_least: f32 = 0, // the lowest over the fact's own questions
    fact_on: f32 = 0, // the share of them at 0.5 or more
    held_mean: f32 = 0,
    held_on: f32 = 0,
    steady_most: f32 = 0, // the highest over other prompts' first reply rows
    other_most: f32 = 0, // the highest over every other row read
    fires: usize = 0, // rows it must stay silent on where it is above zero
};

/// Slot j's neuron fitted on the read rows (a thread a slot; a failure leaves it zero and the slot last in line).
pub fn fitLayer(gpa: std.mem.Allocator, rows: []const u16, kinds: []const Kind, slots: usize, j: usize, d: usize, u: []f32, out: *Fit) void {
    const xs = gpa.alloc([]const u16, kinds.len) catch {
        out.* = .{ .fires = std.math.maxInt(usize) };
        return;
    };
    defer gpa.free(xs);
    for (xs, 0..) |*x, i| x.* = rows[(i * slots + j) * d ..][0..d];
    out.* = fit(gpa, xs, kinds, d, u, false, fit_steps) catch blk: {
        @memset(u, 0);
        break :blk .{ .fires = std.math.maxInt(usize) };
    };
}

/// A neuron fitted on rows `xs` (bf16 [D] each, of kinds `kinds`): its input product at least 1 on the fact's, at most
/// -margin on every row it stays silent on (squared hinges, each kind weighed alike), by `steps` Adam steps on inputs
/// scaled to unit length; from `u` as it is when `warm`, else from the fact's mean less the rest's.
pub fn fit(gpa: std.mem.Allocator, xs: []const []const u16, kinds: []const Kind, d: usize, u: []f32, warm: bool, steps: usize) !Fit {
    const n = kinds.len;
    const x = try gpa.alloc(f32, n * d);
    defer gpa.free(x);
    var scale: f32 = 0;
    for (xs, 0..) |src, i| {
        const dst = x[i * d ..][0..d];
        for (dst, src) |*o, b| o.* = @bitCast(@as(u32, b) << 16);
        scale += @sqrt(subspace.dot(dst, dst));
    }
    scale /= @floatFromInt(@max(n, 1));
    for (x) |*v| v.* /= scale;
    var counts: [4]f32 = @splat(0);
    for (kinds) |k| counts[@backingInt(k)] += 1;
    const w = try gpa.alloc(f32, 4 * d);
    defer gpa.free(w);
    const mean_fact = w[0..d];
    const mean_rest = w[d..][0..d];
    const m = w[2 * d ..][0..d];
    const v = w[3 * d ..][0..d];
    @memset(w, 0);
    if (warm) {
        for (u) |*o| o.* *= scale;
    } else {
        // a start: the fact's mean less the mean of the rows it stays silent on, sized so the fact's mean is 1
        for (kinds, 0..) |k, i| switch (k) {
            .fact => subspace.axpy(mean_fact, 1 / @max(counts[0], 1), x[i * d ..][0..d]),
            .steady, .other => subspace.axpy(mean_rest, 1 / @max(counts[2] + counts[3], 1), x[i * d ..][0..d]),
            .held => {},
        };
        for (u, mean_fact, mean_rest) |*o, a, b| o.* = a - b;
        const at_fact = subspace.dot(u, mean_fact);
        if (at_fact > 0) for (u) |*o| {
            o.* /= at_fact;
        };
    }
    const g = try gpa.alloc(f32, d);
    defer gpa.free(g);
    const z = try gpa.alloc(f32, n);
    defer gpa.free(z);
    for (1..steps + 1) |s| {
        @memset(g, 0);
        // each kind weighed by its rows on the wrong side only: a few rows that still slip are pushed as hard as many
        var wrong: [4]f32 = @splat(0);
        for (0..n) |i| {
            z[i] = subspace.dot(u, x[i * d ..][0..d]);
            const off = switch (kinds[i]) {
                .fact => z[i] < 1,
                .steady, .other => z[i] > -margin,
                .held => false,
            };
            wrong[@backingInt(kinds[i])] += @floatFromInt(@intFromBool(off));
        }
        for (0..n) |i| {
            const zi = z[i];
            const k = @backingInt(kinds[i]);
            const coef: f32 = switch (kinds[i]) {
                .fact => if (zi < 1) -2 * (1 - zi) / @max(wrong[k], 1) else 0,
                .steady, .other => if (zi > -margin) 2 * (zi + margin) / @max(wrong[k], 1) else 0,
                .held => 0,
            };
            if (coef != 0) subspace.axpy(g, coef, x[i * d ..][0..d]);
        }
        const tt: f32 = @floatFromInt(s);
        for (u, m, v, g) |*ui, *mi, *vi, gi| {
            const gg = gi + 2e-4 * ui.*;
            mi.* = 0.9 * mi.* + 0.1 * gg;
            vi.* = 0.999 * vi.* + 0.001 * gg * gg;
            ui.* -= fit_rate * (mi.* / (1 - std.math.pow(f32, 0.9, tt))) / (@sqrt(vi.* / (1 - std.math.pow(f32, 0.999, tt))) + 1e-8);
        }
    }
    var f: Fit = .{ .fact_least = std.math.inf(f32), .steady_most = -std.math.inf(f32), .other_most = -std.math.inf(f32) };
    for (0..n) |i| {
        const zi = subspace.dot(u, x[i * d ..][0..d]);
        switch (kinds[i]) {
            .fact => {
                f.fact_least = @min(f.fact_least, zi);
                f.fact_on += @floatFromInt(@intFromBool(zi >= 0.5));
            },
            .held => {
                f.held_mean += if (zi > 0) zi * zi else 0;
                f.held_on += @floatFromInt(@intFromBool(zi >= 0.5));
            },
            .steady => {
                f.steady_most = @max(f.steady_most, zi);
                f.fires += @intFromBool(zi > 0);
            },
            .other => {
                f.other_most = @max(f.other_most, zi);
                f.fires += @intFromBool(zi > 0);
            },
        }
    }
    f.fact_on /= @max(counts[0], 1);
    f.held_on /= @max(counts[1], 1);
    f.held_mean /= @max(counts[1], 1);
    for (u) |*o| o.* /= scale;
    return f;
}

pub fn dotBf16(a: []const f32, b: []const u16) f32 {
    var s: f32 = 0;
    for (a, b) |x, y| s += x * @as(f32, @bitCast(@as(u32, y) << 16));
    return s;
}

pub fn normBf16(b: []const u16) f32 {
    var s: f32 = 0;
    for (b) |y| {
        const f: f32 = @bitCast(@as(u32, y) << 16);
        s += f * f;
    }
    return @sqrt(s);
}

test "a neuron fires on its fact's rows and stays at zero on the rest; fitted again, it goes quiet on a new fact's" {
    const gpa = std.testing.allocator;
    const d = 64;
    var prng = std.Random.DefaultPrng.init(9);
    const r = prng.random();
    // a shared offset every row has (as hidden states do), and a direction each fact's rows lean along
    var common: [d]f32 = undefined;
    var fact_dir: [d]f32 = undefined;
    var new_dir: [d]f32 = undefined;
    for (&common, &fact_dir, &new_dir) |*c, *f, *g| {
        c.* = 1 + r.floatNorm(f32) * 0.1;
        f.* = r.floatNorm(f32);
        g.* = r.floatNorm(f32);
    }
    const n_fact = 12;
    const n_rest = 200;
    const n_new = 12;
    const rows = try gpa.alloc(u16, (n_fact + n_rest + n_new) * d);
    defer gpa.free(rows);
    for (0..n_fact + n_rest + n_new) |i| {
        const lean: ?*const [d]f32 = if (i < n_fact) &fact_dir else if (i >= n_fact + n_rest) &new_dir else null;
        for (0..d) |k| {
            var v = common[k] + r.floatNorm(f32) * 0.5;
            if (lean) |dir| v += 1.5 * dir[k];
            rows[i * d + k] = @truncate(@as(u32, @bitCast(v)) >> 16);
        }
    }
    const xs = try gpa.alloc([]const u16, n_fact + n_rest);
    defer gpa.free(xs);
    const kinds = try gpa.alloc(Kind, n_fact + n_rest);
    defer gpa.free(kinds);
    for (xs, kinds, 0..) |*x, *k, i| {
        x.* = rows[i * d ..][0..d];
        k.* = if (i < n_fact) .fact else .other;
    }
    var u: [d]f32 = undefined;
    const first = try fit(gpa, xs, kinds, d, &u, false, fit_steps);
    try std.testing.expectEqual(@as(f32, 1), first.fact_on);
    try std.testing.expectEqual(@as(usize, 0), first.fires);
    // the new fact's rows, which nothing told it about, are where it is fitted again to stay quiet
    const all = try gpa.alloc([]const u16, n_fact + n_rest + n_new);
    defer gpa.free(all);
    const kinds2 = try gpa.alloc(Kind, n_fact + n_rest + n_new);
    defer gpa.free(kinds2);
    for (all, kinds2, 0..) |*x, *k, i| {
        x.* = rows[i * d ..][0..d];
        k.* = if (i < n_fact) .fact else if (i < n_fact + n_rest) .other else .steady;
    }
    const again = try fit(gpa, all, kinds2, d, &u, true, fit_steps / 2);
    try std.testing.expectEqual(@as(f32, 1), again.fact_on);
    try std.testing.expectEqual(@as(usize, 0), again.fires);
    try std.testing.expect(again.steady_most <= 0);
}
