//! A pocket neuron's input weights by Adam on squared hinges: at least 1 on its fact's rows, below zero elsewhere.
const std = @import("std");
const subspace = @import("subspace.zig");

pub const margin: f32 = 0.25; // relu^2's margin (families.zig): how far below zero a neuron sits on rows it stays silent on
pub const fit_steps = 500; // Adam steps fitting a neuron's input weights
pub const fit_rate: f32 = 0.01; // their step, the inputs scaled to unit length

/// A read row to the open fact: a key row of a learned, held-out or other prompt, any other row, or left free.
pub const Kind = enum(u8) { fact, held, steady, other, none };

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

/// Slot j's neuron fitted on its read rows, copied side by side first (a failure: zero, its slot last in line).
pub fn fitLayer(gpa: std.mem.Allocator, rows: []const u16, kinds: []const Kind, slots: usize, j: usize, d: usize, u: []f32, out: *Fit, below: f32) void {
    const own = gpa.alloc(u16, kinds.len * d) catch {
        out.* = .{ .fires = std.math.maxInt(usize) };
        return;
    };
    defer gpa.free(own);
    const xs = gpa.alloc([]const u16, kinds.len) catch {
        out.* = .{ .fires = std.math.maxInt(usize) };
        return;
    };
    defer gpa.free(xs);
    for (xs, 0..) |*x, i| {
        @memcpy(own[i * d ..][0..d], rows[(i * slots + j) * d ..][0..d]);
        x.* = own[i * d ..][0..d];
    }
    out.* = fit(gpa, xs, kinds, d, u, false, fit_steps, 0, below) catch blk: {
        @memset(u, 0);
        break :blk .{ .fires = std.math.maxInt(usize) };
    };
}

/// Fits `u` on `xs` (from `u` if `warm`): at least 1 on fact rows, at most -`below` elsewhere; first `fixed` rows kept.
pub fn fit(gpa: std.mem.Allocator, xs: []const []const u16, kinds: []const Kind, d: usize, u: []f32, warm: bool, steps: usize, fixed: usize, below: f32) !Fit {
    const n = kinds.len;
    var scale: f32 = 0;
    for (xs) |x| scale += @sqrt(squares(x));
    scale /= @floatFromInt(@max(n, 1));
    const inv = 1 / scale;
    var counts: [5]f32 = @splat(0);
    for (kinds) |k| counts[@backingInt(k)] += 1;
    const w = try gpa.alloc(f32, 8 * d);
    defer gpa.free(w);
    @memset(w, 0);
    const mean_fact = w[0..d];
    const mean_rest = w[d..][0..d];
    const m = w[2 * d ..][0..d];
    const v = w[3 * d ..][0..d];
    const g = w[4 * d ..][0..d];
    const parts = w[5 * d ..][0 .. 3 * d]; // each kind's rows on the wrong side, their hinges' sum: fact, steady, other
    if (warm) {
        for (u) |*o| o.* *= scale;
    } else {
        // a start: the fact's mean less the mean of the rows it stays silent on, sized so the fact's mean is 1
        for (kinds, xs) |k, x| switch (k) {
            .fact => axpyRow(mean_fact, inv / @max(counts[0], 1), x),
            .steady, .other => axpyRow(mean_rest, inv / @max(counts[2] + counts[3], 1), x),
            .held, .none => {},
        };
        for (u, mean_fact, mean_rest) |*o, a, b| o.* = a - b;
        const at_fact = subspace.dot(u, mean_fact);
        if (at_fact > 0) for (u) |*o| {
            o.* /= at_fact;
        };
    }
    const basis = try gpa.alloc(f32, 2 * fixed * d);
    defer gpa.free(basis);
    const own = basis[fixed * d ..];
    for (0..fixed) |i| for (own[i * d ..][0..d], xs[i]) |*o, b| {
        o.* = bf(b) * inv;
    };
    const rank = orthonormal(basis[0 .. fixed * d], own, d);
    for (1..steps + 1) |s| {
        // each kind weighed by its rows on the wrong side only: a few rows that still slip are pushed as hard as many
        @memset(parts, 0);
        var wrong: [3]f32 = @splat(0);
        for (kinds, xs) |k, x| {
            const z = dotRow(u, x) * inv;
            switch (k) {
                .fact => if (z < 1) {
                    wrong[0] += 1;
                    axpyRow(parts[0..d], -2 * (1 - z), x);
                },
                .steady, .other => if (z > -below) {
                    const at: usize = if (k == .steady) 1 else 2;
                    wrong[at] += 1;
                    axpyRow(parts[at * d ..][0..d], 2 * (z + below), x);
                },
                .held, .none => {},
            }
        }
        @memset(g, 0);
        for (0..3) |at| subspace.axpy(g, inv / @max(wrong[at], 1), parts[at * d ..][0..d]);
        const tt: f32 = @floatFromInt(s);
        for (u, m, v, g) |ui, *mi, *vi, *gi| {
            const gg = gi.* + 2e-4 * ui;
            mi.* = 0.9 * mi.* + 0.1 * gg;
            vi.* = 0.999 * vi.* + 0.001 * gg * gg;
            gi.* = fit_rate * (mi.* / (1 - std.math.pow(f32, 0.9, tt))) / (@sqrt(vi.* / (1 - std.math.pow(f32, 0.999, tt))) + 1e-8);
        }
        // the step, less its part along the fixed rows' span
        for (0..rank) |j| {
            const b = basis[j * d ..][0..d];
            subspace.axpy(g, -subspace.dot(g, b), b);
        }
        for (u, g) |*ui, st| ui.* -= st;
    }
    var f: Fit = .{ .fact_least = std.math.inf(f32), .steady_most = -std.math.inf(f32), .other_most = -std.math.inf(f32) };
    for (kinds, xs) |k, x| {
        const zi = dotRow(u, x) * inv;
        switch (k) {
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
            .none => {},
        }
    }
    f.fact_on /= @max(counts[0], 1);
    f.held_on /= @max(counts[1], 1);
    f.held_mean /= @max(counts[1], 1);
    for (u) |*o| o.* /= scale;
    return f;
}

const V = @Vector(8, f32);

/// Eight bf16 values widened to f32, exactly.
inline fn widen8(b: []const u16) V {
    const raw: @Vector(8, u32) = @as(@Vector(8, u16), b[0..8].*);
    return @bitCast(raw << @splat(16));
}

fn bf(b: u16) f32 {
    return @bitCast(@as(u32, b) << 16);
}

/// a . b for f32 `a` and bf16 `b`: four sums of eight lanes, so no add waits on the one before it.
fn dotRow(a: []const f32, b: []const u16) f32 {
    var acc: [4]V = @splat(@splat(0));
    var i: usize = 0;
    while (i + 32 <= a.len) : (i += 32) {
        inline for (&acc, 0..) |*x, k| x.* += @as(V, a[i + 8 * k ..][0..8].*) * widen8(b[i + 8 * k ..]);
    }
    while (i + 8 <= a.len) : (i += 8) acc[0] += @as(V, a[i..][0..8].*) * widen8(b[i..]);
    var s = @reduce(.Add, (acc[0] + acc[1]) + (acc[2] + acc[3]));
    while (i < a.len) : (i += 1) s += a[i] * bf(b[i]);
    return s;
}

/// y += c b for bf16 `b`.
fn axpyRow(y: []f32, c: f32, b: []const u16) void {
    const cv: V = @splat(c);
    var i: usize = 0;
    while (i + 8 <= y.len) : (i += 8) y[i..][0..8].* = @as(V, y[i..][0..8].*) + cv * widen8(b[i..]);
    while (i < y.len) : (i += 1) y[i] += c * bf(b[i]);
}

/// b . b for bf16 `b`.
fn squares(b: []const u16) f32 {
    var acc: V = @splat(0);
    var i: usize = 0;
    while (i + 8 <= b.len) : (i += 8) {
        const x = widen8(b[i..]);
        acc += x * x;
    }
    var s = @reduce(.Add, acc);
    while (i < b.len) : (i += 1) s += bf(b[i]) * bf(b[i]);
    return s;
}

/// An orthonormal basis of `rows` ([k, d]) into `basis` by Gram-Schmidt run twice, rows in the span left out; its rank.
fn orthonormal(basis: []f32, rows: []const f32, d: usize) usize {
    var rank: usize = 0;
    for (0..rows.len / d) |i| {
        const q = basis[rank * d ..][0..d];
        @memcpy(q, rows[i * d ..][0..d]);
        const before = @sqrt(subspace.dot(q, q));
        for (0..2) |_| for (0..rank) |j| {
            const b = basis[j * d ..][0..d];
            subspace.axpy(q, -subspace.dot(q, b), b);
        };
        const len = @sqrt(subspace.dot(q, q));
        if (len <= 1e-3 * before) continue;
        for (q) |*e| e.* /= len;
        rank += 1;
    }
    return rank;
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
    const first = try fit(gpa, xs, kinds, d, &u, false, fit_steps, 0, margin);
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
    const before = try gpa.dupe(f32, &u);
    defer gpa.free(before);
    const again = try fit(gpa, all, kinds2, d, &u, true, fit_steps / 2, n_fact, margin);
    try std.testing.expectEqual(@as(f32, 1), again.fact_on);
    try std.testing.expectEqual(@as(usize, 0), again.fires);
    try std.testing.expect(again.steady_most <= 0);
    // its own rows were held fixed: each reads as it did before the refit
    for (all[0..n_fact]) |x| try std.testing.expectApproxEqAbs(dotBf16(before, x), dotBf16(&u, x), 1e-3);
}
