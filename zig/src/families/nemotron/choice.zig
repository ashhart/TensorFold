//! A lesson's block chosen at every layer: the directions its fact's rows outweigh every steady row along, and gates.
const std = @import("std");
const subspace = @import("subspace.zig");
const dims = @import("slide_dims.zig");

const n = dims.candidates;
const block = dims.block;

/// What each layer has seen of the steady rows and the fact's answer rows, along its candidate directions.
pub const Choice = struct {
    gpa: std.mem.Allocator,
    steady: []f64, // [layers, n, n]: the steady rows' second moment
    facts: []f64, // [layers, n, n]: the fact's answer rows' second moment
    steady_rows: []std.ArrayList(f32), // per layer, each steady row's n values, for its share
    fact_rows: []std.ArrayList(f32),
    heads: std.ArrayList(usize) = .empty, // where each fact example's first answer row is in fact_rows (in rows)
    steady_heads: std.ArrayList(usize) = .empty, // the same for the steady examples under the gate
    coef: []f32, // [layers, block, n]: each layer's new directions in the candidates' terms
    tau: []f32, // [layers]: each layer's gate, or shut
    hits: []u32, // [layers]: the fact's answer rows that clear it
    opens: []u32, // [layers]: the fact examples whose first answer row clears it

    pub fn init(gpa: std.mem.Allocator, layers: usize) !Choice {
        const c: Choice = .{
            .gpa = gpa,
            .steady = try gpa.alloc(f64, layers * n * n),
            .facts = try gpa.alloc(f64, layers * n * n),
            .steady_rows = try gpa.alloc(std.ArrayList(f32), layers),
            .fact_rows = try gpa.alloc(std.ArrayList(f32), layers),
            .coef = try gpa.alloc(f32, layers * block * n),
            .tau = try gpa.alloc(f32, layers),
            .hits = try gpa.alloc(u32, layers),
            .opens = try gpa.alloc(u32, layers),
        };
        for (c.steady_rows, c.fact_rows) |*x, *y| {
            x.* = .empty;
            y.* = .empty;
        }
        return c;
    }

    pub fn deinit(c: *Choice) void {
        for (c.steady_rows, c.fact_rows) |*x, *y| {
            x.deinit(c.gpa);
            y.deinit(c.gpa);
        }
        inline for (.{ "steady", "facts", "steady_rows", "fact_rows", "coef", "tau", "hits", "opens" }) |f| c.gpa.free(@field(c, f));
        c.heads.deinit(c.gpa);
        c.steady_heads.deinit(c.gpa);
    }

    pub fn reset(c: *Choice) void {
        @memset(c.steady, 0);
        @memset(c.facts, 0);
        for (c.steady_rows, c.fact_rows) |*x, *y| {
            x.clearRetainingCapacity();
            y.clearRetainingCapacity();
        }
        c.heads.clearRetainingCapacity();
        c.steady_heads.clearRetainingCapacity();
    }

    /// The next fact example's rows start here: its first answer row is the one that decides how the answer opens.
    pub fn head(c: *Choice) !void {
        try c.heads.append(c.gpa, c.fact_rows[0].items.len / n);
    }

    /// Layer l's unit rows [rows, n], steady or the fact's; row `first` weighs as all after; `gate`: kept below it.
    pub fn add(c: *Choice, l: usize, rows: []const f32, fact: bool, first: usize, gate: bool) !void {
        const m = (if (fact) c.facts else c.steady)[l * n * n ..][0 .. n * n];
        const count = rows.len / n;
        var r: usize = 0;
        while (r < count) : (r += 1) {
            const p = rows[r * n ..][0..n];
            const w: f64 = if (r == first) @floatFromInt(@max(count - first - 1, 1)) else 1;
            for (0..n) |i| for (0..n) |j| {
                m[i * n + j] += w * p[i] * p[j];
            };
        }
        if (!fact and gate and l == 0) try c.steady_heads.append(c.gpa, c.steady_rows[0].items.len / n + first);
        if (fact or gate) try (if (fact) &c.fact_rows[l] else &c.steady_rows[l]).appendSlice(c.gpa, rows);
    }

    /// Every layer's directions and gate (margin: how far above every steady row's cosine a row must be to open it).
    pub fn choose(c: *Choice, margin: f32) !void {
        var threads: [16]?std.Thread = @splat(null);
        var failed: [16]?anyerror = @splat(null);
        const k = @min(threads.len, c.tau.len);
        for (threads[0..k], 0..) |*t, i| t.* = std.Thread.spawn(.{}, share, .{ c, margin, i, k, &failed[i] }) catch null;
        for (threads[0..k], 0..) |t, i| if (t) |th| th.join() else share(c, margin, i, k, &failed[i]);
        for (failed[0..k]) |f| if (f) |e| return e;
    }

    fn share(c: *Choice, margin: f32, from: usize, step: usize, failed: *?anyerror) void {
        var l = from;
        while (l < c.tau.len) : (l += step) c.layer(l, margin) catch |e| {
            failed.* = e;
        };
    }

    /// Layer l: the gate's direction, the block's others where the fact outweighs the steady rows, the gate above them.
    fn layer(c: *Choice, l: usize, margin: f32) !void {
        const coef = c.coef[l * block * n ..][0 .. block * n];
        var more: [block * n]f32 = undefined;
        try subspace.outweigh(c.gpa, c.facts[l * n * n ..][0 .. n * n], c.steady[l * n * n ..][0 .. n * n], n, block, &more);
        try c.discriminant(l, coef[0..n]);
        @memcpy(coef[n..], more[0 .. (block - 1) * n]);
        subspace.orthonormal(coef, block, n);
        const steady = c.steady_rows[l].items;
        const facts = c.fact_rows[l].items;
        var top: f32 = -1;
        for (0..steady.len / n) |r| top = @max(top, subspace.dot(coef[0..n], steady[r * n ..][0..n]));
        c.tau[l] = top + margin;
        c.hits[l] = 0;
        for (0..facts.len / n) |r| c.hits[l] += @intFromBool(subspace.dot(coef[0..n], facts[r * n ..][0..n]) > c.tau[l]);
        c.opens[l] = 0;
        for (c.heads.items) |r| c.opens[l] += @intFromBool(subspace.dot(coef[0..n], facts[r * n ..][0..n]) > c.tau[l]);
        if (c.hits[l] == 0) {
            c.tau[l] = dims.shut;
            c.opens[l] = 0;
        }
    }

    /// Layer l's directions as a plain weight change, times K = O (F + lambda S + mu)^-1: open rows keep their output.
    pub fn refit(c: *Choice, l: usize, lambda: f64) !void {
        if (c.hits[l] == 0) return;
        const facts = c.fact_rows[l].items;
        const first = c.coef[l * block * n ..][0..n];
        const open = try c.gpa.alloc(f64, 4 * n * n);
        defer c.gpa.free(open);
        const all = open[n * n ..][0 .. n * n];
        const d = open[2 * n * n ..][0 .. n * n];
        const kt = open[3 * n * n ..][0 .. n * n];
        @memset(open[0 .. 2 * n * n], 0);
        for (0..facts.len / n) |r| {
            const p = facts[r * n ..][0..n];
            const on = subspace.dot(first, p) > c.tau[l];
            for (0..n) |i| for (0..n) |j| {
                const v = @as(f64, p[i]) * p[j];
                all[i * n + j] += v;
                if (on) open[i * n + j] += v;
            };
        }
        var trace: f64 = 0;
        for (d, all, c.steady[l * n * n ..][0 .. n * n], 0..) |*x, a, s, k| {
            x.* = a + lambda * s;
            if (k % (n + 1) == 0) trace += x.*;
        }
        for (0..n) |i| d[i * n + i] += 1e-3 * trace / @as(f64, @floatFromInt(n)) + 1e-12;
        var col: [n]f64 = undefined;
        var a: [n * n]f64 = undefined;
        for (0..n) |j| {
            for (&col, 0..) |*x, i| x.* = open[i * n + j];
            @memcpy(&a, d);
            try subspace.solve(&a, n, &col);
            for (col, 0..) |x, i| kt[i * n + j] = x;
        }
        const coef = c.coef[l * block * n ..][0 .. block * n];
        var out: [block * n]f32 = undefined;
        for (0..block) |q| for (0..n) |j| {
            var s: f64 = 0;
            for (0..n) |i| s += coef[q * n + i] * kt[j * n + i];
            out[q * n + j] = @floatCast(s);
        };
        @memcpy(coef, &out);
    }

    /// Fisher's direction between the fact's answer rows and the steady rows: (scatter + lambda)^-1 (mean difference).
    fn discriminant(c: *Choice, l: usize, out: []f32) !void {
        const sets = [2][]const f32{ c.fact_rows[l].items, c.steady_rows[l].items };
        const firsts = [2][]const usize{ c.heads.items, c.steady_heads.items };
        var mean: [2][n]f64 = @splat(@splat(0));
        var total: [2]f64 = .{ 0, 0 };
        for (sets, firsts, 0..) |rows, heads, k| for (0..rows.len / n) |r| {
            const w = weight(heads, r);
            total[k] += w;
            for (0..n) |i| mean[k][i] += w * rows[r * n + i];
        };
        for (0..2) |k| for (&mean[k]) |*v| {
            v.* /= @max(total[k], 1);
        };
        const scatter = try c.gpa.alloc(f64, n * n);
        defer c.gpa.free(scatter);
        @memset(scatter, 0);
        for (sets, firsts, 0..) |rows, heads, k| for (0..rows.len / n) |r| {
            const w = weight(heads, r) / @max(total[k], 1);
            for (0..n) |i| for (0..n) |j| {
                scatter[i * n + j] += w * (rows[r * n + i] - mean[k][i]) * (rows[r * n + j] - mean[k][j]);
            };
        };
        var trace: f64 = 0;
        for (0..n) |i| trace += scatter[i * n + i];
        for (0..n) |i| scatter[i * n + i] += 1e-2 * trace / @as(f64, @floatFromInt(n)) + 1e-12;
        var d: [n]f64 = undefined;
        for (&d, mean[0], mean[1]) |*x, a, b| x.* = a - b;
        try subspace.solve(scatter, n, &d);
        var size: f64 = 0;
        for (d) |x| size += x * x;
        for (out, d) |*o, x| o.* = @floatCast(x / @sqrt(@max(size, 1e-30)));
    }
};

/// A row's weight: a first answer row counts as much as a whole answer, any other row once.
fn weight(heads: []const usize, r: usize) f64 {
    for (heads) |h| if (h == r) return 8;
    return 1;
}
