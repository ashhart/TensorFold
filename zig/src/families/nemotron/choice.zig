//! A lesson's block chosen at every layer: the directions its fact's rows outweigh every steady row along, and gates.
const std = @import("std");
const subspace = @import("subspace.zig");
const adapters = @import("adapters.zig");

const n = adapters.candidates;
const block = adapters.block;

/// What each layer has seen of the steady rows and the fact's answer rows, along its candidate directions.
pub const Choice = struct {
    gpa: std.mem.Allocator,
    steady: []f64, // [layers, n, n]: the steady rows' second moment
    facts: []f64, // [layers, n, n]: the fact's answer rows' second moment
    steady_rows: []std.ArrayList(f32), // per layer, each steady row's n values, for its share
    fact_rows: []std.ArrayList(f32),
    coef: []f32, // [layers, block, n]: each layer's new directions in the candidates' terms
    tau: []f32, // [layers]: each layer's gate, or shut
    hits: []u32, // [layers]: the fact's answer rows that clear it

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
        inline for (.{ "steady", "facts", "steady_rows", "fact_rows", "coef", "tau", "hits" }) |f| c.gpa.free(@field(c, f));
    }

    pub fn reset(c: *Choice) void {
        @memset(c.steady, 0);
        @memset(c.facts, 0);
        for (c.steady_rows, c.fact_rows) |*x, *y| {
            x.clearRetainingCapacity();
            y.clearRetainingCapacity();
        }
    }

    /// Layer l's rows ([rows, n], each a unit input along the candidates), steady ones or the fact's answer rows.
    pub fn add(c: *Choice, l: usize, rows: []const f32, fact: bool) !void {
        const m = (if (fact) c.facts else c.steady)[l * n * n ..][0 .. n * n];
        var r: usize = 0;
        while (r * n < rows.len) : (r += 1) {
            const p = rows[r * n ..][0..n];
            for (0..n) |i| for (0..n) |j| {
                m[i * n + j] += @as(f64, p[i]) * p[j];
            };
        }
        try (if (fact) &c.fact_rows[l] else &c.steady_rows[l]).appendSlice(c.gpa, rows);
    }

    /// Every layer's directions and gate (margin: how far above the largest steady share a row must be to open it).
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

    /// Layer l: the block's directions, then its gate above every steady row's share and the fact rows it lets through.
    fn layer(c: *Choice, l: usize, margin: f32) !void {
        const coef = c.coef[l * block * n ..][0 .. block * n];
        try subspace.outweigh(c.gpa, c.facts[l * n * n ..][0 .. n * n], c.steady[l * n * n ..][0 .. n * n], n, block, coef);
        var top: f32 = 0;
        for (0..c.steady_rows[l].items.len / n) |r| top = @max(top, part(coef, c.steady_rows[l].items[r * n ..][0..n]));
        c.tau[l] = margin * top;
        c.hits[l] = 0;
        for (0..c.fact_rows[l].items.len / n) |r| c.hits[l] += @intFromBool(part(coef, c.fact_rows[l].items[r * n ..][0..n]) > c.tau[l]);
        if (c.hits[l] == 0) c.tau[l] = adapters.shut;
    }
};

/// A unit row's share in a block: its squared length along the block's orthonormal directions.
fn part(coef: []const f32, p: []const f32) f32 {
    var e: f32 = 0;
    for (0..block) |q| {
        const d = subspace.dot(coef[q * n ..][0..n], p);
        e += d * d;
    }
    return e;
}
