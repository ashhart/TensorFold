//! A lesson's block per layer: directions its fact's rows outweigh the steady rows along, the first rank its router's.
const std = @import("std");
const subspace = @import("subspace.zig");
const adapters = @import("learn_shape.zig");

const n = adapters.candidates;
const block = adapters.block;

/// What each layer has seen of the steady rows and the fact's answer rows, along its candidate directions.
pub const Choice = struct {
    gpa: std.mem.Allocator,
    steady: []f64, // [layers, n, n]: the steady rows' second moment
    facts: []f64, // [layers, n, n]: the fact's answer rows' second moment
    coef: []f32, // [layers, block, n]: each layer's new directions in the candidates' terms

    pub fn init(gpa: std.mem.Allocator, layers: usize) !Choice {
        return .{
            .gpa = gpa,
            .steady = try gpa.alloc(f64, layers * n * n),
            .facts = try gpa.alloc(f64, layers * n * n),
            .coef = try gpa.alloc(f32, layers * block * n),
        };
    }

    pub fn deinit(c: *Choice) void {
        inline for (.{ "steady", "facts", "coef" }) |f| c.gpa.free(@field(c, f));
    }

    pub fn reset(c: *Choice) void {
        @memset(c.steady, 0);
        @memset(c.facts, 0);
    }

    /// Layer l's unit rows [rows, n], steady or the fact's; row `first` (how an answer opens) weighs as all after it.
    pub fn add(c: *Choice, l: usize, rows: []const f32, fact: bool, first: usize) void {
        const m = (if (fact) c.facts else c.steady)[l * n * n ..][0 .. n * n];
        const count = rows.len / n;
        for (0..count) |r| {
            const p = rows[r * n ..][0..n];
            const w: f64 = if (r == first) @floatFromInt(@max(count - first - 1, 1)) else 1;
            for (0..n) |i| for (0..n) |j| {
                m[i * n + j] += w * p[i] * p[j];
            };
        }
    }

    /// Every layer's directions.
    pub fn choose(c: *Choice) !void {
        var threads: [16]?std.Thread = @splat(null);
        var failed: [16]?anyerror = @splat(null);
        const layers = c.coef.len / (block * n);
        const k = @min(threads.len, layers);
        for (threads[0..k], 0..) |*t, i| t.* = std.Thread.spawn(.{}, share, .{ c, i, k, &failed[i] }) catch null;
        for (threads[0..k], 0..) |t, i| if (t) |th| th.join() else share(c, i, k, &failed[i]);
        for (failed[0..k]) |f| if (f) |e| return e;
    }

    fn share(c: *Choice, from: usize, step: usize, failed: *?anyerror) void {
        var l = from;
        while (l < c.coef.len / (block * n)) : (l += step) {
            const coef = c.coef[l * block * n ..][0 .. block * n];
            subspace.outweigh(c.gpa, c.facts[l * n * n ..][0 .. n * n], c.steady[l * n * n ..][0 .. n * n], n, block, coef) catch |e| {
                failed.* = e;
            };
        }
    }
};
