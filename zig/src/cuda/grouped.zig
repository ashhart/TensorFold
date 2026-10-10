//! experts.route's plan on experts.cu's kernels: routed pairs grouped by expert into items, shared by MoE families.
const std = @import("std");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");

const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;

pub const small = 1024; // pairs the one-block plan takes; wider plans rank in blocks of 1024 pairs

/// experts.max_items: an item per used expert plus one per `tile` pairs past its first.
pub fn maxItems(pairs: usize, experts: usize, tile: usize) usize {
    return @min(pairs, experts) + pairs / tile;
}

/// Scratch the plan writes: members, items (expert, first, count), counts, and the wide path's rank and hist.
pub const Plan = struct { members: u64, items: u64, counts: u64, rank: u64, hist: u64 };

/// experts.cu's plan instances (cuobjdump -symbols of the experts fatbin).
pub const symbols = struct {
    pub const plan = "_ZN10tf_experts11plan_kernelEPKiiiiPiS2_S2_";
    pub const rank = "_ZN10tf_experts9plan_rankEPKiiiPiS2_";
    pub const offsets = "_ZN10tf_experts12plan_offsetsEiiiPiS0_S0_";
    pub const scatter = "_ZN10tf_experts12plan_scatterEPKiiiS1_S1_Pi";
};

pub const Router = struct {
    plan_small: Function,
    plan_rank: Function,
    plan_offsets: Function,
    plan_scatter: Function,

    /// The plan kernels from a loaded experts module (kernels.experts).
    pub fn resolve(mod: Module) !Router {
        return .{ .plan_small = try mod.function(symbols.plan), .plan_rank = try mod.function(symbols.rank), .plan_offsets = try mod.function(symbols.offsets), .plan_scatter = try mod.function(symbols.scatter) };
    }

    fn go(f: Function, s: Stream, grid: usize, block: u32, a: *launch_.Args) !void {
        try launch_.launch(f, .{ .grid = .{ .x = @intCast(grid), .y = 1, .z = 1 }, .block = .{ .x = block } }, s, a);
    }

    /// experts.route: `pairs` int32 expert ids at `picks` grouped into items of at most `tile` pairs (16 decode).
    pub fn route(r: Router, s: Stream, picks: u64, pairs: usize, count: usize, tile: usize, p: Plan) !void {
        var a: launch_.Args = .{};
        if (pairs <= small) {
            a.add(picks);
            for ([_]usize{ pairs, count, tile }) |v| a.add(@as(c_int, @intCast(v)));
            for ([_]u64{ p.members, p.items, p.counts }) |v| a.add(v);
            return go(r.plan_small, s, 1, 1024, &a);
        }
        const nblk = (pairs + 1023) / 1024;
        a.add(picks);
        a.add(@as(c_int, @intCast(pairs)));
        a.add(@as(c_int, @intCast(count)));
        a.add(p.rank);
        a.add(p.hist);
        try go(r.plan_rank, s, nblk, 1024, &a);
        var b: launch_.Args = .{};
        for ([_]usize{ nblk, count, tile }) |v| b.add(@as(c_int, @intCast(v)));
        for ([_]u64{ p.hist, p.items, p.counts }) |v| b.add(v);
        try go(r.plan_offsets, s, 1, 1024, &b);
        var c: launch_.Args = .{};
        c.add(picks);
        c.add(@as(c_int, @intCast(pairs)));
        c.add(@as(c_int, @intCast(count)));
        for ([_]u64{ p.rank, p.hist, p.members }) |v| c.add(v);
        try go(r.plan_scatter, s, (pairs + 255) / 256, 256, &c);
    }
};

test "plan capacity as experts.max_items" {
    try std.testing.expectEqual(@as(usize, 7), maxItems(7, 385, 16));
    try std.testing.expectEqual(@as(usize, 385 + 2100 / 64), maxItems(2100, 385, 64));
    try std.testing.expectEqual(@as(usize, 21 + 1), maxItems(21, 385, 16));
}
