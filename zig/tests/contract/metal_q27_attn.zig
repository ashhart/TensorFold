//! qwen27 attention's contract helpers: head counts, the engine's source, a round's plan, scratch and binding.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const rig_ = @import("rig.zig");

const c = contract.spec;
const q = tf.qwen27;
const attn = q.attention;
const planning = q.attention_plan;

const D = 256;

/// The head counts main runs: the 27B's 24 query heads over 4, and its rank of a two-Mac split (12 over 2, #1433).
pub const Heads = enum { q27b, rank };

pub fn geometry(h: Heads, shared: bool) planning.Geometry {
    return switch (h) {
        .q27b => .{ .heads = 24, .kv_heads = 4, .shared_prefix = shared },
        .rank => .{ .heads = 12, .kv_heads = 2, .shared_prefix = shared },
    };
}

pub fn label(h: Heads) []const u8 {
    return if (h == .q27b) "27B" else "27B rank";
}

/// A planted control's edit to the shipped source; `from` must occur exactly `count` times.
pub const Edit = struct { from: []const u8, to: []const u8, count: usize = 1 };

/// The source as decode_round.zig builds it here (simd_attention.library, the prompt kernels included).
pub fn source(a: std.mem.Allocator, tensor: bool, edit: ?Edit) ![]u8 {
    const text = try q.simd_attention.library(a, tensor, true);
    const e = edit orelse return text;
    defer a.free(text);
    if (std.mem.count(u8, text, e.from) != e.count) return error.ControlEdit;
    return std.mem.replaceOwned(u8, a, text, e.from, e.to);
}

/// Rounds of the round's rows, query heads and the first stream's cached keys (past 16 rows, streams of 16).
pub fn served(comptime h: Heads) []const c.Shape {
    const n: usize = comptime geometry(h, false).heads;
    // both head counts are GQA6, so the same rows reach the same tiles: 3 and 6 rows fork at 64 and 65 cached keys
    return comptime &.{ .{ .rows = 1, .n = n, .k = 2 }, .{ .rows = 3, .n = n, .k = 64 }, .{ .rows = 6, .n = n, .k = 64 }, .{ .rows = 3, .n = n, .k = 65 }, .{ .rows = 6, .n = n, .k = 65 }, .{ .rows = 16, .n = n, .k = 2046 }, .{ .rows = 16, .n = n, .k = 2047 }, .{ .rows = 128, .n = n, .k = 510 }, .{ .rows = 128, .n = n, .k = 511 }, .{ .rows = 1, .n = n, .k = 8191 }, .{ .rows = 11, .n = n, .k = 8190 }, .{ .rows = 14, .n = n, .k = 8191 } };
}

/// The engine's plan, scratch and binding for one round.
pub const State = struct {
    plan: attn.Plan,
    scratch: attn.Scratch,
    binding: attn.Binding,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, r: *const Round, gm: planning.Geometry, caches: []const attn.Cache) !State {
        var ws: [planning.max_streams]planning.Window = undefined;
        var plan = try attn.Plan.init(a, r.windows(&ws), gm);
        errdefer plan.deinit();
        const limits = attn.Limits{ .rows = @intCast(r.rows), .streams = @intCast(r.streams), .keys = @intCast(r.cap), .shared_prefix = gm.shared_prefix, .geometry = gm };
        var scratch = try attn.Scratch.init(device, limits);
        errdefer scratch.deinit();
        var binding = try attn.Binding.init(device, limits);
        errdefer binding.deinit();
        try scratch.upload(plan);
        try binding.upload(plan, caches);
        return .{ .plan = plan, .scratch = scratch, .binding = binding };
    }

    pub fn deinit(s: *State) void {
        s.binding.deinit();
        s.scratch.deinit();
        s.plan.deinit();
    }
};

/// A round's windows: up to 16 rows each, 97 keys apart; at even starts 3 rows or more fork.
pub const Round = struct {
    rows: usize = 0,
    streams: usize = 0,
    cap: usize = 0,
    counts: [planning.max_streams]usize = @splat(0),
    firsts: [planning.max_streams]usize = @splat(0),
    starts: [planning.max_streams]usize = @splat(0),
    parents: [planning.max_rows]i32 = @splat(-1),

    pub fn init(shape: c.Shape, one: bool, trees: bool, rng: std.Random) Round {
        var r = Round{ .rows = shape.rows, .streams = if (one) 1 else (shape.rows + 15) / 16 };
        var first: usize = 0;
        var end: usize = 0;
        for (0..r.streams) |s| {
            const n = if (one) shape.rows else @min(16, shape.rows - first);
            r.counts[s] = n;
            r.firsts[s] = first;
            r.starts[s] = shape.k + 97 * s;
            const tree = trees and r.starts[s] % 2 == 0 and n >= 3;
            for (0..n) |i| r.parents[first + i] = if (i == 0) -1 else if (tree and (i <= 2 or rng.uintLessThan(u32, 3) != 0)) @intCast(if (i <= 2) 0 else rng.uintLessThan(usize, i)) else @intCast(i - 1);
            first += n;
            end = @max(end, r.starts[s] + n);
        }
        r.cap = end + 64;
        return r;
    }

    pub fn windows(r: *const Round, ws: *[planning.max_streams]planning.Window) []const planning.Window {
        for (0..r.streams) |s| ws[s] = .{ .parents = r.parents[r.firsts[s]..][0..r.counts[s]], .start = @intCast(r.starts[s]), .capacity = @intCast(r.cap) };
        return ws[0..r.streams];
    }

    /// Element 0 of key head `hk`'s cache row `row` in stream `s`'s caches, `kv_heads` heads each.
    pub fn at(r: *const Round, kv_heads: usize, s: usize, hk: usize, row: usize) usize {
        return ((s * kv_heads + hk) * r.cap + row) * D;
    }
};

pub fn ref(r: rig_.Ref) q.projection.Ref {
    return .{ .buffer = r.buf, .offset = r.off };
}
