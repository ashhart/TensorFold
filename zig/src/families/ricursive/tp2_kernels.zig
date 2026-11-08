//! ricursive_tp2.cu's kernels: embed scale, value residual, gate, SwiGLU, post-norms, rank sums, head moves, taps.
const std = @import("std");
const cuda = @import("cuda");

fn blocks(n: usize) u32 {
    return @intCast((n + 255) / 256);
}

pub const Kernels = struct {
    mod: cuda.Module,
    embed_fn: cuda.Function,
    lerp_fn: cuda.Function,
    gate_fn: cuda.Function,
    swiglu_fn: cuda.Function,
    norm_add_fn: cuda.Function,
    rank_sum_fn: cuda.Function,
    heads_fn: cuda.Function,
    taps_fn: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.ricursive_tp2);
        errdefer mod.unload();
        return .{ .mod = mod, .embed_fn = try mod.function("tf_rtp2_embed"), .lerp_fn = try mod.function("tf_rtp2_lerp"), .gate_fn = try mod.function("tf_rtp2_gate"), .swiglu_fn = try mod.function("tf_rtp2_swiglu"), .norm_add_fn = try mod.function("tf_rtp2_norm_add"), .rank_sum_fn = try mod.function("tf_rtp2_rank_sum"), .heads_fn = try mod.function("tf_rtp2_heads"), .taps_fn = try mod.function("tf_rtp2_taps") };
    }

    pub fn unload(k: *Kernels) void {
        k.mod.unload();
    }

    /// x fp32 [rows, width] = bf16(table[id] * scale), seed fp16 the same rows.
    pub fn embed(k: *const Kernels, s: cuda.Stream, ids: u64, table: u64, x: u64, seed: u64, rows: usize, width: usize, scale: f32) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ ids, table, x, seed }) |v| a.add(v);
        a.add(@as(i32, @intCast(width)));
        a.add(scale);
        try cuda.launch.launch(k.embed_fn, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// out fp16 [n] = lerp(seed, y, w).
    pub fn lerp(k: *const Kernels, s: cuda.Stream, seed: u64, y: u64, out: u64, n: usize, w: f32) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ seed, y, out }) |v| a.add(v);
        a.add(@as(i64, @intCast(n)));
        a.add(w);
        try cuda.launch.launch(k.lerp_fn, .{ .grid = .{ .x = blocks(n) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// out fp16 [n] = o (bf16) * sigmoid(g (fp16)).
    pub fn gate(k: *const Kernels, s: cuda.Stream, o: u64, g: u64, out: u64, n: usize) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ o, g, out }) |v| a.add(v);
        a.add(@as(i64, @intCast(n)));
        try cuda.launch.launch(k.gate_fn, .{ .grid = .{ .x = blocks(n) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// out fp16 [n] = silu(min(g, limit)) * clamp(u, -limit, limit).
    pub fn swiglu(k: *const Kernels, s: cuda.Stream, g: u64, u: u64, out: u64, n: usize, limit: f32) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ g, u, out }) |v| a.add(v);
        a.add(@as(i64, @intCast(n)));
        a.add(limit);
        try cuda.launch.launch(k.swiglu_fn, .{ .grid = .{ .x = blocks(n) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// x fp32 [rows, width] += rms(a) * w (fp32 a and w).
    pub fn normAdd(k: *const Kernels, s: cuda.Stream, a_: u64, w: u64, x: u64, rows: usize, width: usize, eps: f32) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ a_, w, x }) |v| a.add(v);
        a.add(@as(i32, @intCast(width)));
        a.add(eps);
        try cuda.launch.launch(k.norm_add_fn, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// out fp32 [n] = parts[0] + parts[1] + ... in rank order (parts: world runs of n).
    pub fn rankSum(k: *const Kernels, s: cuda.Stream, parts: u64, out: u64, n: usize, world: usize) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ parts, out }) |v| a.add(v);
        a.add(@as(i64, @intCast(n)));
        a.add(@as(i32, @intCast(world)));
        try cuda.launch.launch(k.rank_sum_fn, .{ .grid = .{ .x = blocks(n) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// The first g heads of each of `groups` key heads' runs: bf16 [groups, gs, 128] -> [groups, gd, 128].
    pub fn heads(k: *const Kernels, s: cuda.Stream, src: u64, dst: u64, groups: usize, g: usize, gs: usize, gd: usize) !void {
        const n = groups * g * 16; // 16-byte moves
        var a: cuda.Args = .{};
        for ([_]u64{ src, dst }) |v| a.add(v);
        a.add(@as(i64, @intCast(n)));
        for ([_]usize{ g, gs, gd }) |v| a.add(@as(i32, @intCast(v)));
        try cuda.launch.launch(k.heads_fn, .{ .grid = .{ .x = blocks(n) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// Rows' fp32 residuals into a drafter's tap ring: ring[slot][pos % tr][tap] as bf16 (rows of ntaps * h).
    pub fn taps(k: *const Kernels, s: cuda.Stream, x: u64, pos: u64, slots: u64, ring: u64, tr: usize, ntaps: usize, tap: usize, h: usize, rows: usize) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ x, pos, slots, ring }) |v| a.add(v);
        for ([_]usize{ tr, ntaps, tap, h }) |v| a.add(@as(i32, @intCast(v)));
        try cuda.launch.launch(k.taps_fn, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }
};

test "ricursive kernels: one 256-thread block per 256 values" {
    try std.testing.expectEqual(@as(u32, 1), blocks(1));
    try std.testing.expectEqual(@as(u32, 160), blocks(40960));
}
