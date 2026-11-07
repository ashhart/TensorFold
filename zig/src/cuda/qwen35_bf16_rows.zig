//! qwen35_bf16_rows.cu: bf16 token rows and small bf16 dense projections (GDN in_proj_a and in_proj_b).
const std = @import("std");
const Driver = @import("driver.zig").Driver;
const Module = @import("module.zig").Module;
const Function = @import("module.zig").Function;
const Stream = @import("stream.zig").Stream;
const kernels = @import("kernels.zig");
const launch = @import("launch.zig");

const threads = 256;

pub const Kernels = struct {
    mod: Module,
    embed_fn: Function,
    dense_fn: Function,
    dense_f32_fn: ?Function = null,
    stats_fn: ?Function = null,

    pub fn resolve(mod: Module) !Kernels {
        return .{ .mod = mod, .embed_fn = try mod.function("tf_q35_embed_bf16"), .dense_fn = try mod.function("tf_q35_dense_bf16"), .dense_f32_fn = mod.function("tf_q35_dense_f32") catch null, .stats_fn = mod.function("tf_q35_score_stats") catch null };
    }

    /// Load the embedded module (owned: call unload) and resolve its functions.
    pub fn load(d: *const Driver) !Kernels {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try Module.load(d, kernels.qwen35_bf16_rows);
        errdefer mod.unload();
        return resolve(mod);
    }

    pub fn unload(k: *Kernels) void {
        k.mod.unload();
    }

    /// out[r] = table[ids[r]] (ids i32 [rows], table bf16 [vocab][hidden]); params as tf_q35_embed's, or 0.
    pub fn embedRows(k: *const Kernels, s: Stream, ids: u64, table: u64, out: u64, rows: usize, hidden: usize, params: u64) !void {
        if (rows == 0) return;
        if (hidden == 0 or hidden > std.math.maxInt(i32) or rows > std.math.maxInt(u32)) return error.RowShape;
        var a: launch.Args = .{};
        for ([_]u64{ ids, table, out }) |v| a.add(v);
        a.add(@as(i32, @intCast(hidden)));
        a.add(params);
        try launch.launch(k.embed_fn, .{ .grid = .{ .x = @intCast(rows), .y = 1, .z = 1 }, .block = .{ .x = threads } }, s, &a);
    }

    /// out[rows][n] = x[rows][k] @ w[n][k]^T in bf16 with fixed-order fp32 sums (rows <= 128, k % 8 == 0, aligned).
    pub fn denseRows(kn: *const Kernels, s: Stream, x: u64, w: u64, out: u64, rows: usize, n: usize, k: usize) !void {
        if (rows == 0 or n == 0) return;
        if (rows > 128 or k == 0 or k % 8 != 0 or k > std.math.maxInt(i32) or n > std.math.maxInt(i32) or x % 16 != 0 or w % 16 != 0) return error.RowShape;
        var a: launch.Args = .{};
        for ([_]u64{ x, w, out }) |v| a.add(v);
        for ([_]usize{ n, k }) |v| a.add(@as(i32, @intCast(v)));
        try launch.launch(kn.dense_fn, .{ .grid = .{ .x = @intCast(n), .y = @intCast(rows), .z = 1 }, .block = .{ .x = threads } }, s, &a);
    }
    /// Fixed FP32 dots of stored BF16 rows, with FP32 vocabulary logits (no output rounding).
    pub fn denseF32(kn: *const Kernels, s: Stream, x: u64, w: u64, out: u64, rows: usize, n: usize, k: usize) !void {
        if (rows == 0 or n == 0) return;
        if (rows > 128 or k == 0 or k % 8 != 0 or k > std.math.maxInt(i32) or n > std.math.maxInt(i32) or x % 16 != 0 or w % 16 != 0 or out % 4 != 0) return error.RowShape;
        var a: launch.Args = .{};
        for ([_]u64{ x, w, out }) |v| a.add(v);
        for ([_]usize{ n, k }) |v| a.add(@as(i32, @intCast(v)));
        try launch.launch(kn.dense_f32_fn orelse return error.MissingQwenBf16Scores, .{ .grid = .{ .x = @intCast(n), .y = @intCast(rows) }, .block = .{ .x = threads } }, s, &a);
    }

    /// Each row owns 43 words: LSE, selected logit, 20 best IDs/logits, rank; picks are device u32 IDs.
    pub fn stats(kn: *const Kernels, s: Stream, logits: u64, vocab: usize, picks: u64, top: u8, out: u64, rows: usize) !void {
        if (rows == 0) return;
        if (rows > 128 or vocab == 0 or vocab > std.math.maxInt(u32) or top > 20 or top > vocab or (logits | picks | out) % 4 != 0) return error.RowShape;
        var a: launch.Args = .{};
        a.add(logits);
        a.add(@as(u32, @intCast(vocab)));
        a.add(picks);
        a.add(@as(u32, top));
        a.add(out);
        try launch.launch(kn.stats_fn orelse return error.MissingQwenBf16Scores, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = threads } }, s, &a);
    }
};
