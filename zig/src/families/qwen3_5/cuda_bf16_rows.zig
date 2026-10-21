//! qwen35_bf16_rows.cu: bf16 token rows and small bf16 dense projections (GDN in_proj_a and in_proj_b).
const std = @import("std");
const cuda = @import("cuda");

const threads = 256;

pub const Kernels = struct {
    mod: cuda.Module,
    embed_fn: cuda.Function,
    dense_fn: cuda.Function,

    pub fn resolve(mod: cuda.Module) !Kernels {
        return .{ .mod = mod, .embed_fn = try mod.function("tf_q35_embed_bf16"), .dense_fn = try mod.function("tf_q35_dense_bf16") };
    }

    /// Load the embedded module (owned: call unload) and resolve its functions.
    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.qwen35_bf16_rows);
        errdefer mod.unload();
        return resolve(mod);
    }

    pub fn unload(k: *Kernels) void {
        k.mod.unload();
    }

    /// out[r] = table[ids[r]] (ids i32 [rows], table bf16 [vocab][hidden]); params as tf_q35_embed's, or 0.
    pub fn embedRows(k: *const Kernels, s: cuda.Stream, ids: u64, table: u64, out: u64, rows: usize, hidden: usize, params: u64) !void {
        if (rows == 0) return;
        if (hidden == 0 or hidden > std.math.maxInt(i32) or rows > std.math.maxInt(u32)) return error.RowShape;
        var a: cuda.launch.Args = .{};
        for ([_]u64{ ids, table, out }) |v| a.add(v);
        a.add(@as(i32, @intCast(hidden)));
        a.add(params);
        try cuda.launch.launch(k.embed_fn, .{ .grid = .{ .x = @intCast(rows), .y = 1, .z = 1 }, .block = .{ .x = threads } }, s, &a);
    }

    /// out[rows][n] = x[rows][k] @ w[n][k]^T in bf16 with fixed-order fp32 sums (rows <= 128, k % 8 == 0, aligned).
    pub fn denseRows(kn: *const Kernels, s: cuda.Stream, x: u64, w: u64, out: u64, rows: usize, n: usize, k: usize) !void {
        if (rows == 0 or n == 0) return;
        if (rows > 128 or k == 0 or k % 8 != 0 or k > std.math.maxInt(i32) or n > std.math.maxInt(i32) or x % 16 != 0 or w % 16 != 0) return error.RowShape;
        var a: cuda.launch.Args = .{};
        for ([_]u64{ x, w, out }) |v| a.add(v);
        for ([_]usize{ n, k }) |v| a.add(@as(i32, @intCast(v)));
        try cuda.launch.launch(kn.dense_fn, .{ .grid = .{ .x = @intCast(n), .y = @intCast(rows), .z = 1 }, .block = .{ .x = threads } }, s, &a);
    }
};
