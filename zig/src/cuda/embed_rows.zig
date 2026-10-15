//! embed_rows.cu's token rows for every CUDA family: 4-bit affine (MLX layout, groups of 32 or 64) or bf16 tables.
const std = @import("std");
const kernels = @import("kernels.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");

const Driver = @import("driver.zig").Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;

const threads = 256;

pub const EmbedRows = struct {
    mod: Module,
    affine4_fn: Function,
    bf16_fn: Function,

    pub fn load(d: *const Driver) !EmbedRows {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try Module.load(d, kernels.embed_rows);
        errdefer mod.unload();
        return .{ .mod = mod, .affine4_fn = try mod.function("tf_embed_rows_affine4"), .bf16_fn = try mod.function("tf_embed_rows_bf16") };
    }

    pub fn unload(g: *EmbedRows) void {
        g.mod.unload();
    }

    /// out [rows, d] bf16 from ids (i32 [rows]): w u32 [vocab, d / 8], scales and biases bf16 [vocab, d / 64].
    pub fn affine4(g: *const EmbedRows, s: Stream, ids: u64, w: u64, scales: u64, biases: u64, out: u64, rows: usize, d: usize) !void {
        return g.affine4Groups(s, ids, w, scales, biases, out, rows, d, 64, 1);
    }

    /// out [rows, copies, d] bf16, each row `copies` times: scales and biases a group of `group` (32 or 64) values.
    pub fn affine4Groups(g: *const EmbedRows, s: Stream, ids: u64, w: u64, scales: u64, biases: u64, out: u64, rows: usize, d: usize, group: usize, copies: usize) !void {
        if ((group != 32 and group != 64) or d % group != 0 or out % 16 != 0 or copies == 0) return error.RowShape;
        if (rows == 0) return;
        var a: launch_.Args = .{};
        for ([_]u64{ ids, w, scales, biases, out }) |v| a.add(v);
        for ([_]usize{ d, group, copies }) |v| a.add(@as(i32, @intCast(v)));
        try launch_.launch(g.affine4_fn, .{ .grid = .{ .x = @intCast(rows), .y = @intCast((d / 8 + threads - 1) / threads), .z = 1 }, .block = .{ .x = threads } }, s, &a);
    }

    /// out [rows, d] = w [vocab, d] bf16 rows at ids.
    pub fn bf16(g: *const EmbedRows, s: Stream, ids: u64, w: u64, out: u64, rows: usize, d: usize) !void {
        if (d % 8 != 0 or out % 16 != 0 or w % 16 != 0) return error.RowShape;
        if (rows == 0) return;
        var a: launch_.Args = .{};
        for ([_]u64{ ids, w, out }) |v| a.add(v);
        a.add(@as(i32, @intCast(d)));
        try launch_.launch(g.bf16_fn, .{ .grid = .{ .x = @intCast(rows), .y = @intCast((d / 8 + threads - 1) / threads), .z = 1 }, .block = .{ .x = threads } }, s, &a);
    }
};
