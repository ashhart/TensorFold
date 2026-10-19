//! embed_rows.cu's token rows for every CUDA family: MLX affine (2 to 6 or 8 bits, a bf16 scale) or bf16 tables.
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
    bf16_copies_fn: Function,
    affine_fn: Function,

    pub fn load(d: *const Driver) !EmbedRows {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try Module.load(d, kernels.embed_rows);
        errdefer mod.unload();
        return .{ .mod = mod, .affine4_fn = try mod.function("tf_embed_rows_affine4"), .bf16_fn = try mod.function("tf_embed_rows_bf16"), .bf16_copies_fn = try mod.function("tf_embed_rows_bf16_copies"), .affine_fn = try mod.function("tf_embed_rows_affine") };
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

    /// `affine`'s table: codes `bits` wide (2 to 6 or 8) in groups of `group`; `scale` (a bf16 value) on each value.
    pub const Affine = struct { bits: u4 = 4, group: usize = 64, scale: ?f32 = null };

    /// out [rows, d] = MLX's bf16(q s + b) of w u32 [vocab, d bits / 32] at ids, times o.scale and rounded again.
    pub fn affine(g: *const EmbedRows, s: Stream, ids: u64, w: u64, scales: u64, biases: u64, out: u64, rows: usize, d: usize, o: Affine) !void {
        const bits_ok = o.bits >= 2 and o.bits <= 8 and o.bits != 7;
        const scale_ok = if (o.scale) |x| std.math.isFinite(x) and @as(u32, @bitCast(x)) & 0xFFFF == 0 else true;
        if (!bits_ok or !scale_ok or (o.group != 32 and o.group != 64 and o.group != 128) or d == 0 or d % o.group != 0 or
            d > std.math.maxInt(i32) or rows > std.math.maxInt(u32) or out % 16 != 0) return error.RowShape;
        if (o.bits == 4 and o.scale == null and o.group != 128) return g.affine4Groups(s, ids, w, scales, biases, out, rows, d, o.group, 1);
        if (rows == 0) return;
        var a: launch_.Args = .{};
        for ([_]u64{ ids, w, scales, biases, out }) |v| a.add(v);
        for ([_]usize{ d, o.bits, o.group }) |v| a.add(@as(i32, @intCast(v)));
        a.add(o.scale orelse 1);
        try launch_.launch(g.affine_fn, .{ .grid = .{ .x = @intCast(rows), .y = @intCast((d / 8 + threads - 1) / threads), .z = 1 }, .block = .{ .x = threads } }, s, &a);
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

    /// Copy exact BF16 bits into out [rows, copies, d], including signed zeros and NaN payloads.
    pub fn bf16Copies(g: *const EmbedRows, s: Stream, ids: u64, w: u64, out: u64, rows: usize, d: usize, copies: usize) !void {
        if (d == 0 or d % 8 != 0 or d > std.math.maxInt(i32) or copies == 0 or copies > std.math.maxInt(i32) or
            rows > std.math.maxInt(u32) or out % 16 != 0 or w % 16 != 0) return error.RowShape;
        if (rows == 0) return;
        var a: launch_.Args = .{};
        for ([_]u64{ ids, w, out }) |v| a.add(v);
        for ([_]usize{ d, copies }) |v| a.add(@as(i32, @intCast(v)));
        try launch_.launch(g.bf16_copies_fn, .{ .grid = .{ .x = @intCast(rows), .y = @intCast((d / 8 + threads - 1) / threads), .z = 1 }, .block = .{ .x = threads } }, s, &a);
    }
};
