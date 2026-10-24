//! dense_rows.cu's bf16 rows times bf16 weight rows into fp32 (routers, output heads): a row's bits at any row count.
const std = @import("std");
const kernels = @import("kernels.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");

const Driver = @import("driver.zig").Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;

const warps = 8;
/// The launch tiles: rows a tile shares each weight row across.
const tiles = [_]usize{ 1, 4, 16, 32 };

pub const DenseRows = struct {
    mod: Module,
    fns: [tiles.len]Function,
    scaled_fns: [tiles.len]Function,

    pub fn load(d: *const Driver) !DenseRows {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try Module.load(d, kernels.dense_rows);
        errdefer mod.unload();
        var fns: [tiles.len]Function = undefined;
        var scaled: [tiles.len]Function = undefined;
        inline for (tiles, 0..) |t, i| {
            fns[i] = try mod.function(std.fmt.comptimePrint("tf_dense_rows_{d}", .{t}));
            scaled[i] = try mod.function(std.fmt.comptimePrint("tf_dense_rows_scaled_{d}", .{t}));
        }
        return .{ .mod = mod, .fns = fns, .scaled_fns = scaled };
    }

    pub fn unload(g: *DenseRows) void {
        g.mod.unload();
    }

    /// out [m, n] fp32 = x [m, k] bf16 times w [n, k] bf16 transposed; strides in elements, k % 8 == 0, 16 B aligned.
    pub fn rows(g: *const DenseRows, s: Stream, x: u64, x_stride: usize, w: u64, out: u64, out_stride: usize, m: usize, n: usize, k: usize) !void {
        return g.rowsWith(null, s, x, x_stride, w, out, out_stride, m, n, k);
    }

    /// rows with the scaled epilogue: out = fp32(bf16(x . w[n])) * scale[n], scale fp32 [n] (Dory's logits).
    pub fn rowsScaled(g: *const DenseRows, s: Stream, x: u64, x_stride: usize, w: u64, scale: u64, out: u64, out_stride: usize, m: usize, n: usize, k: usize) !void {
        return g.rowsWith(scale, s, x, x_stride, w, out, out_stride, m, n, k);
    }

    fn rowsWith(g: *const DenseRows, scale: ?u64, s: Stream, x: u64, x_stride: usize, w: u64, out: u64, out_stride: usize, m: usize, n: usize, k: usize) !void {
        if (k % 8 != 0 or x % 16 != 0 or w % 16 != 0 or x_stride % 8 != 0) return error.Misaligned;
        if (m == 0 or n == 0) return;
        const i: usize = if (m == 1) 0 else if (m <= 4) 1 else if (m <= 16) 2 else 3;
        var a: launch_.Args = .{};
        a.add(x);
        a.add(@as(i64, @intCast(x_stride)));
        a.add(w);
        if (scale) |v| a.add(v);
        a.add(out);
        a.add(@as(i64, @intCast(out_stride)));
        a.add(@as(i32, @intCast(m)));
        a.add(@as(i32, @intCast(n)));
        a.add(@as(i32, @intCast(k)));
        const grid: launch_.Dim3 = .{ .x = @intCast((n + warps - 1) / warps), .y = @intCast((m + tiles[i] - 1) / tiles[i]), .z = 1 };
        try launch_.launch(if (scale != null) g.scaled_fns[i] else g.fns[i], .{ .grid = grid, .block = .{ .x = warps * 32 } }, s, &a);
    }
};
