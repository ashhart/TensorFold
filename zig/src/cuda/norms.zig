//! norms.cu's RMS norm rows for every CUDA family: a block a row, its bits independent of the row count.
const std = @import("std");
const kernels = @import("kernels.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");

const Driver = @import("driver.zig").Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;

/// Element kinds, numbered as the kernels (and hip/ops/common.hpp) number them.
pub const Kind = enum(u8) { f32 = 0, f16 = 1, bf16 = 2 };

const threads = 1024;
/// norms.cu's 16-byte forms: VNT threads, VEC_MOST loads of 8 a thread.
const vec_threads = 256;
const vec_most = 4;

/// The 16-byte forms take a bf16 row by its width alone: a multiple of 8 within VEC_MOST * 8 * VNT.
fn vectorWidth(width: usize) bool {
    return width % 8 == 0 and width <= vec_most * 8 * vec_threads;
}

/// Every address 16-byte aligned, as the 16-byte forms load.
fn aligned(ptrs: []const u64) bool {
    for (ptrs) |p| if (p % 16 != 0) return false;
    return true;
}

fn vectorOk(width: usize, ptrs: []const u64) bool {
    return vectorWidth(width) and aligned(ptrs);
}

/// addRmsTail's options: b = norm(x) wb (or x), h = bf16(r + b) [* scalar], y = bf16(norm(h) w), xs y's 64-value sums.
pub const Tail = struct {
    /// Each norm as MLX's bf16 steps, bf16(w * bf16(v * inv)), where false gives bf16(v * inv * w).
    round: bool = false,
    /// A norm on x before the add (a sandwich norm): its bf16 weight; null adds x as it is.
    branch: ?u64 = null,
    /// One bf16 the sum is multiplied by, after the sum is rounded, before the output's norm; null: none.
    scalar: ?u64 = null,
};

pub const Norms = struct {
    mod: Module,
    rms_fn: Function,
    add_rms_fn: Function,
    rms_bf16_fn: Function,
    add_rms_bf16_fn: Function,
    add_rms_h_fn: Function,
    add_rms_h_f32w_fn: Function,
    gated_fns: [3]Function, // head dim 64, 128, 256
    rms_sums_fn: Function,
    tail_fns: [2]Function, // each norm rounded once, MLX's bf16 steps
    l2_fn: Function,
    slerp_fn: Function,

    pub fn load(d: *const Driver) !Norms {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try Module.load(d, kernels.norms);
        errdefer mod.unload();
        return .{
            .mod = mod,
            .rms_fn = try mod.function("tf_rms_rows"),
            .add_rms_fn = try mod.function("tf_add_rms_rows"),
            .rms_bf16_fn = try mod.function("tf_rms_rows_bf16"),
            .add_rms_bf16_fn = try mod.function("tf_add_rms_rows_bf16"),
            .add_rms_h_fn = try mod.function("tf_add_rms_rows_bf16h"),
            .add_rms_h_f32w_fn = try mod.function("tf_add_rms_rows_bf16h_f32w"),
            .gated_fns = .{ try mod.function("tf_gated_rms_rows_64"), try mod.function("tf_gated_rms_rows_128"), try mod.function("tf_gated_rms_rows_256") },
            .rms_sums_fn = try mod.function("tf_rms_rows_sums"),
            .tail_fns = .{ try mod.function("tf_add_rms_rows_bf16h_tail"), try mod.function("tf_add_rms_rows_bf16h_tail_round") },
            .l2_fn = try mod.function("tf_l2_rows"),
            .slerp_fn = try mod.function("tf_slerp_rows"),
        };
    }

    pub fn unload(n: *Norms) void {
        n.mod.unload();
    }

    /// `rows` rows of `width`: y = x * rsqrt(mean(x^2) + eps) * w (w null: no weight), fp32 weights.
    pub fn rms(n: *const Norms, s: Stream, x: u64, x_kind: Kind, w: ?u64, y: u64, y_kind: Kind, rows: usize, width: usize, eps: f32) !void {
        if (rows == 0) return;
        if (x_kind == .bf16 and y_kind == .bf16 and vectorWidth(width)) {
            if (!aligned(&.{ x, y, w orelse 0 })) return error.Misaligned; // the other tree would round otherwise
            var a: launch_.Args = .{};
            a.add(x);
            a.add(w orelse 0);
            a.add(y);
            a.add(@as(i32, @intCast(width)));
            a.add(eps);
            return launch_.launch(n.rms_bf16_fn, .{ .grid = .{ .x = @intCast(rows), .y = 1, .z = 1 }, .block = .{ .x = vec_threads } }, s, &a);
        }
        var a: launch_.Args = .{};
        a.add(x);
        a.add(@as(i32, @backingInt(x_kind)));
        a.add(w orelse 0);
        a.add(y);
        a.add(@as(i32, @backingInt(y_kind)));
        a.add(@as(i32, @intCast(width)));
        a.add(eps);
        try launch_.launch(n.rms_fn, .{ .grid = .{ .x = @intCast(rows), .y = 1, .z = 1 }, .block = .{ .x = threads } }, s, &a);
    }

    /// bf16 rows `x_ld` apart: y = bf16(rms(x) * w) [rows, width], fp32 w, and xs the stored y's sums of each `group`.
    pub fn rmsSums(n: *const Norms, s: Stream, x: u64, x_ld: usize, w: u64, y: u64, xs: u64, rows: usize, width: usize, group: usize, eps: f32) !void {
        if (group == 0 or width % group != 0 or x_ld < width) return error.RowShape;
        if (rows == 0) return;
        var a: launch_.Args = .{};
        a.add(x);
        a.add(@as(i64, @intCast(x_ld)));
        for ([_]u64{ w, y, xs }) |v| a.add(v);
        a.add(@as(i32, @intCast(width)));
        a.add(@as(i32, @intCast(group)));
        a.add(eps);
        try launch_.launch(n.rms_sums_fn, .{ .grid = .{ .x = @intCast(rows), .y = 1, .z = 1 }, .block = .{ .x = threads } }, s, &a);
    }

    /// The residual step: res (fp32, in place) += rms(x) * w1 (w1 null: res += x), then y = rms(res) * w2.
    pub fn addRms(n: *const Norms, s: Stream, x: u64, x_kind: Kind, res: u64, w1: ?u64, w2: u64, y: u64, y_kind: Kind, rows: usize, width: usize, eps: f32) !void {
        if (rows == 0) return;
        if (x_kind == .bf16 and y_kind == .bf16 and vectorWidth(width)) {
            if (!aligned(&.{ x, y, res, w1 orelse 0, w2 })) return error.Misaligned;
            var a: launch_.Args = .{};
            a.add(x);
            a.add(res);
            a.add(w1 orelse 0);
            a.add(w2);
            a.add(y);
            a.add(@as(i32, @intCast(width)));
            a.add(eps);
            return launch_.launch(n.add_rms_bf16_fn, .{ .grid = .{ .x = @intCast(rows), .y = 1, .z = 1 }, .block = .{ .x = vec_threads } }, s, &a);
        }
        var a: launch_.Args = .{};
        a.add(x);
        a.add(@as(i32, @backingInt(x_kind)));
        a.add(res);
        a.add(w1 orelse 0);
        a.add(w2);
        a.add(y);
        a.add(@as(i32, @backingInt(y_kind)));
        a.add(@as(i32, @intCast(width)));
        a.add(eps);
        try launch_.launch(n.add_rms_fn, .{ .grid = .{ .x = @intCast(rows), .y = 1, .z = 1 }, .block = .{ .x = threads } }, s, &a);
    }

    /// bf16 residual: h = bf16(x + r) (r null: h = x), y = bf16(rms(h) * w); xs (or null): y's 64-value sums.
    pub fn addRmsH(n: *const Norms, s: Stream, x: u64, r: ?u64, w: u64, w_kind: Kind, h: u64, y: u64, xs: ?u64, rows: usize, width: usize, eps: f32) !void {
        if (width % 64 != 0 or !vectorOk(width, &.{ x, r orelse 0, w, h, y }) or w_kind == .f16) return error.RowShape;
        if (rows == 0) return;
        var a: launch_.Args = .{};
        for ([_]u64{ x, r orelse 0, w, h, y, xs orelse 0 }) |v| a.add(v);
        a.add(@as(i32, @intCast(width)));
        a.add(eps);
        const f = if (w_kind == .bf16) n.add_rms_h_fn else n.add_rms_h_f32w_fn;
        try launch_.launch(f, .{ .grid = .{ .x = @intCast(rows), .y = 1, .z = 1 }, .block = .{ .x = vec_threads } }, s, &a);
    }

    /// addRmsH with bf16 weights and a tail's options; with none set it is addRmsH itself, the same kernel and bits.
    pub fn addRmsTail(n: *const Norms, s: Stream, x: u64, r: ?u64, w: u64, h: u64, y: u64, xs: ?u64, rows: usize, width: usize, eps: f32, t: Tail) !void {
        if (!t.round and t.branch == null and t.scalar == null) return n.addRmsH(s, x, r, w, .bf16, h, y, xs, rows, width, eps);
        if (width % 64 != 0 or !vectorOk(width, &.{ x, r orelse 0, t.branch orelse 0, w, h, y }) or (t.scalar orelse 0) % 2 != 0) return error.RowShape;
        if (rows == 0) return;
        var a: launch_.Args = .{};
        for ([_]u64{ x, r orelse 0, t.branch orelse 0, t.scalar orelse 0, w, h, y, xs orelse 0 }) |v| a.add(v);
        a.add(@as(i32, @intCast(width)));
        a.add(eps);
        const f = n.tail_fns[@intFromBool(t.round)];
        try launch_.launch(f, .{ .grid = .{ .x = @intCast(rows), .y = 1, .z = 1 }, .block = .{ .x = vec_threads } }, s, &a);
    }

    /// nGPT rows: y = bf16(s / sqrt(sum(s^2) + eps)), s = x, or bf16(x + r) with r; width % 64, at most 8,192.
    pub fn l2(n: *const Norms, s: Stream, x: u64, r: ?u64, y: u64, rows: usize, width: usize, eps: f32) !void {
        if (width % 64 != 0 or !vectorOk(width, &.{ x, r orelse 0, y })) return error.RowShape;
        if (rows == 0) return;
        var a: launch_.Args = .{};
        for ([_]u64{ x, r orelse 0, y }) |v| a.add(v);
        a.add(@as(i32, @intCast(width)));
        a.add(eps);
        try launch_.launch(n.l2_fn, .{ .grid = .{ .x = @intCast(rows), .y = 1, .z = 1 }, .block = .{ .x = vec_threads } }, s, &a);
    }

    /// The nGPT residual: h = bf16(l2(a + |alpha scale| (b - a))), a = bf16(l2(res)), b = bf16(l2(out)); bf16 alpha.
    pub fn slerp(n: *const Norms, s: Stream, res: u64, out: u64, alpha: u64, scale: f32, h: u64, rows: usize, width: usize, eps: f32) !void {
        if (width % 64 != 0 or !vectorOk(width, &.{ res, out, alpha, h })) return error.RowShape;
        if (rows == 0) return;
        var a: launch_.Args = .{};
        for ([_]u64{ res, out, alpha }) |v| a.add(v);
        a.add(scale);
        a.add(h);
        a.add(@as(i32, @intCast(width)));
        a.add(eps);
        try launch_.launch(n.slerp_fn, .{ .grid = .{ .x = @intCast(rows), .y = 1, .z = 1 }, .block = .{ .x = vec_threads } }, s, &a);
    }

    /// Per head (dv 64/128/256): out = bf16(silu(z) * rms(y) * w), y/z rows at strides y_ld/z_ld; xs: 64-value sums.
    pub fn gatedRms(n: *const Norms, s: Stream, y: u64, y_ld: usize, z: u64, z_ld: usize, w: u64, out: u64, xs: ?u64, rows: usize, heads: usize, dv: usize, eps: f32) !void {
        const which: usize = switch (dv) {
            64 => 0,
            128 => 1,
            256 => 2,
            else => return error.RowShape,
        };
        if (y_ld % 8 != 0 or z_ld % 8 != 0 or !vectorOk(8, &.{ y, z, w, out })) return error.RowShape;
        if (rows == 0 or heads == 0) return;
        var a: launch_.Args = .{};
        a.add(y);
        a.add(@as(i64, @intCast(y_ld)));
        a.add(z);
        a.add(@as(i64, @intCast(z_ld)));
        for ([_]u64{ w, out, xs orelse 0 }) |v| a.add(v);
        a.add(@as(i32, @intCast(heads)));
        a.add(eps);
        const t = launch_.rowThreads(rows, 256, 32); // a head's lanes stay in one warp either way
        const per_block = t / (dv / 8);
        const grid: launch_.Dim3 = .{ .x = @intCast(rows), .y = @intCast((heads + per_block - 1) / per_block), .z = 1 };
        try launch_.launch(n.gated_fns[which], .{ .grid = grid, .block = .{ .x = t } }, s, &a);
    }
};

test "kinds keep the kernels' numbering" {
    try std.testing.expectEqual(@as(u8, 2), @backingInt(Kind.bf16));
}
