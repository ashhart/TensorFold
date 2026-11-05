//! The audio tower's binding on Metal: qwen3_asr.metal's conv taps, GEMM (bias, GELU, residual), LayerNorm, attention.
const std = @import("std");
const cfg = @import("../config.zig");
const chunks = @import("../chunks.zig");
const shared = @import("../tower.zig");
const gpu = @import("gpu.zig");

/// Where a stage's rows go when a check asks for them: bf16 `rows` x `cols` at `ptr`, channels last for convs.
pub const Dump = struct {
    ctx: *anyopaque,
    put: *const fn (ctx: *anyopaque, stage: []const u8, ptr: u64, rows: usize, cols: usize, conv: ?[3]usize) anyerror!void,
};

pub const Gemm = extern struct { m: u32, n: u32, k: u32, act: u32, res: u32 };
const Conv = extern struct { n: u32, fi: u32, ti: u32, c: u32, fo: u32, to: u32, ld: u32 };
pub const Rows = extern struct { rows: u32, width: u32, eps: f32, has_res: u32 };
pub const Attn = extern struct { q_ld: u32, k_ld: u32, v_ld: u32, o_ld: u32, heads: u32, group: u32, dim: u32, keys: u32, first: u32, scale: f32 };

/// The keys q3a_attn holds in threadgroup memory.
pub const max_keys = 6144;

fn u(x: usize) u32 {
    return @intCast(x);
}

fn ceil(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

/// y [rows, n] = bf16(act(bf16(x w^T + b)) + res), the GEMV when there is one row.
pub fn gemm(g: *gpu.Gpu, x: u64, w: u64, b: u64, res: ?u64, y: u64, rows: usize, n: usize, k: usize, act: shared.Act) void {
    const p: Gemm = .{ .m = u(rows), .n = u(n), .k = u(k), .act = @intFromBool(act == .gelu), .res = @intFromBool(res != null) };
    const bufs = [_]u64{ x, w, b, res orelse y, y };
    if (rows == 1) return g.go(.q3a_gemv, .{ ceil(n, 8), 1, 1 }, 256, &bufs, p);
    g.go(.q3a_gemm, .{ ceil(n, 32), ceil(rows, 32), 1 }, 128, &bufs, p);
}

pub const Tower = struct {
    g: *gpu.Gpu,
    cols: u64, // a conv's taps
    zero: u64, // the bias of a Linear without one
    sink: ?Dump = null,

    /// Scratch for `group` chunks a conv pass.
    pub fn init(g: *gpu.Gpu, a: cfg.Audio, group: usize) !Tower {
        const cf = a.chunkFrames();
        const f1 = (a.mels - 1) / 2 + 1;
        const t1 = (cf - 1) / 2 + 1;
        const p1 = group * f1 * t1;
        const p2 = group * ((f1 - 1) / 2 + 1) * ((t1 - 1) / 2 + 1);
        return .{
            .g = g,
            .cols = try g.alloc(2 * @max(p1 * 9, p2 * 9 * a.conv)),
            .zero = try g.alloc(2 * @max(a.d, a.out, a.ffn, a.conv, 3 * a.d)),
        };
    }

    pub fn upload(t: *Tower, dst: u64, bytes: []const u8) !void {
        try t.g.put(dst, bytes);
    }

    /// y [n, fo, to, co] = bf16(taps(x [n, fi, ti, c]) . w^T + b), a 3x3 stride-2 conv with zero padding 1.
    pub fn conv(t: *Tower, w: u64, b: u64, x: u64, y: u64, n: usize, fi: usize, ti: usize, c: usize, co: usize) !void {
        const fo = (fi - 1) / 2 + 1;
        const to = (ti - 1) / 2 + 1;
        const rows = n * fo * to;
        const p: Conv = .{ .n = u(n), .fi = u(fi), .ti = u(ti), .c = u(c), .fo = u(fo), .to = u(to), .ld = u(9 * c) };
        t.g.go(.q3a_conv_cols, .{ ceil(9 * c, 256), rows, 1 }, 256, &.{ x, t.cols }, p);
        gemm(t.g, t.cols, w, b, null, y, rows, co, 9 * c, .none);
    }

    pub fn dump(t: *Tower, stage: []const u8, ptr: u64, rows: usize, cols: usize, conv_shape: ?[3]usize) !void {
        if (t.sink) |dm| {
            try t.g.flush();
            try dm.put(dm.ctx, stage, ptr, rows, cols, conv_shape);
        }
    }

    pub fn flatten(t: *Tower, x: u64, y: u64, n: usize, f: usize, tt: usize, c: usize) !void {
        const p: Conv = .{ .n = u(n), .fi = u(f), .ti = u(tt), .c = u(c), .fo = 0, .to = 0, .ld = 0 };
        t.g.go(.q3a_flatten, .{ ceil(n * f * tt * c, 256), 1, 1 }, 256, &.{ x, y }, p);
    }

    /// y = bf16(act(bf16(x . w^T + b)) + res); a missing bias is zeros.
    pub fn linear(t: *Tower, x: u64, w: u64, b: ?u64, y: u64, rows: usize, n: usize, k: usize, act: shared.Act, res: ?u64) !void {
        gemm(t.g, x, w, b orelse t.zero, res, y, rows, n, k, act);
    }

    pub fn layernorm(t: *Tower, x: u64, w: u64, b: u64, y: u64, rows: usize, d: usize) !void {
        const p: Rows = .{ .rows = u(rows), .width = u(d), .eps = 1e-5, .has_res = 0 };
        t.g.go(.q3a_layernorm, .{ rows, 1, 1 }, 256, &.{ x, w, b, y }, p);
    }

    /// Bidirectional attention over one window of `len` rows of [q | k | v] (heads x dim each).
    pub fn attention(t: *Tower, qkv: u64, out: u64, len: usize, heads: usize, dim: usize) !void {
        if (len > max_keys or dim > 256) return error.AttentionWindow;
        const d = heads * dim;
        const p: Attn = .{ .q_ld = u(3 * d), .k_ld = u(3 * d), .v_ld = u(3 * d), .o_ld = u(d), .heads = u(heads), .group = 1, .dim = u(dim), .keys = u(len), .first = 0, .scale = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(dim)))) };
        t.g.go(.q3a_attn, .{ len, heads, 1 }, 128, &.{ qkv, qkv + 2 * d, qkv + 4 * d, out }, p);
    }

    pub fn gelu(t: *Tower, x: u64, y: u64, count: usize) !void {
        t.g.go(.q3a_gelu, .{ ceil(count, 256), 1, 1 }, 256, &.{ x, y }, u(count));
    }

    pub fn sync(t: *Tower) !void {
        try t.g.flush();
    }
};
