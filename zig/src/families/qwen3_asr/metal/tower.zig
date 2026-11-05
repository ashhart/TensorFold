//! The audio tower's binding on Metal: conv taps and biased GEMMs here, attention and GELU from the core kernels.
const std = @import("std");
const cfg = @import("../config.zig");
const shared = @import("../tower.zig");
const gpu = @import("gpu.zig");
const core = @import("shared.zig");

/// Where a stage's rows go when a check asks for them: bf16 `rows` x `cols` at `ptr`, channels last for convs.
pub const Dump = struct {
    ctx: *anyopaque,
    put: *const fn (ctx: *anyopaque, stage: []const u8, ptr: u64, rows: usize, cols: usize, conv: ?[3]usize) anyerror!void,
};

pub const Gemm = extern struct { m: u32, n: u32, k: u32, res: u32 };
const Conv = extern struct { n: u32, fi: u32, ti: u32, c: u32, fo: u32, to: u32, ld: u32 };
pub const Rows = extern struct { rows: u32, width: u32, eps: f32, has_res: u32 };

pub fn u(x: usize) u32 {
    return @intCast(x);
}

fn ceil(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

/// y [rows, n] = bf16(bf16(x w^T + b) + res): q3a_gemm, or q3a_gemv for one row.
pub fn gemm(g: *gpu.Gpu, x: u64, w: u64, b: u64, res: ?u64, y: u64, rows: usize, n: usize, k: usize) void {
    const p: Gemm = .{ .m = u(rows), .n = u(n), .k = u(k), .res = @intFromBool(res != null) };
    const bufs = [_]u64{ x, w, b, res orelse y, y };
    if (rows == 1) return g.go(.q3a_gemv, .{ ceil(n, 8), 1, 1 }, 256, &bufs, p);
    g.go(.q3a_gemm, .{ ceil(n, 32), ceil(rows, 32), 1 }, 128, &bufs, p);
}

pub const Tower = struct {
    g: *gpu.Gpu,
    k: *const core.Shared,
    a: cfg.Audio,
    cols: u64, // a conv's taps
    zero: u64, // the bias of a Linear without one, and of a bare GELU
    q: u64, // a window's q, k and v, each [rows, heads, dim]: shared_attention's layout
    kk: u64,
    v: u64,
    mask: u64, // every key visible: encode_rows rows of max_rows bytes
    sink: ?Dump = null,

    /// Scratch for `group` chunks a conv pass and windows of up to `max_rows` rows.
    pub fn init(g: *gpu.Gpu, k: *const core.Shared, a: cfg.Audio, group: usize, max_rows: usize) !Tower {
        const cf = a.chunkFrames();
        const f1 = (a.mels - 1) / 2 + 1;
        const t1 = (cf - 1) / 2 + 1;
        const p1 = group * f1 * t1;
        const p2 = group * ((f1 - 1) / 2 + 1) * ((t1 - 1) / 2 + 1);
        const t: Tower = .{
            .g = g,
            .k = k,
            .a = a,
            .cols = try g.alloc(2 * @max(p1 * 9, p2 * 9 * a.conv)),
            .zero = try g.alloc(2 * @max(a.d, a.out, a.ffn, a.conv)),
            .q = try g.alloc(2 * max_rows * a.d),
            .kk = try g.alloc(2 * max_rows * a.d),
            .v = try g.alloc(2 * max_rows * a.d),
            .mask = try g.alloc(core.encode_rows * max_rows),
        };
        @memset(g.host(t.mask, core.encode_rows * max_rows), 1);
        return t;
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
        gemm(t.g, t.cols, w, b, null, y, rows, co, 9 * c);
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

    /// y = bf16(act(bf16(x . w^T + b)) + res); a missing bias is zeros (no layer has both GELU and residual).
    pub fn linear(t: *Tower, x: u64, w: u64, b: ?u64, y: u64, rows: usize, n: usize, k: usize, act: shared.Act, res: ?u64) !void {
        if (act == .gelu and res != null) return error.GeluWithResidual;
        gemm(t.g, x, w, b orelse t.zero, res, y, rows, n, k);
        if (act == .gelu) t.k.biasGelu(t.g.encoder(), t.g.ref(y), t.g.ref(t.zero), rows, n);
    }

    pub fn layernorm(t: *Tower, x: u64, w: u64, b: u64, y: u64, rows: usize, d: usize) !void {
        const p: Rows = .{ .rows = u(rows), .width = u(d), .eps = 1e-5, .has_res = 0 };
        t.g.go(.q3a_layernorm, .{ rows, 1, 1 }, 256, &.{ x, w, b, y }, p);
    }

    /// Bidirectional attention over one window of `len` rows of [q | k | v] (heads x dim each), by shared_attention.
    pub fn attention(t: *Tower, qkv: u64, out: u64, len: usize, heads: usize, dim: usize) !void {
        const d = heads * dim;
        t.g.go(.q3a_split_qkv, .{ ceil(d, 256), len, 1 }, 256, &.{ qkv, t.q, t.kk, t.v }, [2]u32{ u(len), u(d) });
        const scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(dim))));
        const e = t.g.encoder();
        var r0: usize = 0;
        while (r0 < len) : (r0 += core.encode_rows) {
            const rows = @min(core.encode_rows, len - r0);
            try t.k.tower.encode(e, rows, len, len, scale, t.g.ref(t.q + r0 * d * 2), t.g.ref(t.kk), t.g.ref(t.v), t.g.ref(t.mask), t.g.ref(out + r0 * d * 2));
        }
    }

    /// y = bf16(gelu(x)) through the core GELU (in place, after a copy when y is not x).
    pub fn gelu(t: *Tower, x: u64, y: u64, count: usize) !void {
        if (x != y) t.g.go(.q3a_copy, .{ ceil(count, 256), 1, 1 }, 256, &.{ x, y }, u(count));
        const width = for ([_]usize{ t.a.ffn, t.a.d, t.a.conv }) |w| {
            if (count % w == 0) break w;
        } else return error.GeluWidth;
        t.k.biasGelu(t.g.encoder(), t.g.ref(y), t.g.ref(t.zero), count / width, width);
    }

    pub fn sync(t: *Tower) !void {
        try t.g.flush();
    }
};
