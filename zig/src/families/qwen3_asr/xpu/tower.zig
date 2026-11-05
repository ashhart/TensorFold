//! The audio tower's binding on the Intel GPU without DPAS: qwen3_asr.cl's conv, Linear, LayerNorm, GELU and attention.
const std = @import("std");
const xpu = @import("xpu");
const shared = @import("../tower.zig");
const store = @import("store.zig");
const rt = xpu.rt;

/// Where a stage's device rows go when a check asks for them: bf16 `rows` x `cols` at `ptr`, channels last for convs.
pub const Dump = struct {
    ctx: *anyopaque,
    put: *const fn (ctx: *anyopaque, stage: []const u8, ptr: u64, rows: usize, cols: usize, conv: ?[3]usize) anyerror!void,
};

/// A launch: device addresses (u64) as buffers, f32 as floats, every other integer as a 32-bit argument.
pub fn run(r: *rt.Runtime, k: *rt.Kernel, groups: [3]usize, args: anytype) !void {
    inline for (args, 0..) |v, i| {
        const T = @TypeOf(v);
        if (T == u64) {
            try k.setBuffer(i, store.view(r, v, 0));
        } else if (T == f32) {
            try k.setF32(i, v);
        } else try k.setU32(i, @intCast(v));
    }
    try k.launch(.{ @intCast(groups[0]), @intCast(groups[1]), @intCast(groups[2]) });
}

pub const Tower = struct {
    r: *rt.Runtime,
    md: rt.Module,
    gemm: rt.Kernel,
    conv_k: rt.Kernel,
    flat: rt.Kernel,
    ln: rt.Kernel,
    gelu_k: rt.Kernel,
    attn: rt.Kernel,
    zero: u64, // a missing bias or residual
    sink: ?Dump = null,

    pub fn init(r: *rt.Runtime, st: *store.Store) !Tower {
        if (!xpu.kernels.available) return error.BuiltWithoutKernels;
        var md = try r.module(xpu.kernels.qwen3_asr);
        errdefer md.deinit();
        return .{
            .r = r,
            .md = md,
            .gemm = try md.kernel("asr_gemm", .{ 16, 16, 1 }),
            .conv_k = try md.kernel("asr_conv", .{ 16, 16, 1 }),
            .flat = try md.kernel("asr_flatten", .{ 256, 1, 1 }),
            .ln = try md.kernel("asr_layernorm", .{ 256, 1, 1 }),
            .gelu_k = try md.kernel("asr_gelu", .{ 256, 1, 1 }),
            .attn = try md.kernel("asr_attn_full", .{ 16, 1, 1 }),
            .zero = try st.alloc(256),
        };
    }

    pub fn deinit(t: *Tower) void {
        inline for (.{ &t.gemm, &t.conv_k, &t.flat, &t.ln, &t.gelu_k, &t.attn }) |k| k.deinit();
        t.md.deinit();
    }

    pub fn upload(t: *Tower, dst: u64, bytes: []const u8) !void {
        try t.r.upload(store.view(t.r, dst, bytes.len), bytes);
    }

    /// y [n, fo, to, co] = bf16(taps(x [n, fi, ti, c]) . w^T + b), a 3x3 stride-2 conv with zero padding 1.
    pub fn conv(t: *Tower, w: u64, b: u64, x: u64, y: u64, n: usize, fi: usize, ti: usize, c: usize, co: usize) !void {
        const m = n * ((fi - 1) / 2 + 1) * ((ti - 1) / 2 + 1);
        try run(t.r, &t.conv_k, .{ (co + 63) / 64, (m + 63) / 64, 1 }, .{ x, w, b, y, m, co, 9 * c, fi, ti, c });
    }

    pub fn dump(t: *Tower, stage: []const u8, ptr: u64, rows: usize, cols: usize, conv_shape: ?[3]usize) !void {
        if (t.sink) |dm| try dm.put(dm.ctx, stage, ptr, rows, cols, conv_shape);
    }

    pub fn flatten(t: *Tower, x: u64, y: u64, n: usize, f: usize, tt: usize, c: usize) !void {
        const total = n * tt * c * f;
        try run(t.r, &t.flat, .{ (total + 255) / 256, 1, 1 }, .{ x, y, f, tt, c, total });
    }

    /// y = bf16(act(bf16(x . w^T + b)) + res); a missing bias is none, `res` may be y itself.
    pub fn linear(t: *Tower, x: u64, w: u64, b: ?u64, y: u64, rows: usize, n: usize, k: usize, act: shared.Act, res: ?u64) !void {
        const a: u32 = switch (act) {
            .none => 0,
            .gelu => 1,
        };
        try run(t.r, &t.gemm, .{ (n + 63) / 64, (rows + 63) / 64, 1 }, .{ x, w, b orelse t.zero, @as(u32, @intFromBool(b != null)), res orelse t.zero, @as(u32, @intFromBool(res != null)), y, rows, n, k, n, a });
    }

    pub fn layernorm(t: *Tower, x: u64, w: u64, b: u64, y: u64, rows: usize, d: usize) !void {
        try run(t.r, &t.ln, .{ rows, 1, 1 }, .{ x, w, b, y, d, @as(f32, 1e-5) });
    }

    /// Bidirectional attention over one window of `len` rows of [q | k | v] (heads x 64 each).
    pub fn attention(t: *Tower, qkv: u64, out: u64, len: usize, heads: usize, dim: usize) !void {
        if (dim != 64) return error.UnsupportedHeadDim;
        try run(t.r, &t.attn, .{ len, heads, 1 }, .{ qkv, out, len, heads, @as(f32, 0.125) });
    }

    pub fn gelu(t: *Tower, x: u64, y: u64, count: usize) !void {
        try run(t.r, &t.gelu_k, .{ (count + 255) / 256, 1, 1 }, .{ x, y, count });
    }

    pub fn sync(t: *Tower) !void {
        try t.r.sync();
    }
};
