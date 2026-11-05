//! The audio tower's binding on CUDA: vit.cu's conv taps, LayerNorm and pointwise kernels, cuBLASLt, attention at 64.
const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("../config.zig");
const chunks = @import("../chunks.zig");
const shared = @import("../tower.zig");
const store = @import("store.zig");
const vit = cuda.vit;

/// Where a stage's device rows go when a check asks for them: bf16 `rows` x `cols` at `ptr`, channels last for convs.
pub const Dump = struct {
    ctx: *anyopaque,
    put: *const fn (ctx: *anyopaque, stage: []const u8, ptr: u64, rows: usize, cols: usize, conv: ?[3]usize) anyerror!void,
};

pub const Tower = struct {
    d: *const cuda.Driver,
    s: cuda.Stream,
    vk: vit.Kernels,
    g: vit.Gemms,
    att: vit.Attention,
    cols: u64, // a conv's taps
    tmp: u64, // a fused GEMM's bf16 output before its act and residual
    acc: u64, // an unfused GEMM's f32 sums
    zero: u64, // the bias of a Linear without one
    sink: ?Dump = null,

    /// Kernels and scratch for `group` chunks a conv pass and up to `max_rows` rows.
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, s: cuda.Stream, a: cfg.Audio, group: usize, max_rows: usize, st: *store.Store) !Tower {
        var vk = try vit.Kernels.load(d);
        errdefer vk.unload();
        var g = try vit.Gemms.init(gpa, d);
        errdefer g.deinit();
        var att = try vit.Attention.load(d);
        errdefer att.unload();
        const cf = a.chunkFrames();
        const p1 = group * ((a.mels - 1) / 2 + 1) * ((cf - 1) / 2 + 1); // conv1's output positions a group
        const rows = @max(max_rows, group) + chunks.rows(cf);
        const most = @max(p1 * a.conv, rows * @max(a.ffn, 3 * a.d, a.out));
        const wide = @max(a.d, a.out, a.ffn, a.conv);
        const zero = try st.alloc(2 * wide);
        try store.view(d, zero, 2 * wide).fill8(0, null);
        return .{
            .d = d,
            .s = s,
            .vk = vk,
            .g = g,
            .att = att,
            .cols = try st.alloc(2 * @max(p1 * store.tapsLd(9), (p1 / 4 + 1) * store.tapsLd(9 * a.conv))),
            .tmp = try st.alloc(2 * most),
            .acc = try st.alloc(4 * most),
            .zero = zero,
        };
    }

    pub fn deinit(t: *Tower) void {
        t.att.unload();
        t.g.deinit();
        t.vk.unload();
    }

    pub fn upload(t: *Tower, dst: u64, bytes: []const u8) !void {
        try store.view(t.d, dst, bytes.len).uploadAsync(0, bytes, t.s.handle);
    }

    /// y [n, fo, to, co] = bf16(taps(x [n, fi, ti, c]) . w^T + b), a 3x3 stride-2 conv with zero padding 1.
    pub fn conv(t: *Tower, w: u64, b: u64, x: u64, y: u64, n: usize, fi: usize, ti: usize, c: usize, co: usize) !void {
        const ld = store.tapsLd(9 * c);
        try t.vk.convCols(t.s, x, t.cols, n, fi, ti, c, ld);
        const rows = n * ((fi - 1) / 2 + 1) * ((ti - 1) / 2 + 1);
        try vit.linear(&t.g, &t.vk, t.s, t.acc, t.cols, w, b, y, rows, co, ld);
    }

    pub fn dump(t: *Tower, stage: []const u8, ptr: u64, rows: usize, cols: usize, conv_shape: ?[3]usize) !void {
        if (t.sink) |dm| try dm.put(dm.ctx, stage, ptr, rows, cols, conv_shape);
    }

    pub fn flatten(t: *Tower, x: u64, y: u64, n: usize, f: usize, tt: usize, c: usize) !void {
        try t.vk.convFlatten(t.s, x, y, n, f, tt, c);
    }

    /// y = bf16(act(bf16(x . w^T + b)) + res); a missing bias is zeros.
    pub fn linear(t: *Tower, x: u64, w: u64, b: ?u64, y: u64, rows: usize, n: usize, k: usize, act: shared.Act, res: ?u64) !void {
        const bias = b orelse t.zero;
        if (act == .none and res == null) return vit.linear(&t.g, &t.vk, t.s, t.acc, x, w, bias, y, rows, n, k);
        const va: vit.Act = switch (act) {
            .none => .none,
            .gelu => .gelu,
        };
        try vit.linearAct(&t.g, &t.vk, t.s, t.acc, x, w, bias, t.tmp, rows, n, k, res, y, va);
    }

    pub fn layernorm(t: *Tower, x: u64, w: u64, b: u64, y: u64, rows: usize, d: usize) !void {
        try t.vk.layerNorm(t.s, x, w, b, y, rows, d, 1e-5);
    }

    /// Bidirectional attention over one window of `len` rows of [q | k | v] (heads x dim each).
    pub fn attention(t: *Tower, qkv: u64, out: u64, len: usize, heads: usize, dim: usize) !void {
        const scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(dim))));
        try t.att.full(t.s, qkv, out, len, heads, dim, scale);
    }

    pub fn gelu(t: *Tower, x: u64, y: u64, count: usize) !void {
        try t.vk.pointwise(t.s, x, null, y, count, .gelu);
    }

    pub fn sync(t: *Tower) !void {
        try t.s.synchronize();
    }
};
