//! The decoders' own CUDA kernels (idol_decoder.cu); their projections, norms and self attention are idol.cu's.
const std = @import("std");
const cuda = @import("cuda");

pub const hidden = 1024;
pub const heads = 16;
pub const head_dim = 64;
/// The latent the cross attention reads: 8,192 rows of 16 channels.
pub const context_rows = 8192;

/// Query rows the fp32 short form takes: torch on MPS (the cases' SDK) runs its fp32 vector kernel up to 8.
pub const few_rows = 8;
/// The short form's keys: 256 threads of 32 each.
pub const few_keys = 256 * 32;

/// The tiled attention's geometry: idol.cu's (8 warps of 16 query rows, two buffers of 64 K and V rows).
pub const attention_warps = 8;
pub const attention_smem: u32 = 2 * 2 * 64 * (head_dim / 8 + 1) * 16;

pub const Kernels = struct {
    mod: cuda.Module,
    embed: cuda.Function,
    half: cuda.Function,
    kv: cuda.Function,
    heads: cuda.Function,
    attention: cuda.Function,
    few: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.idol_decoder);
        errdefer mod.unload();
        var k: Kernels = undefined;
        k.mod = mod;
        inline for (.{ "embed", "half", "kv", "heads", "attention", "few" }) |n|
            @field(k, n) = try mod.function("tf_idol_dec_" ++ n);
        return k;
    }

    pub fn unload(k: *Kernels) void {
        k.mod.unload();
    }
};

fn int(x: usize) c_int {
    return @intCast(x);
}

fn ceil(n: usize, d: usize) u32 {
    return @intCast((n + d - 1) / d);
}

fn go(f: cuda.Function, s: cuda.Stream, grid: cuda.Dim3, a: *cuda.Args) !void {
    try cuda.launch.launch(f, .{ .grid = grid, .block = .{ .x = 256 } }, s, a);
}

fn addAll(a: *cuda.Args, ptrs: []const u64) void {
    for (ptrs) |p| a.add(p);
}

/// h [rows, 1024] fp32 = in_proj(p) + the points' sinusoidal embedding, p [rows, 3] fp32, frequencies [170].
pub fn embed(k: *const Kernels, s: cuda.Stream, p: u64, w: u64, b: u64, freqs: u64, h: u64, rows: usize) !void {
    var a: cuda.Args = .{};
    addAll(&a, &.{ p, w, b, freqs, h });
    a.add(int(rows));
    try go(k.embed, s, .{ .x = @intCast(rows), .y = ceil(hidden, 256) }, &a);
}

/// o [rows, width] fp16 = y + b by column, y and b fp32.
pub fn toHalf(k: *const Kernels, s: cuda.Stream, y: u64, b: u64, o: u64, rows: usize, width: usize) !void {
    var a: cuda.Args = .{};
    addAll(&a, &.{ y, b, o });
    for ([_]usize{ width, rows }) |v| a.add(int(v));
    try go(k.half, s, .{ .x = @intCast(rows), .y = ceil(width, 256) }, &a);
}

/// kv [rows, 2048] fp16 = to_kv of the fp16 latent rows [rows, 16] (fp16 weights and bias, fp32 sums).
pub fn kv(k: *const Kernels, s: cuda.Stream, ctx: u64, w: u64, b: u64, out: u64, rows: usize) !void {
    var a: cuda.Args = .{};
    addAll(&a, &.{ ctx, w, b, out });
    a.add(int(rows));
    try go(k.kv, s, .{ .x = @intCast(rows), .y = ceil(2 * hidden, 256) }, &a);
}

/// The 16 heads at column `offset` of rows `stride` apart RMS-normed by gamma [16, 64] times 8; dst may be src.
pub fn rmsHeads(k: *const Kernels, s: cuda.Stream, src: u64, gamma: u64, dst: u64, rows: usize, stride: usize, offset: usize) !void {
    var a: cuda.Args = .{};
    addAll(&a, &.{ src, gamma, dst });
    for ([_]usize{ stride, offset, rows }) |v| a.add(int(v));
    try go(k.heads, s, .{ .x = @intCast(rows), .y = ceil(heads, 8) }, &a);
}

/// The attention's forms: fp32 throughout for up to `few_rows` query rows, else tiled (P rounded to fp16).
pub const Form = enum { tiled, few };

pub fn form(lq: usize) Form {
    return if (lq <= few_rows) .few else .tiled;
}

/// o [lq, 1024] = softmax(q k^T scale) v: q rows `qs` apart, k rows `ks` apart in `kvp` with v at column 1024.
pub fn attention(k: *const Kernels, s: cuda.Stream, q: u64, qs: usize, kvp: u64, ks: usize, o: u64, lq: usize, lk: usize, scale: f32) !void {
    return attentionAs(k, s, form(lq), q, qs, kvp, ks, o, lq, lk, scale);
}

/// The attention in the form asked (the contract checks each at shapes the decoders never call).
pub fn attentionAs(k: *const Kernels, s: cuda.Stream, f: Form, q: u64, qs: usize, kvp: u64, ks: usize, o: u64, lq: usize, lk: usize, scale: f32) !void {
    if (lq == 0 or lk == 0) return;
    var a: cuda.Args = .{};
    a.add(q);
    a.add(int(qs));
    a.add(kvp);
    a.add(int(ks));
    a.add(o);
    for ([_]usize{ lq, lk, heads }) |v| a.add(int(v));
    a.add(scale);
    switch (f) {
        .few => {
            if (lq > few_rows or lk > few_keys) return error.ShortFormShape;
            try go(k.few, s, .{ .x = @intCast(lq), .y = heads }, &a);
        },
        .tiled => {
            const grid: cuda.Dim3 = .{ .x = ceil(lq, 16 * attention_warps), .y = heads };
            try cuda.launch.launch(k.attention, .{ .grid = grid, .block = .{ .x = 32 * attention_warps }, .shared = attention_smem }, s, &a);
        },
    }
}

test "launch geometry: points on grid x, 256-column strips, 8 heads a block, attention's tile under 48 KiB" {
    try std.testing.expectEqual(@as(u32, 4), ceil(hidden, 256));
    try std.testing.expectEqual(@as(u32, 8), ceil(2 * hidden, 256));
    try std.testing.expectEqual(@as(u32, 2), ceil(heads, 8));
    try std.testing.expectEqual(@as(u32, 256), ceil(32768, 16 * attention_warps));
    try std.testing.expectEqual(@as(u32, 36864), attention_smem);
    try std.testing.expect(attention_smem <= 48 * 1024);
    try std.testing.expect(form(1) == .few and form(8) == .few and form(9) == .tiled);
    try std.testing.expect(context_rows <= few_keys);
}
