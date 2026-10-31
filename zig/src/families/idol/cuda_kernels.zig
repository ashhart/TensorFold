//! Idol's own CUDA kernels (idol.cu): LayerNorms, RePo's angles, q/k turned and normed, epilogues, head-64 attention.
const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("config.zig");

fn int(x: usize) c_int {
    return @intCast(x);
}

fn ceil(n: usize, d: usize) u32 {
    return @intCast((n + d - 1) / d);
}

/// The LayerNorm's form: plain, affine (fp32 weight and bias), or modulated (after the fp16 rounding).
pub const Norm = enum(c_int) { plain = 0, affine = 1, modulated = 2 };

/// The attention's tile: 8 warps of 16 query rows a block, two buffers of 64 K and V rows padded to 9 chunks of 16 B.
pub const attention_warps = 8;
pub const attention_smem: u32 = 2 * 2 * 64 * (cfg.head_dim / 8 + 1) * 16;

pub const Kernels = struct {
    mod: cuda.Module,
    norm: cuda.Function,
    norm32: cuda.Function,
    repo: cuda.Function,
    qk: cuda.Function,
    gate_add: cuda.Function,
    bias: cuda.Function,
    embed: cuda.Function,
    cond: cuda.Function,
    bias32: cuda.Function,
    attention: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.idol);
        errdefer mod.unload();
        var k: Kernels = undefined;
        k.mod = mod;
        inline for (.{ "norm", "norm32", "repo", "qk", "gate_add", "bias", "embed", "cond", "bias32", "attention" }) |n|
            @field(k, n) = try mod.function("tf_idol_" ++ n);
        return k;
    }

    pub fn unload(k: *Kernels) void {
        k.mod.unload();
    }
};

fn go(f: cuda.Function, s: cuda.Stream, grid: cuda.Dim3, threads: u32, a: *cuda.Args) !void {
    try cuda.launch.launch(f, .{ .grid = grid, .block = .{ .x = threads } }, s, a);
}

fn addAll(a: *cuda.Args, ptrs: []const u64) void {
    for (ptrs) |p| a.add(p);
}

/// y fp16 = the LayerNorm32 of x's `rows` rows, rounded to fp16: plain, affine (w, b fp32) or modulated (s1, sh fp16).
pub fn norm(k: *const Kernels, s: cuda.Stream, x: u64, w: u64, b: u64, s1: u64, sh: u64, y: u64, rows: usize, eps: f32, mode: Norm) !void {
    var a: cuda.Args = .{};
    addAll(&a, &.{ x, w, b, s1, sh, y });
    a.add(int(cfg.hidden));
    a.add(eps);
    a.add(@backingInt(mode));
    try go(k.norm, s, .{ .x = @intCast(rows) }, 256, &a);
}

/// y fp32 = the LayerNorm of x's rows times w plus b (affine), or times (1 + w) plus b (modulated).
pub fn norm32(k: *const Kernels, s: cuda.Stream, x: u64, w: u64, b: u64, y: u64, rows: usize, eps: f32, mode: Norm) !void {
    var a: cuda.Args = .{};
    addAll(&a, &.{ x, w, b, y });
    a.add(int(cfg.hidden));
    a.add(eps);
    a.add(@backingInt(mode));
    try go(k.norm32, s, .{ .x = @intCast(rows) }, 256, &a);
}

/// cos, sin [rows, 16, 32] fp32 of RePo from its gate | content sums [rows, 256] fp32, final map and frequencies.
pub fn repo(k: *const Kernels, s: cuda.Stream, gc: u64, final: u64, freqs: u64, cos: u64, sin: u64, rows: usize) !void {
    var a: cuda.Args = .{};
    addAll(&a, &.{ gc, final, freqs, cos, sin });
    a.add(int(rows));
    try go(k.repo, s, .{ .x = @intCast(rows) }, 64, &a);
}

/// Attention's packed rows [rows, q | k | v] from qkv rows: q, k turned by cos, sin, normed by gq, gk times 8.
pub fn qk(k: *const Kernels, s: cuda.Stream, qkv: u64, gq: u64, gk: u64, cos: u64, sin: u64, out: u64, rows: usize) !void {
    var a: cuda.Args = .{};
    addAll(&a, &.{ qkv, gq, gk, cos, sin, out });
    a.add(int(rows));
    a.add(@as(f32, @sqrt(@as(f32, cfg.head_dim))));
    try go(k.qk, s, .{ .x = @intCast(rows), .y = ceil(cfg.heads, 8) }, 256, &a);
}

/// x [rows, width] += y gated by the fp16 vector g, in place.
pub fn gateAdd(k: *const Kernels, s: cuda.Stream, x: u64, y: u64, g: u64, rows: usize, width: usize) !void {
    var a: cuda.Args = .{};
    addAll(&a, &.{ x, y, g });
    for ([_]usize{ width, rows }) |v| a.add(int(v));
    try go(k.gate_add, s, .{ .x = @intCast(rows), .y = ceil(width, 256) }, 256, &a);
}

/// y fp16 = fp16(acc + b) from fp32 sums, then fp16(GELU) when `gelu`.
pub fn bias(k: *const Kernels, s: cuda.Stream, acc: u64, b: u64, y: u64, rows: usize, width: usize, gelu: bool) !void {
    var a: cuda.Args = .{};
    addAll(&a, &.{ acc, b, y });
    for ([_]usize{ width, rows }) |v| a.add(int(v));
    a.add(@as(c_int, @intFromBool(gelu)));
    try go(k.bias, s, .{ .x = @intCast(rows), .y = ceil(width, 256) }, 256, &a);
}

/// x [rows, 1024] = fp16(fp16(z w^T + b) + pe), z [rows, 16] fp32.
pub fn embed(k: *const Kernels, s: cuda.Stream, z: u64, w: u64, b: u64, pe: u64, x: u64, rows: usize) !void {
    var a: cuda.Args = .{};
    addAll(&a, &.{ z, w, b, pe, x });
    a.add(int(rows));
    try go(k.embed, s, .{ .x = @intCast(rows), .y = ceil(cfg.hidden, 256) }, 256, &a);
}

/// y [rows, width] = fp16((a + ba) + (c + bc)).
pub fn cond(k: *const Kernels, s: cuda.Stream, a_: u64, ba: u64, c: u64, bc: u64, y: u64, rows: usize, width: usize) !void {
    var a: cuda.Args = .{};
    addAll(&a, &.{ a_, ba, c, bc, y });
    for ([_]usize{ width, rows }) |v| a.add(int(v));
    try go(k.cond, s, .{ .x = @intCast(rows), .y = ceil(width, 256) }, 256, &a);
}

/// y [rows, width] += b by column, fp32.
pub fn bias32(k: *const Kernels, s: cuda.Stream, y: u64, b: u64, rows: usize, width: usize) !void {
    var a: cuda.Args = .{};
    addAll(&a, &.{ y, b });
    for ([_]usize{ width, rows }) |v| a.add(int(v));
    try go(k.bias32, s, .{ .x = @intCast(rows), .y = ceil(width, 256) }, 256, &a);
}

/// out [rows, heads 64] = softmax(q k^T scale) v of packed rows [rows, q | k | v heads 64], fp16, every key.
pub fn attention(k: *const Kernels, s: cuda.Stream, qkv: u64, out: u64, rows: usize, heads: usize, scale: f32) !void {
    if (rows == 0) return;
    var a: cuda.Args = .{};
    addAll(&a, &.{ qkv, out });
    for ([_]usize{ rows, heads }) |v| a.add(int(v));
    a.add(scale);
    const grid: cuda.Dim3 = .{ .x = ceil(rows, 16 * attention_warps), .y = @intCast(heads) };
    try cuda.launch.launch(k.attention, .{ .grid = grid, .block = .{ .x = 32 * attention_warps }, .shared = attention_smem }, s, &a);
}

test "launch geometry: rows on grid x, 256-column strips on y, attention's tile under 48 KiB" {
    try std.testing.expectEqual(@as(u32, 4), ceil(cfg.hidden, 256));
    try std.testing.expectEqual(@as(u32, 2), ceil(cfg.heads, 8));
    try std.testing.expectEqual(@as(u32, 97), ceil(cfg.joint_rows, 16 * attention_warps));
    try std.testing.expectEqual(@as(u32, 36864), attention_smem);
    try std.testing.expect(attention_smem <= 48 * 1024);
}
