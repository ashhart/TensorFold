//! Idol's DINOv3 kernels on CUDA (idol_dino.cu): LayerNorms, RoPE heads, the linears' bf16 epilogues, attention.
const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("dino_config.zig");

fn int(x: usize) c_int {
    return @intCast(x);
}

fn ceil(n: usize, d: usize) u32 {
    return @intCast((n + d - 1) / d);
}

/// The attention's tile, as idol_attention.cuh's full<64, 8>: 8 warps of 16 query rows, two buffers of 64 K and V rows.
pub const attention_warps = 8;
pub const attention_smem: u32 = 2 * 2 * 64 * (cfg.head_dim / 8 + 1) * 16;
/// The heads kernel's threads: one a head's pair (j, j + 32), 20 heads of 32.
pub const head_pairs = cfg.heads * 32;

pub const names = [_][]const u8{ "bias", "norm", "final", "heads", "scale_add", "swiglu", "attention" };

pub const Kernels = struct {
    mod: cuda.Module,
    bias: cuda.Function,
    norm: cuda.Function,
    final: cuda.Function,
    heads: cuda.Function,
    scale_add: cuda.Function,
    swiglu: cuda.Function,
    attention: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.idol_dino);
        errdefer mod.unload();
        var k: Kernels = undefined;
        k.mod = mod;
        inline for (names) |n| @field(k, n) = try mod.function("tf_idol_dino_" ++ n);
        return k;
    }

    pub fn unload(k: *Kernels) void {
        k.mod.unload();
    }
};

fn go(f: cuda.Function, s: cuda.Stream, grid: cuda.Dim3, a: *cuda.Args) !void {
    try cuda.launch.launch(f, .{ .grid = grid, .block = .{ .x = 256 } }, s, a);
}

fn addAll(a: *cuda.Args, ptrs: []const u64) void {
    for (ptrs) |p| a.add(p);
}

/// Rows on grid x, 256-column strips on y.
fn strips(rows: usize, width: usize) cuda.Dim3 {
    return .{ .x = @intCast(rows), .y = ceil(width, 256) };
}

/// y bf16 [rows, width] = bf16(acc + b): a linear's fp32 sums and its bf16 bias.
pub fn bias(k: *const Kernels, s: cuda.Stream, acc: u64, b: u64, y: u64, rows: usize, width: usize) !void {
    if (rows == 0) return;
    var a: cuda.Args = .{};
    addAll(&a, &.{ acc, b, y });
    for ([_]usize{ width, rows }) |v| a.add(int(v));
    try go(k.bias, s, strips(rows, width), &a);
}

/// y = nn.LayerNorm of x's bf16 rows [rows, 1280] (fp32 statistics), bf16(n w + b) with bf16 w, b.
pub fn norm(k: *const Kernels, s: cuda.Stream, x: u64, w: u64, b: u64, y: u64, rows: usize) !void {
    if (rows == 0) return;
    var a: cuda.Args = .{};
    addAll(&a, &.{ x, w, b, y });
    a.add(int(cfg.width));
    a.add(cfg.eps);
    try go(k.norm, s, .{ .x = @intCast(rows) }, &a);
}

/// h = the last norm of x's rows (bf16), y = F.layer_norm of h in fp32, no affine: ImageEncoder's feature1.
pub fn final(k: *const Kernels, s: cuda.Stream, x: u64, w: u64, b: u64, h: u64, y: u64, rows: usize) !void {
    if (rows == 0) return;
    var a: cuda.Args = .{};
    addAll(&a, &.{ x, w, b, h, y });
    a.add(int(cfg.width));
    a.add(cfg.eps);
    try go(k.final, s, .{ .x = @intCast(rows) }, &a);
}

/// out bf16 [rows, q | k | v] from the qkv linear's fp32 sums [rows, 3840]; past the prefix, q and k turned.
pub fn heads(k: *const Kernels, s: cuda.Stream, qkv: u64, cos: u64, sin: u64, out: u64, rows: usize) !void {
    if (rows == 0) return;
    var a: cuda.Args = .{};
    addAll(&a, &.{ qkv, cos, sin, out });
    for ([_]usize{ rows, cfg.prefix }) |v| a.add(int(v));
    try go(k.heads, s, .{ .x = @intCast(rows), .y = ceil(head_pairs, 256) }, &a);
}

/// x [rows, 1280] = bf16(x + bf16(g bf16(acc + b))), in place: a linear's bias, LayerScale, the residual.
pub fn scaleAdd(k: *const Kernels, s: cuda.Stream, x: u64, acc: u64, b: u64, g: u64, rows: usize) !void {
    if (rows == 0) return;
    var a: cuda.Args = .{};
    addAll(&a, &.{ x, acc, b, g });
    for ([_]usize{ cfg.width, rows }) |v| a.add(int(v));
    try go(k.scale_add, s, strips(rows, cfg.width), &a);
}

/// h [rows, 5120] = bf16(bf16(silu(g)) u) from fc1's fp32 sums [rows, gate | up] and its bf16 bias [gate | up].
pub fn swiglu(k: *const Kernels, s: cuda.Stream, acc: u64, b: u64, h: u64, rows: usize) !void {
    if (rows == 0) return;
    var a: cuda.Args = .{};
    addAll(&a, &.{ acc, b, h });
    for ([_]usize{ cfg.ffn, rows }) |v| a.add(int(v));
    try go(k.swiglu, s, strips(rows, cfg.ffn), &a);
}

/// out [rows, heads 64] = softmax(q k^T scale) v of packed rows [rows, q | k | v heads 64], bf16, every key.
pub fn attention(k: *const Kernels, s: cuda.Stream, qkv: u64, out: u64, rows: usize, heads_: usize, scale: f32) !void {
    if (rows == 0) return;
    var a: cuda.Args = .{};
    addAll(&a, &.{ qkv, out });
    for ([_]usize{ rows, heads_ }) |v| a.add(int(v));
    a.add(scale);
    const grid: cuda.Dim3 = .{ .x = ceil(rows, 16 * attention_warps), .y = @intCast(heads_) };
    try cuda.launch.launch(k.attention, .{ .grid = grid, .block = .{ .x = 32 * attention_warps }, .shared = attention_smem }, s, &a);
}

test "launch geometry: rows on grid x, 256-column strips on y, attention's tile under 48 KiB" {
    try std.testing.expectEqual(@as(u32, 5), ceil(cfg.width, 256));
    try std.testing.expectEqual(@as(u32, 20), ceil(cfg.ffn, 256));
    try std.testing.expectEqual(@as(u32, 3), ceil(head_pairs, 256));
    try std.testing.expectEqual(@as(u32, 33), ceil(cfg.rows, 16 * attention_warps));
    try std.testing.expectEqual(@as(u32, 36864), attention_smem);
    try std.testing.expect(attention_smem <= 48 * 1024);
    // idol_dino.cu's tf_idol_dino_final keeps a row of the width in shared memory; the heads kernel fixes the width
    try std.testing.expectEqual(@as(usize, 1280), cfg.width);
    try std.testing.expectEqual(@as(usize, 64), cfg.head_dim);
}
