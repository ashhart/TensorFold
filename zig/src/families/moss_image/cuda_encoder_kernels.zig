//! MOSS-Image's prompt encoder kernels (moss_image_encoder.cu): their launches, geometry independent.
const std = @import("std");
const cuda = @import("cuda");

fn int(x: usize) c_int {
    return @intCast(x);
}

fn ceil(n: usize, d: usize) u32 {
    return @intCast((n + d - 1) / d);
}

pub const Kernels = struct {
    mod: cuda.Module,
    norm: cuda.Function,
    head: cuda.Function,
    attention: cuda.Function,
    gate: cuda.Function,
    silu_mul: cuda.Function,
    conv: cuda.Function,
    gdn_prep: cuda.Function,
    gdn: cuda.Function,
    gated_norm: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.moss_image_encoder);
        errdefer mod.unload();
        return .{
            .mod = mod,
            .norm = try mod.function("tf_me_norm"),
            .head = try mod.function("tf_me_head"),
            .attention = try mod.function("tf_me_attention"),
            .gate = try mod.function("tf_me_gate"),
            .silu_mul = try mod.function("tf_me_silu_mul"),
            .conv = try mod.function("tf_me_conv"),
            .gdn_prep = try mod.function("tf_me_gdn_prep"),
            .gdn = try mod.function("tf_me_gdn"),
            .gated_norm = try mod.function("tf_me_gated_norm"),
        };
    }

    pub fn unload(k: *Kernels) void {
        k.mod.unload();
    }
};

fn go(f: cuda.Function, s: cuda.Stream, grid: cuda.launch.Dim3, threads: u32, a: *cuda.Args) !void {
    try cuda.launch.launch(f, .{ .grid = grid, .block = .{ .x = threads } }, s, a);
}

/// y = bf16(x rsqrt(mean x^2 + eps) (1 + w)): a block a row, rows `in_stride` and `out_stride` apart.
pub fn norm(k: *const Kernels, s: cuda.Stream, x: u64, w: u64, y: u64, rows: usize, width: usize, in_stride: usize, out_stride: usize, eps: f32) !void {
    var a: cuda.Args = .{};
    for ([_]u64{ x, w, y }) |v| a.add(v);
    for ([_]usize{ width, in_stride, out_stride }) |v| a.add(int(v));
    a.add(eps);
    try go(k.norm, s, .{ .x = @intCast(rows) }, 256, &a);
}

/// 256-wide heads to [heads, rows, 256], normed and the first 64 channels turned by cos, sin when `normed`.
pub fn head(k: *const Kernels, s: cuda.Stream, x: u64, w: u64, cos: u64, sin: u64, y: u64, rows: usize, heads: usize, in_stride: usize, head_stride: usize, eps: f32, normed: bool) !void {
    var a: cuda.Args = .{};
    for ([_]u64{ x, w, cos, sin, y }) |v| a.add(v);
    for ([_]usize{ rows, heads, in_stride, head_stride }) |v| a.add(int(v));
    a.add(eps);
    a.add(@as(c_int, @intFromBool(normed)));
    try go(k.head, s, .{ .x = @intCast(rows), .y = ceil(heads, 8) }, 256, &a);
}

/// o [rows, heads * 256] = causal softmax(q k^T scale) v, q, k, v packed [heads, rows, 256].
pub fn attention(k: *const Kernels, s: cuda.Stream, q: u64, kb: u64, vb: u64, o: u64, rows: usize, heads: usize, kv_heads: usize, scale: f32) !void {
    var a: cuda.Args = .{};
    for ([_]u64{ q, kb, vb, o }) |v| a.add(v);
    for ([_]usize{ rows, heads, kv_heads }) |v| a.add(int(v));
    a.add(scale);
    try go(k.attention, s, .{ .x = @intCast(rows), .y = @intCast(heads) }, 256, &a);
}

/// o = bf16(a bf16(sigmoid(g))) where g is the second half of each 512-wide q_proj head: rows [rows, width].
pub fn gate(k: *const Kernels, s: cuda.Stream, a: u64, qg: u64, o: u64, rows: usize, width: usize) !void {
    var args: cuda.Args = .{};
    for ([_]u64{ a, qg, o }) |v| args.add(v);
    args.add(int(width));
    try go(k.gate, s, .{ .x = ceil(width, 256), .y = @intCast(rows) }, 256, &args);
}

/// o = bf16(bf16(silu(g)) u) of rows [rows, width].
pub fn siluMul(k: *const Kernels, s: cuda.Stream, g: u64, u: u64, o: u64, rows: usize, width: usize) !void {
    var a: cuda.Args = .{};
    for ([_]u64{ g, u, o }) |v| a.add(v);
    a.add(int(width));
    try go(k.silu_mul, s, .{ .x = ceil(width, 256), .y = @intCast(rows) }, 256, &a);
}

/// y = bf16(silu(bf16(the 4-tap causal conv of x))) by channel of rows [rows, width].
pub fn conv(k: *const Kernels, s: cuda.Stream, x: u64, w: u64, y: u64, rows: usize, width: usize) !void {
    var a: cuda.Args = .{};
    for ([_]u64{ x, w, y }) |v| a.add(v);
    a.add(int(width));
    try go(k.conv, s, .{ .x = ceil(width, 256), .y = @intCast(rows) }, 256, &a);
}

/// The DeltaNet's geometry: rows, key and value heads, the mixed row's stride, q's scale.
pub const Gdn = struct { rows: usize, kheads: usize, vheads: usize, mix_stride: usize, scale: f32 };

/// qn, kn [rows, kheads, 128] and g, beta [rows, vheads] in fp32 from the conv's rows and the a, b projections.
pub fn gdnPrep(k: *const Kernels, s: cuda.Stream, mix: u64, a: u64, b: u64, a_log: u64, dt_bias: u64, qn: u64, kn: u64, g: u64, beta: u64, p: Gdn) !void {
    var args: cuda.Args = .{};
    for ([_]u64{ mix, a, b, a_log, dt_bias, qn, kn, g, beta }) |v| args.add(v);
    for ([_]usize{ p.rows, p.kheads, p.vheads, p.mix_stride }) |v| args.add(int(v));
    args.add(p.scale);
    try go(k.gdn_prep, s, .{ .x = @intCast(p.rows), .y = ceil(p.kheads, 8) }, 256, &args);
}

/// The gated delta rule's out [rows, vheads, 128] bf16: a block a value head, a thread a value column.
pub fn gdn(k: *const Kernels, s: cuda.Stream, qn: u64, kn: u64, mix: u64, g: u64, beta: u64, out: u64, p: Gdn) !void {
    var a: cuda.Args = .{};
    for ([_]u64{ qn, kn, mix, g, beta, out }) |v| a.add(v);
    for ([_]usize{ p.rows, p.kheads, p.vheads, p.mix_stride }) |v| a.add(int(v));
    try go(k.gdn, s, .{ .x = @intCast(p.vheads) }, 128, &a);
}

/// y [rows, width] = the gated norm of x by its 128-wide heads with gate z and weight w.
pub fn gatedNorm(k: *const Kernels, s: cuda.Stream, x: u64, z: u64, w: u64, y: u64, rows: usize, width: usize, eps: f32) !void {
    var a: cuda.Args = .{};
    for ([_]u64{ x, z, w, y }) |v| a.add(v);
    a.add(int(width));
    a.add(eps);
    try go(k.gated_norm, s, .{ .x = @intCast(rows), .y = ceil(width / 128, 8) }, 256, &a);
}

test "launch geometry: heads in eights, 256-wide strips on y, a value head a block" {
    try std.testing.expectEqual(@as(u32, 2), ceil(16, 8));
    try std.testing.expectEqual(@as(u32, 4), ceil(32, 8));
    try std.testing.expectEqual(@as(u32, 32), ceil(8192, 256));
    try std.testing.expectEqual(@as(u32, 16), ceil(4096, 256));
    try std.testing.expectEqual(@as(u32, 4), ceil(4096 / 128, 8));
    try std.testing.expectEqual(@as(u32, 1), ceil(1, 256));
}
