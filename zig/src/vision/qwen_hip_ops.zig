//! Qwen vision's native HIP launchers: BF16 rows and losslessly widened FP32 checkpoint weights.
const std = @import("std");
const hip = @import("hip");

pub const Error = hip.Error || error{VisionShape};
pub const max_rows = 16384;

pub const Kernels = struct {
    mod: hip.Module,
    linear_fn: hip.Function,
    embed_fn: hip.Function,
    layernorm_fn: hip.Function,
    qkv_fn: hip.Function,
    attention_fn: hip.Function,
    gelu_fn: hip.Function,
    residual_fn: hip.Function,
    cast_fn: hip.Function,
    text_rope_fn: hip.Function,

    pub fn load(r: *const hip.Runtime, image: []const u8) !Kernels {
        var mod = try hip.Module.load(r, image);
        errdefer mod.unload();
        return .{
            .mod = mod,
            .linear_fn = try mod.function("tf_qvh_linear"),
            .embed_fn = try mod.function("tf_qvh_embed"),
            .layernorm_fn = try mod.function("tf_qvh_layernorm"),
            .qkv_fn = try mod.function("tf_qvh_qkv_rope"),
            .attention_fn = try mod.function("tf_qvh_attention64"),
            .gelu_fn = try mod.function("tf_qvh_bias_gelu"),
            .residual_fn = try mod.function("tf_qvh_bias_residual"),
            .cast_fn = try mod.function("tf_qvh_bias_cast"),
            .text_rope_fn = try mod.function("tf_qvh_text_rope"),
        };
    }

    pub fn deinit(k: *Kernels) void {
        k.mod.unload();
    }

    /// FP32 out = BF16 x [rows, inputs] times FP32 w [outputs, inputs] transposed.
    pub fn linear(k: *const Kernels, s: hip.Stream, x: u64, w: u64, out: u64, rows: usize, outputs: usize, inputs: usize) Error!void {
        try shape(rows, outputs);
        if (inputs == 0 or inputs > 8192 or x == 0 or w == 0 or out == 0) return error.VisionShape;
        try go(k.linear_fn, s, outputs, rows, 32, .{ x, w, out, int(rows), int(outputs), int(inputs) });
    }

    pub fn embed(k: *const Kernels, s: hip.Stream, acc: u64, bias: u64, pos: u64, idx: u64, wts: u64, out: u64, rows: usize, width: usize) Error!void {
        try shape(rows, width);
        try go(k.embed_fn, s, rows, 1, 256, .{ acc, bias, pos, idx, wts, out, int(width) });
    }

    pub fn layernorm(k: *const Kernels, s: hip.Stream, x: u64, w: u64, b: u64, out: u64, rows: usize, width: usize, eps: f32) Error!void {
        try shape(rows, width);
        if (!std.math.isFinite(eps) or eps <= 0) return error.VisionShape;
        try go(k.layernorm_fn, s, rows, 1, 256, .{ x, w, b, out, int(width), eps });
    }

    /// Split biased projection rows and apply FP32 vision RoPE; outputs are [heads, rows, 64].
    pub fn qkv(k: *const Kernels, s: hip.Stream, acc: u64, bias: u64, hw: u64, inv: u64, q: u64, key: u64, v: u64, rows: usize, heads: usize) Error!void {
        if (heads == 0 or heads > 32) return error.VisionShape;
        try shape(rows, heads * 64);
        try go(k.qkv_fn, s, rows, heads, 64, .{ acc, bias, hw, inv, q, key, v, int(rows), int(heads) });
    }

    /// One image's independent full attention; score and probability BF16 boundaries match the eager reference.
    pub fn attention(k: *const Kernels, s: hip.Stream, q: u64, key: u64, v: u64, out: u64, rows: usize, heads: usize, start: usize, count: usize) Error!void {
        if (heads == 0 or heads > 32 or count == 0 or start > rows or count > rows - start) return error.VisionShape;
        try shape(rows, heads * 64);
        try go(k.attention_fn, s, (count + 63) / 64, heads, 64, .{ q, key, v, out, int(rows), int(heads), int(start), int(count), @as(f32, 0.125) });
    }

    pub fn gelu(k: *const Kernels, s: hip.Stream, acc: u64, bias: u64, out: u64, rows: usize, width: usize, tanh_form: bool) Error!void {
        try shape(rows, width);
        try go(k.gelu_fn, s, rows, 1, 256, .{ acc, bias, out, int(width), @as(c_int, @intFromBool(tanh_form)) });
    }

    pub fn residual(k: *const Kernels, s: hip.Stream, acc: u64, bias: u64, x: u64, rows: usize, width: usize) Error!void {
        try shape(rows, width);
        try go(k.residual_fn, s, rows, 1, 256, .{ acc, bias, x, int(width) });
    }

    pub fn cast(k: *const Kernels, s: hip.Stream, acc: u64, bias: u64, out: u64, rows: usize, width: usize) Error!void {
        try shape(rows, width);
        try go(k.cast_fn, s, rows, 1, 256, .{ acc, bias, out, int(width) });
    }

    /// Qwen's 64 rotated text channels use interleaved 11/11/10 axes and Torch's BF16 branch rounding.
    pub fn textRope(k: *const Kernels, s: hip.Stream, x: hip.ops.Tensor, out: hip.ops.Tensor, s_head: usize, s_row: usize, rows: usize, heads: usize, dim: usize, rotary: usize, positions3: u64, theta: f32) Error!void {
        if (x.ptr == 0 or out.ptr == 0 or heads == 0 or heads > 32 or
            x.kind != .bf16 or (out.kind != .bf16 and out.kind != .f32) or dim != 256 or rotary != 64 or
            s_row < dim or s_head < dim or positions3 == 0 or !std.math.isFinite(theta) or theta <= 0)
            return error.VisionShape;
        try shape(rows, heads * dim);
        try go(k.text_rope_fn, s, (rows * heads * dim + 255) / 256, 1, 256, .{
            x.ptr,                      @as(c_int, @backingInt(x.kind)), out.ptr,    @as(c_int, @backingInt(out.kind)),
            @as(i64, @intCast(s_head)), @as(i64, @intCast(s_row)),       int(rows),  int(heads),
            int(dim),                   int(rotary),                     positions3, theta,
        });
    }
};

fn shape(rows: usize, width: usize) Error!void {
    if (rows == 0 or rows > max_rows or width == 0 or width > 8192) return error.VisionShape;
}

fn int(v: usize) c_int {
    return @intCast(v);
}

fn go(f: hip.Function, s: hip.Stream, gx: usize, gy: usize, threads: u32, values: anytype) hip.Error!void {
    var args: hip.launch.Args = .{};
    inline for (values) |v| try args.add(v);
    try hip.launch.launch(f, .{ .grid = .{ .x = @intCast(gx), .y = @intCast(gy) }, .block = .{ .x = threads } }, s, &args);
}

test "Qwen HIP vision shape refusals happen before device use" {
    const k: Kernels = undefined;
    const s: hip.Stream = undefined;
    try std.testing.expectError(error.VisionShape, k.linear(s, 1, 2, 3, 0, 768, 1536));
    try std.testing.expectError(error.VisionShape, k.linear(s, 1, 2, 3, 1, 0, 1536));
    try std.testing.expectError(error.VisionShape, k.linear(s, 0, 2, 3, 1, 768, 1536));
    try std.testing.expectError(error.VisionShape, k.qkv(s, 1, 2, 3, 4, 5, 6, 7, 1, 0));
    try std.testing.expectError(error.VisionShape, k.qkv(s, 1, 2, 3, 4, 5, 6, 7, 1, std.math.maxInt(usize)));
    try std.testing.expectError(error.VisionShape, k.attention(s, 1, 2, 3, 4, 8, 12, 7, 2));
    try std.testing.expectError(error.VisionShape, k.textRope(s, .{ .ptr = 1, .kind = .bf16 }, .{ .ptr = 2, .kind = .bf16 }, 256, 256, 1, 1, 256, 64, 0, 10000));
}
