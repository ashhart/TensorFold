//! Host-only LFM3 text tensor admission; no weights are uploaded and a vision tower is not qualified here.
const std = @import("std");
const st = @import("safetensors");
const c = @import("config.zig");
const Allocator = std.mem.Allocator;

pub const Namespace = enum {
    bare,
    wrapped,

    pub fn prefix(n: Namespace) []const u8 {
        return if (n == .bare) "model." else "language_model.model.";
    }
};

pub const Expected = struct {
    name: []const u8,
    dtype: st.DType,
    rank: u8,
    shape: [st.max_rank]usize,
};

pub const Manifest = struct { namespace: Namespace, tensors: usize, bytes: usize };

fn dense(a: Allocator, out: *std.ArrayList(Expected), name: []const u8, shape: []const usize, dtype: st.DType) !void {
    var e: Expected = .{ .name = try a.dupe(u8, name), .dtype = dtype, .rank = @intCast(shape.len), .shape = @splat(1) };
    @memcpy(e.shape[0..shape.len], shape);
    try out.append(a, e);
}

fn named(a: Allocator, out: *std.ArrayList(Expected), prefix: []const u8, suffix: []const u8, shape: []const usize) !void {
    var buf: [256]u8 = undefined;
    try dense(a, out, try std.fmt.bufPrint(&buf, "{s}{s}", .{ prefix, suffix }), shape, .bf16);
}

fn affine(a: Allocator, out: *std.ArrayList(Expected), cfg: *const c.Config, prefix: []const u8, suffix: []const u8, n: usize, k: usize) !void {
    if (cfg.scheme == .modelopt_nvfp4) return modelopt(a, out, prefix, suffix, n, k);
    if (cfg.scheme == .fp8_block) return fp8Block(a, out, prefix, suffix, n, k);
    var buf: [256]u8 = undefined;
    const module = try std.fmt.bufPrint(&buf, "{s}{s}", .{ prefix, suffix });
    const q = try cfg.width(module);
    if (k % q.group != 0 or k * q.bits % 32 != 0) return error.UnsupportedQuantization;
    var name: [272]u8 = undefined;
    try dense(a, out, try std.fmt.bufPrint(&name, "{s}.weight", .{module}), &.{ n, k * q.bits / 32 }, .u32);
    try dense(a, out, try std.fmt.bufPrint(&name, "{s}.scales", .{module}), &.{ n, k / q.group }, .bf16);
    try dense(a, out, try std.fmt.bufPrint(&name, "{s}.biases", .{module}), &.{ n, k / q.group }, .bf16);
}

/// ModelOpt NVFP4: the FFN's codes [n, k / 2], e4m3 scales [n, k / 16], fp32 scalar scales; the rest bf16 [n, k].
fn modelopt(a: Allocator, out: *std.ArrayList(Expected), prefix: []const u8, suffix: []const u8, n: usize, k: usize) !void {
    var name: [272]u8 = undefined;
    if (!std.mem.startsWith(u8, suffix, "feed_forward.")) return dense(a, out, try std.fmt.bufPrint(&name, "{s}{s}.weight", .{ prefix, suffix }), &.{ n, k }, .bf16);
    if (k % 16 != 0) return error.UnsupportedQuantization;
    try dense(a, out, try std.fmt.bufPrint(&name, "{s}{s}.weight", .{ prefix, suffix }), &.{ n, k / 2 }, .u8);
    try dense(a, out, try std.fmt.bufPrint(&name, "{s}{s}.weight_scale", .{ prefix, suffix }), &.{ n, k / 16 }, .f8_e4m3);
    try dense(a, out, try std.fmt.bufPrint(&name, "{s}{s}.weight_scale_2", .{ prefix, suffix }), &.{}, .f32);
    try dense(a, out, try std.fmt.bufPrint(&name, "{s}{s}.input_scale", .{ prefix, suffix }), &.{}, .f32);
}

/// Block FP8: q/k/v/o and the FFN as e4m3 [n, k] with bf16 scales [n / 128, k / 128]; the rest bf16 [n, k].
fn fp8Block(a: Allocator, out: *std.ArrayList(Expected), prefix: []const u8, suffix: []const u8, n: usize, k: usize) !void {
    var name: [272]u8 = undefined;
    const quantized = std.mem.startsWith(u8, suffix, "feed_forward.") or
        (std.mem.startsWith(u8, suffix, "self_attn.") and !std.mem.eql(u8, suffix, "self_attn.gate_proj"));
    if (!quantized) return dense(a, out, try std.fmt.bufPrint(&name, "{s}{s}.weight", .{ prefix, suffix }), &.{ n, k }, .bf16);
    const b = c.fp8_block;
    if (n % b != 0 or k % b != 0) return error.UnsupportedQuantization;
    try dense(a, out, try std.fmt.bufPrint(&name, "{s}{s}.weight", .{ prefix, suffix }), &.{ n, k }, .f8_e4m3);
    try dense(a, out, try std.fmt.bufPrint(&name, "{s}{s}.weight_scale", .{ prefix, suffix }), &.{ n / b, k / b }, .bf16);
}

/// Expected metadata for one text decoder; allocate the returned names and list in a caller-owned arena.
pub fn schema(a: Allocator, cfg: *const c.Config, namespace: Namespace) ![]Expected {
    var out: std.ArrayList(Expected) = .empty;
    const prefix = namespace.prefix();
    try affine(a, &out, cfg, prefix, "embed_tokens", c.vocab, c.hidden);
    try named(a, &out, prefix, "embedding_norm.weight", &.{c.hidden});
    for (cfg.layer_types, 0..) |kind, i| {
        var buf: [128]u8 = undefined;
        const p = try std.fmt.bufPrint(&buf, "{s}layers.{d}.", .{ prefix, i });
        for ([_][]const u8{ "operator_norm.weight", "ffn_norm.weight", "post_operator_norm.weight", "post_ffn_norm.weight" }) |n|
            try named(a, &out, p, n, &.{c.hidden});
        try affine(a, &out, cfg, p, "feed_forward.w1", c.intermediate, c.hidden);
        try affine(a, &out, cfg, p, "feed_forward.w3", c.intermediate, c.hidden);
        try affine(a, &out, cfg, p, "feed_forward.w2", c.hidden, c.intermediate);
        if (kind == .conv) {
            try affine(a, &out, cfg, p, "conv.in_proj", 3 * c.hidden, c.hidden);
            try affine(a, &out, cfg, p, "conv.out_proj", c.hidden, c.hidden);
            // MLX keeps the depthwise kernel as [channels, taps, 1], torch as [channels, 1, taps]: same values
            const taps: []const usize = if (cfg.scheme != .mlx_affine) &.{ c.hidden, 1, c.conv_taps } else &.{ c.hidden, c.conv_taps, 1 };
            try named(a, &out, p, "conv.conv.weight", taps);
        } else {
            try affine(a, &out, cfg, p, "self_attn.q_proj", c.query_heads * c.head_dim, c.hidden);
            try affine(a, &out, cfg, p, "self_attn.k_proj", c.kv_heads * c.head_dim, c.hidden);
            try affine(a, &out, cfg, p, "self_attn.v_proj", c.kv_heads * c.head_dim, c.hidden);
            try affine(a, &out, cfg, p, "self_attn.o_proj", c.hidden, c.query_heads * c.head_dim);
            try affine(a, &out, cfg, p, "self_attn.gate_proj", c.query_heads, c.hidden);
            try named(a, &out, p, "self_attn.q_norm.weight", &.{c.head_dim});
            try named(a, &out, p, "self_attn.k_norm.weight", &.{c.head_dim});
        }
    }
    return out.toOwnedSlice(a);
}

/// Validate a merged safetensors header; the shared parser must have checked shard byte ranges first.
pub fn validate(a: Allocator, cfg: *const c.Config, header: *const st.Header) !Manifest {
    const bare = header.contains("model.embed_tokens.weight");
    const wrapped = header.contains("language_model.model.embed_tokens.weight");
    if (bare and wrapped) return error.MixedNamespaces;
    if (!bare and !wrapped) return error.MissingTensor;
    const namespace: Namespace = if (bare) .bare else .wrapped;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const expected = try schema(arena.allocator(), cfg, namespace);
    var names: std.StringHashMapUnmanaged(void) = .empty;
    var bytes: usize = 0;
    for (expected) |e| {
        const actual = header.get(e.name) orelse return error.MissingTensor;
        if (actual.dtype != e.dtype) return error.TensorDType;
        if (actual.rank != e.rank or !std.mem.eql(usize, actual.shape[0..actual.rank], e.shape[0..e.rank])) return error.TensorShape;
        var size: usize = e.dtype.size();
        for (e.shape[0..e.rank]) |dim| size = std.math.mul(usize, size, dim) catch return error.TensorBytes;
        if (actual.end < actual.begin or actual.end - actual.begin != size) return error.TensorBytes;
        bytes = std.math.add(usize, bytes, size) catch return error.TensorBytes;
        try names.put(arena.allocator(), e.name, {});
    }
    for (header.keys()) |name| {
        if (std.mem.startsWith(u8, name, "model.") or std.mem.startsWith(u8, name, "language_model.")) {
            if (!names.contains(name)) return error.UnexpectedTensor;
        }
    }
    return .{ .namespace = namespace, .tensors = expected.len, .bytes = bytes };
}

test {
    _ = @import("weights_test.zig");
}
