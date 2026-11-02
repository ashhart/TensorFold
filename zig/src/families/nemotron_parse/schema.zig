//! Every tensor required by the pinned RADIO/mBART graph, before device allocation.
const std = @import("std");
const st = @import("core").safetensors;
const c = @import("config.zig");
pub const radio = "encoder.model_encoder.radio_model.";
pub const tower = radio ++ "model.";
pub const patch_generator = tower ++ "patch_generator.";
pub const Error = error{ MissingParseTensor, BadParseTensor, UnexpectedParseTensor };

/// Used for both the real header and disposable malformed header fixtures.
pub fn validate(header: *const st.Header) Error!void {
    var checker: Checker = .{ .header = header };
    each(&checker) catch |err| switch (err) {
        error.NoSpaceLeft => unreachable, // fixed tensor stems fit the local name buffers
        else => |e| return e,
    };
    if (checker.count != header.count()) return error.UnexpectedParseTensor;
}

const Checker = struct {
    header: *const st.Header,
    count: usize = 0,
    fn tensor(s: *Checker, name: []const u8, dtype: st.DType, shape: []const usize) Error!void {
        const t = s.header.get(name) orelse return error.MissingParseTensor;
        if (t.dtype != dtype or t.rank != shape.len or !std.mem.eql(usize, t.shape[0..t.rank], shape))
            return error.BadParseTensor;
        s.count += 1;
    }
};

/// Emit the complete expected schema. Names are borrowed only for the callback.
pub fn each(sink: anytype) !void {
    try sink.tensor("decoder.embed_tokens.weight", .f32, &.{ c.vocab, c.hidden });
    for ([_][]const u8{ "decoder.layer_norm", "decoder.layernorm_embedding" }) |stem|
        try norm(sink, stem, c.hidden);
    try sink.tensor(radio ++ "summary_idxs", .i64, &.{2});
    try sink.tensor(radio ++ "input_conditioner.norm_mean", .f32, &.{ 3, 1, 1 });
    try sink.tensor(radio ++ "input_conditioner.norm_std", .f32, &.{ 3, 1, 1 });
    try sink.tensor(patch_generator ++ "cls_token.token", .f32, &.{ c.vision_prefix, c.vision_hidden });
    try sink.tensor(patch_generator ++ "embedder.weight", .f32, &.{ c.vision_hidden, 768 });
    try sink.tensor(patch_generator ++ "pos_embed", .f32, &.{ 1, 128 * 128, c.vision_hidden });
    try linear(sink, "encoder.conv1", &.{ c.hidden, c.vision_hidden, 1 }, c.hidden);
    try sink.tensor("encoder.conv2.weight", .f32, &.{ c.hidden, c.hidden, 1, 4 });
    try linear(sink, "encoder.sum_proj", &.{ c.hidden, c.vision_hidden * 2 }, c.hidden);
    for ([_][]const u8{ "encoder.layer_norm1", "encoder.layer_norm2", "encoder.layer_norm3" }) |stem|
        try norm(sink, stem, c.hidden);
    var buf: [192]u8 = undefined;
    for (0..c.layers) |i| {
        const stem = try std.fmt.bufPrint(&buf, "decoder.layers.{d}", .{i});
        try subNorm(sink, stem, "self_attn_layer_norm", c.hidden);
        try subNorm(sink, stem, "final_layer_norm", c.hidden);
        try subLinear(sink, stem, "self_attn.q_proj", c.hidden, c.hidden);
        try subLinear(sink, stem, "self_attn.k_proj", c.kv_heads * c.head_dim, c.hidden);
        try subLinear(sink, stem, "self_attn.v_proj", c.kv_heads * c.head_dim, c.hidden);
        try subLinear(sink, stem, "self_attn.out_proj", c.hidden, c.hidden);
        try subLinear(sink, stem, "fc1", c.ffn, c.hidden);
        try subLinear(sink, stem, "fc2", c.hidden, c.ffn);
    }
    try subLinear(sink, "decoder", "extra_heads.0", c.hidden, c.hidden);
    try subLinear(sink, "decoder", "extra_proj.0", c.hidden, c.hidden);
    for (0..c.vision_layers) |i| {
        const stem = try std.fmt.bufPrint(&buf, tower ++ "blocks.{d}", .{i});
        try subNorm(sink, stem, "norm1", c.vision_hidden);
        try subNorm(sink, stem, "norm2", c.vision_hidden);
        try subLinear(sink, stem, "attn.qkv", 3 * c.vision_hidden, c.vision_hidden);
        try subLinear(sink, stem, "attn.proj", c.vision_hidden, c.vision_hidden);
        try subLinear(sink, stem, "mlp.fc1", c.vision_ffn, c.vision_hidden);
        try subLinear(sink, stem, "mlp.fc2", c.vision_hidden, c.vision_ffn);
    }
}

fn norm(sink: anytype, stem: []const u8, n: usize) !void {
    var buf: [256]u8 = undefined;
    for ([_][]const u8{ "weight", "bias" }) |end|
        try sink.tensor(try std.fmt.bufPrint(&buf, "{s}.{s}", .{ stem, end }), .f32, &.{n});
}
fn linear(sink: anytype, stem: []const u8, shape: []const usize, n: usize) !void {
    var buf: [256]u8 = undefined;
    try sink.tensor(try std.fmt.bufPrint(&buf, "{s}.weight", .{stem}), .f32, shape);
    try sink.tensor(try std.fmt.bufPrint(&buf, "{s}.bias", .{stem}), .f32, &.{n});
}
fn subNorm(sink: anytype, stem: []const u8, sub: []const u8, n: usize) !void {
    var buf: [224]u8 = undefined;
    try norm(sink, try std.fmt.bufPrint(&buf, "{s}.{s}", .{ stem, sub }), n);
}
fn subLinear(sink: anytype, stem: []const u8, sub: []const u8, n: usize, k: usize) !void {
    var buf: [224]u8 = undefined;
    try linear(sink, try std.fmt.bufPrint(&buf, "{s}.{s}", .{ stem, sub }), &.{ n, k }, n);
}
