//! Tensor admission for all 526 BF16 tensors, including the checkpoint's MTP block.
const std = @import("std");
const st = @import("core").safetensors;
const c = @import("config.zig");
pub const text = "model.language_model.";
pub const visual = "model.visual.";
pub const mtp = text ++ "layers.16.";
pub const Error = error{ MissingGlmOcrTensor, BadGlmOcrTensor, UnexpectedGlmOcrTensor };

pub fn validate(header: *const st.Header) Error!void {
    var checker: Checker = .{ .header = header };
    each(&checker) catch |err| switch (err) {
        error.NoSpaceLeft => unreachable,
        else => |e| return e,
    };
    if (checker.count != header.count()) return error.UnexpectedGlmOcrTensor;
}

const Checker = struct {
    header: *const st.Header,
    count: usize = 0,
    pub fn tensor(s: *Checker, name: []const u8, shape: []const usize) Error!void {
        const t = s.header.get(name) orelse return error.MissingGlmOcrTensor;
        if (t.dtype != .bf16 or t.rank != shape.len or !std.mem.eql(usize, t.shape[0..t.rank], shape))
            return error.BadGlmOcrTensor;
        s.count += 1;
    }
};

/// Names are borrowed for the callback; callers retain them only by making a copy.
pub fn each(sink: anytype) !void {
    try sink.tensor("lm_head.weight", &.{ c.vocab, c.hidden });
    try sink.tensor(text ++ "embed_tokens.weight", &.{ c.vocab, c.hidden });
    try sink.tensor(text ++ "norm.weight", &.{c.hidden});
    var buf: [160]u8 = undefined;
    for (0..c.layers + 1) |i| {
        const stem = try std.fmt.bufPrint(&buf, text ++ "layers.{d}.", .{i});
        for ([_][]const u8{ "input_layernorm", "post_attention_layernorm", "post_self_attn_layernorm", "post_mlp_layernorm" }) |norm|
            try linear(sink, stem, norm, &.{c.hidden}, false);
        try linear(sink, stem, "self_attn.q_proj", &.{ c.query, c.hidden }, false);
        try linear(sink, stem, "self_attn.k_proj", &.{ c.kv, c.hidden }, false);
        try linear(sink, stem, "self_attn.v_proj", &.{ c.kv, c.hidden }, false);
        try linear(sink, stem, "self_attn.o_proj", &.{ c.hidden, c.query }, false);
        try linear(sink, stem, "mlp.gate_up_proj", &.{ c.ffn * 2, c.hidden }, false);
        try linear(sink, stem, "mlp.down_proj", &.{ c.hidden, c.ffn }, false);
    }
    try sink.tensor(mtp ++ "eh_proj.weight", &.{ c.hidden, 2 * c.hidden });
    try sink.tensor(mtp ++ "embed_tokens.weight", &.{ c.vocab, c.hidden });
    try sink.tensor(mtp ++ "enorm.weight", &.{c.hidden});
    try sink.tensor(mtp ++ "hnorm.weight", &.{c.hidden});
    try sink.tensor(mtp ++ "shared_head.head.weight", &.{ c.vocab, c.hidden });
    try sink.tensor(mtp ++ "shared_head.norm.weight", &.{c.hidden});
    try linear(sink, visual, "patch_embed.proj", &.{ c.vision_hidden, 3, c.temporal, c.patch, c.patch }, true);
    try linear(sink, visual, "post_layernorm", &.{c.vision_hidden}, false);
    try linear(sink, visual, "downsample", &.{ c.hidden, c.vision_hidden, c.merge, c.merge }, true);
    try linear(sink, visual, "merger.proj", &.{ c.hidden, c.hidden }, false);
    try linear(sink, visual, "merger.post_projection_norm", &.{c.hidden}, true);
    try linear(sink, visual, "merger.gate_proj", &.{ c.ffn, c.hidden }, false);
    try linear(sink, visual, "merger.up_proj", &.{ c.ffn, c.hidden }, false);
    try linear(sink, visual, "merger.down_proj", &.{ c.hidden, c.ffn }, false);
    for (0..c.vision_layers) |i| {
        const stem = try std.fmt.bufPrint(&buf, visual ++ "blocks.{d}.", .{i});
        try linear(sink, stem, "norm1", &.{c.vision_hidden}, false);
        try linear(sink, stem, "norm2", &.{c.vision_hidden}, false);
        try linear(sink, stem, "attn.q_norm", &.{c.vision_head_dim}, false);
        try linear(sink, stem, "attn.k_norm", &.{c.vision_head_dim}, false);
        try linear(sink, stem, "attn.qkv", &.{ 3 * c.vision_hidden, c.vision_hidden }, true);
        try linear(sink, stem, "attn.proj", &.{ c.vision_hidden, c.vision_hidden }, true);
        try linear(sink, stem, "mlp.gate_proj", &.{ c.vision_ffn, c.vision_hidden }, true);
        try linear(sink, stem, "mlp.up_proj", &.{ c.vision_ffn, c.vision_hidden }, true);
        try linear(sink, stem, "mlp.down_proj", &.{ c.vision_hidden, c.vision_ffn }, true);
    }
}

fn linear(sink: anytype, stem: []const u8, sub: []const u8, shape: []const usize, bias: bool) !void {
    var buf: [256]u8 = undefined;
    try sink.tensor(try std.fmt.bufPrint(&buf, "{s}{s}.weight", .{ stem, sub }), shape);
    if (bias) try sink.tensor(try std.fmt.bufPrint(&buf, "{s}{s}.bias", .{ stem, sub }), shape[0..1]);
}
