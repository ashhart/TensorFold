//! LFM3 text decoder tensor names: the two namespaces and every module's path, without the safetensors parser.
const std = @import("std");

pub const Namespace = enum {
    bare,
    wrapped,

    pub fn prefix(n: Namespace) []const u8 {
        return if (n == .bare) "model." else "language_model.model.";
    }
};

/// One module of the text decoder; its path follows the namespace prefix (and `layers.{i}.` inside a layer).
pub const Module = enum {
    embed_tokens,
    embedding_norm,
    operator_norm,
    ffn_norm,
    post_operator_norm,
    post_ffn_norm,
    w1,
    w3,
    w2,
    conv_in,
    conv_out,
    conv,
    q_proj,
    k_proj,
    v_proj,
    o_proj,
    gate_proj,
    q_norm,
    k_norm,

    /// The module path; an affine module's tensors add `.weight`, `.scales` and `.biases`.
    pub fn path(m: Module) []const u8 {
        return switch (m) {
            .embed_tokens => "embed_tokens",
            .embedding_norm => "embedding_norm.weight",
            .operator_norm => "operator_norm.weight",
            .ffn_norm => "ffn_norm.weight",
            .post_operator_norm => "post_operator_norm.weight",
            .post_ffn_norm => "post_ffn_norm.weight",
            .w1 => "feed_forward.w1",
            .w3 => "feed_forward.w3",
            .w2 => "feed_forward.w2",
            .conv_in => "conv.in_proj",
            .conv_out => "conv.out_proj",
            .conv => "conv.conv.weight",
            .q_proj => "self_attn.q_proj",
            .k_proj => "self_attn.k_proj",
            .v_proj => "self_attn.v_proj",
            .o_proj => "self_attn.o_proj",
            .gate_proj => "self_attn.gate_proj",
            .q_norm => "self_attn.q_norm.weight",
            .k_norm => "self_attn.k_norm.weight",
        };
    }
};

/// A module's full name: the namespace prefix, `layers.{layer}.` for a layer module, then its path.
pub fn moduleName(buf: []u8, namespace: Namespace, layer: ?usize, m: Module) ![]const u8 {
    if (layer) |i| return std.fmt.bufPrint(buf, "{s}layers.{d}.{s}", .{ namespace.prefix(), i, m.path() });
    return std.fmt.bufPrint(buf, "{s}{s}", .{ namespace.prefix(), m.path() });
}

/// The namespace of a header from whether it holds the bare and the wrapped embedding.
pub fn detect(bare: bool, wrapped: bool) !Namespace {
    if (bare and wrapped) return error.MixedNamespaces;
    if (!bare and !wrapped) return error.MissingTensor;
    return if (bare) .bare else .wrapped;
}

/// The embedding's packed weight under `namespace`, the tensor `detect` looks for.
pub fn embedName(buf: []u8, namespace: Namespace) ![]const u8 {
    var m: [64]u8 = undefined;
    return std.fmt.bufPrint(buf, "{s}.weight", .{try moduleName(&m, namespace, null, .embed_tokens)});
}
