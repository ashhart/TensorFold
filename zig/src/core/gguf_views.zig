//! Typed, borrowed GGUF metadata views. Absent settings stay absent; family adapters choose defaults.
const std = @import("std");
const meta = @import("gguf_value.zig");

pub const Config = struct {
    architecture: []const u8,
    layers: ?u64,
    embedding: ?u64,
    feed_forward: ?u64,
    heads: ?u64,
    kv_heads: ?u64,
    key_length: ?u64,
    value_length: ?u64,
    context: ?u64,
    rope_dimensions: ?u64,
    rope_base: ?f64,
    rope_scaling: ?[]const u8,
    rope_factor: ?f64,
    rope_original_context: ?u64,
    rope_finetuned: ?bool,
    rms_epsilon: ?f64,
    layer_norm_epsilon: ?f64,
};
fn string(v: ?meta.Value) !?[]const u8 {
    const x = v orelse return null;
    return if (x == .string) x.string else error.MetadataTypeMismatch;
}
fn integer(v: ?meta.Value) !?u64 {
    return if (v) |x| try x.unsigned() else null;
}
fn real(v: ?meta.Value) !?f64 {
    return if (v) |x| try x.float() else null;
}
fn boolean(v: ?meta.Value) !?bool {
    const x = v orelse return null;
    return if (x == .boolean) x.boolean else error.MetadataTypeMismatch;
}
/// No temporary keys or allocations. Nonstandard and per-layer array settings remain available through get().
pub fn scoped(doc: anytype, architecture: []const u8, suffix: []const u8) ?meta.Value {
    for (doc.metadata) |e| {
        if (e.key.len <= architecture.len or !std.mem.startsWith(u8, e.key, architecture)) continue;
        if (e.key[architecture.len] == '.' and std.mem.eql(u8, e.key[architecture.len + 1 ..], suffix)) return e.value;
    }
    return null;
}
pub fn config(doc: anytype) !Config {
    const arch = (try string(doc.get("general.architecture"))) orelse return error.MissingArchitecture;
    if (arch.len == 0 or std.mem.indexOfScalar(u8, arch, 0) != null) return error.InvalidArchitecture;
    return .{
        .architecture = arch,
        .layers = try integer(scoped(doc, arch, "block_count")),
        .embedding = try integer(scoped(doc, arch, "embedding_length")),
        .feed_forward = try integer(scoped(doc, arch, "feed_forward_length")),
        .heads = try integer(scoped(doc, arch, "attention.head_count")),
        .kv_heads = try integer(scoped(doc, arch, "attention.head_count_kv")),
        .key_length = try integer(scoped(doc, arch, "attention.key_length")),
        .value_length = try integer(scoped(doc, arch, "attention.value_length")),
        .context = try integer(scoped(doc, arch, "context_length")),
        .rope_dimensions = try integer(scoped(doc, arch, "rope.dimension_count")),
        .rope_base = try real(scoped(doc, arch, "rope.freq_base")),
        .rope_scaling = try string(scoped(doc, arch, "rope.scaling.type")),
        .rope_factor = try real(scoped(doc, arch, "rope.scaling.factor")),
        .rope_original_context = try integer(scoped(doc, arch, "rope.scaling.original_context_length")),
        .rope_finetuned = try boolean(scoped(doc, arch, "rope.scaling.finetuned")),
        .rms_epsilon = try real(scoped(doc, arch, "attention.layer_norm_rms_epsilon")),
        .layer_norm_epsilon = try real(scoped(doc, arch, "attention.layer_norm_epsilon")),
    };
}
pub const Tokenizer = struct {
    model: ?[]const u8,
    pre: ?[]const u8,
    tokens: ?meta.Array,
    scores: ?meta.Array,
    types: ?meta.Array,
    merges: ?meta.Array,
    bos: ?u64,
    eos: ?u64,
    add_bos: ?bool,
    add_eos: ?bool,
};
fn array(v: ?meta.Value, kind: meta.Kind) !?meta.Array {
    const x = v orelse return null;
    if (x != .array or x.array.kind != kind) return error.MetadataTypeMismatch;
    return x.array;
}
pub fn tokenizer(doc: anytype) !Tokenizer {
    const out: Tokenizer = .{
        .model = try string(doc.get("tokenizer.ggml.model")),
        .pre = try string(doc.get("tokenizer.ggml.pre")),
        .tokens = try array(doc.get("tokenizer.ggml.tokens"), .string),
        .scores = try array(doc.get("tokenizer.ggml.scores"), .float32),
        .types = try array(doc.get("tokenizer.ggml.token_type"), .int32),
        .merges = try array(doc.get("tokenizer.ggml.merges"), .string),
        .bos = try integer(doc.get("tokenizer.ggml.bos_token_id")),
        .eos = try integer(doc.get("tokenizer.ggml.eos_token_id")),
        .add_bos = try boolean(doc.get("tokenizer.ggml.add_bos_token")),
        .add_eos = try boolean(doc.get("tokenizer.ggml.add_eos_token")),
    };
    if (out.tokens) |tokens| {
        if (out.scores) |a| if (a.len != tokens.len) return error.TokenizerLengthMismatch;
        if (out.types) |a| if (a.len != tokens.len) return error.TokenizerLengthMismatch;
        if (out.bos) |id| if (id >= tokens.len) return error.TokenIdOutsideVocabulary;
        if (out.eos) |id| if (id >= tokens.len) return error.TokenIdOutsideVocabulary;
    }
    return out;
}
