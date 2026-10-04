//! FlashNext text dimensions and quantization declarations are checked before weight or cache allocation.
const std = @import("std");
const affine = @import("affine.zig");

pub const Kind = enum { linear_attention, sparse_attention };
pub const Gate = enum { sigmoid, silu };
pub const max_layers = 512;
pub const Mtp = struct { layers: usize = 0, kinds: [16]Kind = @splat(.sparse_attention), hybrid: bool = false, dedicated_embeddings: bool = false, source_layer: ?usize = null, rope_theta: f64 = 10_000_000 };

pub const Config = struct {
    parsed: std.json.Parsed(std.json.Value),
    text: std.json.ObjectMap,
    quant: std.json.ObjectMap,
    hidden: usize,
    vocab: usize,
    layers: usize,
    kinds: [max_layers]Kind = undefined,
    eps: f32,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    rope_theta: f64,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    conv: usize,
    gate: Gate,
    experts: usize,
    topk: usize,
    expert_width: usize,
    shared_width: usize,
    hc_count: usize,
    hc_lowrank: usize,
    index_heads: usize,
    index_kv_heads: usize,
    index_dim: usize,
    index_budget: usize,
    index_ratio: usize,
    ple: [max_layers]bool = @splat(false),
    ple_count: usize = 0,
    ple_dim: usize,
    ple_conv: usize,
    ngram: usize,
    ngram_heads: usize,
    ngram_vocab: usize,
    ngram_divisor: usize,
    ngram_shards: usize,
    seed: u64,
    mtp: Mtp,
    mrope_interleaved: bool,
    mrope_section: [3]usize,
    eos: [16]u32 = @splat(0),
    eos_count: usize = 0,
    global_affine: affine.Spec,

    pub fn deinit(c: *Config) void {
        c.parsed.deinit();
        c.* = undefined;
    }

    pub fn read(gpa: std.mem.Allocator, io: std.Io, directory: []const u8) !Config {
        const path = try std.fs.path.join(gpa, &.{ directory, "config.json" });
        defer gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 24));
        defer gpa.free(bytes);
        return parse(gpa, bytes);
    }

    pub fn wide(c: *const Config) !usize {
        return std.math.mul(usize, c.hidden, c.hc_count);
    }

    pub fn keyWidth(c: *const Config) !usize {
        return std.math.mul(usize, c.key_heads, c.key_dim);
    }

    pub fn valueWidth(c: *const Config) !usize {
        return std.math.mul(usize, c.value_heads, c.value_dim);
    }

    pub fn convWidth(c: *const Config) !usize {
        return std.math.add(usize, try std.math.mul(usize, 2, try c.keyWidth()), try c.valueWidth());
    }

    pub fn ngramHeadCount(c: *const Config) !usize {
        return std.math.mul(usize, c.ngram - 1, c.ngram_heads);
    }

    pub fn kind(c: *const Config, i: usize) !Kind {
        if (i >= c.layers) return error.LayerOutOfBounds;
        return c.kinds[i];
    }

    pub fn quantization(c: *const Config, path: []const u8) !?affine.Spec {
        const wanted = canonical(path);
        var found = false;
        var answer: ?affine.Spec = c.global_affine;
        var it = c.quant.iterator();
        while (it.next()) |item| {
            if (!std.mem.eql(u8, canonical(item.key_ptr.*), wanted)) continue;
            const candidate: ?affine.Spec = switch (item.value_ptr.*) {
                .bool => |enabled| if (enabled) c.global_affine else null,
                .object => |value| if (value.count() == 0) null else try spec(value, 4),
                else => return error.InvalidAffineOverride,
            };
            if (found and !same(answer, candidate)) return error.ConflictingAffineAlias;
            answer = candidate;
            found = true;
        }
        return answer;
    }
};

fn canonical(value: []const u8) []const u8 {
    var path = if (std.mem.endsWith(u8, value, ".weight")) value[0 .. value.len - 7] else value;
    for ([_][]const u8{ "model.language_model.", "language_model.", "text_model.", "model." }) |prefix| {
        if (std.mem.startsWith(u8, path, prefix)) path = path[prefix.len..];
    }
    return path;
}

fn same(a: ?affine.Spec, b: ?affine.Spec) bool {
    if (a == null or b == null) return a == null and b == null;
    return a.?.bits == b.?.bits and a.?.group == b.?.group;
}

fn object(v: std.json.Value) !std.json.ObjectMap {
    return if (v == .object) v.object else error.InvalidFlashConfig;
}

fn integer(v: std.json.Value) !usize {
    if (v != .integer or v.integer < 0) return error.InvalidFlashInteger;
    return std.math.cast(usize, v.integer) orelse error.InvalidFlashInteger;
}

fn required(o: std.json.ObjectMap, name: []const u8) !usize {
    const value = try integer(o.get(name) orelse return error.MissingFlashField);
    if (value == 0) return error.InvalidFlashInteger;
    return value;
}

fn optional(o: std.json.ObjectMap, name: []const u8, default: usize) !usize {
    return if (o.get(name)) |v| try integer(v) else default;
}

fn number(v: std.json.Value) !f64 {
    const value: f64 = switch (v) {
        .integer => |x| @floatFromInt(x),
        .float => |x| x,
        else => return error.InvalidFlashNumber,
    };
    if (!std.math.isFinite(value)) return error.InvalidFlashNumber;
    return value;
}

fn boolean(o: std.json.ObjectMap, name: []const u8, fallback: bool) !bool {
    const v = o.get(name) orelse return fallback;
    return if (v == .bool) v.bool else error.InvalidFlashFlag;
}

fn mtpConfig(text: std.json.ObjectMap) !Mtp {
    const o = if (text.get("mtp")) |v| if (v == .null) std.json.ObjectMap.empty else try object(v) else std.json.ObjectMap.empty;
    var m = Mtp{
        .layers = try optional(o, "num_hidden_layers", try optional(text, "mtp_num_hidden_layers", 0)),
        .hybrid = try boolean(o, "hybrid", false),
        .dedicated_embeddings = try boolean(text, "mtp_use_dedicated_embeddings", false),
        .rope_theta = try number(o.get("rope_theta") orelse .{ .float = 10_000_000 }),
    };
    if (m.layers > m.kinds.len or m.rope_theta <= 0) return error.InvalidFlashMtp;
    if (o.get("mtp_use_hidden_state_from_layer")) |v| {
        if (v != .null) m.source_layer = try integer(v);
    }
    if (m.layers > 0) {
        const types = o.get("layer_types") orelse return error.InvalidFlashMtp;
        if (types != .array or types.array.items.len != m.layers) return error.InvalidFlashMtp;
        for (types.array.items, 0..) |v, i| {
            if (v != .string) return error.InvalidFlashMtp;
            m.kinds[i] = if (std.mem.eql(u8, v.string, "linear_attention")) .linear_attention else if (std.mem.eql(u8, v.string, "full_attention") or std.mem.eql(u8, v.string, "sparse_attention")) .sparse_attention else return error.InvalidFlashMtp;
        }
    }
    return m;
}

fn spec(o: std.json.ObjectMap, default_bits: ?usize) !affine.Spec {
    const bits = if (o.get("bits")) |v| try integer(v) else default_bits orelse return error.MissingAffineBits;
    if (o.get("mode")) |v| if (v != .null and (v != .string or (v.string.len > 0 and !std.mem.eql(u8, v.string, "affine")))) return error.UnsupportedAffineMode;
    return affine.Spec.init(bits, try optional(o, "group_size", 64));
}

fn quantBlock(root: std.json.ObjectMap, text: std.json.ObjectMap) !std.json.ObjectMap {
    for ([_]std.json.ObjectMap{ root, text }) |source| for ([_][]const u8{ "quantization", "quantization_config" }) |key| {
        if (source.get(key)) |v| if (v == .object and v.object.count() > 0) return v.object;
    };
    return error.MissingAffineConfig;
}

pub fn parse(gpa: std.mem.Allocator, bytes: []const u8) !Config {
    var parsed = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
    errdefer parsed.deinit();
    const root = try object(parsed.value);
    const text = if (root.get("text_config")) |v| if (v == .object and v.object.count() > 0) v.object else root else root;
    const tag = root.get("model_type") orelse text.get("model_type") orelse return error.MissingFlashField;
    if (tag != .string or !std.mem.eql(u8, tag.string, "qwen4_exp")) return error.NotFlashNext;
    const quant = try quantBlock(root, text);
    if (quant.get("quant_method")) |v| if (v != .null and (v != .string or (!std.mem.eql(u8, v.string, "mlx") and !std.mem.eql(u8, v.string, "affine")))) return error.UnsupportedAffineMode;
    const rope = if (text.get("rope_parameters")) |v| if (v == .null) std.json.ObjectMap.empty else try object(v) else std.json.ObjectMap.empty;
    for ([_][]const u8{ "rope_type", "type" }) |key| if (rope.get(key)) |v| if (v != .string or !std.mem.eql(u8, v.string, "default")) return error.UnsupportedFlashRope;
    const hidden = try required(text, "hidden_size");
    const heads = try required(text, "num_attention_heads");
    const head_dim = if (text.get("head_dim")) |v| if (v == .null or (v == .integer and v.integer == 0)) hidden / heads else try integer(v) else hidden / heads;
    const partial = try number(rope.get("partial_rotary_factor") orelse text.get("partial_rotary_factor") orelse .{ .float = 0.25 });
    if (partial < 0 or partial > 1 or head_dim == 0) return error.UnsupportedFlashRope;
    const gate: std.json.Value = text.get("output_gate_type") orelse .{ .string = "sigmoid" };
    if (gate != .string) return error.UnsupportedFlashGate;
    var c = Config{
        .parsed = parsed,
        .text = text,
        .quant = quant,
        .hidden = hidden,
        .heads = heads,
        .vocab = try required(text, "vocab_size"),
        .layers = try required(text, "num_hidden_layers"),
        .eps = @floatCast(try number(text.get("rms_norm_eps") orelse return error.MissingFlashField)),
        .kv_heads = try required(text, "num_key_value_heads"),
        .head_dim = head_dim,
        .rotary_dim = @intFromFloat(@as(f64, @floatFromInt(head_dim)) * partial),
        .rope_theta = try number(rope.get("rope_theta") orelse .{ .float = 10_000_000 }),
        .key_heads = try required(text, "linear_num_key_heads"),
        .value_heads = try required(text, "linear_num_value_heads"),
        .key_dim = try required(text, "linear_key_head_dim"),
        .value_dim = try required(text, "linear_value_head_dim"),
        .conv = try required(text, "linear_conv_kernel_dim"),
        .gate = std.meta.stringToEnum(Gate, gate.string) orelse return error.UnsupportedFlashGate,
        .experts = try required(text, "num_experts"),
        .topk = try required(text, "num_experts_per_tok"),
        .expert_width = try required(text, "moe_intermediate_size"),
        .shared_width = try required(text, "shared_expert_intermediate_size"),
        .hc_count = try optional(text, "hc_count", 4),
        .hc_lowrank = try optional(text, "hc_lowrank", 320),
        .index_heads = try optional(text, "indexer_n_heads", 4),
        .index_kv_heads = try optional(text, "indexer_kv_heads", 1),
        .index_dim = try optional(text, "indexer_head_dim", 128),
        .index_budget = try optional(text, "indexer_budget", 2048),
        .index_ratio = try optional(text, "indexer_compress_ratio", 4),
        .ple_dim = try optional(text, "ple_embed_dim", hidden),
        .ple_conv = try optional(text, "ple_conv_kernel_size", 4),
        .ngram = try optional(text, "ngram_size", 3),
        .ngram_heads = try optional(text, "heads_per_ngram", 8),
        .ngram_vocab = try optional(text, "ngram_vocab_size_base", 20_000_000),
        .ngram_divisor = try optional(text, "make_ngram_vocab_size_divisible_by", 128),
        .ngram_shards = try optional(text, "split_ngram_parts", 128),
        .seed = try optional(text, "seed", 1234),
        .mtp = try mtpConfig(text),
        .mrope_interleaved = try boolean(rope, "mrope_interleaved", true),
        .mrope_section = .{ 11, 11, 10 },
        .global_affine = try spec(quant, null),
    };
    if (c.layers > max_layers or c.heads % c.kv_heads != 0 or c.value_heads % c.key_heads != 0 or c.topk > c.experts) return error.InvalidFlashShape;
    if (c.conv < 2 or c.hc_count == 0 or c.hc_lowrank == 0 or c.index_ratio == 0 or c.index_heads == 0 or c.index_kv_heads == 0 or c.index_heads % c.index_kv_heads != 0 or c.index_dim == 0 or c.index_budget == 0 or c.index_budget % c.index_ratio != 0) return error.InvalidFlashShape;
    if (c.mtp.source_layer) |i| if (i >= c.layers) return error.InvalidFlashMtp;
    if (rope.get("mrope_section")) |v| {
        if (v != .array or v.array.items.len != 3) return error.UnsupportedFlashRope;
        for (v.array.items, 0..) |section, i| c.mrope_section[i] = try integer(section);
    }
    const sections = try std.math.add(usize, try std.math.add(usize, c.mrope_section[0], c.mrope_section[1]), c.mrope_section[2]);
    if (c.rotary_dim > 0 and try std.math.mul(usize, sections, 2) != c.rotary_dim and rope.get("mrope_section") != null) return error.UnsupportedFlashRope;
    if (c.eps <= 0 or !std.math.isFinite(c.eps) or c.rope_theta <= 0 or c.rotary_dim % 2 != 0 or c.rotary_dim > c.head_dim or c.rotary_dim > c.index_dim) return error.UnsupportedFlashRope;
    if (c.ngram < 2 or c.ngram_heads == 0 or c.ngram_vocab == 0 or c.ngram_divisor == 0 or c.ngram_shards == 0 or c.ple_conv < 2) return error.InvalidFlashNgram;
    const table_heads = try c.ngramHeadCount();
    if (c.ple_dim == 0 or c.ple_dim % table_heads != 0) return error.InvalidFlashNgram;
    _ = try c.wide();
    _ = try c.convWidth();
    const kinds = text.get("layer_types") orelse return error.MissingFlashField;
    if (kinds != .array or kinds.array.items.len != c.layers) return error.InvalidFlashLayers;
    for (kinds.array.items, 0..) |v, i| {
        if (v != .string) return error.InvalidFlashLayers;
        c.kinds[i] = if (std.mem.eql(u8, v.string, "linear_attention")) .linear_attention else if (std.mem.eql(u8, v.string, "sparse_attention") or std.mem.eql(u8, v.string, "full_attention")) .sparse_attention else return error.InvalidFlashLayers;
    }
    if (text.get("ple_layer_ids")) |v| if (v != .null) {
        if (v != .array) return error.InvalidFlashNgram;
        for (v.array.items) |item| {
            const index = try integer(item);
            if (index == 0 or index > c.layers) return error.InvalidFlashNgram;
            if (!c.ple[index - 1]) c.ple_count += 1;
            c.ple[index - 1] = true;
        }
    };
    if (root.get("eos_token_id") orelse text.get("eos_token_id")) |v| if (v != .null) {
        const ids = if (v == .array) v.array.items else &.{v};
        if (ids.len > c.eos.len) return error.InvalidFlashEos;
        for (ids) |id| {
            const value = try integer(id);
            if (value >= c.vocab or value > std.math.maxInt(u32)) return error.InvalidFlashEos;
            c.eos[c.eos_count] = @intCast(value);
            c.eos_count += 1;
        }
    };
    return c;
}
