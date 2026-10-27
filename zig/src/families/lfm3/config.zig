//! Admission for LFM3-1.7B and LFM3-VL-2B MLX checkpoints; this parser does not advertise an engine.
const std = @import("std");

pub const hidden = 2048;
pub const vocab = 128000;
pub const layers = 24;
pub const intermediate = 7680;
pub const conv_taps = 7;
pub const query_heads = 16;
pub const kv_heads = 4;
pub const head_dim = 128;
pub const sliding_window = 256;
pub const eps: f32 = 1e-5;
pub const Kind = enum { text, vision_language };
pub const LayerType = enum { conv, full_attention, sliding_attention };
pub const Rotary = struct { theta: f32, partial: f32 };
pub const schedule: [layers]LayerType = .{
    .conv,              .conv, .full_attention,    .conv, .sliding_attention, .conv,
    .full_attention,    .conv, .sliding_attention, .conv, .full_attention,    .conv,
    .sliding_attention, .conv, .full_attention,    .conv, .sliding_attention, .conv,
    .full_attention,    .conv, .sliding_attention, .conv, .full_attention,    .conv,
};

pub const Width = struct { bits: u8, group: u16 };
/// The projections' storage: MLX affine at `quant`'s widths, or ModelOpt NVFP4 (FP4 FFN only); decoders refuse others.
pub const Scheme = enum { mlx_affine, modelopt_nvfp4 };
/// The width `quant` reports for a ModelOpt NVFP4 export (its FFN): 4 bits in groups of 16.
pub const nvfp4_width: Width = .{ .bits = 4, .group = 16 };
pub const Quant = struct {
    global: Width,
    modules: std.StringHashMapUnmanaged(Width) = .empty,

    pub fn width(self: *const Quant, key: []const u8) Width {
        return self.modules.get(canonical(key)) orelse self.global;
    }
};

fn canonical(key: []const u8) []const u8 {
    var k = key;
    if (std.mem.endsWith(u8, k, ".weight")) k = k[0 .. k.len - 7];
    for ([_][]const u8{ "model.language_model.", "language_model.", "text_model.", "model." }) |prefix| {
        if (std.mem.startsWith(u8, k, prefix)) k = k[prefix.len..];
    }
    return k;
}

fn projection(key: []const u8) bool {
    const k = canonical(key);
    if (std.mem.eql(u8, k, "embed_tokens")) return true;
    if (!std.mem.startsWith(u8, k, "layers.")) return false;
    const tail = k[7..];
    const dot = std.mem.indexOfScalar(u8, tail, '.') orelse return false;
    if (dot == 0 or (dot > 1 and tail[0] == '0')) return false;
    for (tail[0..dot]) |digit| if (digit < '0' or digit > '9') return false;
    const index = std.fmt.parseInt(usize, tail[0..dot], 10) catch return false;
    if (index >= layers) return false;
    const name = tail[dot + 1 ..];
    const mixer = schedule[index];
    const names: []const []const u8 = if (mixer == .conv)
        &.{ "conv.in_proj", "conv.out_proj", "feed_forward.w1", "feed_forward.w2", "feed_forward.w3" }
    else
        &.{ "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "self_attn.gate_proj", "feed_forward.w1", "feed_forward.w2", "feed_forward.w3" };
    for (names) |name_| if (std.mem.eql(u8, name, name_)) return true;
    return false;
}

fn quantWidth(o: std.json.ObjectMap) !Width {
    const bits = try integer(o, "bits");
    const group = try integer(o, "group_size");
    if (o.get("mode")) |mode| {
        if (mode != .string or !std.mem.eql(u8, mode.string, "affine")) return error.UnsupportedLfm3Quantization;
    }
    const w: Width = .{
        .bits = std.math.cast(u8, bits) orelse return error.UnsupportedLfm3Quantization,
        .group = std.math.cast(u16, group) orelse return error.UnsupportedLfm3Quantization,
    };
    try width(w);
    return w;
}

pub const Config = struct {
    arena: std.heap.ArenaAllocator,
    kind: Kind,
    context: usize,
    layer_types: [layers]LayerType = schedule,
    full_rotary: Rotary = .{ .theta = 1e6, .partial = 0.5 },
    sliding_rotary: Rotary = .{ .theta = 1e4, .partial = 1 },
    scheme: Scheme = .mlx_affine,
    quant: Quant,

    pub fn deinit(self: *Config) void {
        self.arena.deinit();
        self.* = undefined;
    }

    pub fn width(self: *const Config, key: []const u8) !Width {
        return self.quant.width(key);
    }

    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Config {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
        defer gpa.free(bytes);
        return parse(gpa, bytes);
    }
};

const Object = std.json.ObjectMap;

fn object(v: std.json.Value) !Object {
    return if (v == .object) v.object else error.BadLfm3Config;
}

fn field(o: Object, key: []const u8) !std.json.Value {
    return o.get(key) orelse error.BadLfm3Config;
}

fn integer(o: Object, key: []const u8) !usize {
    const v = try field(o, key);
    if (v != .integer or v.integer < 1) return error.BadLfm3Config;
    return std.math.cast(usize, v.integer) orelse error.BadLfm3Config;
}

fn string(o: Object, key: []const u8, expected: []const u8) !void {
    const v = try field(o, key);
    if (v != .string or !std.mem.eql(u8, v.string, expected)) return error.UnsupportedLfm3Config;
}

fn boolean(o: Object, key: []const u8, expected: bool) !void {
    const v = try field(o, key);
    if (v != .bool or v.bool != expected) return error.UnsupportedLfm3Config;
}

fn number(o: Object, key: []const u8, expected: f64) !void {
    const v = try field(o, key);
    const actual: f64 = switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => return error.BadLfm3Config,
    };
    if (actual != expected) return error.UnsupportedLfm3Config;
}

fn geometry(o: Object, dims: anytype) !void {
    inline for (dims) |d| if (try integer(o, d[0]) != d[1]) return error.UnsupportedLfm3Geometry;
}

fn layerTypes(o: Object, expected: []const LayerType) !void {
    const v = try field(o, "layer_types");
    if (v != .array or v.array.items.len != expected.len) return error.UnsupportedLfm3LayerSchedule;
    for (v.array.items, expected) |actual, wanted| {
        if (actual != .string or !std.mem.eql(u8, actual.string, @tagName(wanted)))
            return error.UnsupportedLfm3LayerSchedule;
    }
}

fn text(o: Object, kind: Kind) !usize {
    try string(o, "model_type", "lfm3");
    try geometry(o, .{
        .{ "hidden_size", hidden },             .{ "vocab_size", vocab },       .{ "num_hidden_layers", layers },
        .{ "intermediate_size", intermediate }, .{ "conv_L_cache", conv_taps }, .{ "num_attention_heads", query_heads },
        .{ "num_key_value_heads", kv_heads },   .{ "head_dim", head_dim },      .{ "sliding_window", sliding_window },
    });
    try boolean(o, "conv_bias", false);
    try boolean(o, "tie_word_embeddings", true);
    try boolean(o, "use_sandwich_norm", true);
    try string(o, "dtype", "bfloat16");
    try number(o, "norm_eps", 1e-5);
    try layerTypes(o, &schedule);
    const rope = try object(try field(o, "rope_parameters"));
    if (rope.count() != 2) return error.UnsupportedLfm3Rotary;
    const full = try object(try field(rope, "full_attention"));
    const sliding = try object(try field(rope, "sliding_attention"));
    try rotary(full, 1e6, 0.5);
    try rotary(sliding, 1e4, 1);
    const context = try integer(o, "max_position_embeddings");
    if (context != @as(usize, if (kind == .text) 262144 else 128000)) return error.UnsupportedLfm3Context;
    return context;
}

fn rotary(o: Object, theta: f64, partial: f64) !void {
    try string(o, "rope_type", "default");
    try number(o, "rope_theta", theta);
    if (o.contains("partial_rotary_factor")) try number(o, "partial_rotary_factor", partial) else if (partial != 1)
        return error.UnsupportedLfm3Rotary;
    var it = o.iterator();
    while (it.next()) |e| {
        const k = e.key_ptr.*;
        if (!std.mem.eql(u8, k, "rope_type") and !std.mem.eql(u8, k, "rope_theta") and
            !std.mem.eql(u8, k, "partial_rotary_factor")) return error.UnsupportedLfm3Rotary;
    }
}

fn vision(root: Object) !void {
    try boolean(root, "tie_word_embeddings", true);
    try boolean(root, "projector_bias", true);
    try string(root, "projector_hidden_act", "gelu");
    try number(root, "projector_layer_norm_eps", 1e-6);
    try geometry(root, .{
        .{ "downsample_factor", 2 },   .{ "projector_hidden_size", 4096 },   .{ "image_token_id", 125030 },
        .{ "video_token_id", 125029 }, .{ "vision_start_token_id", 125009 }, .{ "vision_end_token_id", 125010 },
    });
    const o = try object(try field(root, "vision_config"));
    try string(o, "model_type", "lfm3_vl_vision");
    try string(o, "dtype", "bfloat16");
    try number(o, "norm_eps", 1e-5);
    try geometry(o, .{
        .{ "hidden_size", 1024 },       .{ "intermediate_size", 2560 }, .{ "num_hidden_layers", 14 },
        .{ "num_attention_heads", 16 }, .{ "num_key_value_heads", 8 },  .{ "in_channels", 3 },
        .{ "conv_kernel_size", 3 },     .{ "patch_size", 16 },          .{ "spatial_merge_size", 2 },
        .{ "temporal_patch_size", 2 },
    });
    try layerTypes(o, &.{ .conv, .conv, .full_attention, .conv, .full_attention, .conv, .full_attention, .conv, .full_attention, .conv, .full_attention, .conv, .full_attention, .conv });
    const rope = try object(try field(o, "rope_parameters"));
    try string(rope, "rope_type", "axial");
    try number(rope, "rope_theta", 1e6);
    if (rope.count() != 2) return error.UnsupportedLfm3Rotary;
}

fn quantization(a: std.mem.Allocator, value: std.json.Value) !Quant {
    const o = try object(value);
    var it = o.iterator();
    while (it.next()) |e| {
        const key = e.key_ptr.*;
        if (std.mem.eql(u8, key, "bits") or std.mem.eql(u8, key, "group_size") or std.mem.eql(u8, key, "mode")) continue;
        if (e.value_ptr.* != .object) return error.UnsupportedLfm3Quantization;
        const entry = e.value_ptr.object;
        if (entry.count() < 2 or entry.count() > 3) return error.UnsupportedLfm3Quantization;
        var fields = entry.iterator();
        while (fields.next()) |f| {
            const k = f.key_ptr.*;
            if (!std.mem.eql(u8, k, "bits") and !std.mem.eql(u8, k, "group_size") and !std.mem.eql(u8, k, "mode"))
                return error.UnsupportedLfm3Quantization;
        }
        if (!projection(key)) return error.UnsupportedLfm3Quantization;
    }
    var q: Quant = .{ .global = try quantWidth(o) };
    it = o.iterator();
    while (it.next()) |e| {
        if (e.value_ptr.* != .object) continue;
        const w = try quantWidth(e.value_ptr.object);
        const slot = try q.modules.getOrPut(a, canonical(e.key_ptr.*));
        if (slot.found_existing and !std.meta.eql(slot.value_ptr.*, w)) return error.UnsupportedLfm3Quantization;
        slot.value_ptr.* = w;
    }
    return q;
}

/// ModelOpt's NVFP4 text export: true; another quant_method: false; any other recipe or ignore list is refused.
fn modelopt(value: std.json.Value, kind: Kind) !bool {
    const o = try object(value);
    const method = o.get("quant_method") orelse return false;
    if (method != .string or !std.mem.eql(u8, method.string, "modelopt")) return false;
    if (kind != .text or !named(o, "quant_algo", "NVFP4")) return error.UnsupportedLfm3Quantization;
    const groups = try object(try field(o, "config_groups"));
    if (groups.count() != 1) return error.UnsupportedLfm3Quantization;
    const group = try object(groups.values()[0]);
    try fp4(try object(try field(group, "weights")));
    try fp4(try object(try field(group, "input_activations")));
    const targets = try field(group, "targets");
    if (targets != .array or targets.array.items.len != 1 or targets.array.items[0] != .string or
        !std.mem.eql(u8, targets.array.items[0].string, "Linear")) return error.UnsupportedLfm3Quantization;
    try excluded(try field(o, "ignore"));
    return true;
}

fn named(o: Object, key: []const u8, expected: []const u8) bool {
    const v = o.get(key) orelse return false;
    return v == .string and std.mem.eql(u8, v.string, expected);
}

/// A weights or input_activations entry: static FP4 (e2m1) in groups of 16.
fn fp4(o: Object) !void {
    const dynamic = o.get("dynamic") orelse return error.UnsupportedLfm3Quantization;
    const bits = o.get("num_bits") orelse return error.UnsupportedLfm3Quantization;
    const group = o.get("group_size") orelse return error.UnsupportedLfm3Quantization;
    if (dynamic != .bool or dynamic.bool or bits != .integer or bits.integer != 4 or group != .integer or
        group.integer != 16 or !named(o, "type", "float")) return error.UnsupportedLfm3Quantization;
}

/// The ignore list as the export writes it: the head, the embedding and every layer's mixer (conv* or self_attn*).
fn excluded(value: std.json.Value) !void {
    if (value != .array or value.array.items.len != layers + 2) return error.UnsupportedLfm3Quantization;
    var seen: [layers + 2]bool = @splat(false);
    for (value.array.items) |item| {
        if (item != .string) return error.UnsupportedLfm3Quantization;
        const at = ignoredAt(item.string) orelse return error.UnsupportedLfm3Quantization;
        if (seen[at]) return error.UnsupportedLfm3Quantization;
        seen[at] = true;
    }
}

/// Where an ignore entry sits: 0 the head, 1 the embedding, 2 + i layer i's mixer; null for any other name.
fn ignoredAt(name: []const u8) ?usize {
    if (std.mem.eql(u8, name, "lm_head")) return 0;
    if (std.mem.eql(u8, name, "model.embed_tokens")) return 1;
    const prefix = "model.layers.";
    if (!std.mem.startsWith(u8, name, prefix)) return null;
    const tail = name[prefix.len..];
    const dot = std.mem.indexOfScalar(u8, tail, '.') orelse return null;
    if (dot == 0 or (dot > 1 and tail[0] == '0')) return null;
    const index = std.fmt.parseInt(usize, tail[0..dot], 10) catch return null;
    if (index >= layers) return null;
    const mixer: []const u8 = if (schedule[index] == .conv) "conv*" else "self_attn*";
    return if (std.mem.eql(u8, tail[dot + 1 ..], mixer)) 2 + index else null;
}

fn width(w: Width) !void {
    if ((w.bits != 4 and w.bits != 6 and w.bits != 8) or (w.group != 32 and w.group != 64))
        return error.UnsupportedLfm3Quantization;
}

fn equalQuant(a: Quant, b: Quant) bool {
    if (!std.meta.eql(a.global, b.global) or a.modules.count() != b.modules.count()) return false;
    var it = a.modules.iterator();
    while (it.next()) |e| {
        const other = b.modules.get(e.key_ptr.*) orelse return false;
        if (!std.meta.eql(e.value_ptr.*, other)) return false;
    }
    return true;
}

pub fn parse(gpa: std.mem.Allocator, bytes: []const u8) !Config {
    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();
    const root = try object(try std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{ .allocate = .alloc_always }));
    const tag = try field(root, "model_type");
    if (tag != .string) return error.BadLfm3Config;
    const kind: Kind = if (std.mem.eql(u8, tag.string, "lfm3")) .text else if (std.mem.eql(u8, tag.string, "lfm3_vl")) .vision_language else return error.UnsupportedLfm3Config;
    const language = if (kind == .text) root else try object(try field(root, "text_config"));
    const context = try text(language, kind);
    if (kind == .vision_language) try vision(root);
    if (root.get("quantization") == null) if (root.get("quantization_config")) |qc| if (try modelopt(qc, kind))
        return .{ .arena = arena, .kind = kind, .context = context, .scheme = .modelopt_nvfp4, .quant = .{ .global = nvfp4_width } };
    const q = try quantization(a, root.get("quantization") orelse root.get("quantization_config") orelse
        return error.UnsupportedLfm3Quantization);
    if (root.get("quantization_config")) |other| {
        if (!equalQuant(q, try quantization(a, other))) return error.UnsupportedLfm3Quantization;
    }
    return .{ .arena = arena, .kind = kind, .context = context, .quant = q };
}
