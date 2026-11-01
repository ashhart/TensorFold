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
pub const LayerType = enum {
    conv,
    full_attention,
    sliding_attention,

    /// Keys before a row its attention reads: the sliding window, or 0 for every one (a conv layer reads none).
    pub fn window(t: LayerType) usize {
        return if (t == .sliding_attention) sliding_window else 0;
    }
};

pub const Rotary = struct {
    theta: f32,
    partial: f32,

    /// RoPE's inverse frequencies as torch's float32 rows: 1 / theta^(2i / 128) rotated, zero past them (cos 1, sin 0).
    pub fn inverseFrequencies(rot: Rotary) [head_dim / 2]f32 {
        var out: [head_dim / 2]f32 = undefined;
        const angles: usize = @intFromFloat(rot.partial * @as(f32, head_dim) / 2);
        for (&out, 0..) |*v, i| {
            if (i >= angles) {
                v.* = 0;
                continue;
            }
            const e: f32 = @as(f32, @floatFromInt(2 * i)) / @as(f32, head_dim);
            const pw: f32 = @floatCast(std.math.pow(f64, rot.theta, e));
            v.* = 1.0 / pw;
        }
        return out;
    }
};

pub const schedule: [layers]LayerType = .{
    .conv,              .conv, .full_attention,    .conv, .sliding_attention, .conv,
    .full_attention,    .conv, .sliding_attention, .conv, .full_attention,    .conv,
    .sliding_attention, .conv, .full_attention,    .conv, .sliding_attention, .conv,
    .full_attention,    .conv, .sliding_attention, .conv, .full_attention,    .conv,
};

/// Layers of `kind` in the admitted schedule.
pub fn count(kind: LayerType) usize {
    var n: usize = 0;
    for (schedule) |t| n += @intFromBool(t == kind);
    return n;
}

/// Keys a sliding layer's ring holds when `rows` rows are written before the first of them reads its window back.
pub fn ring(rows: usize) usize {
    return sliding_window + rows;
}

pub const Width = struct { bits: u8, group: u16 };
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
    const q = try quantization(a, root.get("quantization") orelse root.get("quantization_config") orelse
        return error.UnsupportedLfm3Quantization);
    if (root.get("quantization_config")) |other| {
        if (!equalQuant(q, try quantization(a, other))) return error.UnsupportedLfm3Quantization;
    }
    return .{ .arena = arena, .kind = kind, .context = context, .quant = q };
}
