//! LFM3-VL's vision tower and projector for every backend: geometry, every tensor's name and shape, image rows.
const std = @import("std");
const config = @import("config.zig");

/// A tower layer's mixer: the gated 3 x 3 depthwise conv or full attention (vision_config's layer_types).
pub const Mixer = enum { conv, attention };
pub const max_layers = 32;

/// vision_config's geometry and the projector's widths, as the backends read them.
pub const Vision = struct {
    hidden: usize,
    intermediate: usize,
    layers: usize,
    heads: usize,
    kv_heads: usize,
    channels: usize,
    kernel: usize, // conv_kernel_size: the depthwise conv's taps a side
    patch: usize,
    temporal: usize,
    merge: usize, // spatial_merge_size: merge x merge patches make one image row
    projector_hidden: usize,
    text_hidden: usize, // a projected image row: the text decoder's width
    mixers: [max_layers]Mixer = @splat(.conv),

    pub fn headDim(v: *const Vision) usize {
        return v.hidden / v.heads;
    }

    /// K and V rows: every KV head's head dim.
    pub fn kvWidth(v: *const Vision) usize {
        return v.kv_heads * v.headDim();
    }

    /// A patch row: channels, temporal frames, patch x patch pixels (1536).
    pub fn patchWidth(v: *const Vision) usize {
        return v.channels * v.temporal * v.patch * v.patch;
    }

    /// The projector's input row: one merge block's patch rows side by side (4096).
    pub fn merged(v: *const Vision) usize {
        return v.hidden * v.merge * v.merge;
    }

    /// Image rows (prompt tokens) of a gh x gw patch grid: one a merge block.
    pub fn tokens(v: *const Vision, gh: usize, gw: usize) usize {
        return gh / v.merge * (gw / v.merge);
    }
};

/// The geometry config.zig admits: LFM3-VL-2B (attention at every even layer from 2 on).
pub const lfm3_vl_2b: Vision = blk: {
    var v: Vision = .{ .hidden = 1024, .intermediate = 2560, .layers = 14, .heads = 16, .kv_heads = 8, .channels = 3, .kernel = 3, .patch = 16, .temporal = 2, .merge = 2, .projector_hidden = 4096, .text_hidden = config.hidden };
    for (0..v.layers) |i| v.mixers[i] = if (i >= 2 and i % 2 == 0) .attention else .conv;
    break :blk v;
};

const Object = std.json.ObjectMap;

fn integer(o: Object, key: []const u8) !usize {
    const v = o.get(key) orelse return error.BadLfm3VisionConfig;
    if (v != .integer or v.integer < 1) return error.BadLfm3VisionConfig;
    return std.math.cast(usize, v.integer) orelse error.BadLfm3VisionConfig;
}

fn object(o: Object, key: []const u8) !Object {
    const v = o.get(key) orelse return error.BadLfm3VisionConfig;
    return if (v == .object) v.object else error.BadLfm3VisionConfig;
}

/// The geometry of an LFM3-VL config.json; config.zig's `parse` decides whether the engines admit it.
pub fn read(gpa: std.mem.Allocator, bytes: []const u8) !Vision {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const doc = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), bytes, .{});
    if (doc != .object) return error.BadLfm3VisionConfig;
    const root = doc.object;
    const o = try object(root, "vision_config");
    var v: Vision = .{
        .hidden = try integer(o, "hidden_size"),
        .intermediate = try integer(o, "intermediate_size"),
        .layers = try integer(o, "num_hidden_layers"),
        .heads = try integer(o, "num_attention_heads"),
        .kv_heads = try integer(o, "num_key_value_heads"),
        .channels = try integer(o, "in_channels"),
        .kernel = try integer(o, "conv_kernel_size"),
        .patch = try integer(o, "patch_size"),
        .temporal = try integer(o, "temporal_patch_size"),
        .merge = try integer(o, "spatial_merge_size"),
        .projector_hidden = try integer(root, "projector_hidden_size"),
        .text_hidden = try integer(try object(root, "text_config"), "hidden_size"),
    };
    if (v.layers > max_layers or v.hidden % v.heads != 0 or v.heads % v.kv_heads != 0) return error.BadLfm3VisionConfig;
    if (try integer(root, "downsample_factor") != v.merge) return error.BadLfm3VisionConfig;
    if (o.get("head_dim")) |_| if (try integer(o, "head_dim") != v.headDim()) return error.BadLfm3VisionConfig;
    const types = o.get("layer_types") orelse return error.BadLfm3VisionConfig;
    if (types != .array or types.array.items.len != v.layers) return error.BadLfm3VisionConfig;
    for (types.array.items, 0..) |t, i| {
        if (t != .string) return error.BadLfm3VisionConfig;
        v.mixers[i] = if (std.mem.eql(u8, t.string, "conv")) .conv else if (std.mem.eql(u8, t.string, "full_attention")) .attention else return error.BadLfm3VisionConfig;
    }
    return v;
}

/// One vision tensor; its name is `prefix()`, then `layers.{i}.` for a layer tensor, then `path()`.
pub const Module = enum {
    patchifier,
    embedding_norm,
    operator_norm,
    ffn_norm,
    w1,
    w3,
    w2,
    conv_in,
    conv_out,
    spatial_conv,
    q_proj,
    k_proj,
    v_proj,
    out_proj,
    q_norm,
    k_norm,
    projector_norm,
    projector_norm_bias,
    linear_1,
    linear_1_bias,
    linear_2,
    linear_2_bias,

    pub fn prefix(m: Module) []const u8 {
        return if (@backingInt(m) >= @backingInt(Module.projector_norm)) "multi_modal_projector." else "vision_tower.";
    }

    /// Whether the tensor sits in a tower layer (`layers.{i}.`).
    pub fn layered(m: Module) bool {
        return @backingInt(m) >= @backingInt(Module.operator_norm) and @backingInt(m) < @backingInt(Module.projector_norm);
    }

    pub fn path(m: Module) []const u8 {
        return switch (m) {
            .patchifier => "patchifier.weight",
            .embedding_norm => "embedding_norm.weight",
            .operator_norm => "operator_norm.weight",
            .ffn_norm => "ffn_norm.weight",
            .w1 => "feed_forward.w1.weight",
            .w3 => "feed_forward.w3.weight",
            .w2 => "feed_forward.w2.weight",
            .conv_in => "conv.in_proj.weight",
            .conv_out => "conv.out_proj.weight",
            .spatial_conv => "conv.spatial_conv.weight",
            .q_proj => "self_attn.q_proj.weight",
            .k_proj => "self_attn.k_proj.weight",
            .v_proj => "self_attn.v_proj.weight",
            .out_proj => "self_attn.out_proj.weight",
            .q_norm => "self_attn.q_layernorm.weight",
            .k_norm => "self_attn.k_layernorm.weight",
            .projector_norm => "layer_norm.weight",
            .projector_norm_bias => "layer_norm.bias",
            .linear_1 => "linear_1.weight",
            .linear_1_bias => "linear_1.bias",
            .linear_2 => "linear_2.weight",
            .linear_2_bias => "linear_2.bias",
        };
    }
};

/// The tensors before the layers, a layer's by its mixer, and the projector's, in table order.
pub const tower_modules = [_]Module{ .patchifier, .embedding_norm };
pub const ffn_modules = [_]Module{ .operator_norm, .ffn_norm, .w1, .w3, .w2 };
pub const conv_modules = [_]Module{ .conv_in, .conv_out, .spatial_conv };
pub const attention_modules = [_]Module{ .q_proj, .k_proj, .v_proj, .out_proj, .q_norm, .k_norm };
pub const projector_modules = [_]Module{ .projector_norm, .projector_norm_bias, .linear_1, .linear_1_bias, .linear_2, .linear_2_bias };

/// Every tensor is stored bf16 in the MLX exports, quantized text decoder or not.
pub const Dtype = enum { bf16 };

pub const Shape = struct {
    dims: [4]usize = @splat(0),
    rank: u8 = 0,

    pub fn of(dims: []const usize) Shape {
        var s: Shape = .{ .rank = @intCast(dims.len) };
        @memcpy(s.dims[0..dims.len], dims);
        return s;
    }

    pub fn slice(s: *const Shape) []const usize {
        return s.dims[0..s.rank];
    }

    pub fn elems(s: Shape) usize {
        var n: usize = 1;
        for (s.dims[0..s.rank]) |d| n *= d;
        return n;
    }
};

/// A module's stored shape, rows first as safetensors holds them (the spatial conv in MLX's [C, kh, kw, 1]).
pub fn shape(v: Vision, m: Module) Shape {
    const h = v.hidden;
    return switch (m) {
        .patchifier => .of(&.{ h, v.patchWidth() }),
        .embedding_norm, .operator_norm, .ffn_norm, .projector_norm, .projector_norm_bias => .of(&.{h}),
        .w1, .w3 => .of(&.{ v.intermediate, h }),
        .w2 => .of(&.{ h, v.intermediate }),
        .conv_in => .of(&.{ 3 * h, h }),
        .conv_out, .q_proj, .out_proj => .of(&.{ h, h }),
        .spatial_conv => .of(&.{ h, v.kernel, v.kernel, 1 }),
        .k_proj, .v_proj => .of(&.{ v.kvWidth(), h }),
        .q_norm, .k_norm => .of(&.{v.headDim()}),
        .linear_1 => .of(&.{ v.projector_hidden, v.merged() }),
        .linear_1_bias => .of(&.{v.projector_hidden}),
        .linear_2 => .of(&.{ v.text_hidden, v.projector_hidden }),
        .linear_2_bias => .of(&.{v.text_hidden}),
    };
}

/// A module's full name; `layer` is required for a layer tensor and refused for the others.
pub fn name(buf: []u8, layer: ?usize, m: Module) ![]const u8 {
    if (m.layered() != (layer != null)) return error.BadLfm3VisionModule;
    if (layer) |i| return std.fmt.bufPrint(buf, "{s}layers.{d}.{s}", .{ m.prefix(), i, m.path() });
    return std.fmt.bufPrint(buf, "{s}{s}", .{ m.prefix(), m.path() });
}

pub const Tensor = struct {
    name: []const u8,
    shape: Shape,
    dtype: Dtype = .bf16,

    pub fn bytes(t: Tensor) usize {
        return t.shape.elems() * 2;
    }
};

/// One tensor's name (in `buf`), shape and dtype.
pub fn tensor(buf: []u8, v: Vision, layer: ?usize, m: Module) !Tensor {
    if (layer) |i| if (i >= v.layers or !inLayer(v, i, m)) return error.BadLfm3VisionModule;
    return .{ .name = try name(buf, layer, m), .shape = shape(v, m) };
}

/// Whether layer `i` holds module `m`: the norms and SwiGLU always, then its mixer's tensors.
pub fn inLayer(v: Vision, i: usize, m: Module) bool {
    const mixer: []const Module = if (v.mixers[i] == .attention) &attention_modules else &conv_modules;
    return std.mem.indexOfScalar(Module, &ffn_modules, m) != null or std.mem.indexOfScalar(Module, mixer, m) != null;
}

/// Tensors in the table (138 for LFM3-VL-2B).
pub fn count(v: Vision) usize {
    var n: usize = tower_modules.len + projector_modules.len;
    for (v.mixers[0..v.layers]) |mx| n += ffn_modules.len + if (mx == .attention) attention_modules.len else conv_modules.len;
    return n;
}

/// Every tower and projector tensor in load order: patchifier, norm, layer by layer, projector; names in `a`.
pub fn table(a: std.mem.Allocator, v: Vision) ![]Tensor {
    const out = try a.alloc(Tensor, count(v));
    errdefer a.free(out);
    var n: usize = 0;
    errdefer for (out[0..n]) |t| a.free(t.name);
    var buf: [128]u8 = undefined;
    for (tower_modules) |m| {
        out[n] = try owned(a, try tensor(&buf, v, null, m));
        n += 1;
    }
    for (0..v.layers) |i| {
        const mixer: []const Module = if (v.mixers[i] == .attention) &attention_modules else &conv_modules;
        for ([_][]const Module{ ffn_modules[0..2], mixer, ffn_modules[2..] }) |group| for (group) |m| {
            out[n] = try owned(a, try tensor(&buf, v, i, m));
            n += 1;
        };
    }
    for (projector_modules) |m| {
        out[n] = try owned(a, try tensor(&buf, v, null, m));
        n += 1;
    }
    return out;
}

fn owned(a: std.mem.Allocator, t: Tensor) !Tensor {
    return .{ .name = try a.dupe(u8, t.name), .shape = t.shape, .dtype = t.dtype };
}

pub fn freeTable(a: std.mem.Allocator, t: []Tensor) void {
    for (t) |x| a.free(x.name);
    a.free(t);
}

/// Whether a checkpoint key belongs to the tower or the projector.
pub fn isVision(key: []const u8) bool {
    return std.mem.startsWith(u8, key, "vision_tower.") or std.mem.startsWith(u8, key, "multi_modal_projector.");
}

/// A checkpoint's keys hold exactly the table's vision tensors (other keys are the decoder's and pass).
pub fn check(t: []const Tensor, keys: []const []const u8) !void {
    var found: usize = 0;
    for (keys) |k| {
        if (!isVision(k)) continue;
        for (t) |x| {
            if (std.mem.eql(u8, x.name, k)) break;
        } else return error.UnexpectedLfm3VisionTensor;
        found += 1;
    }
    for (t) |x| {
        for (keys) |k| {
            if (std.mem.eql(u8, x.name, k)) break;
        } else return error.MissingLfm3VisionTensor;
    }
    if (found != t.len) return error.UnexpectedLfm3VisionTensor;
}

/// A picture's projected rows in the prompt: positions [at, at + rows) read image rows from `source` on.
pub const Span = struct { at: u32, rows: u32, source: u32 };

/// What a prompt chunk takes of a span: `rows` image rows from `source` into chunk rows from `row`.
pub const Copy = struct { source: usize, row: usize, rows: usize };

/// The part of `span` a prompt chunk of `rows` rows at position `at` reads; null when they don't overlap.
pub fn overlap(span: Span, at: usize, rows: usize) ?Copy {
    const first = @max(at, span.at);
    const end = @min(at + rows, @as(usize, span.at) + span.rows);
    if (first >= end) return null;
    return .{ .source = @as(usize, span.source) + first - span.at, .row = first - at, .rows = end - first };
}

/// Spans of the pictures (`.at`, `.tokens`) a prompt chunk [from, to) reads, packed from image row 0 in order.
pub fn place(out: []Span, images: anytype, from: usize, to: usize, capacity: usize) ![]Span {
    var used: usize = 0;
    var n: usize = 0;
    for (images) |img| {
        if (@as(usize, img.at) + img.tokens <= from or img.at >= to) continue;
        if (img.tokens > capacity -| used) return error.ImageTooLarge;
        if (n == out.len) return error.TooManyImages;
        out[n] = .{ .at = img.at, .rows = img.tokens, .source = @intCast(used) };
        n += 1;
        used += img.tokens;
    }
    return out[0..n];
}
