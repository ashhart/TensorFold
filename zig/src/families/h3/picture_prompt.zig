//! MiniMax-H3-Image's prompt rules on the host: the encoder's ids, tags and positions, references' sizes, configs.
const std = @import("std");
const rope = @import("qwen_vl_rope");
const picture = @import("picture");
const bf16_round = @import("bf16_round");

/// Each text segment (a reference's label, the prompt) is cut to its first this many tokens on its own.
pub const segment_tokens = 4096;
/// The transformer reads the hidden state leaving the encoder's decoder layer index 49: 50 layers are run.
pub const encoder_layers = 50;
pub const head_dim = 128;
/// The Qwen copy of a reference: its short edge exactly this, both sides multiples of `qwen_factor`.
pub const qwen_edge = 1024;
/// Patch 16 times merge 2: the processor's side unit, and one vision token per 2 x 2 patches.
pub const qwen_factor = 32;
pub const merge = 2;
/// Each row's tag selects its AdaLN group in the transformer: vision-block rows 0, all other text rows 1.
pub const tag_visual: i32 = 0;
pub const tag_text: i32 = 1;

/// The encoder's text model as run: `layers` of its decoder layers (50 of the released 64), heads of 128.
pub const Text = struct {
    hidden: usize,
    heads: usize,
    kv_heads: usize,
    ffn: usize,
    vocab: usize,
    layers: usize,
    eps: f32,
    theta: f32,
    /// Interleaved M-RoPE's frequencies per axis (t, h, w).
    sections: [3]u32,
};

pub const released_text: Text = .{ .hidden = 5120, .heads = 64, .kv_heads = 8, .ffn = 25600, .vocab = 151936, .layers = encoder_layers, .eps = 1e-6, .theta = 5e6, .sections = .{ 24, 20, 20 } };

/// The transformer's text front end: condition_proj `text` to `hidden` wide, then refiner blocks of `heads` x 128.
pub const Refiner = struct {
    text: usize,
    hidden: usize,
    heads: usize,
    mlp: usize,
    blocks: usize,
    eps: f32,
    qk_eps: f32,
    final_eps: f32,
};

pub const released_refiner: Refiner = .{ .text = 5120, .hidden = 5376, .heads = 56, .mlp = 14336, .blocks = 2, .eps = 1e-5, .qk_eps = 1e-5, .final_eps = 1e-5 };

/// The vision block's tokens: `vision_start_token_id`, `image_token_id` and `vision_end_token_id`.
pub const Marks = struct { start: u32, image: u32, end: u32 };

/// The encoder's input: the ids, each id's tag, and where each reference's image-pad run starts.
pub const Presentation = struct {
    ids: []u32,
    tags: []i32,
    at: []u32,

    pub fn deinit(p: *Presentation, a: std.mem.Allocator) void {
        a.free(p.ids);
        a.free(p.tags);
        a.free(p.at);
    }
};

/// The merged vision tokens of a reference whose patch grid is `gh` x `gw`: one per 2 x 2 patches.
pub fn visionTokens(gh: usize, gw: usize) !usize {
    if (gh == 0 or gw == 0 or gh % merge != 0 or gw % merge != 0) return error.H3BadVisionGrid;
    return gh / merge * (gw / merge);
}

/// `tok`'s ids of each segment on its own, cut to 4,096: per reference `Picture i: ` and its vision block, the prompt.
pub fn present(a: std.mem.Allocator, tok: anytype, marks: Marks, references: []const usize, prompt: []const u8) !Presentation {
    var ids: std.ArrayList(u32) = .empty;
    errdefer ids.deinit(a);
    var tags: std.ArrayList(i32) = .empty;
    errdefer tags.deinit(a);
    const at = try a.alloc(u32, references.len);
    errdefer a.free(at);
    var label: [32]u8 = undefined;
    for (references, 0..) |n, i| {
        if (n == 0) return error.H3BadVisionGrid;
        try segment(a, tok, try std.fmt.bufPrint(&label, "Picture {d}: ", .{i + 1}), &ids, &tags);
        try ids.append(a, marks.start);
        at[i] = std.math.cast(u32, ids.items.len) orelse return error.H3PromptTooLong;
        try ids.appendNTimes(a, marks.image, n);
        try ids.append(a, marks.end);
        try tags.appendNTimes(a, tag_visual, n + 2);
    }
    try segment(a, tok, prompt, &ids, &tags);
    if (ids.items.len == 0) return error.H3EmptyPrompt;
    if (ids.items.len > std.math.maxInt(u32)) return error.H3PromptTooLong;
    const out_tags = try tags.toOwnedSlice(a);
    errdefer a.free(out_tags);
    return .{ .ids = try ids.toOwnedSlice(a), .tags = out_tags, .at = at };
}

fn segment(a: std.mem.Allocator, tok: anytype, text: []const u8, ids: *std.ArrayList(u32), tags: *std.ArrayList(i32)) !void {
    const all = try tok.encode(a, text);
    defer a.free(all);
    const kept = all[0..@min(all.len, segment_tokens)];
    try ids.appendSlice(a, kept);
    try tags.appendNTimes(a, tag_text, kept.len);
}

/// Each row's (t, h, w): text counts up; a reference's pads sit at its grid; text resumes past the larger side.
pub fn positions(a: std.mem.Allocator, p: Presentation, marks: Marks, grids: []const [2]u32) ![][3]i32 {
    if (grids.len != p.at.len or grids.len > rope.max_images) return error.H3BadVisionGrid;
    var images: [rope.max_images]struct { at: u32, gh: u32, gw: u32, tokens: u32 } = undefined;
    for (grids, p.at, 0..) |g, at, i| images[i] = .{ .at = at, .gh = g[0], .gw = g[1], .tokens = @intCast(try visionTokens(g[0], g[1])) };
    const layout = rope.Layout.init(p.ids, images[0..grids.len], .{ .start = marks.start, .image = marks.image, .end = marks.end }, merge) catch return error.H3BadVisionGrid;
    const out = try a.alloc([3]i32, p.ids.len);
    for (out, 0..) |*o, i| o.* = layout.position(@intCast(i));
    return out;
}

/// The encoder's rotary tables [rows, 64]: frequency i at its interleaved axis' position, fp32 angles, bf16 cos, sin.
pub fn rotary(cos: []f32, sin: []f32, pos: []const [3]i32, theta: f32, sections: [3]u32) void {
    const half = head_dim / 2;
    std.debug.assert(cos.len == pos.len * half and sin.len == cos.len);
    for (0..half) |i| {
        const exponent: f32 = @as(f32, @floatFromInt(2 * i)) / @as(f32, head_dim);
        const freq: f32 = 1.0 / std.math.pow(f32, theta, exponent);
        const axis = rope.axis(@intCast(i), sections);
        for (pos, 0..) |p, r| {
            const angle: f32 = @as(f32, @floatFromInt(p[axis])) * freq;
            cos[r * half + i] = bf16_round.round32(@floatCast(@cos(@as(f64, angle))));
            sin[r * half + i] = bf16_round.round32(@floatCast(@sin(@as(f64, angle))));
        }
    }
}

pub const Size = struct { width: usize, height: usize };

/// x rounded to the nearest integer, ties to even, as Python's round() does on a float64.
fn roundEven(x: f64) f64 {
    const f = @floor(x);
    const d = x - f;
    if (d > 0.5) return f + 1;
    if (d < 0.5) return f;
    return if (@mod(f, 2) == 0) f else f + 1;
}

/// The Qwen copy's size: a 1,024 short edge kept, else both sides times 1024 / short edge, snapped to 32 (half even).
pub fn qwenBound(width: usize, height: usize) !Size {
    if (width < qwen_factor or height < qwen_factor) return error.H3ReferenceTooSmall;
    const short = @min(width, height);
    if (short == qwen_edge) return .{ .width = width, .height = height };
    const scale = @as(f64, qwen_edge) / @as(f64, @floatFromInt(short));
    const snap = struct {
        fn side(v: f64) usize {
            return @as(usize, @intFromFloat(@max(1.0, roundEven(v / qwen_factor)))) * qwen_factor;
        }
    };
    return .{ .width = snap.side(@as(f64, @floatFromInt(width)) * scale), .height = snap.side(@as(f64, @floatFromInt(height)) * scale) };
}

/// The processor's size for a Qwen copy: smart resize to multiples of 32 in its pixel bounds, a no-op when snapped.
pub fn qwenProcessed(s: Size, min_pixels: u64, max_pixels: u64) !Size {
    const r = picture.patches.smartResize(@intCast(s.height), @intCast(s.width), qwen_factor, min_pixels, max_pixels) catch return error.H3ReferenceAspect;
    return .{ .width = r.w, .height = r.h };
}

/// The Qwen copy's RGB pixels: the original when its short edge is 1,024, else Pillow's LANCZOS to `qwenBound`.
pub fn qwenCopy(a: std.mem.Allocator, rgb: picture.Rgb) !picture.Rgb {
    const s = try qwenBound(rgb.width, rgb.height);
    if (s.width == rgb.width and s.height == rgb.height) return .{ .width = s.width, .height = s.height, .pixels = try a.dupe(u8, rgb.pixels) };
    const view = picture.resample.View.of(rgb.pixels, rgb.width, rgb.height);
    return .{ .width = s.width, .height = s.height, .pixels = try picture.resample.resizeWith(a, .pillow, .lanczos, view, s.width, s.height) };
}

fn field(o: std.json.ObjectMap, key: []const u8) !std.json.Value {
    return o.get(key) orelse {
        std.log.warn("h3 prompt: the config has no {s}", .{key});
        return error.H3BadPromptConfig;
    };
}

fn object(o: std.json.ObjectMap, key: []const u8) !std.json.ObjectMap {
    const v = try field(o, key);
    return if (v == .object) v.object else error.H3BadPromptConfig;
}

fn integer(o: std.json.ObjectMap, key: []const u8) !i64 {
    const v = try field(o, key);
    return if (v == .integer) v.integer else error.H3BadPromptConfig;
}

fn number(o: std.json.ObjectMap, key: []const u8) !f64 {
    return switch (try field(o, key)) {
        .integer => |n| @floatFromInt(n),
        .float => |f| f,
        else => error.H3BadPromptConfig,
    };
}

/// Each `want` pair's integer field equal to its value, else the released geometry is refused.
fn expect(o: std.json.ObjectMap, comptime what: []const u8, want: anytype) !void {
    inline for (want) |kv| if (try integer(o, kv[0]) != kv[1]) {
        std.log.warn("h3 prompt: {s} {s} is {d}, not the released {d}", .{ what, kv[0], try integer(o, kv[0]), kv[1] });
        return error.H3UnsupportedPromptModel;
    };
}

fn expectString(o: std.json.ObjectMap, key: []const u8, want: []const u8) !void {
    const v = try field(o, key);
    if (v != .string or !std.mem.eql(u8, v.string, want)) return error.H3UnsupportedPromptModel;
}

/// The checked text encoder: its text model as run, the vision block's ids and the vision tower's merge.
pub const Encoder = struct { text: Text, marks: Marks };

/// MiniMax-H3's `text_encoder/config.json`: Qwen3-VL at the released geometry, else an error.
pub fn checkEncoder(a: std.mem.Allocator, json: []const u8) !Encoder {
    const doc = std.json.parseFromSliceLeaky(std.json.Value, a, json, .{}) catch return error.H3BadPromptConfig;
    if (doc != .object) return error.H3BadPromptConfig;
    try expectString(doc.object, "model_type", "qwen3_vl");
    const t = try object(doc.object, "text_config");
    const r = released_text;
    try expect(t, "text_config", .{ .{ "hidden_size", r.hidden }, .{ "num_hidden_layers", 64 }, .{ "num_attention_heads", r.heads }, .{ "num_key_value_heads", r.kv_heads }, .{ "head_dim", head_dim }, .{ "intermediate_size", r.ffn }, .{ "vocab_size", r.vocab } });
    try expectString(t, "hidden_act", "silu");
    const bias = try field(t, "attention_bias");
    if (bias != .bool or bias.bool) return error.H3UnsupportedPromptModel;
    if (@as(f32, @floatCast(try number(t, "rms_norm_eps"))) != r.eps or try number(t, "rope_theta") != r.theta) return error.H3UnsupportedPromptModel;
    const rs = try object(t, "rope_scaling");
    const inter = try field(rs, "mrope_interleaved");
    if (inter != .bool or !inter.bool) return error.H3UnsupportedPromptModel;
    try expectString(rs, "rope_type", "default");
    const sec = try field(rs, "mrope_section");
    if (sec != .array or sec.array.items.len != 3) return error.H3UnsupportedPromptModel;
    for (sec.array.items, r.sections) |v, want| if (v != .integer or v.integer != want) return error.H3UnsupportedPromptModel;
    const v = try object(doc.object, "vision_config");
    try expect(v, "vision_config", .{ .{ "spatial_merge_size", merge }, .{ "patch_size", qwen_factor / merge }, .{ "temporal_patch_size", 2 }, .{ "out_hidden_size", r.hidden } });
    var marks: Marks = undefined;
    inline for (.{ .{ "start", "vision_start_token_id" }, .{ "image", "image_token_id" }, .{ "end", "vision_end_token_id" } }) |kv| {
        const id = try integer(doc.object, kv[1]);
        if (id < 0 or id >= r.vocab) return error.H3BadPromptConfig;
        @field(marks, kv[0]) = @intCast(id);
    }
    return .{ .text = r, .marks = marks };
}

/// `transformer/config.json`'s text front end at the released geometry: condition_proj and the 2-block refiner.
pub fn checkRefiner(a: std.mem.Allocator, json: []const u8) !Refiner {
    const doc = std.json.parseFromSliceLeaky(std.json.Value, a, json, .{}) catch return error.H3BadPromptConfig;
    if (doc != .object) return error.H3BadPromptConfig;
    const o = doc.object;
    var r = released_refiner;
    try expect(o, "transformer", .{ .{ "text_dim", r.text }, .{ "hidden_size", r.hidden }, .{ "num_attention_heads", r.heads }, .{ "attention_head_dim", head_dim }, .{ "ffn_hidden_size", r.mlp }, .{ "token_refiner_num_layers", r.blocks } });
    inline for (.{ .{ "eps", "norm_eps" }, .{ "qk_eps", "qk_norm_eps" }, .{ "final_eps", "final_norm_eps" } }) |kv| {
        const e = try number(o, kv[1]);
        if (!(e > 0 and e < 1)) return error.H3BadPromptConfig;
        @field(r, kv[0]) = @floatCast(e);
    }
    return r;
}

/// The text model's tensor prefix in a checkpoint index's names: what precedes its first layer's input norm.
pub fn textPrefix(names: []const []const u8) ![]const u8 {
    const tail = "layers.0.input_layernorm.weight";
    var found: ?[]const u8 = null;
    for (names) |n| {
        if (!std.mem.endsWith(u8, n, tail) or std.mem.indexOf(u8, n, "visual") != null) continue;
        const p = n[0 .. n.len - tail.len];
        if (found) |f| if (!std.mem.eql(u8, f, p)) return error.H3AmbiguousTextModel;
        found = p;
    }
    return found orelse error.H3NoTextModel;
}
