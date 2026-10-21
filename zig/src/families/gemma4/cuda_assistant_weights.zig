//! Gemma 4's MTP assistant weights on CUDA: config admitted as Metal's, bf16 linears and the tied head made 4-bit.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const cfg = @import("cuda_config.zig");
const kern = @import("cuda_kernels.zig");

const Affine4 = cuda.qlinear.Affine4;

/// Layers an admitted assistant has at most.
pub const max_layers = 8;
/// The stock assistant's sizes (google/gemma-4-12B-it-assistant); another config is refused with its reason.
const stock = .{ .hidden = 1024, .layers = 4, .intermediate = 8192, .heads = 16 };

pub const Config = struct {
    layers: usize,
    hidden: usize,
    intermediate: usize,
    heads: usize,
    /// Each layer's kind: true for a sliding layer (the target's last sliding cache), false for a full one.
    sliding: [max_layers]bool = @splat(true),

    pub fn headDim(c: *const Config, i: usize) usize {
        return if (c.sliding[i]) cfg.sliding_dim else cfg.global_dim;
    }

    /// Admission as Metal's assistant_weights.Config.parse, refusal for refusal.
    pub fn parse(gpa: std.mem.Allocator, bytes: []const u8, geo: cfg.Geometry) !Config {
        var parsed = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
        defer parsed.deinit();
        const o = if (parsed.value == .object) parsed.value.object else return error.BadDraftConfig;
        if (!isString(o.get("model_type"), "gemma4_unified_assistant")) return error.UnsupportedAssistantType;
        if (isTrue(o, "use_ordered_embeddings")) return error.AssistantOrderedEmbeddingsNotServed;
        if (try integer(o, "backbone_hidden_size") != geo.hidden) return error.AssistantForAnotherTarget;
        if (o.get("quantization") != null or o.get("quantization_config") != null) return error.UnsupportedAssistantQuantization;
        const t = if (o.get("text_config")) |v| (if (v == .object) v.object else return error.BadDraftConfig) else return error.BadDraftConfig;
        if (!isString(t.get("model_type"), "gemma4_unified_text")) return error.BadDraftConfig;
        var c: Config = .{ .layers = @intCast(try integer(t, "num_hidden_layers")), .hidden = @intCast(try integer(t, "hidden_size")), .intermediate = @intCast(try integer(t, "intermediate_size")), .heads = @intCast(try integer(t, "num_attention_heads")) };
        if (c.hidden != stock.hidden or c.layers != stock.layers or c.intermediate != stock.intermediate or c.heads != stock.heads) return error.UnsupportedAssistantGeometry;
        if (try integer(t, "num_key_value_heads") != geo.sliding_kv_heads or try integer(t, "num_global_key_value_heads") != geo.global_kv_heads) return error.AssistantForAnotherTarget;
        if (try integer(t, "num_kv_shared_layers") != c.layers) return error.UnsupportedAssistantGeometry;
        const dims = .{ .{ "head_dim", cfg.sliding_dim }, .{ "global_head_dim", cfg.global_dim }, .{ "sliding_window", cfg.window }, .{ "vocab_size", cfg.vocab } };
        inline for (dims) |d| if (try integer(t, d[0]) != d[1]) return error.UnsupportedAssistantGeometry;
        if (isTrue(t, "enable_moe_block") or isTrue(t, "use_double_wide_mlp") or isTrue(t, "attention_bias")) return error.UnsupportedAssistantGeometry;
        if (t.get("hidden_size_per_layer_input")) |v| if (v != .integer or v.integer != 0) return error.UnsupportedAssistantGeometry;
        if (!isTrue(t, "attention_k_eq_v") or !isTrue(t, "tie_word_embeddings")) return error.UnsupportedAssistantGeometry;
        if (!isString(t.get("hidden_activation"), "gelu_pytorch_tanh")) return error.UnsupportedAssistantGeometry;
        if (t.get("final_logit_softcapping")) |v| if (v != .null) return error.UnsupportedAssistantGeometry;
        if (try number(t, "rms_norm_eps") != 1e-6) return error.UnsupportedAssistantGeometry;
        const ropes = if (t.get("rope_parameters")) |v| (if (v == .object) v.object else return error.BadDraftConfig) else return error.BadDraftConfig;
        try rope(ropes, "sliding_attention", "default", cfg.sliding_theta, null);
        try rope(ropes, "full_attention", "proportional", cfg.global_theta, 0.25);
        const kinds = t.get("layer_types") orelse return error.BadDraftConfig;
        if (kinds != .array or kinds.array.items.len != c.layers) return error.BadDraftConfig;
        for (kinds.array.items, 0..) |k, i| {
            c.sliding[i] = isString(k, "sliding_attention");
            if (!c.sliding[i] and !isString(k, "full_attention")) return error.UnsupportedAssistantGeometry;
        }
        return c;
    }
};

fn isString(v: ?std.json.Value, expected: []const u8) bool {
    const s = v orelse return false;
    return s == .string and std.mem.eql(u8, s.string, expected);
}

fn isTrue(o: std.json.ObjectMap, key: []const u8) bool {
    const v = o.get(key) orelse return false;
    return v == .bool and v.bool;
}

fn integer(o: std.json.ObjectMap, key: []const u8) !i64 {
    const v = o.get(key) orelse return error.BadDraftConfig;
    return if (v == .integer) v.integer else error.BadDraftConfig;
}

fn number(o: std.json.ObjectMap, key: []const u8) !f64 {
    const v = o.get(key) orelse return error.BadDraftConfig;
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => error.BadDraftConfig,
    };
}

fn rope(o: std.json.ObjectMap, kind: []const u8, kind_type: []const u8, theta: f64, partial: ?f64) !void {
    const v = o.get(kind) orelse return error.BadDraftConfig;
    const r = if (v == .object) v.object else return error.BadDraftConfig;
    if (!isString(r.get("rope_type"), kind_type) or try number(r, "rope_theta") != theta) return error.UnsupportedAssistantRotary;
    if (partial) |p| if (try number(r, "partial_rotary_factor") != p) return error.UnsupportedAssistantRotary;
}

/// Whether `dir` holds a Gemma 4 assistant (its config's model_type names one).
pub fn isAssistant(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !bool {
    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20));
    defer gpa.free(bytes);
    var parsed = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
    defer parsed.deinit();
    const t = if (parsed.value == .object) parsed.value.object.get("model_type") else null;
    return t != null and t.? == .string and std.mem.startsWith(u8, t.?.string, "gemma4") and std.mem.endsWith(u8, t.?.string, "assistant");
}

/// A group of 64 bf16 values as MLX quantizes it to 4 bits (Metal's dflash_weights.quantizeGroup).
fn quantizeGroup(x: []const u16, words: []u32, scale: *u16, bias: *u16) void {
    var lo: f32 = std.math.inf(f32);
    var hi: f32 = -std.math.inf(f32);
    for (x) |v| {
        lo = @min(lo, bf(v));
        hi = @max(hi, bf(v));
    }
    const bins: f32 = 15;
    const scale0 = @max(hi - lo, 1e-7) / bins;
    const from_min = @abs(lo) > @abs(hi);
    const edge = if (from_min) lo else hi;
    const q0 = @round(@abs(edge) / scale0);
    const s_f: f32 = if (q0 == 0) scale0 else (if (from_min) @abs(edge) else -@abs(edge)) / q0;
    scale.* = toBf(s_f);
    bias.* = toBf(if (q0 == 0) lo else edge);
    const s = bf(scale.*);
    const b = bf(bias.*);
    @memset(words, 0);
    for (x, 0..) |v, i| {
        const q = std.math.clamp(@round((bf(v) - b) / s), 0, bins);
        words[i / 8] |= @as(u32, @intFromFloat(q)) << @intCast((i % 8) * 4);
    }
}

fn bf(v: u16) f32 {
    return @bitCast(@as(u32, v) << 16);
}

/// Round to nearest even, as MLX casts.
fn toBf(f: f32) u16 {
    const b: u32 = @bitCast(f);
    return @intCast((b + 0x7FFF + ((b >> 16) & 1)) >> 16);
}

const Job = struct {
    src: []const u16,
    k: usize,
    words: []u32,
    scales: []u16,
    biases: []u16,
    fn run(j: Job, from: usize, to: usize) void {
        const groups = j.k / 64;
        for (from..to) |r| for (0..groups) |g| {
            quantizeGroup(j.src[r * j.k + g * 64 ..][0..64], j.words[(r * groups + g) * 8 ..][0..8], &j.scales[r * groups + g], &j.biases[r * groups + g]);
        };
    }
};

pub const Layer = struct {
    q: Affine4,
    o: Affine4,
    gate: Affine4,
    up: Affine4,
    down: Affine4,
    q_norm: u64,
    input_norm: u64,
    post_attention: u64,
    pre_feedforward: u64,
    post_feedforward: u64,
    scalar: u64,
};

pub const Weights = struct {
    pre: Affine4, // [embedding | hidden] to the assistant's width
    post: Affine4, // its output back to the target's
    head: Affine4, // the tied head: embed_tokens as a 4-bit linear over the vocabulary
    norm: u64,
    layers: [max_layers]Layer,
    buffers: std.ArrayList(cuda.DeviceBuffer) = .empty,

    pub fn deinit(w: *Weights, gpa: std.mem.Allocator) void {
        for (w.buffers.items) |*b| b.free();
        w.buffers.deinit(gpa);
    }
};

pub const Loader = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    s: cuda.Stream,
    k: *const kern.Kernels,
    ck: *core.checkpoint.Checkpoint,
    bufs: *std.ArrayList(cuda.DeviceBuffer), // owns what it allocates

    pub fn keep(x: *Loader, b: cuda.DeviceBuffer) !u64 {
        errdefer {
            var y = b;
            y.free();
        }
        try x.bufs.append(x.gpa, b);
        return b.ptr;
    }

    pub fn vector(x: *Loader, name: []const u8, len: usize) !u64 {
        return x.keep(try cuda.DeviceBuffer.fromHost(x.d, (try x.ck.expect(name, .bf16, &.{len})).bytes));
    }

    /// An n-by-k bf16 matrix quantized to MLX affine 4/64 on every core, packed into qlinear's tiles.
    pub fn affine4(x: *Loader, name: []const u8, n: usize, k: usize) !Affine4 {
        const t = try x.ck.expect(name, .bf16, &.{ n, k });
        const src = std.mem.bytesAsSlice(u16, @as([]align(1) const u8, t.bytes));
        const host = try x.gpa.alloc(u16, n * k);
        defer x.gpa.free(host);
        for (host, src) |*h, v| h.* = v;
        const words = try x.gpa.alloc(u32, n * k / 8);
        defer x.gpa.free(words);
        const sb = try x.gpa.alloc(u16, 2 * n * k / 64);
        defer x.gpa.free(sb);
        const job: Job = .{ .src = host, .k = k, .words = words, .scales = sb[0 .. n * k / 64], .biases = sb[n * k / 64 ..] };
        const threads = @min(std.Thread.getCpuCount() catch 1, 16);
        const per = (n + threads - 1) / threads;
        var handles: [16]std.Thread = undefined;
        var spawned: usize = 0;
        var at: usize = 0;
        while (at < n) : (at += per) {
            handles[spawned] = try std.Thread.spawn(.{}, Job.run, .{ job, at, @min(n, at + per) });
            spawned += 1;
        }
        for (handles[0..spawned]) |h| h.join();
        var raw_w = try cuda.DeviceBuffer.fromHost(x.d, std.mem.sliceAsBytes(words));
        defer raw_w.free();
        var raw_sb = try cuda.DeviceBuffer.fromHost(x.d, std.mem.sliceAsBytes(sb));
        defer raw_sb.free();
        const l = Affine4.layout(n, k);
        const q: Affine4 = .{ .w = try x.keep(try cuda.DeviceBuffer.alloc(x.d, l.words)), .s = try x.keep(try cuda.DeviceBuffer.alloc(x.d, l.scales)), .b = try x.keep(try cuda.DeviceBuffer.alloc(x.d, l.scales)), .n = n, .k = k, .npad = l.npad };
        try x.k.affine.pack(x.s, raw_w.ptr, raw_sb.ptr, raw_sb.ptr + n * k / 64 * 2, q);
        try x.s.synchronize(); // the raw buffers are freed on return
        return q;
    }
};

/// Every tensor checked by name, dtype and shape; the bf16 linears and the tied head made 4-bit as Metal's.
pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, s: cuda.Stream, k: *const kern.Kernels, dir: []const u8, c: *const Config, geo: cfg.Geometry) !Weights {
    var ck = try core.checkpoint.Checkpoint.openModel(gpa, io, dir);
    defer ck.close();
    var w: Weights = .{ .pre = undefined, .post = undefined, .head = undefined, .norm = 0, .layers = undefined };
    errdefer w.deinit(gpa);
    var x: Loader = .{ .gpa = gpa, .d = d, .s = s, .k = k, .ck = &ck, .bufs = &w.buffers };
    const hid = c.hidden;
    w.pre = try x.affine4("pre_projection.weight", hid, 2 * geo.hidden);
    w.post = try x.affine4("post_projection.weight", geo.hidden, hid);
    w.head = try x.affine4("model.embed_tokens.weight", cfg.vocab, hid);
    w.norm = try x.vector("model.norm.weight", hid);
    var buf: [160]u8 = undefined;
    for (w.layers[0..c.layers], 0..) |*l, i| {
        const q = c.heads * c.headDim(i);
        const name = struct {
            fn of(b: []u8, layer: usize, suffix: []const u8) ![]const u8 {
                return std.fmt.bufPrint(b, "model.layers.{d}.{s}", .{ layer, suffix });
            }
        }.of;
        l.q = try x.affine4(try name(&buf, i, "self_attn.q_proj.weight"), q, hid);
        l.o = try x.affine4(try name(&buf, i, "self_attn.o_proj.weight"), hid, q);
        l.gate = try x.affine4(try name(&buf, i, "mlp.gate_proj.weight"), c.intermediate, hid);
        l.up = try x.affine4(try name(&buf, i, "mlp.up_proj.weight"), c.intermediate, hid);
        l.down = try x.affine4(try name(&buf, i, "mlp.down_proj.weight"), hid, c.intermediate);
        l.q_norm = try x.vector(try name(&buf, i, "self_attn.q_norm.weight"), c.headDim(i));
        l.input_norm = try x.vector(try name(&buf, i, "input_layernorm.weight"), hid);
        l.post_attention = try x.vector(try name(&buf, i, "post_attention_layernorm.weight"), hid);
        l.pre_feedforward = try x.vector(try name(&buf, i, "pre_feedforward_layernorm.weight"), hid);
        l.post_feedforward = try x.vector(try name(&buf, i, "post_feedforward_layernorm.weight"), hid);
        l.scalar = try x.vector(try name(&buf, i, "layer_scalar"), 1);
    }
    return w;
}

const fixture = @embedFile("fixtures/assistant_config.json");

test "the 12B's assistant is admitted against the 12B, and refused against the 31B" {
    const c = try Config.parse(std.testing.allocator, fixture, cfg.unified12b);
    try std.testing.expectEqual(@as(usize, 4), c.layers);
    try std.testing.expect(c.sliding[0] and c.sliding[2] and !c.sliding[3]);
    try std.testing.expectEqual(@as(usize, 512), c.headDim(3));
    try std.testing.expectError(error.AssistantForAnotherTarget, Config.parse(std.testing.allocator, fixture, cfg.dense31b));
}

test "a group quantizes as MLX's affine 4-bit: every value within half a step" {
    var x: [64]u16 = undefined;
    for (&x, 0..) |*v, i| v.* = toBf(@as(f32, @floatFromInt(i)) * 0.03 - 0.7);
    var words: [8]u32 = undefined;
    var scale: u16 = 0;
    var bias: u16 = 0;
    quantizeGroup(&x, &words, &scale, &bias);
    for (x, 0..) |v, i| {
        const q: f32 = @floatFromInt((words[i / 8] >> @intCast((i % 8) * 4)) & 0xF);
        try std.testing.expect(@abs(q * bf(scale) + bf(bias) - bf(v)) <= @abs(bf(scale)) * 0.51);
    }
}
