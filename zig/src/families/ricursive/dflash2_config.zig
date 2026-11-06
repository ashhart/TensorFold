//! ricursiveAI's DFlash2 drafter for Ripple (Ripple-65B-Preview-DFlash2): its config and bf16 weights as published.
const std = @import("std");
const ck = @import("../../core/checkpoint_metal.zig");
const wts = @import("weights.zig");
const cfg = @import("config.zig");

/// The release's 16,384-token draft window: the global layer's training rows (`inference/server/README.md`).
pub const global_window = 16384;
/// Streams a drafted server serves at most; at 4 and 7 an M5 Ultra decodes drafted rounds slower than plain ones.
pub const max_streams = 2;
/// Layers, tapped target layers and convolution taps this drafter has (the release's).
pub const max_layers = 5;
pub const max_taps = 5;

pub const Config = struct {
    layers: u32,
    heads: u32,
    kv_heads: u32,
    hidden: u32,
    ffn: u32,
    vocab: u32,
    window: u32, // the sliding layers' window
    block: u32,
    mask: u32,
    rank: u32,
    top: u32,
    group: u32,
    conv: u32, // the block convolution's taps
    taps: [max_taps]u32,
    global: [max_layers]bool = @splat(false), // full attention over the draft window
    rope: [max_layers]bool = @splat(true), // false on the NoPE layer
    theta: f32,
    eps: f32,

    /// Keys a layer sees before a row: the sliding window, or the draft window on the global layer.
    pub fn span(c: *const Config, l: usize) u32 {
        return if (c.global[l]) global_window else c.window;
    }

    /// A layer's ring of keys: its span and a block past it, whole 128-key blocks.
    pub fn ring(c: *const Config, l: usize) u32 {
        return std.mem.alignForward(u32, c.span(l) + c.block, 128);
    }

    /// The coefficients a row's kernel projection writes: [side][tap][group].
    pub fn dyn(c: *const Config) u32 {
        return 2 * c.conv * (c.hidden / c.group);
    }
};

pub const Error = error{ NotRippleDFlash2, UnqualifiedRippleDFlash2 } || std.mem.Allocator.Error;

/// The drafter's config.json; `target` its Ripple: the published drafter's geometry exactly, or a refusal.
pub fn parse(gpa: std.mem.Allocator, bytes: []const u8, target: *const cfg.Config) Error!Config {
    var doc = std.json.parseFromSlice(std.json.Value, gpa, bytes, .{}) catch return error.NotRippleDFlash2;
    defer doc.deinit();
    if (doc.value != .object) return error.NotRippleDFlash2;
    const o = doc.value.object;
    const arch = o.get("architectures") orelse return error.NotRippleDFlash2;
    if (arch != .array or arch.array.items.len != 1 or arch.array.items[0] != .string or !std.mem.eql(u8, arch.array.items[0].string, "DFlash2DraftModel")) return error.NotRippleDFlash2;
    const d = o.get("dflash_config") orelse return error.NotRippleDFlash2;
    if (d != .object) return error.NotRippleDFlash2;
    const dc = d.object;
    var c: Config = .{
        .layers = try int(o, "num_hidden_layers"),
        .heads = try int(o, "num_attention_heads"),
        .kv_heads = try int(o, "num_key_value_heads"),
        .hidden = try int(o, "hidden_size"),
        .ffn = try int(o, "intermediate_size"),
        .vocab = try int(o, "vocab_size"),
        .window = try int(o, "sliding_window"),
        .block = try int(dc, "block_size"),
        .mask = try int(dc, "mask_token_id"),
        .rank = try int(dc, "selector_rank"),
        .top = try int(dc, "selector_top_k"),
        .group = try int(dc, "conv_group_size"),
        .conv = try int(dc, "conv_kernel_size"),
        .taps = undefined,
        .theta = try float(o, "rope_theta"),
        .eps = try float(o, "rms_norm_eps"),
    };
    const taps = dc.get("target_layer_ids") orelse return error.NotRippleDFlash2;
    if (taps != .array or taps.array.items.len != max_taps) return error.UnqualifiedRippleDFlash2;
    for (taps.array.items, &c.taps) |x, *y| {
        if (x != .integer or x.integer < 0 or x.integer >= target.layers) return error.UnqualifiedRippleDFlash2;
        y.* = @intCast(x.integer);
    }
    if (c.layers != max_layers) return error.UnqualifiedRippleDFlash2;
    const types = o.get("layer_types") orelse return error.NotRippleDFlash2;
    if (types != .array or types.array.items.len != c.layers) return error.UnqualifiedRippleDFlash2;
    for (types.array.items, 0..) |x, i| {
        if (x != .string) return error.UnqualifiedRippleDFlash2;
        if (std.mem.eql(u8, x.string, "full_attention")) c.global[i] = true else if (!std.mem.eql(u8, x.string, "sliding_attention")) return error.UnqualifiedRippleDFlash2;
    }
    // the NoPE layer: `no_rope_layers` in the drafter's config.json
    if (o.get("no_rope_layers")) |nr| {
        if (nr != .array) return error.UnqualifiedRippleDFlash2;
        for (nr.array.items) |x| {
            if (x != .integer or x.integer < 0 or x.integer >= c.layers) return error.UnqualifiedRippleDFlash2;
            c.rope[@intCast(x.integer)] = false;
        }
    }
    if (!(try qualified(o, &c, target))) return error.UnqualifiedRippleDFlash2;
    return c;
}

/// The published drafter's shape beside this target, the one the kernels and the release's server were checked at.
fn qualified(o: std.json.ObjectMap, c: *const Config, target: *const cfg.Config) Error!bool {
    if ((try int(o, "head_dim")) != cfg.head_dim) return false;
    if (o.get("attention_bias")) |b| if (b != .bool or b.bool) return false;
    if (o.get("ffn_clamp_threshold")) |x| if (x != .null) return false; // the MLP runs unclamped
    const expected = [_]bool{ false, false, false, false, true };
    const roped = [_]bool{ true, true, true, true, false };
    return c.hidden == target.hidden and c.vocab == target.vocab and c.mask < c.vocab and c.heads == 80 and
        c.kv_heads == 8 and c.ffn == 40960 and c.window == 4096 and c.block == 16 and c.rank == 256 and c.top == 32 and
        c.group == 16 and c.conv == 4 and c.hidden % c.group == 0 and c.theta == 10000 and
        std.mem.eql(bool, &c.global, &expected) and std.mem.eql(bool, &c.rope, &roped);
}

fn int(o: std.json.ObjectMap, k: []const u8) Error!u32 {
    const v = o.get(k) orelse return error.NotRippleDFlash2;
    if (v != .integer or v.integer <= 0 or v.integer > std.math.maxInt(u32)) return error.NotRippleDFlash2;
    return @intCast(v.integer);
}

fn float(o: std.json.ObjectMap, k: []const u8) Error!f32 {
    const v = o.get(k) orelse return error.NotRippleDFlash2;
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| @floatCast(f),
        else => error.NotRippleDFlash2,
    };
}

pub const Conv = struct { base: wts.Tensor, proj: wts.Dense };
pub const Layer = struct {
    in_norm: wts.Tensor,
    post_norm: wts.Tensor,
    q_norm: wts.Tensor,
    k_norm: wts.Tensor,
    q: wts.Dense,
    k: wts.Dense,
    v: wts.Dense,
    o: wts.Dense,
    gate: wts.Dense,
    up: wts.Dense,
    down: wts.Dense,
    attn_conv: Conv,
    mlp_conv: Conv,
};

pub const Weights = struct {
    fc: wts.Dense, // the taps' concatenation into the hidden width
    hidden_norm: wts.Tensor,
    norm: wts.Tensor,
    hproj: wts.Dense, // the selector's hidden projection
    pred: wts.Tensor,
    succ: wts.Tensor,
    layers: [max_layers]Layer,
};

/// Every tensor the release stores, each at its exact shape in bf16.
pub fn load(c: *const ck.Checkpoint, f: *const Config) !Weights {
    const h: usize = f.hidden;
    const kvw: usize = @as(usize, f.kv_heads) * cfg.head_dim;
    const qw: usize = @as(usize, f.heads) * cfg.head_dim;
    var w: Weights = .{
        .fc = try wts.dense(c, "", "fc.weight", h, h * max_taps),
        .hidden_norm = try wts.get(c, "", "hidden_norm.weight", &.{h}),
        .norm = try wts.get(c, "", "norm.weight", &.{h}),
        .hproj = try wts.dense(c, "", "candidate_selector.hidden_projection.weight", f.rank, h),
        .pred = try wts.get(c, "", "candidate_selector.predecessor_codebook", &.{ f.vocab, f.rank }),
        .succ = try wts.get(c, "", "candidate_selector.successor_codebook", &.{ f.vocab, f.rank }),
        .layers = undefined,
    };
    for (&w.layers, 0..) |*l, i| {
        var nb: [32]u8 = undefined;
        const p = try std.fmt.bufPrint(&nb, "layers.{d}.", .{i});
        l.* = .{
            .in_norm = try wts.get(c, p, "input_layernorm.weight", &.{h}),
            .post_norm = try wts.get(c, p, "post_attention_layernorm.weight", &.{h}),
            .q_norm = try wts.get(c, p, "self_attn.q_norm.weight", &.{cfg.head_dim}),
            .k_norm = try wts.get(c, p, "self_attn.k_norm.weight", &.{cfg.head_dim}),
            .q = try wts.dense(c, p, "self_attn.q_proj.weight", qw, h),
            .k = try wts.dense(c, p, "self_attn.k_proj.weight", kvw, h),
            .v = try wts.dense(c, p, "self_attn.v_proj.weight", kvw, h),
            .o = try wts.dense(c, p, "self_attn.o_proj.weight", h, qw),
            .gate = try wts.dense(c, p, "mlp.gate_proj.weight", f.ffn, h),
            .up = try wts.dense(c, p, "mlp.up_proj.weight", f.ffn, h),
            .down = try wts.dense(c, p, "mlp.down_proj.weight", h, f.ffn),
            .attn_conv = .{ .base = try wts.get(c, p, "attention_conv.base_kernel", &.{ 2, f.conv, h }), .proj = try wts.dense(c, p, "attention_conv.kernel_projection.weight", f.dyn(), h) },
            .mlp_conv = .{ .base = try wts.get(c, p, "mlp_conv.base_kernel", &.{ 2, f.conv, h }), .proj = try wts.dense(c, p, "mlp_conv.kernel_projection.weight", f.dyn(), h) },
        };
    }
    return w;
}

test "ricursive dflash2: the published drafter's config is qualified beside Ripple, another refused" {
    const a = std.testing.allocator;
    const target = try cfg.parse(a, @embedFile("fixtures/config.json"));
    const bytes = @embedFile("fixtures/dflash2_config.json");
    const c = try parse(a, bytes, &target);
    try std.testing.expectEqualSlices(u32, &.{ 1, 20, 39, 58, 77 }, &c.taps);
    try std.testing.expectEqual(@as(u32, 4224), c.ring(0));
    try std.testing.expectEqual(@as(u32, 16512), c.ring(4));
    try std.testing.expectEqual(@as(u32, 2560), c.dyn());
    try std.testing.expect(!c.rope[4] and c.rope[3] and c.global[4] and !c.global[0]);
    const other = try std.mem.replaceOwned(u8, a, bytes, "\"block_size\": 16", "\"block_size\": 8");
    defer a.free(other);
    try std.testing.expectError(error.UnqualifiedRippleDFlash2, parse(a, other, &target));
    const clamped = try std.mem.replaceOwned(u8, a, bytes, "\"no_rope_layers\"", "\"ffn_clamp_threshold\": 20, \"no_rope_layers\"");
    defer a.free(clamped);
    try std.testing.expectError(error.UnqualifiedRippleDFlash2, parse(a, clamped, &target));
    const not = try std.mem.replaceOwned(u8, a, bytes, "DFlash2DraftModel", "Qwen3ForCausalLM");
    defer a.free(not);
    try std.testing.expectError(error.NotRippleDFlash2, parse(a, not, &target));
}
