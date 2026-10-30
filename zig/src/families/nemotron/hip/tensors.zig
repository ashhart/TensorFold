//! Nemotron-H's tensor names as upstream's loaders read them (cuda_weights.zig), and those a checkpoint must hold.
const std = @import("std");
const config = @import("../config.zig");

const Kind = config.Kind;
const Config = config.Config;

pub const embed = "backbone.embeddings";
pub const norm_f = "backbone.norm_f.weight";
pub const lm_head = "lm_head";
/// The MTP head's file beside the shards; its names have no `backbone.` prefix.
pub const mtp_file = "mtp-4bit.safetensors";

/// Block `i`'s stem: its norm is `{stem}.norm.weight`, its mixer's tensors `{stem}.mixer.*`.
pub fn layer(buf: []u8, i: usize) ![]const u8 {
    return std.fmt.bufPrint(buf, "backbone.layers.{d}", .{i});
}

/// The Mamba-2 mixer under `{stem}.mixer.`: two projections, then its float vectors.
pub const mamba = struct {
    pub const in_proj = "in_proj";
    pub const out_proj = "out_proj";
    pub const conv_w = "conv1d.weight";
    pub const conv_b = "conv1d.bias";
    pub const a_log = "A_log";
    pub const d = "D";
    pub const dt_bias = "dt_bias";
    pub const norm = "norm.weight";
    pub const projections = [_][]const u8{ in_proj, out_proj };
    pub const vectors = [_][]const u8{ conv_w, conv_b, a_log, d, dt_bias, norm };
};

/// The attention mixer under `{stem}.mixer.`: no norms and no rotary embedding in Nemotron-H.
pub const attention = struct {
    pub const q = "q_proj";
    pub const k = "k_proj";
    pub const v = "v_proj";
    pub const o = "o_proj";
    pub const projections = [_][]const u8{ q, k, v, o };
    pub const vectors = [_][]const u8{};
};

/// The MoE mixer under `{stem}.mixer.`: router, its score bias, routed stacks and the shared expert (relu2, no gate).
pub const moe = struct {
    pub const router = "gate.weight";
    pub const bias = "gate.e_score_correction_bias";
    pub const fc1 = "switch_mlp.fc1";
    pub const fc2 = "switch_mlp.fc2";
    pub const shared_up = "shared_experts.up_proj";
    pub const shared_down = "shared_experts.down_proj";
    pub const projections = [_][]const u8{ fc1, fc2, shared_up, shared_down };
    pub const vectors = [_][]const u8{ router, bias };
};

/// The MTP head in mtp-4bit.safetensors: layer 0 holds its input norms, eh_proj and attention, layer 1 its MoE.
pub const mtp = struct {
    pub const enorm = "layers.0.enorm.weight";
    pub const hnorm = "layers.0.hnorm.weight";
    pub const eh_proj = "layers.0.eh_proj";
    pub const attn_norm = "layers.0.norm.weight";
    pub const attn = "layers.0.mixer";
    pub const moe_norm = "layers.1.norm.weight";
    pub const moe_stem = "layers.1.mixer";
    pub const final_norm = "layers.1.final_layernorm.weight";
};

/// The affine parts of a projection `p`: `{p}.weight` (u32 words), `{p}.scales` and `{p}.biases`.
pub const parts = [_][]const u8{ "weight", "scales", "biases" };

/// What each kind's mixer stores: its projections (three tensors each) and its plain tensors.
pub fn mixerNames(kind: Kind) struct { projections: []const []const u8, vectors: []const []const u8 } {
    return switch (kind) {
        .mamba => .{ .projections = &mamba.projections, .vectors = &mamba.vectors },
        .attention => .{ .projections = &attention.projections, .vectors = &attention.vectors },
        .moe => .{ .projections = &moe.projections, .vectors = &moe.vectors },
    };
}

/// Calls `f(ctx, name)` for every tensor name the text checkpoint of `c` holds, in no set order.
pub fn eachText(c: *const Config, ctx: anytype, comptime f: fn (@TypeOf(ctx), []const u8) anyerror!void) !void {
    var buf: [192]u8 = undefined;
    for (parts) |p| {
        try f(ctx, try std.fmt.bufPrint(&buf, "{s}.{s}", .{ embed, p }));
        try f(ctx, try std.fmt.bufPrint(&buf, "{s}.{s}", .{ lm_head, p }));
    }
    try f(ctx, norm_f);
    for (c.kinds[0..c.layers], 0..) |kind, i| {
        var sb: [64]u8 = undefined;
        const stem = try layer(&sb, i);
        try f(ctx, try std.fmt.bufPrint(&buf, "{s}.norm.weight", .{stem}));
        try eachMixer(kind, stem, ctx, f);
    }
}

/// Every tensor of one mixer of `kind` under `{stem}.mixer.`.
pub fn eachMixer(
    kind: Kind,
    stem: []const u8,
    ctx: anytype,
    comptime f: fn (@TypeOf(ctx), []const u8) anyerror!void,
) !void {
    var buf: [192]u8 = undefined;
    const names = mixerNames(kind);
    for (names.projections) |p| for (parts) |s| try f(
        ctx,
        try std.fmt.bufPrint(&buf, "{s}.mixer.{s}.{s}", .{ stem, p, s }),
    );
    for (names.vectors) |v| try f(ctx, try std.fmt.bufPrint(&buf, "{s}.mixer.{s}", .{ stem, v }));
}

/// Every tensor name of the MTP head's file.
pub fn eachMtp(ctx: anytype, comptime f: fn (@TypeOf(ctx), []const u8) anyerror!void) !void {
    var buf: [192]u8 = undefined;
    for ([_][]const u8{ mtp.enorm, mtp.hnorm, mtp.attn_norm, mtp.moe_norm, mtp.final_norm }) |n| try f(ctx, n);
    for (parts) |s| try f(ctx, try std.fmt.bufPrint(&buf, "{s}.{s}", .{ mtp.eh_proj, s }));
    try eachMixer(.attention, "layers.0", ctx, f);
    try eachMixer(.moe, "layers.1", ctx, f);
}

const Count = struct {
    n: usize = 0,
    fn add(self: *Count, _: []const u8) anyerror!void {
        self.n += 1;
    }
};

test "the names cover every tensor of the stock checkpoint and its MTP head" {
    var why: config.Why = .{};
    const c = try config.parse(std.testing.allocator, @embedFile("../fixtures/config.json"), &why);
    var text: Count = .{};
    try eachText(&c, &text, Count.add);
    // the stock index's 729: 23 Mamba mixers of 12 tensors, 23 MoE of 14, 6 attention of 12, 52 norms and 7 more
    try std.testing.expectEqual(@as(usize, 729), text.n);
    try std.testing.expectEqual(@as(usize, 23 * 12 + 23 * 14 + 6 * 12 + 52 + 7), text.n);
    var head: Count = .{};
    try eachMtp(&head, Count.add);
    try std.testing.expectEqual(@as(usize, 34), head.n);
}
