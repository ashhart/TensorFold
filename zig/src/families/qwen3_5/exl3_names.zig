// SPDX-License-Identifier: Apache-2.0

//! EXL3 projections of a Qwen3.5-MoE checkpoint under their HF tensor names.

//! No device work: names, shapes, and the stored-group check. CUDA and XPU read the same groups.

const std = @import("std");

/// The widths the projections take, from config.json.
pub const Dims = struct {
    hidden: u32,
    vocab: u32,
    experts: u32,
    /// moe_intermediate_size (routed and shared experts).
    intermediate: u32,
    /// Linear-attention (Gated DeltaNet) layers: qkv conv width and value width.
    conv: u32,
    value: u32,
    /// Full-attention layers: q_proj output (query and gate) and the attention width o_proj reads.
    q_out: u32,
    kv_out: u32,
    attn: u32,
};

pub const Layer = enum { linear, full };

/// One projection: `base` names the group (base.trellis, .suh, .svh, .mul1); y[n] = x[k] @ W.
pub const Proj = struct { base: []const u8, k: u32, n: u32 };

pub const max_name = 128;

fn print(buf: *[max_name]u8, comptime fmt: []const u8, args: anytype) []const u8 {
    return std.fmt.bufPrint(buf, fmt, args) catch unreachable; // every name here fits max_name
}

/// The attention projections of layer `i` (four), names written into `bufs`.
pub fn attention(d: Dims, layer: Layer, i: u32, bufs: *[4][max_name]u8) [4]Proj {
    const p = "model.language_model.layers.";
    return switch (layer) {
        .linear => .{
            .{ .base = print(&bufs[0], p ++ "{d}.linear_attn.in_proj_qkv", .{i}), .k = d.hidden, .n = d.conv },
            .{ .base = print(&bufs[1], p ++ "{d}.linear_attn.in_proj_z", .{i}), .k = d.hidden, .n = d.value },
            .{ .base = print(&bufs[2], p ++ "{d}.linear_attn.out_proj", .{i}), .k = d.value, .n = d.hidden },
            .{ .base = "", .k = 0, .n = 0 },
        },
        .full => .{
            .{ .base = print(&bufs[0], p ++ "{d}.self_attn.q_proj", .{i}), .k = d.hidden, .n = d.q_out },
            .{ .base = print(&bufs[1], p ++ "{d}.self_attn.k_proj", .{i}), .k = d.hidden, .n = d.kv_out },
            .{ .base = print(&bufs[2], p ++ "{d}.self_attn.v_proj", .{i}), .k = d.hidden, .n = d.kv_out },
            .{ .base = print(&bufs[3], p ++ "{d}.self_attn.o_proj", .{i}), .k = d.attn, .n = d.hidden },
        },
    };
}

pub const ExpertProj = enum { gate_proj, up_proj, down_proj };

/// Expert `e` of layer `i` (e == experts: the shared expert).
pub fn expert(d: Dims, i: u32, e: u32, which: ExpertProj, buf: *[max_name]u8) Proj {
    const p = "model.language_model.layers.";
    const base = if (e < d.experts)
        print(buf, p ++ "{d}.mlp.experts.{d}.{s}", .{ i, e, @tagName(which) })
    else
        print(buf, p ++ "{d}.mlp.shared_expert.{s}", .{ i, @tagName(which) });
    return switch (which) {
        .gate_proj, .up_proj => .{ .base = base, .k = d.hidden, .n = d.intermediate },
        .down_proj => .{ .base = base, .k = d.intermediate, .n = d.hidden },
    };
}

pub fn lmHead(d: Dims) Proj {
    return .{ .base = "lm_head", .k = d.hidden, .n = d.vocab };
}

/// The mul1 codebook marker's value.
pub const mul1_magic: u32 = 0x83DCD12D;

/// A stored group's tensor shapes and marker, as the checkpoint gives them (the caller reads the header).
pub const Stored = struct { trellis: []const usize, suh: []const usize, svh: []const usize, mul1: u32 };

/// Checks a stored group against the projection and returns its halves (twice the bits).
pub fn check(p: Proj, s: Stored) error{ UnexpectedTensor, UnsupportedQuantization }!u32 {
    if (s.mul1 != mul1_magic) return error.UnsupportedQuantization;
    const t = s.trellis;
    if (t.len != 3 or t[0] != p.k / 16 or t[1] != p.n / 16 or t[2] % 8 != 0) return error.UnexpectedTensor;
    if (s.suh.len != 1 or s.suh[0] != p.k or s.svh.len != 1 or s.svh[0] != p.n) return error.UnexpectedTensor;
    const halves: u32 = @intCast(t[2] / 8);
    if (halves < 2 or halves > 16) return error.UnsupportedQuantization;
    return halves;
}

/// Widths of a 256-expert checkpoint with hidden 2048, the one the EXL3 checks ran on.
const a3b: Dims = .{ .hidden = 2048, .vocab = 248320, .experts = 256, .intermediate = 512, .conv = 8192, .value = 4096, .q_out = 8192, .kv_out = 512, .attn = 4096 };

test "names and shapes follow the HF layout" {
    var bufs: [4][max_name]u8 = undefined;
    const lin = attention(a3b, .linear, 0, &bufs);
    try std.testing.expectEqualStrings("model.language_model.layers.0.linear_attn.in_proj_qkv", lin[0].base);
    try std.testing.expectEqual(@as(u32, 8192), lin[0].n);
    try std.testing.expectEqual(@as(u32, 4096), lin[2].k);
    const full = attention(a3b, .full, 3, &bufs);
    try std.testing.expectEqualStrings("model.language_model.layers.3.self_attn.o_proj", full[3].base);
    try std.testing.expectEqual(@as(u32, 4096), full[3].k);
    var buf: [max_name]u8 = undefined;
    const e = expert(a3b, 39, 255, .down_proj, &buf);
    try std.testing.expectEqualStrings("model.language_model.layers.39.mlp.experts.255.down_proj", e.base);
    try std.testing.expectEqual(@as(u32, 512), e.k);
    try std.testing.expectEqualStrings("model.language_model.layers.1.mlp.shared_expert.gate_proj", expert(a3b, 1, 256, .gate_proj, &buf).base);
}

test "stored groups are checked against the projection" {
    const p = lmHead(a3b);
    try std.testing.expectEqual(@as(u32, 12), try check(p, .{ .trellis = &.{ 128, 15520, 96 }, .suh = &.{2048}, .svh = &.{248320}, .mul1 = mul1_magic }));
    try std.testing.expectError(error.UnexpectedTensor, check(p, .{ .trellis = &.{ 128, 15519, 96 }, .suh = &.{2048}, .svh = &.{248320}, .mul1 = mul1_magic }));
    try std.testing.expectError(error.UnsupportedQuantization, check(p, .{ .trellis = &.{ 128, 15520, 96 }, .suh = &.{2048}, .svh = &.{248320}, .mul1 = 0xCBAC1FED }));
}
