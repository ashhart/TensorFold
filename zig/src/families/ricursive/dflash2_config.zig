//! ricursiveAI's DFlash2 drafter for Ripple (Ripple-65B-Preview-DFlash2): its config and bf16 weights as published.
const std = @import("std");
const ck = @import("../../core/checkpoint_metal.zig");
const wts = @import("weights.zig");
const cfg = @import("config.zig");
const spec = @import("dflash2_spec.zig");

pub const global_window = spec.global_window;
/// Streams a drafted server serves at most; at 4 and 7 an M5 Ultra decodes drafted rounds slower than plain ones.
pub const max_streams = 2;
pub const max_layers = spec.max_layers;
pub const max_taps = spec.max_taps;
pub const Config = spec.Config;
pub const Error = spec.Error;
pub const parse = spec.parse;

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

test {
    _ = spec; // the moved config tests still run with the Metal family
}
