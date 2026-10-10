//! The device weights the forward reads: projections, fp32 norms, conv taps, DeltaNet tables, MLPs and the MTP head.

const std = @import("std");
const hip = @import("hip");

pub const Projection = hip.quant.Projection;
pub const Kind = hip.ops.Kind;

/// The text model's dimensions (Python's Spec), as the checkpoint's config gives them.
pub const Spec = @import("../../../weights/config.zig").Spec;

/// The conv's channels: q, k and v of the linear attention.
pub fn convChannels(s: Spec) usize {
    return 2 * s.keyWidth() + s.valueWidth();
}

/// A layer's experts, shared expert last: gate and up stacked as one (E + 1, 2 NI, ...) weight, down (E + 1, D, ...).
pub const Experts = struct {
    fused: Projection,
    down: Projection,
    count: usize,
    width: usize,
    dims: usize,
    limit: f32 = 0,
};

/// The router's rows in fp32 (E + 1, D), the shared expert's gate last, and the experts.
pub const Routed = struct {
    rows32: u64,
    experts: Experts,
    top_k: usize,
    /// The router's rows, E + 1: the model's expert count, whatever share of the experts this rank holds.
    rows: usize,
    /// A rank's expert ids (int32, E + 1): each id's place among its own, -1 for another rank's; zero on one rank.
    remap: u64 = 0,

    /// Routed experts, the shared one not counted.
    pub fn count(r: Routed) usize {
        return r.rows - 1;
    }
};

pub const Mlp = union(enum) {
    dense: struct { gate: Projection, up: Projection, down: Projection },
    moe: Routed,
};

pub const Full = struct {
    input_norm: u64,
    post_norm: u64,
    q: Projection,
    k: Projection,
    v: Projection,
    o: Projection,
    q_norm: u64,
    k_norm: u64,
    mlp: Mlp,
};

pub const Linear = struct {
    input_norm: u64,
    post_norm: u64,
    qkv: Projection,
    z: Projection,
    a: Projection,
    b: Projection,
    out: Projection,
    conv: u64,
    a_log: u64,
    dt_bias: u64,
    gnorm: u64,
    mlp: Mlp,
};

pub const Layer = union(enum) { full: Full, linear: Linear };

/// The MTP head: the Qwen3 shape (gated attention, an MLP or experts) with its own or the model's output head.
pub const Mtp = struct {
    fc_e_norm: u64,
    fc_h_norm: u64,
    fc_e: Projection,
    fc_h: Projection,
    q_norm: u64,
    k_norm: u64,
    q: Projection,
    k: Projection,
    v: Projection,
    o: Projection,
    final_norm: u64,
    head: ?Projection,
    input_norm: ?u64,
    post_norm: ?u64,
    mlp: ?Mlp,
    gated: bool,
};

/// normalize_qk's constant norm weights (key_dim ** -0.5 squared for q, once for k) and its eps.
pub const QkNorm = struct { q_weight: u64, k_weight: u64, eps: f32 };

pub const Model = struct {
    /// What this rank sees: under tp its heads and value heads.
    spec: Spec,
    /// The tensor-parallel communicator; null on one rank.
    tp: ?hip.rccl.Comm = null,
    act: Kind,
    embed: Projection,
    layers: []const Layer,
    final_norm: u64,
    head: Projection,
    qk: QkNorm,
    mtp: ?Mtp = null,
};

/// q and k's normalize_qk weights and eps exactly as Python makes them: double arithmetic, then fp32.
pub fn qkConstants(key_dim: usize, eps: f64) struct { q: f32, k: f32, eps: f32 } {
    const inv = std.math.pow(f64, @floatFromInt(key_dim), -0.5);
    return .{ .q = @floatCast(inv * inv), .k = @floatCast(inv), .eps = @floatCast(eps * inv * inv) };
}

test "normalize_qk constants for a 128-wide key" {
    const c = qkConstants(128, 1e-6);
    try std.testing.expectEqual(@as(f32, 0.0078125), c.q);
    try std.testing.expect(@abs(c.k - 0.08838834764) < 1e-8);
}
