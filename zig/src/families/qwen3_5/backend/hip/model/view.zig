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

/// A layer's experts and router rows (hip/moe.zig).
pub const Experts = hip.moe.Experts;
pub const Routed = hip.moe.Routed;

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

pub const ImageRun = struct {
    at: usize,
    rows: usize,
    features: u64,
    pub const Clip = struct { first: usize, rows: usize, features: u64 };
    pub fn clip(run: ImageRun, first: usize, rows: usize, hidden: usize) ?Clip {
        const lo = @max(run.at, first);
        const hi = @min(run.at + run.rows, first + rows);
        if (lo >= hi) return null;
        return .{ .first = lo - first, .rows = hi - lo, .features = run.features + (lo - run.at) * hidden * 2 };
    }
};
pub const ImageFrame = struct { positions: u64, runs: []const ImageRun };

pub const Model = struct {
    /// Torch's interleaved M-RoPE operator, engine-owned: HF BF16 text, and image prompts' rows.
    rope: ?*const @import("qwen_hip_vision").Kernels = null,
    /// Every text row rotates with `rope` (an HF BF16 checkpoint), with or without --vision.
    torch_text: bool = false,
    /// [p, p, p] for each position: a text-only prompt's rows under `torch_text`.
    positions: u64 = 0,
    frame: ?ImageFrame = null,
    /// What this rank sees: under tp its heads and value heads.
    spec: Spec,
    /// The tensor-parallel communicator; null on one rank.
    tp: ?hip.collective.Comm = null,
    act: Kind,
    embed: Projection,
    layers: []const Layer,
    final_norm: u64,
    head: Projection,
    qk: QkNorm,
    mtp: ?Mtp = null,
    /// A drafter's tapped layers: after each, every row's residual lands in its stream's tap ring.
    taps: Taps = .{},
};

/// The layers a drafter taps and the rows a stream's ring keeps (`Caches.tap`, row = position % rows).
pub const Taps = struct {
    layers: [8]u32 = undefined,
    n: usize = 0,
    rows: usize = 0,

    /// The tap index of layer `layer`, or null when it is not tapped.
    pub fn of(t: *const Taps, layer: usize) ?usize {
        for (t.layers[0..t.n], 0..) |l, i| if (l == layer) return i;
        return null;
    }

    /// Bytes of one stream's ring: `rows` rows of every tapped layer's `hidden` 16-bit values.
    pub fn ringBytes(t: *const Taps, hidden: usize) usize {
        return t.rows * t.n * hidden * 2;
    }
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

test "Qwen HIP image features splice correctly across prompt chunks" {
    const run: ImageRun = .{ .at = 100, .rows = 300, .features = 0x10000 };
    try std.testing.expectEqual(ImageRun.Clip{ .first = 36, .rows = 28, .features = 0x10000 }, run.clip(64, 64, 1024).?);
    try std.testing.expectEqual(ImageRun.Clip{ .first = 0, .rows = 128, .features = 0x10000 + 28 * 2048 }, run.clip(128, 128, 1024).?);
    try std.testing.expectEqual(ImageRun.Clip{ .first = 0, .rows = 16, .features = 0x10000 + 284 * 2048 }, run.clip(384, 128, 1024).?);
    try std.testing.expect(run.clip(0, 100, 1024) == null);
    try std.testing.expect(run.clip(400, 100, 1024) == null);
}
