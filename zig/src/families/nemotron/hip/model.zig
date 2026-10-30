//! Nemotron-H on a HIP rank: the device weights the forward reads, and the dimensions this rank sees.
const std = @import("std");
const hip = @import("hip");
const config = @import("../config.zig");

pub const Projection = hip.ops.Projection;
pub const Kind = hip.ops.Kind;

/// The checkpoint's dimensions, and this rank's share of what tensor parallelism splits.
pub const Dims = struct {
    hidden: usize,
    vocab: usize,
    eps: f32,
    world: usize,
    rank: usize,
    /// This rank's query heads and the kv heads they read.
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    /// This rank's Mamba heads and groups.
    mamba_heads: usize,
    mamba_head_dim: usize,
    groups: usize,
    state: usize,
    conv_kernel: usize,
    dt_min: f32,
    dt_max: f32,
    /// Every routed expert (the router scores them all), and this rank's run of them from `first_expert`.
    experts: usize,
    rank_experts: usize,
    first_expert: usize,
    top_k: usize,
    expert_width: usize,
    shared_width: usize,
    routed_scaling: f32,
    norm_topk: bool,
    /// This rank's rows of the vocabulary head.
    vocab_rows: usize,

    pub fn inner(d: Dims) usize {
        return d.mamba_heads * d.mamba_head_dim;
    }

    /// The conv's channels on this rank: x, then B and C.
    pub fn convDim(d: Dims) usize {
        return d.inner() + 2 * d.groups * d.state;
    }

    /// in_proj's rows on this rank: z, x, B, C, dt.
    pub fn projDim(d: Dims) usize {
        return d.inner() + d.convDim() + d.mamba_heads;
    }

    pub fn of(c: *const config.Config, world: usize, rank: usize) error{UnevenSplit}!Dims {
        for ([_]usize{
            c.heads,
            c.mamba_heads,
            c.groups,
            c.experts,
            c.vocab,
        }) |n| if (n % world != 0) return error.UnevenSplit;
        if (world > c.kv_heads and world % c.kv_heads != 0) return error.UnevenSplit;
        return .{
            .hidden = c.hidden,
            .vocab = c.vocab,
            .eps = c.eps,
            .world = world,
            .rank = rank,
            .heads = c.heads / world,
            .kv_heads = @max(1, c.kv_heads / world),
            .head_dim = c.head_dim,
            .mamba_heads = c.mamba_heads / world,
            .mamba_head_dim = c.mamba_head_dim,
            .groups = c.groups / world,
            .state = c.state,
            .conv_kernel = c.conv_kernel,
            .dt_min = c.dt_min,
            .dt_max = c.dt_max,
            .experts = c.experts,
            .rank_experts = c.experts / world,
            .first_expert = rank * (c.experts / world),
            .top_k = c.top_k,
            .expert_width = c.expert_width,
            .shared_width = c.shared_width,
            .routed_scaling = c.routed_scaling,
            .norm_topk = c.norm_topk,
            .vocab_rows = c.vocab / world,
        };
    }

    /// The MTP head's dimensions: whole on the drafting rank.
    pub fn whole(d: Dims, c: *const config.Config) Dims {
        var w = Dims.of(c, 1, 0) catch unreachable;
        w.vocab_rows = d.vocab_rows;
        return w;
    }
};

/// A Mamba-2 mixer: in_proj (projDim rows), out_proj (partial under tp), fp32 vectors on this rank.
pub const Mamba = struct {
    in_proj: Projection,
    out_proj: Projection,
    /// fp32 (convDim, conv_kernel) taps, channel-major as stored, and the fp32 conv bias (convDim).
    conv_w: u64,
    conv_b: u64,
    /// fp32 A_log, D and dt_bias (mamba_heads each), the gated norm's fp32 weight (inner).
    a_log: u64,
    d: u64,
    dt_bias: u64,
    gnorm: u64,
};

/// An attention mixer: q, k, v and o (o partial under tp); Nemotron-H has no q/k norm and no rotary embedding.
pub const Attention = struct { q: Projection, k: Projection, v: Projection, o: Projection };

/// A MoE mixer: the router over every expert, this rank's routed stacks, the shared expert on rank 0.
pub const Moe = struct {
    /// fp32 (experts, hidden) router rows and the fp32 score correction bias (experts).
    router: u64,
    bias: u64,
    /// fc1 stacks (rank_experts, expert_width, hidden), fc2 (rank_experts, hidden, expert_width): relu2 between.
    fc1: Projection,
    fc2: Projection,
    /// up (shared_width, hidden) and down (hidden, shared_width); null on ranks above 0.
    shared_up: ?Projection,
    shared_down: ?Projection,
};

pub const Mixer = union(config.Kind) { mamba: Mamba, moe: Moe, attention: Attention };

/// One block: its fp32 RMSNorm weight, then its mixer.
pub const Block = struct { norm: u64, mixer: Mixer };

/// The MTP head (whole on the drafting rank): input norms, eh_proj in halves, an attention and a MoE block.
pub const Mtp = struct {
    enorm: u64,
    hnorm: u64,
    /// eh_proj's columns over the embedding and over the hidden state: one product each, summed in fp32.
    fc_e: Projection,
    fc_h: Projection,
    attn_norm: u64,
    attn: Attention,
    moe_norm: u64,
    moe: Moe,
    final_norm: u64,
    /// The vocabulary head's rows of the draft ids, and the ids (int32) on the device.
    draft: Projection,
    draft_ids: u64,
    draft_count: usize,
    dims: Dims,
};

pub const Model = struct {
    dims: Dims,
    act: Kind,
    /// The tensor-parallel communicator; null on one rank.
    tp: ?hip.collective.Comm = null,
    embed: Projection,
    blocks: []const Block,
    norm_f: u64,
    /// This rank's rows of the vocabulary head.
    head: Projection,
    mtp: ?Mtp = null,
};

test "a rank's dimensions: heads, Mamba groups, experts and the vocabulary split; kv heads shared past two" {
    var why: config.Why = .{};
    const c = try config.parse(std.testing.allocator, @embedFile("../fixtures/config.json"), &why);
    const one = try Dims.of(&c, 1, 0);
    try std.testing.expectEqual(@as(usize, 10304), one.projDim());
    try std.testing.expectEqual(@as(usize, 6144), one.convDim());
    const d = try Dims.of(&c, 4, 3);
    try std.testing.expectEqual(@as(usize, 8), d.heads);
    try std.testing.expectEqual(@as(usize, 1), d.kv_heads);
    try std.testing.expectEqual(@as(usize, 16), d.mamba_heads);
    try std.testing.expectEqual(@as(usize, 2), d.groups);
    try std.testing.expectEqual(@as(usize, 96), d.first_expert);
    try std.testing.expectEqual(@as(usize, 10304 / 4), d.projDim());
    try std.testing.expectError(error.UnevenSplit, Dims.of(&c, 3, 0));
    try std.testing.expectError(error.UnevenSplit, Dims.of(&c, 16, 0));
}
