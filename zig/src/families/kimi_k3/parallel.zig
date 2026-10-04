//! How a K3 deployment splits work over nodes: heads and intermediates by column, o/down by row, experts by set.
const std = @import("std");
const Config = @import("config.zig").Config;

/// What a node holds of a weight: all of it, a slice of its output rows (columns), or of its inputs (rows).
pub const Split = enum { replicated, columns, rows };

/// After a row-split projection the nodes' fp32 partials are summed (fixed node order), then rounded to bf16.
pub const Sync = enum { none, all_reduce, all_gather };

pub const Part = struct { lo: u32, hi: u32 };

/// One node's share of a tensor-, expert- and pipeline-parallel deployment.
pub const Plan = struct {
    tp: u32 = 1,
    rank: u32 = 0,
    ep: u32 = 1,
    ep_rank: u32 = 0,
    first_layer: u32 = 0,
    last_layer: u32 = std.math.maxInt(u32),

    /// [lo, hi) of `n` units split as evenly as whole units allow, earlier ranks taking the remainder.
    pub fn share(n: u32, parts: u32, rank: u32) Part {
        const base = n / parts;
        const extra = n % parts;
        const lo = rank * base + @min(rank, extra);
        return .{ .lo = lo, .hi = lo + base + @intFromBool(rank < extra) };
    }

    pub fn kdaHeads(p: Plan, c: *const Config) Part {
        return share(c.kda_heads, p.tp, p.rank);
    }

    pub fn mlaHeads(p: Plan, c: *const Config) Part {
        return share(c.mla_heads, p.tp, p.rank);
    }

    /// The shared experts' intermediate columns, in 32-wide units (the row kernels' lane block).
    pub fn sharedCols(p: Plan, c: *const Config) Part {
        const s = share(c.shared_inter / 32, p.tp, p.rank);
        return .{ .lo = s.lo * 32, .hi = s.hi * 32 };
    }

    pub fn denseCols(p: Plan, c: *const Config) Part {
        const s = share(c.dense_inter / 32, p.tp, p.rank);
        return .{ .lo = s.lo * 32, .hi = s.hi * 32 };
    }

    /// LM head vocabulary rows; each node's logits are gathered (or each node's best token compared).
    pub fn vocab(p: Plan, c: *const Config) Part {
        return share(c.vocab, p.tp, p.rank);
    }

    /// The routed experts this node owns: router, plan and combine see all, the expert kernels only these.
    pub fn experts(p: Plan, c: *const Config) Part {
        return share(c.experts, p.ep, p.ep_rank);
    }

    pub fn ownsLayer(p: Plan, i: u32) bool {
        return i >= p.first_layer and i < p.last_layer;
    }
};

/// Each weight's split, and where a layer synchronises: the cluster layer's map of K3.
pub const rules = [_]struct { tensor: []const u8, split: Split, sync: Sync }{
    .{ .tensor = "self_attn.{q,k,v,g}_proj, f_b_proj, b_proj (KDA)", .split = .columns, .sync = .none },
    .{ .tensor = "self_attn.{A_log,dt_bias,conv1d} and the KDA state (per head)", .split = .columns, .sync = .none },
    .{ .tensor = "self_attn.f_a_proj, o_norm (KDA)", .split = .replicated, .sync = .none },
    .{ .tensor = "self_attn.q_a_proj, kv_a_proj_with_mqa, their norms, the latent cache (MLA)", .split = .replicated, .sync = .none },
    .{ .tensor = "self_attn.q_b_proj, kv_b_proj, g_proj (MLA, by head)", .split = .columns, .sync = .none },
    .{ .tensor = "self_attn.o_proj (both kinds)", .split = .rows, .sync = .all_reduce },
    .{ .tensor = "mlp.{gate,up}_proj, shared_experts.{gate,up}_proj", .split = .columns, .sync = .none },
    .{ .tensor = "mlp.down_proj, shared_experts.down_proj", .split = .rows, .sync = .all_reduce },
    .{ .tensor = "block_sparse_moe.gate, routed_expert_{down,up}_proj, routed_expert_norm", .split = .replicated, .sync = .none },
    .{ .tensor = "block_sparse_moe.experts.* (a set a node, fp32 partial latent sums)", .split = .columns, .sync = .all_reduce },
    .{ .tensor = "lm_head (by vocabulary)", .split = .columns, .sync = .all_gather },
    .{ .tensor = "embed_tokens, norms, attention-residual weights", .split = .replicated, .sync = .none },
};

test "shares cover every unit once" {
    const c = Config{};
    for ([_]u32{ 1, 2, 3, 4, 8 }) |n| {
        var next: u32 = 0;
        for (0..n) |r| {
            const p = Plan{ .tp = n, .rank = @intCast(r), .ep = n, .ep_rank = @intCast(r) };
            try std.testing.expectEqual(next, p.kdaHeads(&c).lo);
            next = p.kdaHeads(&c).hi;
        }
        try std.testing.expectEqual(c.kda_heads, next);
    }
    const p = Plan{ .tp = 4, .rank = 3, .ep = 4, .ep_rank = 3 };
    try std.testing.expectEqual(Part{ .lo = 672, .hi = 896 }, p.experts(&c));
    try std.testing.expectEqual(Part{ .lo = 4608, .hi = 6144 }, p.sharedCols(&c));
}
