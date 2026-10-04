//! A converged round's cost on the slowest rank: weights, picked experts (deduplicated over rows), caches, FLOPs, collectives.
const std = @import("std");
const node = @import("node.zig");
const model = @import("model.zig");
const plan_mod = @import("plan.zig");
const budget = @import("budget.zig");
const traffic = @import("traffic.zig");

const Plan = plan_mod.Plan;

pub const Efficiency = struct {
    /// Fraction of nominal bandwidth decode kernels reach.
    memory: f64 = 0.8,
    /// Fraction of the GPU's fp32-rate peak that batched matmuls reach.
    compute: f64 = 0.6,
    /// Host work a round (planning, the descriptor, sampling dispatch), in ms.
    host_ms: f64 = 0.5,
};

pub const Round = struct {
    /// Sampled decode rows (one a stream unless it drafts).
    rows: u32 = 1,
    /// Streams whose caches the round reads.
    streams: u32 = 1,
    /// Tokens each stream has cached.
    context: u64 = 2048,
    /// Prompt chunk rows in the same round.
    prompt: u32 = 0,
    /// Tokens of the prompt cached before this chunk; its rows attend over that plus half the chunk on average.
    prompt_context: u64 = 0,
};

pub const Cost = struct {
    /// Bytes the slowest rank reads: weights other than experts, the experts its rows picked, and caches.
    dense_bytes: u64 = 0,
    expert_bytes: u64 = 0,
    cache_bytes: u64 = 0,
    /// Distinct experts a MoE layer reads over the whole round (all ranks).
    distinct: f64 = 0,
    flops: f64 = 0,
    dense_ms: f64 = 0,
    expert_ms: f64 = 0,
    cache_ms: f64 = 0,
    compute_ms: f64 = 0,
    comm: traffic.Traffic = .{},
    host_ms: f64 = 0,
    total_ms: f64 = 0,

    pub fn memoryMs(c: Cost) f64 {
        return c.dense_ms + c.expert_ms + c.cache_ms;
    }

    pub fn tokensPerSecond(c: Cost, tokens_per_round: f64) f64 {
        return tokens_per_round * 1000.0 / c.total_ms;
    }
};

/// P(X >= x) for X ~ Binomial(n, q), all x, written to `out[0..n+1]`.
fn tail(n: u32, q: f64, out: []f64) void {
    var pmf: [1024]f64 = undefined;
    const lq = @log(q);
    const l1 = std.math.log1p(-q);
    var lc: f64 = 0;
    for (0..n + 1) |k| {
        if (k > 0) lc += @log(@as(f64, @floatFromInt(n - k + 1))) - @log(@as(f64, @floatFromInt(k)));
        const kk: f64 = @floatFromInt(k);
        pmf[k] = if (q <= 0) @as(f64, if (k == 0) 1 else 0) else if (q >= 1) @as(f64, if (k == n) 1 else 0) else @exp(lc + kk * lq + (@as(f64, @floatFromInt(n)) - kk) * l1);
    }
    var acc: f64 = 0;
    var k: usize = n + 1;
    while (k > 0) {
        k -= 1;
        acc += pmf[k];
        out[k] = acc;
    }
}

/// E[max over `ranks` blocks] of the distinct experts a block reads: each holds `per` experts, each picked with prob q.
pub fn expectedMax(ranks: u32, per: u32, q: f64) f64 {
    var t: [1025]f64 = undefined;
    tail(per, q, &t);
    var e: f64 = 0;
    for (1..per + 1) |x| e += 1.0 - std.math.pow(f64, 1.0 - t[x], @floatFromInt(ranks));
    return e;
}

/// Chance an expert is picked by at least one of `rows` rows, each choosing `k` of `experts` (uniform routing).
pub fn picked(experts: u32, k: u32, rows: u32) f64 {
    return 1.0 - std.math.pow(f64, 1.0 - @as(f64, @floatFromInt(k)) / @as(f64, @floatFromInt(experts)), @floatFromInt(rows));
}

fn gflops(inv: node.Inventory) f64 {
    const cores: f64 = @floatFromInt(@max(inv.gpu_cores, 1));
    return if (inv.backend == .metal) cores * 128 * 2 * 1.4 else 100_000;
}

fn f(x: anytype) f64 {
    return @floatFromInt(x);
}

/// Cache bytes rank `r` reads (and for linear attention writes back) in a round, and its attention FLOPs.
fn attention(p: *const Plan, s: *const model.Shape, st: plan_mod.Stage, x: Round) struct { bytes: f64, flops: f64 } {
    const t = f(st.tensor);
    const by_streams = p.opts.mla == .streams;
    const held = if (by_streams) @ceil(f(x.streams) / t) else f(x.streams);
    const share = if (by_streams) 1.0 / t else 1.0;
    const prompt_seen = f(x.prompt_context) + f(x.prompt) / 2;
    var bytes: f64 = 0;
    var flops: f64 = 0;
    for (st.layers.begin..st.layers.end) |l| {
        const layer: u32 = @intCast(l);
        switch (s.kind(layer)) {
            .mla => {
                const seen = f(s.attended(x.context));
                bytes += held * (seen * f(s.kvPerToken(layer)) + f(x.context) * f(s.indexPerToken(layer)));
                const heads = if (by_streams) f(s.heads) else f(s.heads) / t;
                const per_token = heads * 2 * f(2 * s.kv_lora + s.rope_dim);
                const index = f(s.index_heads) * f(s.index_dim) * 2;
                const decode = @ceil(f(x.rows) * share);
                const prompt = @ceil(f(x.prompt) * share);
                flops += decode * (per_token * seen + index * f(x.context));
                flops += prompt * (per_token * f(s.attended(@intFromFloat(prompt_seen))) + index * prompt_seen);
            },
            .gqa => {
                bytes += f(x.streams) * f(x.context) * f(s.kvPerToken(layer)) / t;
                flops += (f(x.rows) * f(x.context) + f(x.prompt) * prompt_seen) * f(s.heads) / t * f(s.head_dim) * 4;
            },
            .linear => {
                bytes += 2 * f(x.streams) * f(s.statePerStream(layer)) / t;
                flops += (f(x.rows) + f(x.prompt)) * f(s.linear_heads) / t * f(s.linear_head_dim) * f(s.linear_head_dim) * 8;
            },
        }
    }
    return .{ .bytes = bytes, .flops = flops };
}

/// The round on every stage; stages run one after another, each as slow as its slowest rank.
pub fn round(p: *const Plan, ns: []const budget.Need, s: *const model.Shape, x: Round, link: traffic.Link, eff: Efficiency) Cost {
    const rows = x.rows + x.prompt;
    const q = picked(s.experts, s.top_k, rows);
    var c: Cost = .{ .distinct = q * f(s.experts) };
    var total: f64 = 0;
    for (p.stages) |st| {
        var moe_layers: u64 = 0;
        for (st.layers.begin..st.layers.end) |l| moe_layers += @intFromBool(s.moe(@intCast(l)));
        const att = attention(p, s, st, x);
        var worst: Cost = .{};
        var worst_ms: f64 = -1;
        for (0..st.tensor) |k| {
            const rank = st.first + @as(u32, @intCast(k));
            const n = ns[rank];
            const inv = p.nodes[rank];
            const dense = n.weightBytes() - n.of(.experts) - n.of(.embed) - n.of(.vision) - n.of(.draft);
            const per_layer = if (moe_layers == 0) 0 else f(n.of(.experts)) / f(moe_layers);
            const held_experts: u32 = if (st.expert == 1) s.experts else s.experts / st.expert;
            const read = if (st.expert == 1) q * f(held_experts) else expectedMax(st.expert, held_experts, q);
            const experts = read * per_layer / f(@max(held_experts, 1)) * f(moe_layers);
            const expert_params = f(3 * s.moe_inter * s.latent);
            const weight_flops = 2 * f(rows) * (f(dense) / 2 + f(s.top_k) * expert_params * f(moe_layers) / f(st.tensor));
            const bw = f(@max(inv.bandwidth, 1)) * eff.memory;
            const r: Cost = .{
                .dense_bytes = dense,
                .expert_bytes = @intFromFloat(experts),
                .cache_bytes = @intFromFloat(att.bytes),
                .flops = weight_flops + att.flops,
                .dense_ms = f(dense) / bw * 1000,
                .expert_ms = experts / bw * 1000,
                .cache_ms = att.bytes / bw * 1000,
                .compute_ms = (weight_flops + att.flops) / (gflops(inv) * 1e9 * eff.compute) * 1000,
            };
            const ms = @max(r.memoryMs(), r.compute_ms);
            if (ms <= worst_ms) continue;
            worst_ms = ms;
            worst = r;
        }
        total += worst_ms;
        c.dense_bytes += worst.dense_bytes;
        c.expert_bytes += worst.expert_bytes;
        c.cache_bytes += worst.cache_bytes;
        c.flops += worst.flops;
        c.dense_ms += worst.dense_ms;
        c.expert_ms += worst.expert_ms;
        c.cache_ms += worst.cache_ms;
        c.compute_ms += worst.compute_ms;
    }
    c.comm = traffic.round(p, s, .{ .rows = x.rows, .prompt = x.prompt }, link);
    c.host_ms = eff.host_ms + f(x.rows) * 0.002;
    c.total_ms = total + c.comm.ms + c.host_ms;
    return c;
}

test "picked experts and their expected maximum per rank" {
    try std.testing.expectApproxEqRel(@as(f64, 16.0 / 896.0), picked(896, 16, 1), 1e-12);
    const one = expectedMax(4, 224, picked(896, 16, 1));
    try std.testing.expect(one > 5.0 and one < 6.5);
    try std.testing.expectApproxEqRel(@as(f64, 224), expectedMax(4, 224, 1.0), 1e-9);
    try std.testing.expect(expectedMax(4, 224, picked(896, 16, 32)) > 100);
    try std.testing.expectApproxEqAbs(@as(f64, 0), expectedMax(4, 224, 0), 1e-12);
}

test {
    _ = @import("cost_test.zig");
}
