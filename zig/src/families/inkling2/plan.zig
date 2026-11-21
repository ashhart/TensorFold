//! The layer plan: which layers slide or see everything, where attention-residual blocks end, what layers sum over.
const std = @import("std");
const config = @import("config.zig");

/// A routed-expert layer's stored format, read from its tensors: ModelOpt NVFP4 (groups of 16) or MXFP8 (groups of 32).
pub const Precision = enum { nvfp4, mxfp8 };

pub const Kind = enum { local, global };

pub const Layer = struct {
    kind: Kind,
    heads: config.Heads,
    extent: u32, // backward tokens the relative bias reaches
    boundary: bool, // the layer's input joins the finished blocks after its attention, before its MLP input is formed
    blocks_in: u32, // finished block states when the layer starts
    attn_queries: u8, // learned queries of the attention input: none for layer 0, one per q/k/v for a global layer
    attn_sources: u32, // states the attention input attends over: finished blocks plus the running partial one
    mlp_sources: u32, // the same for the MLP input; a boundary layer first adds its own input as a block
    precision: Precision,
};

pub const Visit = struct {
    layer: u32, // physical weight layer; core layers repeat with the same weights
    pass: u32, // zero outside the core, otherwise one-based loop pass
    cache_slot: u32, // this visit's private KV and four short-convolution states
    previous_core_start: ?u32, // previous pass's first visit; its two bank states feed this visit's messages

    pub fn hasMessages(v: Visit) bool {
        return v.previous_core_start != null;
    }
};

pub const Plan = struct {
    layers: [config.max_layers]Layer,
    n: u32,
    visits: [config.max_visits]Visit,
    n_visits: u32,
    final_sources: u32, // the output query's states: every finished block and the last layer's output

    pub fn at(p: *const Plan, i: u32) *const Layer {
        return &p.layers[i];
    }

    pub fn visit(p: *const Plan, i: u32) *const Visit {
        return &p.visits[i];
    }

    pub fn count(p: *const Plan, kind: Kind) u32 {
        var k: u32 = 0;
        for (p.layers[0..p.n]) |l| k += @intFromBool(l.kind == kind);
        return k;
    }

    pub fn visitCount(p: *const Plan, kind: Kind) u32 {
        var k: u32 = 0;
        for (p.visits[0..p.n_visits]) |v| k += @intFromBool(p.layers[v.layer].kind == kind);
        return k;
    }
};

/// The plan for a parsed config and each layer's expert precision (`prec` holds one entry per layer).
fn appendVisit(p: *Plan, c: *const config.Text, layer: u32, pass: u32) void {
    const previous: ?u32 = if (c.loop) |core|
        (if (pass >= 2) core.core_start + (pass - 2) * (core.core_end - core.core_start) else null)
    else
        null;
    p.visits[p.n_visits] = .{
        .layer = layer,
        .pass = pass,
        .cache_slot = p.n_visits,
        .previous_core_start = previous,
    };
    p.n_visits += 1;
}
pub fn build(c: *const config.Text, prec: []const Precision) error{PrecisionCount}!Plan {
    if (prec.len != c.layers) return error.PrecisionCount;
    var p: Plan = undefined;
    p.n = c.layers;
    p.n_visits = 0;
    for (0..c.layers) |k| {
        const i: u32 = @intCast(k);
        const boundary = i % c.block == 0;
        const blocks_in = (i + c.block - 1) / c.block;
        const global = config.isGlobal(i);
        p.layers[i] = .{
            .kind = if (global) .global else .local,
            .heads = c.heads(i),
            .extent = c.extent(i),
            .boundary = boundary,
            .blocks_in = blocks_in,
            .attn_queries = if (i == 0) 0 else if (global) 3 else 1,
            .attn_sources = blocks_in + 1,
            .mlp_sources = blocks_in + 1 + @intFromBool(boundary),
            .precision = prec[i],
        };
    }
    p.final_sources = (c.layers + c.block - 1) / c.block + 1;
    if (c.loop) |loop| {
        for (0..loop.core_start) |i| appendVisit(&p, c, @intCast(i), 0);
        for (1..loop.passes + 1) |pass| for (loop.core_start..loop.core_end) |i| appendVisit(&p, c, @intCast(i), @intCast(pass));
        for (loop.core_end..c.layers) |i| appendVisit(&p, c, @intCast(i), 0);
    } else {
        for (0..c.layers) |i| appendVisit(&p, c, @intCast(i), 0);
    }
    return p;
}
