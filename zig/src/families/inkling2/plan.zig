//! The layer plan: which layers slide or see everything, where attention-residual blocks end, what layers sum over, and
//! the logical visits (a looped model runs its core layers once a pass over one set of weights).
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

/// One logical layer visit: the physical layer whose weights it runs and where it stands in the loop.
pub const Visit = struct {
    layer: u32, // physical layer: weights, geometry, block structure
    pass: u32, // core pass (0 outside the core and in its first pass)
    messages: bool, // a core visit of pass >= 1: gated messages of the previous pass enter its inputs
    core_index: u32, // layer - core start of a core visit (the message queries' row), else 0
};

pub const Plan = struct {
    layers: [config.max_layers]Layer,
    n: u32,
    final_sources: u32, // the output query's states: every finished block and the last layer's output
    visits: [config.max_visits]Visit,
    n_visits: u32,
    loop: ?config.Loop,
    /// Finished blocks entering the core (the blocks of the layers before it).
    pre_blocks: u32,
    /// Blocks a core pass adds that the next pass receives as messages: all but the first (the core's entry state,
    /// which every pass rebuilds identically).
    bank_blocks: u32,

    pub fn at(p: *const Plan, i: u32) *const Layer {
        return &p.layers[i];
    }

    pub fn count(p: *const Plan, kind: Kind) u32 {
        var k: u32 = 0;
        for (p.layers[0..p.n]) |l| k += @intFromBool(l.kind == kind);
        return k;
    }

    pub fn visit(p: *const Plan, k: u32) *const Visit {
        return &p.visits[k];
    }

    /// Visit `k` starts a core pass after the first: the state restarts from the core's entry.
    pub fn restarts(p: *const Plan, k: u32) bool {
        const l = p.loop orelse return false;
        return k < p.n_visits and p.visits[k].pass > 0 and p.visits[k].layer == l.start;
    }

    /// Sources of a message attention: the previous pass's banked blocks and its final hidden state.
    pub fn messageSources(p: *const Plan) u32 {
        return p.bank_blocks + 1;
    }
};

/// The plan for a parsed config and each layer's expert precision (`prec` holds one entry per layer).
pub fn build(c: *const config.Text, prec: []const Precision) error{PrecisionCount}!Plan {
    if (prec.len != c.layers) return error.PrecisionCount;
    var p: Plan = undefined;
    p.n = c.layers;
    p.loop = c.loop;
    p.pre_blocks = 0;
    p.bank_blocks = 0;
    if (c.loop) |l| {
        p.pre_blocks = l.start / c.block;
        p.bank_blocks = (l.end - l.start) / c.block - 1;
    }
    p.n_visits = c.executionLayers();
    for (0..p.n_visits) |k| {
        const v: u32 = @intCast(k);
        const layer = c.physicalLayer(v);
        var pass: u32 = 0;
        var core_index: u32 = 0;
        if (c.loop) |l| if (v >= l.start and v < l.start + l.passes * (l.end - l.start)) {
            pass = (v - l.start) / (l.end - l.start);
            core_index = layer - l.start;
        };
        p.visits[k] = .{ .layer = layer, .pass = pass, .messages = pass > 0, .core_index = core_index };
    }
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
    return p;
}
