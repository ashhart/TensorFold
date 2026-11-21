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

pub const Plan = struct {
    layers: [config.max_layers]Layer,
    n: u32,
    final_sources: u32, // the output query's states: every finished block and the last layer's output

    pub fn at(p: *const Plan, i: u32) *const Layer {
        return &p.layers[i];
    }

    pub fn count(p: *const Plan, kind: Kind) u32 {
        var k: u32 = 0;
        for (p.layers[0..p.n]) |l| k += @intFromBool(l.kind == kind);
        return k;
    }
};

/// The plan for a parsed config and each layer's expert precision (`prec` holds one entry per layer).
pub fn build(c: *const config.Text, prec: []const Precision) error{PrecisionCount}!Plan {
    if (prec.len != c.layers) return error.PrecisionCount;
    var p: Plan = undefined;
    p.n = c.layers;
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
