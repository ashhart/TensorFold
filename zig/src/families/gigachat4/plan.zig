//! The layer plan: each decoder layer's mixer and feed-forward, and the multi-token-prediction layer after them.
const config = @import("config.zig");

/// Gated DeltaNet, MLA with a DSA indexer, or MLA attending to every earlier key (dense_attention_layers).
pub const Mixer = enum { gated_deltanet, mla_dsa, mla_dense };
pub const Ffn = enum { dense, moe };

pub const Layer = struct { mixer: Mixer, ffn: Ffn };

/// The stored MTP layer, `model.layers.{num_hidden_layers}`: MLA whose checkpoint keeps an indexer, a dense MLP.
pub const Mtp = struct {
    index: u32,
    /// nextn_is_sparse: whether its attention selects through the indexer it stores (false in both releases).
    sparse: bool,
};

pub const Plan = struct {
    layers: [config.max_layers]Layer,
    n: u32,
    mtp: Mtp,

    pub fn count(p: *const Plan, kind: Mixer) u32 {
        var k: u32 = 0;
        for (p.layers[0..p.n]) |l| k += @intFromBool(l.mixer == kind);
        return k;
    }

    pub fn dense(p: *const Plan) u32 {
        var k: u32 = 0;
        for (p.layers[0..p.n]) |l| k += @intFromBool(l.ffn == .dense);
        return k;
    }
};

pub fn build(c: *const config.Config) Plan {
    var p: Plan = .{ .layers = undefined, .n = c.layers, .mtp = .{ .index = c.layers, .sparse = false } };
    for (0..c.layers) |k| {
        const i: u32 = @intCast(k);
        const mixer: Mixer = if (!c.isMla(i)) .gated_deltanet else if (c.isDsa(i)) .mla_dsa else .mla_dense;
        p.layers[i] = .{ .mixer = mixer, .ffn = if (c.isDense(i)) .dense else .moe };
    }
    return p;
}
