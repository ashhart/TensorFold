//! The layer plan: HCA or DSA attention and a dense or MoE feed-forward per layer, and the draft head's layers.
const config = @import("config.zig");

pub const Attention = enum { hca, dsa };
pub const Ffn = enum { dense, moe };

pub const Layer = struct {
    attention: Attention,
    ffn: Ffn,
    ratio: u32, // tokens per compressed key of an HCA layer; 0 on a DSA layer
};

pub const Plan = struct {
    layers: [config.max_layers]Layer,
    n: u32,
    draft_layers: u32, // the bundled draft head's layers: one per tapped target layer, each attending in its window

    pub fn count(p: *const Plan, kind: Attention) u32 {
        var k: u32 = 0;
        for (p.layers[0..p.n]) |l| k += @intFromBool(l.attention == kind);
        return k;
    }
};

pub fn build(c: *const config.Config) Plan {
    var p: Plan = .{ .layers = undefined, .n = c.layers, .draft_layers = c.dspark.taps_n };
    for (0..c.layers) |k| {
        const i: u32 = @intCast(k);
        const dsa = c.isDsa(i);
        p.layers[i] = .{
            .attention = if (dsa) .dsa else .hca,
            .ffn = if (c.isDense(i)) .dense else .moe,
            .ratio = if (dsa) 0 else @intCast(c.compress[i]),
        };
    }
    return p;
}

test "the release plan: 37 HCA and 6 DSA layers, one dense MLP first, a three-layer draft" {
    const std = @import("std");
    var pr: config.Problem = .{};
    const c = try config.parse(std.testing.allocator, @embedFile("fixtures/config.json"), &pr);
    const p = build(&c);
    try std.testing.expectEqual(@as(u32, 37), p.count(.hca));
    try std.testing.expectEqual(@as(u32, 6), p.count(.dsa));
    try std.testing.expectEqual(Layer{ .attention = .hca, .ffn = .dense, .ratio = 128 }, p.layers[0]);
    try std.testing.expectEqual(Layer{ .attention = .dsa, .ffn = .moe, .ratio = 0 }, p.layers[7]);
    try std.testing.expectEqual(Layer{ .attention = .dsa, .ffn = .moe, .ratio = 0 }, p.layers[42]);
    try std.testing.expectEqual(@as(u32, 3), p.draft_layers);
}
