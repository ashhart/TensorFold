//! A round through one node's layers: embedding on the first stage, the layers, drafter taps, head on the last.
const std = @import("std");
const mtl = @import("metal");
const weights = @import("weights.zig");
const layer = @import("layer.zig");
const Tensor = @import("store.zig").Tensor;

/// Layers whose output (the residual block's prefix sum) a drafter reads, one bf16 plane of rows each.
pub const Taps = struct {
    layers: []const u32,
    out: mtl.Buffer,

    /// DSpark (RadixArk/Kimi-K3-DSpark) reads layers 7, 23, 51, 67 and 83.
    pub const dspark = [_]u32{ 7, 23, 51, 67, 83 };
    /// DFlash2 (lightseekorg/kimi-k3-dflash2) reads layers 19, 37, 66, 78 and 90.
    pub const dflash2 = [_]u32{ 19, 37, 66, 78, 90 };
};

/// A node's pipeline stage: layers [first, last) present in `layers` (indexed by layer), the head on the last stage.
pub const Stage = struct {
    first: u32,
    last: u32,
    layers: []const weights.Layer,
    head: ?*const weights.Head = null,
    embed: ?Tensor = null,

    /// The round's rows through this stage; a later stage takes over prefix, delta and the block planes.
    pub fn encode(s: Stage, x: layer.Ctx, e: mtl.ComputeEncoder, taps: ?Taps) !void {
        const sc = x.sc;
        if (s.first == 0) x.k.embedRows(e, sc.ref("ids"), (s.embed orelse return error.NoEmbedding).ref, sc.ref("prefix"), sc.rows, x.c.hidden);
        for (s.first..s.last) |li| {
            const i: u32 = @intCast(li);
            try layer.encode(x, e, i, &s.layers[i - s.first]);
            const t = taps orelse continue;
            for (t.layers, 0..) |tl, j| if (tl == i) {
                const plane = @as(usize, sc.rows_max) * x.c.hidden * 2;
                x.k.add(e, sc.ref("prefix"), sc.ref("delta"), .{ .buf = t.out, .off = j * plane }, sc.rows * x.c.hidden);
            };
        }
        if (s.head) |h| {
            layer.head(x, e, h, s.last);
            x.k.greedy(e, sc.ref("logits"), sc.ref("tokens"), sc.rows, x.c.vocab);
        }
    }
};

test {
    std.testing.refAllDecls(@This());
}
