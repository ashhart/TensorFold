//! Hy4-flash's layer plan from its config: what each backbone layer runs and what the KV cache holds for it.
const std = @import("std");
const config = @import("config.zig");

/// One backbone layer as the engine will build it.
pub const Layer = struct {
    kind: config.Kind,
    heads: config.Heads,
    moe: bool, // false: the dense MLP
    engram: bool, // the layer adds its n-gram lookup to the residual streams
    drafter_tap: bool, // the drafter reads this layer's keys and indices
};

pub fn layer(c: *const config.Config, i: u32) Layer {
    var tap = false;
    if (c.drafter) |d| for (d.taps[0..d.taps_n]) |at| {
        tap = tap or at == i;
    };
    return .{
        .kind = c.text.kinds[i],
        .heads = c.text.heads(i),
        .moe = c.text.moe[i],
        .engram = c.text.engram[i],
        .drafter_tap = tap,
    };
}

/// KV rows a layer keeps for a stream `rows` tokens long: sliding layers keep their window, sparse layers every row.
pub fn kvRows(c: *const config.Config, i: u32, rows: u64) u64 {
    return if (c.text.kinds[i] == .sliding) @min(rows, c.text.window) else rows;
}

/// Bytes of fp8 KV one stream of `rows` tokens holds over all layers, before page rounding.
pub fn kvBytes(c: *const config.Config, rows: u64) u64 {
    var total: u64 = 0;
    for (0..c.text.layers) |i| total += kvRows(c, @intCast(i), rows) * c.text.kvRowBytes();
    return total;
}

const t = std.testing;
const real = @embedFile("fixtures/config.json");

test "layers 0, 1 and 2: dense sparse, MoE sparse with Engram, MoE sliding" {
    var why: config.Why = .{};
    const c = try config.parse(t.allocator, real, &why);
    const l0 = layer(&c, 0);
    try t.expectEqual(config.Kind.sparse, l0.kind);
    try t.expect(!l0.moe and !l0.engram and !l0.drafter_tap);
    const l1 = layer(&c, 1);
    try t.expect(l1.moe and l1.engram and l1.heads.q == 32);
    const l2 = layer(&c, 2);
    try t.expectEqual(config.Kind.sliding, l2.kind);
    try t.expect(l2.moe and !l2.engram and l2.heads.q == 64);
}

test "the drafter taps layers 25, 29, 33, 37 and 41, all sparse" {
    var why: config.Why = .{};
    const c = try config.parse(t.allocator, real, &why);
    var taps: u32 = 0;
    for (0..c.text.layers) |i| {
        const l = layer(&c, @intCast(i));
        if (!l.drafter_tap) continue;
        taps += 1;
        try t.expectEqual(config.Kind.sparse, l.kind);
    }
    try t.expectEqual(@as(u32, 5), taps);
}

test "a 256K stream's KV: 12 sparse layers hold every row, 30 sliding layers hold 512" {
    var why: config.Why = .{};
    const c = try config.parse(t.allocator, real, &why);
    try t.expectEqual(@as(u64, 512), kvRows(&c, 2, 262144));
    try t.expectEqual(@as(u64, 262144), kvRows(&c, 0, 262144));
    try t.expectEqual(@as(u64, 100), kvRows(&c, 2, 100));
    const rows: u64 = 12 * 262144 + 30 * 512;
    try t.expectEqual(rows * 656, kvBytes(&c, 262144));
}
