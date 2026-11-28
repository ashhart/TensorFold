//! Memory floors for one device: stored weights from the weight map, per-position caches and per-stream state.
const std = @import("std");
const config = @import("config.zig");
const names = @import("names.zig");
const plan = @import("plan.zig");

const Config = config.Config;

/// GB10's memory (121.7 GiB) as the ricursive family budgets it, until the server reads the room itself.
pub const gb10_bytes: u64 = @intFromFloat(121.7 * (1 << 30));

/// The weights a device holds: the decoder alone, or with the stored MTP layer.
pub const Weights = enum { backbone, with_mtp };

/// Stored bytes of the checkpoint's tensors, as the weight map names them (no packing or prepared copies).
pub fn weightBytes(c: *const Config, which: Weights) !u64 {
    var s: names.Sizer = .{};
    try names.backbone(c, &s);
    if (which == .with_mtp) try names.mtp(c, &s);
    return s.bytes;
}

/// One position's BF16 cache in the decoder: each MLA layer's latent and rope key, and each DSA layer's index key.
pub fn tokenBytes(c: *const Config) u64 {
    const p = plan.build(c);
    const latent: u64 = (@as(u64, c.mla.kv_lora) + c.mla.rope) * 2;
    const index: u64 = @as(u64, c.index.dim) * 2;
    var n: u64 = 0;
    for (p.layers[0..p.n]) |l| n += switch (l.mixer) {
        .gated_deltanet => 0,
        .mla_dsa => latent + index,
        .mla_dense => latent,
    };
    return n;
}

/// One position's BF16 cache in the MTP layer: its latent and rope key (it does not select, nextn_is_sparse).
pub fn mtpTokenBytes(c: *const Config) u64 {
    return (@as(u64, c.mla.kv_lora) + c.mla.rope) * 2;
}

/// One stream's Gated DeltaNet state: the convolution's last (span - 1) inputs in BF16, the recurrent state in FP32.
pub fn streamBytes(c: *const Config) error{Overflow}!u64 {
    const d = c.delta;
    const mul = std.math.mul;
    const width = try std.math.add(u64, try mul(u64, 2 * @as(u64, d.key_heads), d.key_dim), try mul(u64, d.value_heads, d.value_dim));
    const conv = try mul(u64, try mul(u64, d.conv - 1, width), 2);
    const recurrent = try mul(u64, try mul(u64, try mul(u64, d.value_heads, d.value_dim), d.key_dim), 4);
    const p = plan.build(c);
    return mul(u64, p.count(.gated_deltanet), try std.math.add(u64, conv, recurrent));
}

/// One stream's bytes at `context` positions: its caches and its Gated DeltaNet state.
pub fn perStream(c: *const Config, which: Weights, context: u64) error{Overflow}!u64 {
    var per_position = tokenBytes(c);
    if (which == .with_mtp) per_position += mtpTokenBytes(c);
    return std.math.add(u64, try std.math.mul(u64, per_position, context), try streamBytes(c));
}

/// The bytes a device must hold for `streams` streams of `context` positions: weights, caches and state, no scratch.
pub fn floor(c: *const Config, which: Weights, context: u64, streams: u64) !u64 {
    const weights = try weightBytes(c, which);
    if (streams == 0) return weights;
    return std.math.add(u64, weights, try std.math.mul(u64, try perStream(c, which, context), streams));
}

/// The most streams of `context` positions whose floor fits `room`; maxInt(u64) when a stream costs nothing.
pub fn streamsIn(c: *const Config, which: Weights, context: u64, room: u64) !u64 {
    const weights = try weightBytes(c, which);
    if (weights > room) return 0;
    const one = perStream(c, which, context) catch return 0; // a stream past u64 bytes fits no room
    if (one == 0) return std.math.maxInt(u64);
    return (room - weights) / one;
}
