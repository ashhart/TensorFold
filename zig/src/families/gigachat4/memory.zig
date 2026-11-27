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
pub fn streamBytes(c: *const Config) u64 {
    const d = c.delta;
    const width: u64 = 2 * @as(u64, d.key_heads) * d.key_dim + @as(u64, d.value_heads) * d.value_dim;
    const conv = (@as(u64, d.conv) - 1) * width * 2;
    const recurrent = @as(u64, d.value_heads) * d.value_dim * d.key_dim * 4;
    const p = plan.build(c);
    return p.count(.gated_deltanet) * (conv + recurrent);
}

/// The bytes a device must hold for `streams` streams of `context` positions: weights, caches and state, no scratch.
pub fn floor(c: *const Config, which: Weights, context: u64, streams: u64) !u64 {
    var per_position = tokenBytes(c);
    if (which == .with_mtp) per_position += mtpTokenBytes(c);
    const per_stream = try std.math.add(u64, try std.math.mul(u64, per_position, context), streamBytes(c));
    return std.math.add(u64, try weightBytes(c, which), try std.math.mul(u64, per_stream, streams));
}

/// The most streams of `context` positions whose floor fits `room` bytes; 0 when not even the weights fit.
pub fn streamsIn(c: *const Config, which: Weights, context: u64, room: u64) !u64 {
    const weights = try weightBytes(c, which);
    if (weights > room) return 0;
    const one = try floor(c, which, context, 1) - weights;
    return (room - weights) / one;
}
