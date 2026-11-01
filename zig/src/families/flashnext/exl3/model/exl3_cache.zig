//! The KV bytes and the resident-set budget.

const std = @import("std");
const shape = @import("../pack/exl3_shape.zig");

/// One attention layer's K and V for `tokens` tokens, bf16.
pub fn layerBytes(tokens: usize, kv_heads: usize, head_dim: usize) u64 {
    const t: u64 = @intCast(tokens);
    const h: u64 = @intCast(kv_heads);
    const d: u64 = @intCast(head_dim);
    return t * 2 * h * d * 2;
}

/// A kept prefix stops one token before the prompt end. An empty prompt stays at 0.
pub fn resumeAt(prompt_len: u32) u32 {
    if (prompt_len == 0) return 0;
    return prompt_len - 1;
}

pub const Budget = struct {
    device_bytes: u64,
    /// Dense + shared + router + HC + MTP + graph workspace, already summed by the caller.
    resident_bytes: u64,
    kv_bytes: u64,
    routed_expert_bytes: u64,
    ngram_bytes: u64,
};

pub const Place = enum { host, refuse };

/// Resident weights plus KV. Routed experts and the n-gram table are not in this sum.
pub fn deviceSum(b: Budget) u64 {
    return b.resident_bytes + b.kv_bytes;
}

pub fn fits(b: Budget) error{ExceedsDevice}!void {
    if (deviceSum(b) > b.device_bytes) return error.ExceedsDevice;
}

/// One RTX 5060 Ti. The learned expert lines are whatever this does not already use.
pub const card_bytes: u64 = 16 << 30;
/// CUDA context and the batch-1 workspace. Not a weight, and not a second copy of the bank.
pub const device_reserve: u64 = 256 << 20;

/// Bytes left for routed experts after pinned GEMMs, KV, and the reserve.
pub fn learnedBytes(device: u64, pinned: u64, kv: u64) u64 {
    if (pinned >= device) return 0;
    const after_pinned = device - pinned;
    if (kv >= after_pinned) return 0;
    const after_kv = after_pinned - kv;
    if (after_kv <= device_reserve) return 0;
    return after_kv - device_reserve;
}

fn matrixBytes(k: usize, n: usize) u64 {
    const kn: u64 = @as(u64, k) * @as(u64, n);
    // 3.05 bpw trellis, plus one fp16 scale along K and one along N.
    return kn * 3050 / 8000 + @as(u64, k + n) * 2;
}

/// One routed expert as stored.
pub fn expertBytes() u64 {
    const k = shape.hidden;
    const n = shape.expert_intermediate;
    return matrixBytes(k, n) * 2 + matrixBytes(n, k);
}

pub fn expertLines(room: u64) u64 {
    return expertLinesOf(room, expertBytes());
}

/// Budget stride for one owned expert until a header span is larger. Not a line count.
pub const full_expert_bytes: u64 = 1_901_213;

/// Lines that fit in `room` at `line` bytes. A zero stride is no lines, not a panic.
pub fn expertLinesOf(room: u64, line: u64) u64 {
    if (line == 0) return 0;
    return room / line;
}

/// KV for every full-attention layer.
pub fn contextKv(tokens: usize) u64 {
    const layers: u64 = shape.layers / shape.full_attention_interval;
    return layerBytes(tokens, shape.kv_heads, shape.head_dim) * layers;
}

/// Routed experts and the n-gram table always live on the host.
pub fn placeWeights(b: Budget) Place {
    _ = b;
    return .host;
}

/// Neither the routed-expert bank nor the n-gram table may be counted as resident.
pub fn refusesResident(which: enum { routed, ngram }) bool {
    _ = which;
    return true;
}

test "resumeAt stops one token before the end" {
    try std.testing.expectEqual(@as(u32, 0), resumeAt(0));
    try std.testing.expectEqual(@as(u32, 7), resumeAt(8));
    try std.testing.expectEqual(@as(u32, 0), resumeAt(1));
}

test "layer bytes match v0.6.6 row_bytes" {
    try std.testing.expectEqual(@as(u64, 2048), layerBytes(1, 2, 256));
    try std.testing.expectEqual(@as(u64, 2048), layerBytes(1, shape.kv_heads, shape.head_dim));
    try std.testing.expectEqual(@as(u64, 6144), layerBytes(3, 2, 256));
    try std.testing.expectEqual(@as(u64, 192), layerBytes(1, 2, 24));
}

test "device sum is resident plus kv, and a 16 GiB card refuses the overflow" {
    const device: u64 = 16 << 30;
    const resident: u64 = 6 << 30;
    const kv: u64 = 8 << 30;
    const held = Budget{
        .device_bytes = device,
        .resident_bytes = resident,
        .kv_bytes = kv,
        .routed_expert_bytes = 70 << 30,
        .ngram_bytes = 29 << 30,
    };
    try std.testing.expectEqual(resident + kv, deviceSum(held));
    try fits(held);
    try std.testing.expectEqual(Place.host, placeWeights(held));
    try std.testing.expect(refusesResident(.routed));
    try std.testing.expect(refusesResident(.ngram));

    try std.testing.expectError(error.ExceedsDevice, fits(.{
        .device_bytes = device,
        .resident_bytes = resident,
        .kv_bytes = 11 << 30,
        .routed_expert_bytes = 0,
        .ngram_bytes = 0,
    }));

    const rest = device - resident;
    try fits(.{
        .device_bytes = device,
        .resident_bytes = resident,
        .kv_bytes = rest,
        .routed_expert_bytes = 1,
        .ngram_bytes = 1,
    });
    try std.testing.expectError(error.ExceedsDevice, fits(.{
        .device_bytes = device,
        .resident_bytes = resident,
        .kv_bytes = rest + 1,
        .routed_expert_bytes = 0,
        .ngram_bytes = 0,
    }));

    try std.testing.expectEqual(@as(u64, 1_893_120), expertBytes());
    try std.testing.expectEqual(@as(u64, 1_901_213), full_expert_bytes);
    try std.testing.expectEqual(@as(u64, 0), expertLinesOf(1, 0));
    try std.testing.expectEqual(@as(u64, 1), expertLinesOf(full_expert_bytes, full_expert_bytes));

    try std.testing.expectEqual(@as(u64, 0), learnedBytes(card_bytes, card_bytes, 0));

    // Counting the expert bank as resident is what blows the same card.
    try std.testing.expectError(error.ExceedsDevice, fits(.{
        .device_bytes = device,
        .resident_bytes = resident + (70 << 30),
        .kv_bytes = kv,
        .routed_expert_bytes = 0,
        .ngram_bytes = 0,
    }));
}
