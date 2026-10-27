//! KV quant and the resident-set budget.

const std = @import("std");
const shape = @import("../pack/exl3_shape.zig");

pub const Dtype = enum { bf16, int8, int4 };

const group: usize = 32;
const inv_sqrt32: f32 = 1.0 / @sqrt(32.0);

pub fn layerBytes(tokens: usize, kv_heads: usize, head_dim: usize, dtype: Dtype) error{HeadDim}!u64 {
    const t: u64 = @intCast(tokens);
    const h: u64 = @intCast(kv_heads);
    const d: u64 = @intCast(head_dim);
    switch (dtype) {
        .bf16 => return t * 2 * h * d * 2,
        .int8, .int4 => {
            if (head_dim % group != 0) return error.HeadDim;
            const scales = 2 * h * (d / group) * 2;
            const width: u64 = if (dtype == .int8) d else d / 2;
            return t * (2 * h * width + scales);
        },
    }
}

/// A kept prefix stops one token before the prompt end. An empty prompt stays at 0.
pub fn resumeAt(prompt_len: u32) u32 {
    if (prompt_len == 0) return 0;
    return prompt_len - 1;
}

/// In-place FWHT butterflies on 32 lanes, then one multiply by 1/sqrt(32).
fn hadamard(x: *[group]f32) void {
    var lo: usize = 1;
    while (lo <= 16) : (lo *= 2) {
        const hi = group / (2 * lo);
        var nxt: [group]f32 = undefined;
        for (0..hi) |h| {
            for (0..lo) |i| {
                const a = x[(h * 2 + 0) * lo + i];
                const b = x[(h * 2 + 1) * lo + i];
                nxt[(h * 2 + 0) * lo + i] = a + b;
                nxt[(h * 2 + 1) * lo + i] = a - b;
            }
        }
        x.* = nxt;
    }
    for (x) |*v| v.* *= inv_sqrt32;
}

pub fn rotate32(x: []const f32, out: []f32) error{Length}!void {
    if (x.len != group or out.len != group) return error.Length;
    var cur: [group]f32 = undefined;
    @memcpy(cur[0..], x);
    hadamard(&cur);
    @memcpy(out, cur[0..]);
}

fn groupScale(rot: []const f32) f32 {
    var peak: f32 = 0;
    for (rot) |v| peak = @max(peak, @abs(v));
    return peak + 1e-10;
}

pub fn quantize8(x: []const f32, codes: []i8, scale: *f16) error{Length}!void {
    if (x.len != group or codes.len != group) return error.Length;
    var rot: [group]f32 = undefined;
    try rotate32(x, &rot);
    const s = groupScale(&rot);
    for (rot, codes) |v, *c| {
        const q = std.math.clamp(@floor(v / s * 128.0) + 128.0, 0.0, 255.0);
        const as_i: i32 = @intFromFloat(q);
        c.* = @intCast(as_i - 128);
    }
    scale.* = @floatCast(s);
}

pub fn dequant8(codes: []const i8, scale: f16, out: []f32) error{Length}!void {
    if (codes.len != group or out.len != group) return error.Length;
    const s: f32 = @floatCast(scale);
    for (codes, out) |c, *o| {
        o.* = (@as(f32, @floatFromInt(c)) + 0.5) * s * (1.0 / 128.0);
    }
}

/// Low nibble is the even code, high nibble the odd code. `codes` are 32 values in 0..15.
pub fn packNibbles(codes: []const u8, out: []u8) error{Length}!void {
    if (codes.len != group or out.len != group / 2) return error.Length;
    for (out, 0..) |*byte, i| {
        const lo = codes[i * 2] & 0xf;
        const hi = codes[i * 2 + 1] & 0xf;
        byte.* = lo | (hi << 4);
    }
}

pub fn unpackNibbles(bytes: []const u8, out: []u8) error{Length}!void {
    if (bytes.len != group / 2 or out.len != group) return error.Length;
    for (bytes, 0..) |byte, i| {
        out[i * 2] = byte & 0xf;
        out[i * 2 + 1] = byte >> 4;
    }
}

pub fn quantize4(x: []const f32, bytes: []u8, scale: *f16) error{Length}!void {
    if (x.len != group or bytes.len != group / 2) return error.Length;
    var rot: [group]f32 = undefined;
    try rotate32(x, &rot);
    const s = groupScale(&rot);
    var codes: [group]u8 = undefined;
    for (rot, &codes) |v, *c| {
        const q = std.math.clamp(@floor(v / s * 8.0) + 8.0, 0.0, 15.0);
        c.* = @intFromFloat(q);
    }
    try packNibbles(&codes, bytes);
    scale.* = @floatCast(s);
}

pub fn dequant4(bytes: []const u8, scale: f16, out: []f32) error{Length}!void {
    if (bytes.len != group / 2 or out.len != group) return error.Length;
    var codes: [group]u8 = undefined;
    try unpackNibbles(bytes, &codes);
    const s: f32 = @floatCast(scale);
    for (codes, out) |q, *o| {
        o.* = (@as(f32, @floatFromInt(q)) - 7.5) * s * 0.125;
    }
}

pub const Budget = struct {
    device_bytes: u64,
    /// Dense + shared + router + HC + MTP workspace, already summed by the caller.
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

/// One routed expert as this rank stores it.
pub fn expertBytes(ranks: usize) u64 {
    const k = shape.hidden;
    const n = shape.expert_intermediate;
    if (ranks == 2) return matrixBytes(k / 2, n) * 2 + matrixBytes(n, k / 2);
    return matrixBytes(k, n) * 2 + matrixBytes(n, k);
}

pub fn expertLines(room: u64, ranks: usize) u64 {
    return expertLinesOf(room, expertBytes(ranks));
}

/// Budget stride for one owned expert until a header span is larger. Not a line count.
pub const full_expert_bytes: u64 = 1_901_213;

/// Lines that fit in `room` at `line` bytes. A zero stride is no lines, not a panic.
pub fn expertLinesOf(room: u64, line: u64) u64 {
    if (line == 0) return 0;
    return room / line;
}

/// KV for every full-attention layer.
pub fn contextKv(tokens: usize, dtype: Dtype) error{ HeadDim, Ranks }!u64 {
    return contextKvRank(tokens, dtype, 1);
}

/// This rank's KV. Heads split evenly.
pub fn contextKvRank(tokens: usize, dtype: Dtype, ranks: usize) error{ HeadDim, Ranks }!u64 {
    if (ranks == 0 or shape.kv_heads % ranks != 0) return error.Ranks;
    const layers: u64 = shape.layers / shape.full_attention_interval;
    const heads = shape.kv_heads / ranks;
    return (try layerBytes(tokens, heads, shape.head_dim, dtype)) * layers;
}

pub const KvFit = struct {
    dtype: Dtype,
    /// False when even int4 misses `expert_floor`. Not a dtype to start on.
    floor_held: bool,
};

/// Widest KV that still leaves `expert_floor` bytes for learned experts.
pub fn kvFit(device: u64, pinned: u64, tokens: usize, expert_floor: u64, ranks: usize) error{ HeadDim, Ranks }!KvFit {
    const order = [_]Dtype{ .bf16, .int8, .int4 };
    for (order) |dtype| {
        const kv = try contextKvRank(tokens, dtype, ranks);
        if (learnedBytes(device, pinned, kv) >= expert_floor) return .{ .dtype = dtype, .floor_held = true };
    }
    return .{ .dtype = .int4, .floor_held = false };
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

fn roundTrip8(x: []const f32) !void {
    var codes: [group]i8 = undefined;
    var scale: f16 = undefined;
    try quantize8(x, &codes, &scale);
    var back: [group]f32 = undefined;
    try dequant8(&codes, scale, &back);
    var rot: [group]f32 = undefined;
    try rotate32(x, &rot);
    const s: f32 = @floatCast(scale);
    const limit = (0.5 * s / 128.0) + s * 0.01;
    for (back, rot) |got, want| try std.testing.expect(@abs(got - want) <= limit);
}

test "constant and ramp groups stay inside half an int8 bin of the rotation" {
    var constant: [group]f32 = @splat(0.75);
    try roundTrip8(&constant);

    var negative: [group]f32 = @splat(-1.5);
    try roundTrip8(&negative);

    var ramp: [group]f32 = undefined;
    for (&ramp, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 0.17 - 2.4;
    try roundTrip8(&ramp);

    var rot: [group]f32 = undefined;
    try rotate32(&constant, &rot);
    try std.testing.expectApproxEqAbs(rot[0], 0.75 * @sqrt(32.0), 1e-4);
    for (rot[1..]) |v| try std.testing.expectApproxEqAbs(v, 0, 1e-5);
}

test "nibbles pack low then high" {
    var codes: [group]u8 = @splat(0);
    codes[0] = 1;
    codes[1] = 2;
    codes[2] = 15;
    codes[3] = 0;
    var bytes: [group / 2]u8 = undefined;
    try packNibbles(&codes, &bytes);
    try std.testing.expectEqual(@as(u8, 0x21), bytes[0]);
    try std.testing.expectEqual(@as(u8, 0x0f), bytes[1]);

    var back: [group]u8 = undefined;
    try unpackNibbles(&bytes, &back);
    try std.testing.expectEqualSlices(u8, &codes, &back);

    var x: [group]f32 = undefined;
    for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) - 8.0;
    var qpacked: [group / 2]u8 = undefined;
    var scale: f16 = undefined;
    try quantize4(&x, &qpacked, &scale);
    var qcodes: [group]u8 = undefined;
    try unpackNibbles(&qpacked, &qcodes);
    try std.testing.expectEqual(qcodes[0], qpacked[0] & 0xf);
    try std.testing.expectEqual(qcodes[1], qpacked[0] >> 4);

    var restored: [group]f32 = undefined;
    try dequant4(&qpacked, scale, &restored);
    var rot: [group]f32 = undefined;
    try rotate32(&x, &rot);
    const s: f32 = @floatCast(scale);
    const limit = (0.5 * s / 8.0) + s * 0.01;
    for (restored, rot) |got, want| try std.testing.expect(@abs(got - want) <= limit);
}

test "resumeAt stops one token before the end" {
    try std.testing.expectEqual(@as(u32, 0), resumeAt(0));
    try std.testing.expectEqual(@as(u32, 7), resumeAt(8));
    try std.testing.expectEqual(@as(u32, 0), resumeAt(1));
}

test "layer bytes match v0.6.6 row_bytes" {
    try std.testing.expectEqual(@as(u64, 2048), try layerBytes(1, 2, 256, .bf16));
    try std.testing.expectEqual(@as(u64, 1088), try layerBytes(1, 2, 256, .int8));
    try std.testing.expectEqual(@as(u64, 576), try layerBytes(1, 2, 256, .int4));
    try std.testing.expectEqual(@as(u64, 2048), try layerBytes(1, shape.kv_heads, shape.head_dim, .bf16));
    try std.testing.expectEqual(@as(u64, 1088), try layerBytes(1, shape.kv_heads, shape.head_dim, .int8));
    try std.testing.expectEqual(@as(u64, 576), try layerBytes(1, shape.kv_heads, shape.head_dim, .int4));
    try std.testing.expectEqual(@as(u64, 6144), try layerBytes(3, 2, 256, .bf16));
    try std.testing.expectEqual(@as(u64, 192), try layerBytes(1, 2, 24, .bf16));
    try std.testing.expectError(error.HeadDim, layerBytes(1, 2, 24, .int8));
    try std.testing.expectError(error.HeadDim, layerBytes(1, 2, 24, .int4));
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

    try std.testing.expectEqual(@as(u64, 1_893_120), expertBytes(1));
    try std.testing.expectEqual(@as(u64, 948_480), expertBytes(2));
    try std.testing.expectEqual(@as(u64, 1_901_213), full_expert_bytes);
    try std.testing.expectEqual(@as(u64, 0), expertLinesOf(1, 0));
    try std.testing.expectEqual(@as(u64, 1), expertLinesOf(full_expert_bytes, full_expert_bytes));

    // One head leaves 672 MiB at bf16, so 700 MiB steps to int8. 741 MiB is this pin's int4 room; one more MiB misses.
    const ranks: usize = 2;
    const tight_pin: u64 = 15 << 30;
    try std.testing.expectEqual(KvFit{ .dtype = .int8, .floor_held = true }, try kvFit(card_bytes, tight_pin, 8192, 700 << 20, ranks));
    try std.testing.expectEqual(KvFit{ .dtype = .int4, .floor_held = true }, try kvFit(card_bytes, tight_pin, 8192, 741 << 20, ranks));
    try std.testing.expectEqual(KvFit{ .dtype = .int4, .floor_held = false }, try kvFit(card_bytes, tight_pin, 8192, 742 << 20, ranks));
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

test "one kv head keeps bf16 through 8205 on the census floor" {
    const pin: u64 = 2_072_396_202;
    // Fixture floor only. Not a line count the loader hardcodes.
    const floor = full_expert_bytes * 7752;
    const ranks: usize = 2;
    try std.testing.expectEqual(@as(u64, 100_663_296), try contextKvRank(8192, .bf16, ranks));

    const at_8192 = try kvFit(card_bytes, pin, 8192, floor, ranks);
    try std.testing.expectEqual(Dtype.bf16, at_8192.dtype);
    try std.testing.expect(at_8192.floor_held);
    const at_8205 = try kvFit(card_bytes, pin, 8205, floor, ranks);
    try std.testing.expectEqual(Dtype.bf16, at_8205.dtype);
    try std.testing.expect(at_8205.floor_held);
    const at_8206 = try kvFit(card_bytes, pin, 8206, floor, ranks);
    try std.testing.expectEqual(Dtype.int8, at_8206.dtype);
    try std.testing.expect(at_8206.floor_held);

    // Both-heads divisor, 24,576 bytes per token. Not a call to kvFit.
    const both: u64 = 24_576;
    try std.testing.expectEqual(both, try contextKv(1, .bf16));
    const kv_room = card_bytes - pin - device_reserve - floor;
    try std.testing.expectEqual(@as(u64, 100_834_350), kv_room);
    try std.testing.expect(4_102 * both <= kv_room);
    try std.testing.expect(4_103 * both > kv_room);
    try std.testing.expectEqual(try contextKv(8192, .bf16), try contextKvRank(8192, .bf16, 1));
}

test "a zero rank does not panic and is not a held bf16 fit" {
    // Floor 0 would be bf16 with the floor held if a 0-head charge were accepted.
    try std.testing.expectError(error.Ranks, kvFit(card_bytes, 0, 8192, 0, 0));
    try std.testing.expectError(error.Ranks, contextKvRank(8192, .bf16, 0));
    try std.testing.expectError(error.Ranks, contextKvRank(8192, .bf16, 3));
    try std.testing.expectError(error.Ranks, kvFit(card_bytes, 0, 8192, 0, 4));
}
