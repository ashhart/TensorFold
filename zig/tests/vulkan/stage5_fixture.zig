// Stage 5: reference re-walks widened bits (not production helpers), q.backend.draw rule; align=256B, tail=16x0xffff
const std = @import("std");
const contract = @import("contract");
const c = contract.spec;

/// Poisoned elements after each row's data, so an overrun reads poison.
pub const tail = 16;
pub const poison16: u16 = 0xffff;

/// The real Qwen3.5 vocab and the argreduce chunk width (greedy.zig block_elements).
pub const vocab: usize = 248320;
pub const chunk: usize = 4096;

// The offset is 128 bf16 = 256 bytes, Vulkan's largest minStorageBufferOffsetAlignment; stages 2-4 use the same.
pub const at: usize = 128;

pub const ReferenceError = error{ NonFinite, Invalid };

// Reference rejects non-finite values; strict ascending comparison keeps the largest and lowest index (+0/-0 equal).
pub fn referenceBits(data: []const u16) ReferenceError!u32 {
    if (data.len == 0) return error.Invalid;
    var best = -std.math.inf(f32);
    var index: u32 = 0;
    for (data, 0..) |b, i| {
        const v = c.fromBf16(b);
        if (!std.math.isFinite(v)) return error.NonFinite;
        if (v > best) {
            best = v;
            index = @intCast(i);
        }
    }
    return index;
}

// The first non-finite element of a row — the evidence decode() must report; the caller guarantees one exists.
pub fn firstBadBits(data: []const u16) usize {
    for (data, 0..) |b, i| {
        if (!std.math.isFinite(c.fromBf16(b))) return i;
    }
    unreachable;
}

pub const NonFinite = enum { nan, pinf, ninf };
pub const ZeroOrder = enum { plus_first, minus_first };

pub const Row = struct {
    a: std.mem.Allocator,
    /// The whole poisoned shell; the row's data is bits[at..][0..count].
    bits: []u16,
    count: usize,

    pub fn deinit(r: Row) void {
        r.a.free(r.bits);
    }

    pub fn data(r: Row) []u16 {
        return r.bits[at..][0..r.count];
    }

    pub fn reference(r: Row) ReferenceError!u32 {
        return referenceBits(r.data());
    }

    pub fn firstBad(r: Row) usize {
        return firstBadBits(r.data());
    }

    // The downloaded shell equals the uploaded shell bit for bit; guards keep poison and logits stay unchanged.
    pub fn checkShell(r: Row, got: []const u16) !void {
        if (got.len != r.bits.len) return error.LengthMismatch;
        if (std.mem.indexOfDiff(u16, r.bits, got) != null) return error.Changed;
    }
};

fn shell(a: std.mem.Allocator, count: usize) ![]u16 {
    const buf = try a.alloc(u16, at + count + tail);
    @memset(buf, poison16);
    return buf;
}

// Random finite bf16 magnitudes use [0.25, cap), both signs unless negative-only; rounding never reaches or passes cap.
fn fillRandom(data: []u16, rng: std.Random, cap: f32, neg_only: bool) void {
    for (data) |*d| {
        const mag = rng.float(f32) * (cap - 0.25) + 0.25;
        const v = if (neg_only) -mag else if (rng.boolean()) mag else -mag;
        d.* = c.toBf16(v);
    }
}

// A realistic full-vocab row: random finite logits with a unique max of exactly 5.0 at count/2.
pub fn random(a: std.mem.Allocator, count: usize, seed: u64) !Row {
    const bits = try shell(a, count);
    errdefer a.free(bits);
    var prng = std.Random.DefaultPrng.init(seed);
    fillRandom(bits[at..][0..count], prng.random(), 4.75, false);
    bits[at + count / 2] = c.toBf16(5.0);
    return .{ .a = a, .bits = bits, .count = count };
}

// Ties at 1, chunk starts, final partial start, and row end test whether lowest index wins across chunk/combine.
pub fn ties(a: std.mem.Allocator, count: usize, seed: u64) !Row {
    const bits = try shell(a, count);
    errdefer a.free(bits);
    var prng = std.Random.DefaultPrng.init(seed);
    const data = bits[at..][0..count];
    fillRandom(data, prng.random(), 3.0, false);
    const tie_value = c.toBf16(3.5);
    const tie_indices = [_]usize{ 1, chunk, 2 * chunk, (count - 1) / chunk * chunk, count - 1 };
    for (tie_indices) |i| data[i] = tie_value;
    return .{ .a = a, .bits = bits, .count = count };
}

/// All-negative logits with a unique max of exactly -0.125 at count/2.
pub fn negative(a: std.mem.Allocator, count: usize, seed: u64) !Row {
    const bits = try shell(a, count);
    errdefer a.free(bits);
    var prng = std.Random.DefaultPrng.init(seed);
    fillRandom(bits[at..][0..count], prng.random(), 4.0, true);
    bits[at + count / 2] = c.toBf16(-0.125);
    return .{ .a = a, .bits = bits, .count = count };
}

// With -1.0 background, +0 at element 5 and -0 at count-7 win in either order because zeros compare equal.
pub fn zero(a: std.mem.Allocator, count: usize, seed: u64, order: ZeroOrder) !Row {
    const bits = try shell(a, count);
    errdefer a.free(bits);
    _ = seed;
    @memset(bits[at..][0..count], c.toBf16(-1.0));
    bits[at + 5] = switch (order) {
        .plus_first => 0x0000,
        .minus_first => 0x8000,
    };
    bits[at + count - 7] = switch (order) {
        .plus_first => 0x8000,
        .minus_first => 0x0000,
    };
    return .{ .a = a, .bits = bits, .count = count };
}

// A finite row has a unique max at count/2; a non-finite `bad` (nan/+inf/-inf) at that spot must still be refused.
pub fn nonFinite(a: std.mem.Allocator, count: usize, seed: u64, kind: NonFinite, bad: usize) !Row {
    const bits = try shell(a, count);
    errdefer a.free(bits);
    var prng = std.Random.DefaultPrng.init(seed);
    const data = bits[at..][0..count];
    fillRandom(data, prng.random(), 4.75, false);
    data[count / 2] = c.toBf16(5.0);
    data[bad] = switch (kind) {
        .nan => 0x7fc0,
        .pinf => 0x7f80,
        .ninf => 0xff80,
    };
    return .{ .a = a, .bits = bits, .count = count };
}

/// The smallest legal row: two elements, the second one largest.
pub fn minTwo(a: std.mem.Allocator) !Row {
    const bits = try shell(a, 2);
    errdefer a.free(bits);
    bits[at] = c.toBf16(-2.0);
    bits[at + 1] = c.toBf16(3.5);
    return .{ .a = a, .bits = bits, .count = 2 };
}

const testing = std.testing;

fn guardsPoisoned(r: Row) !void {
    for (r.bits, 0..) |g, i| {
        if (i >= at and i < at + r.count) continue;
        try testing.expectEqual(poison16, g);
    }
}

test "the reference applies the draw rule on raw bf16 bits" {
    const neg = [_]u16{ c.toBf16(-1.5), c.toBf16(-1.5), c.toBf16(-0.5), c.toBf16(-2.0) };
    try testing.expectEqual(@as(u32, 2), try referenceBits(&neg));
    const tie = [_]u16{ c.toBf16(-1.0), c.toBf16(3.5), c.toBf16(3.5), c.toBf16(3.5) };
    try testing.expectEqual(@as(u32, 1), try referenceBits(&tie));
    const zeros_a = [_]u16{ 0x0000, 0x8000, c.toBf16(-1.0) };
    try testing.expectEqual(@as(u32, 0), try referenceBits(&zeros_a));
    const zeros_b = [_]u16{ 0x8000, 0x0000, c.toBf16(-1.0) };
    try testing.expectEqual(@as(u32, 0), try referenceBits(&zeros_b));
    try testing.expectError(error.Invalid, referenceBits(&.{}));
    try testing.expectError(error.NonFinite, referenceBits(&.{ c.toBf16(1.0), 0x7fc0 }));
    try testing.expectError(error.NonFinite, referenceBits(&.{ c.toBf16(1.0), 0x7f80 }));
    try testing.expectError(error.NonFinite, referenceBits(&.{ 0xff80, c.toBf16(-1.0) }));
}

test "every builder's row agrees with the reference and keeps its guards poisoned" {
    const a = testing.allocator;
    {
        var r = try random(a, 512, 7);
        defer r.deinit();
        _ = try r.reference();
        try guardsPoisoned(r);
    }
    {
        var r = try ties(a, vocab, 8);
        defer r.deinit();
        try testing.expectEqual(@as(u32, 1), try r.reference());
        try guardsPoisoned(r);
    }
    {
        var r = try negative(a, 4096, 9);
        defer r.deinit();
        _ = try r.reference();
        try guardsPoisoned(r);
    }
}

test "zero rows pick element 5 in both orders; non-finite rows refuse and name the offender" {
    const a = testing.allocator;
    {
        var r = try zero(a, 64, 10, .plus_first);
        defer r.deinit();
        try testing.expectEqual(@as(u32, 5), try r.reference());
    }
    {
        var r = try zero(a, 64, 11, .minus_first);
        defer r.deinit();
        try testing.expectEqual(@as(u32, 5), try r.reference());
    }
    {
        var r = try nonFinite(a, 4096, 12, .nan, 100);
        defer r.deinit();
        try testing.expectError(error.NonFinite, r.reference());
        try testing.expectEqual(@as(usize, 100), r.firstBad());
    }
    {
        var m = try minTwo(a);
        defer m.deinit();
        try testing.expectEqual(@as(u32, 1), try m.reference());
        try guardsPoisoned(m);
    }
}
