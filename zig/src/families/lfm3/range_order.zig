//! Host-checkable LFM3 range-repair rules: shared-cache residency and the key, part and chunk merge order.
const std = @import("std");

/// Declares each segment's keys and values, which shared kernels reach only by GPU address.
pub fn useCaches(e: anytype, segments: anytype, layer: usize, usage: anytype) void {
    for (segments) |seg| {
        const cache = seg.cache.blocks[layer].attention;
        e.useResource(cache.keys, usage);
        e.useResource(cache.values, usage);
    }
}

const per_part = 1024;
const per_chunk = 4 * per_part;
const Sum = struct { maximum: f32 = -std.math.inf(f32), total: f32 = 0, output: f32 = 0 };

fn addKey(s: *Sum, score: f32, value: f32) void {
    const next = @max(s.maximum, score);
    const a = if (s.maximum == -std.math.inf(f32)) 0 else @exp(s.maximum - next);
    const v = @exp(score - next);
    s.total = s.total * a + v;
    s.output = s.output * a + v * value;
    s.maximum = next;
}

fn merge(into: *Sum, part: Sum) void {
    if (part.total == 0) return;
    const next = @max(into.maximum, part.maximum);
    const a = if (into.maximum == -std.math.inf(f32)) 0 else @exp(into.maximum - next);
    const b = @exp(part.maximum - next);
    into.total = into.total * a + part.total * b;
    into.output = into.output * a + part.output * b;
    into.maximum = next;
}

fn foldPart(scores: []const f32, values: []const f32, chunk: usize, part: usize, cut: usize) Sum {
    const first = @max(chunk * per_chunk + part * per_part, cut);
    const end = @min(chunk * per_chunk + (part + 1) * per_part, scores.len);
    var out: Sum = .{};
    var key = first;
    while (key < end) : (key += 1) addKey(&out, scores[key], values[key]);
    return out;
}

fn serial(scores: []const f32, values: []const f32, cut: usize) Sum {
    var out: Sum = .{};
    for (cut / per_chunk..(scores.len + per_chunk - 1) / per_chunk) |chunk| {
        var merged: Sum = .{};
        for (0..4) |part| merge(&merged, foldPart(scores, values, chunk, part, cut));
        merge(&out, merged);
    }
    return out;
}

fn parallel(scores: []const f32, values: []const f32, cut: usize, reversed: bool) Sum {
    var slots: [16][4]Sum = undefined;
    const low = cut / per_chunk;
    const high = (scores.len + per_chunk - 1) / per_chunk;
    var chunk = high;
    while (chunk > low) {
        chunk -= 1;
        for ([_]usize{ 2, 0, 3, 1 }) |part| slots[chunk][part] = foldPart(scores, values, chunk, part, cut);
    }
    var out: Sum = .{};
    for (low..high) |index| {
        var merged: Sum = .{};
        for (0..4) |i| merge(&merged, slots[index][if (reversed) 3 - i else i]);
        merge(&out, merged);
    }
    return out;
}

fn sameBits(a: Sum, b: Sum) bool {
    return std.meta.eql(
        [_]u32{ @bitCast(a.maximum), @bitCast(a.total), @bitCast(a.output) },
        [_]u32{ @bitCast(b.maximum), @bitCast(b.total), @bitCast(b.output) },
    );
}

fn fill(random: std.Random, scores: []f32, values: []f32) void {
    for (scores, values) |*score, *value| {
        score.* = random.float(f32) * 40 - 20;
        value.* = random.float(f32) * 8 - 4;
    }
}

test "LFM3 range order: out-of-order parts merged by index match the serial replay's bits" {
    const scores = try std.testing.allocator.alloc(f32, 32768);
    defer std.testing.allocator.free(scores);
    const values = try std.testing.allocator.alloc(f32, 32768);
    defer std.testing.allocator.free(values);
    var prng = std.Random.DefaultPrng.init(4505);
    fill(prng.random(), scores, values);
    for ([_]usize{ 1, 1023, 1024, 1025, 4095, 4096, 4097, 8192, 32768 }) |count| {
        for ([_]usize{ 0, 1, 1024, 4097 }) |cut| {
            if (cut >= count) continue;
            try std.testing.expect(sameBits(serial(scores[0..count], values[0..count], cut), parallel(scores[0..count], values[0..count], cut, false)));
        }
    }
}

test "LFM3 range order: the planted reversed part order changes bits" {
    const scores = try std.testing.allocator.alloc(f32, 8192);
    defer std.testing.allocator.free(scores);
    const values = try std.testing.allocator.alloc(f32, 8192);
    defer std.testing.allocator.free(values);
    var prng = std.Random.DefaultPrng.init(4505);
    fill(prng.random(), scores, values);
    try std.testing.expect(!sameBits(serial(scores, values, 0), parallel(scores, values, 0, true)));
}

test "LFM3 range order: every shared segment's keys and values are declared" {
    const Attention = struct { keys: u32, values: u32 };
    const Block = struct { attention: Attention };
    const Cache = struct { blocks: [2]Block };
    const Segment = struct { cache: *const Cache };
    const Recorder = struct {
        seen: [4]u32 = undefined,
        count: usize = 0,
        fn useResource(self: *@This(), resource: u32, usage: u32) void {
            std.debug.assert(usage == 1);
            self.seen[self.count] = resource;
            self.count += 1;
        }
    };
    const first: Cache = .{ .blocks = .{ .{ .attention = .{ .keys = 1, .values = 2 } }, .{ .attention = .{ .keys = 3, .values = 4 } } } };
    const second: Cache = .{ .blocks = .{ .{ .attention = .{ .keys = 5, .values = 6 } }, .{ .attention = .{ .keys = 7, .values = 8 } } } };
    const segments = [_]Segment{ .{ .cache = &first }, .{ .cache = &second } };
    var recorder: Recorder = .{};
    useCaches(&recorder, &segments, 1, @as(u32, 1));
    try std.testing.expectEqual(@as(usize, 4), recorder.count);
    try std.testing.expectEqualSlices(u32, &.{ 3, 4, 7, 8 }, &recorder.seen);
}
