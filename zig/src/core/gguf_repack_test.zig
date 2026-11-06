//! Exhaust physical byte fields and all half encodings, then combine independently randomized blocks.
const std = @import("std");
const q = @import("gguf_repack.zig");
const ref = @import("gguf_repack_ref.zig");
const t = std.testing;
fn next(seed: *u64) u8 {
    seed.* = seed.* *% 6364136223846793005 +% 1442695040888963407;
    return @truncate(seed.* >> 37);
}
fn fixture(kind: q.Kind, bytes: []u8, seed: *u64) void {
    const g = ref.geometry(kind);
    for (bytes) |*v| v.* = next(seed);
    var start: usize = 0;
    while (start < bytes.len) : (start += g.bytes) {
        for (g.halves, 0..) |at, j| {
            const bits: u16 = if (j == 0) 0x3c01 else 0xb3fd;
            std.mem.writeInt(u16, bytes[start + at ..][0..2], bits, .little);
        }
    }
}
fn code(words: []const u32, bits: u8, i: usize) u32 {
    var v: u32 = 0;
    for (0..bits) |j| {
        const at = i * bits + j;
        v |= ((words[at / 32] >> @intCast(at % 32)) & 1) << @intCast(j);
    }
    return v;
}
const Storage = struct {
    words: [194]u32 = @splat(0xa5a5a5a5),
    scales: [50]f32 = @splat(-12345),
    biases: [50]f32 = @splat(-23456),
    fn write(s: *Storage, kind: q.Kind, src: []const u8) !q.Layout {
        const l = try q.layout(kind, src.len);
        try q.into(kind, src, s.words[1..][0..l.words], s.scales[1..][0..l.groups], s.biases[1..][0..l.groups]);
        return l;
    }
    fn guards(s: *const Storage, l: q.Layout) !void {
        try t.expectEqual(@as(u32, 0xa5a5a5a5), s.words[0]);
        for (s.words[l.words + 1 ..]) |v| try t.expectEqual(@as(u32, 0xa5a5a5a5), v);
        try t.expectEqual(@as(f32, -12345), s.scales[0]);
        for (s.scales[l.groups + 1 ..]) |v| try t.expectEqual(@as(f32, -12345), v);
        try t.expectEqual(@as(f32, -23456), s.biases[0]);
        for (s.biases[l.groups + 1 ..]) |v| try t.expectEqual(@as(f32, -23456), v);
    }
};
fn equal(want: f64, got: f64) !void {
    if (want != got) {
        std.debug.print("GGUF exact mismatch: want {d}, got {d}\n", .{ want, got });
        return error.AffineValueMismatch;
    }
    // Signed-zero policy is reported separately. Every nonzero finite value must also have identical FP64 bits.
    if (want != 0) try t.expectEqual(@as(u64, @bitCast(want)), @as(u64, @bitCast(got)));
}
fn compare(kind: q.Kind, src: []const u8, s: *const Storage, l: q.Layout, all: bool) !void {
    const g = ref.geometry(kind);
    for (0..l.groups) |group| {
        const block = group / (g.values / g.group);
        const b = src[block * g.bytes ..][0..g.bytes];
        const p = try ref.coefficients(kind, b, group % (g.values / g.group));
        try equal(p.s, s.scales[1 + group]);
        try equal(p.b, s.biases[1 + group]);
        for (0..g.group) |j| {
            if (!all and j != 0 and j != g.group - 1) continue;
            const i = group * g.group + j;
            const got = code(s.words[1..][0..l.words], g.bits, i);
            try t.expectEqual(ref.code(kind, b, i % g.values), got);
            const want = try ref.value(kind, b, i % g.values);
            const actual = @as(f64, s.scales[1 + group]) * @as(f64, @floatFromInt(got)) + s.biases[1 + group];
            try equal(want, actual);
            if (want != 0) try t.expectEqual(@as(u32, @bitCast(@as(f32, @floatCast(want)))), @as(u32, @bitCast(@as(f32, @floatCast(actual)))));
        }
    }
}
test "GGUF repack: deterministic combined blocks, compact words, guards and five identical runs" {
    var seed: u64 = 0x7d9a4b16;
    for (ref.kinds) |kind| {
        const g = ref.geometry(kind);
        for (0..128) |round| {
            var input: [630]u8 = undefined;
            const src = input[0 .. 3 * g.bytes];
            fixture(kind, src, &seed);
            if (round != 0) {
                for (0..3) |block| for (g.halves, 0..) |h, j| {
                    var bits = @as(u16, next(&seed)) + 256 * @as(u16, next(&seed));
                    if ((bits & 0x7c00) == 0x7c00) bits &= 0xfbff;
                    const edges = [_]u16{ 0, 0x8000, 1, 0x03ff, 0x0400, 0x7bff, 0xfbff, 0x3c00, 0xbc00 };
                    if (round < edges.len) bits = edges[(round + j + block) % edges.len];
                    std.mem.writeInt(u16, src[block * g.bytes + h ..][0..2], bits, .little);
                };
            }
            const before = input;
            var s: Storage = .{};
            const l = try s.write(kind, src);
            try t.expectEqual(@as(usize, 3 * g.values), l.elements);
            try t.expectEqual(@as(usize, g.bits), l.format.bits);
            try t.expectEqual(g.group, l.format.group);
            try t.expectEqual(l.elements / 32 * g.bits, l.words);
            try compare(kind, src, &s, l, true);
            const expected = s;
            for (0..5) |_| {
                _ = try s.write(kind, src);
                try t.expectEqualSlices(u8, std.mem.sliceAsBytes(&expected.words), std.mem.sliceAsBytes(&s.words));
                try t.expectEqualSlices(u8, std.mem.sliceAsBytes(&expected.scales), std.mem.sliceAsBytes(&s.scales));
                try t.expectEqualSlices(u8, std.mem.sliceAsBytes(&expected.biases), std.mem.sliceAsBytes(&s.biases));
                try t.expectEqualSlices(u8, before[0..src.len], src);
                try s.guards(l);
            }
        }
    }
}
test "GGUF repack: every byte value at every non-half field position against original FP64 values" {
    var seed: u64 = 0xe3c726a4;
    for (ref.kinds) |kind| {
        const g = ref.geometry(kind);
        var input: [210]u8 = undefined;
        fixture(kind, input[0..g.bytes], &seed);
        const original = input;
        for (0..g.bytes) |at| {
            var is_half = false;
            for (g.halves) |h| if (at == h or at == h + 1) {
                is_half = true;
            };
            if (is_half) continue;
            for (0..256) |value| {
                input = original;
                input[at] = @intCast(value);
                var s: Storage = .{};
                const l = try s.write(kind, input[0..g.bytes]);
                try compare(kind, input[0..g.bytes], &s, l, true);
                try s.guards(l);
            }
        }
    }
}
test "GGUF repack: every binary16 pattern at every half field, exact group metadata and endpoint values" {
    var seed: u64 = 0x01b1fc73;
    for (ref.kinds) |kind| {
        const g = ref.geometry(kind);
        var input: [210]u8 = undefined;
        fixture(kind, input[0..g.bytes], &seed);
        const original = input;
        for (g.halves) |at| for (0..65536) |value| {
            input = original;
            std.mem.writeInt(u16, input[at..][0..2], @intCast(value), .little);
            var s: Storage = .{};
            if ((value & 0x7c00) == 0x7c00) {
                const before = s;
                const l = try q.layout(kind, g.bytes);
                try t.expectError(error.NonFiniteQuantScale, s.write(kind, input[0..g.bytes]));
                try t.expectEqualSlices(u8, std.mem.asBytes(&before), std.mem.asBytes(&s));
                try s.guards(l);
            } else {
                const l = try s.write(kind, input[0..g.bytes]);
                try compare(kind, input[0..g.bytes], &s, l, false);
                try s.guards(l);
            }
        };
    }
}
test "GGUF repack: manual FP64 half decoder agrees with every finite native half and refuses every nonfinite" {
    var bytes: [2]u8 = undefined;
    for (0..65536) |v| {
        std.mem.writeInt(u16, &bytes, @intCast(v), .little);
        if ((v & 0x7c00) == 0x7c00) {
            try t.expectError(error.NonFiniteQuantScale, ref.half(&bytes, 0));
        } else {
            const want: f64 = @floatCast(@as(f16, @bitCast(@as(u16, @intCast(v)))));
            try t.expectEqual(@as(u64, @bitCast(want)), @as(u64, @bitCast(try ref.half(&bytes, 0))));
        }
    }
}
test "GGUF repack: malformed lengths, nonfinite tails, storage and aliases are refused before allocation or writes" {
    var input: [420]u8 align(4) = @splat(0);
    var fail = t.FailingAllocator.init(t.allocator, .{ .fail_index = 0 });
    try t.expectError(error.UnsupportedRepackType, q.repack(fail.allocator(), .f32, &input));
    try t.expectError(error.InvalidBlockLength, q.repack(fail.allocator(), .q8_0, input[0..33]));
    try t.expectError(error.InvalidBlockLength, q.repack(fail.allocator(), .q8_0, input[0..0]));
    input[66] = 0;
    input[67] = 0x7c;
    try t.expectError(error.NonFiniteQuantScale, q.repack(fail.allocator(), .q8_0, input[32..100]));
    try t.expect(!fail.has_induced_failure);
    @memset(&input, 0);
    var s: Storage = .{};
    const before = s;
    try t.expectError(error.InvalidRepackStorage, q.into(.q8_0, input[0..34], s.words[1..8], s.scales[1..2], s.biases[1..2]));
    try t.expectEqualSlices(u8, std.mem.asBytes(&before), std.mem.asBytes(&s));
    try t.expectError(error.OverlappingRepackStorage, q.into(.q8_0, input[0..34], @as([*]u32, @ptrCast(&input))[0..8], s.scales[1..2], s.biases[1..2]));
    try t.expectError(error.OverlappingRepackStorage, q.into(.q8_0, input[0..34], s.words[1..9], s.scales[1..2], s.scales[1..2]));
    try t.expectError(error.RepackSizeOverflow, q.layout(.q4_0, std.math.maxInt(usize) / 18 * 18));
}
fn allocationTrial(a: std.mem.Allocator) !void {
    var input: [210]u8 = undefined;
    var seed: u64 = 44;
    fixture(.q6_k, &input, &seed);
    var result = try q.repack(a, .q6_k, &input);
    defer result.deinit();
    try t.expectEqual(@as(usize, 48), result.words.len);
    try t.expectEqual(@as(usize, 16), result.scales.len);
    try t.expectEqual(@as(usize, 16), result.biases.len);
}
test "GGUF repack: allocation failures release all prior owned arrays" {
    try t.checkAllAllocationFailures(t.allocator, allocationTrial, .{});
}
test "GGUF repack: FP32 coefficient witnesses and unaligned serialized fields" {
    var input: [211]u8 align(4) = @splat(0);
    var src: []u8 = input[1..35];
    std.mem.writeInt(u16, src[0..2], 0x7bff, .little);
    src[2] = 127;
    var s: Storage = .{};
    const l = try s.write(.q8_0, src);
    try compare(.q8_0, src, &s, l, true);
    try t.expect(std.math.isFinite(s.biases[1]));
    try t.expect(std.math.isInf(@as(f16, @floatCast(s.biases[1]))));
    @memset(&input, 0);
    src = input[1..145];
    std.mem.writeInt(u16, src[0..2], 0x3c01, .little);
    std.mem.writeInt(u16, src[2..4], 0x3c01, .little);
    src[4] = 63;
    src[8] = 63;
    s = .{};
    const k = try s.write(.q4_k, src);
    try compare(.q4_k, src, &s, k, true);
    try t.expect(s.scales[1] != @as(f32, @floatCast(@as(f16, @floatCast(s.scales[1])))));
    try t.expect(s.biases[1] != @as(f32, @floatCast(@as(f16, @floatCast(s.biases[1])))));
}
test "GGUF repack: all finite half times every signed byte are exact in FP32, including centered bias" {
    var bytes: [2]u8 = undefined;
    for (0..65536) |v| {
        if ((v & 0x7c00) == 0x7c00) continue;
        std.mem.writeInt(u16, &bytes, @intCast(v), .little);
        const d = try ref.half(&bytes, 0);
        const native: f32 = @floatCast(d);
        for (0..256) |s| {
            const integer = @as(i32, @intCast(s)) - 128;
            const want = d * @as(f64, @floatFromInt(integer));
            const actual = native * @as(f32, @floatFromInt(integer));
            try t.expectEqual(@as(u64, @bitCast(want)), @as(u64, @bitCast(@as(f64, actual))));
            const bias = -128 * actual;
            try t.expectEqual(@as(u64, @bitCast(-128 * want)), @as(u64, @bitCast(@as(f64, bias))));
        }
    }
}
test "GGUF repack: IEEE counterexamples distinguish numerical exactness from every weight bit" {
    var input: [18]u8 = @splat(0x88);
    std.mem.writeInt(u16, input[0..2], 0xbc00, .little);
    var s: Storage = .{};
    const l = try s.write(.q4_0, &input);
    const want = try ref.value(.q4_0, &input, 0);
    const got = @as(f64, s.scales[1]) * @as(f64, @floatFromInt(code(s.words[1..][0..l.words], 4, 0))) + s.biases[1];
    try t.expectEqual(@as(u64, 0x8000000000000000), @as(u64, @bitCast(want)));
    try t.expectEqual(@as(u64, 0), @as(u64, @bitCast(got)));
    try t.expect(want == got);
    const d = std.math.inf(f64);
    const signed = -d;
    const unsigned = d * 127 - 128 * d;
    try t.expect(std.math.isInf(signed) and std.math.isNan(unsigned));
}
