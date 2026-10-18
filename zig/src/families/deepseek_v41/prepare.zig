//! The run-time weight forms built at load from named host tensors: linear strips, routed + shared experts' tables.

const std = @import("std");

/// Trellis int16 [kt, nt, words] -> int32 [nt/8, kt, 8, words/2]: each 128-column block's tiles in K order.
pub fn strips(gpa: std.mem.Allocator, src: []const u8, kt: usize, nt: usize, words: usize) ![]align(16) u8 {
    if (nt % 8 != 0 or src.len != kt * nt * words * 2) return error.BadTrellis;
    const run = 8 * words * 2; // 8 tiles of one K row inside one 128-column block
    const out = try gpa.alignedAlloc(u8, .@"16", src.len);
    const blocks = nt / 8;
    for (0..blocks * kt) |i| {
        const nb = i / kt;
        const k = i % kt;
        @memcpy(out[i * run ..][0..run], src[(k * nt + nb * 8) * words * 2 ..][0..run]);
    }
    return out;
}

/// Strips words -> dense3's lanes layout (dense3.cu to_lanes_kernel): per strip, lane L's bits of each G-step group.
pub fn lanes(gpa: std.mem.Allocator, strip_words: []const u8, kt: usize, k2: usize) ![]align(16) u8 {
    const G: usize = if (k2 == 16) 1 else 2;
    const nb = k2 / 2; // bytes of a lane's run of one tile (4 K2 bits)
    const strip = 4 * kt * 32 * k2; // bytes a strip
    if (k2 < 4 or k2 % 2 != 0 or kt % G != 0 or strip_words.len % strip != 0) return error.BadStrips;
    const out = try gpa.alignedAlloc(u8, .@"16", strip_words.len);
    const tile = 16 * k2; // bytes a tile (4 K2 words)
    for (0..strip_words.len / strip) |s| {
        const src = strip_words[s * strip ..][0..strip];
        const dst = out[s * strip ..][0..strip];
        for (0..kt / G) |gi| {
            const g0 = gi * G * 8 * tile; // the group's first byte (src and dst)
            for (0..32) |L| for (0..8 * G) |tt| for (0..nb) |j| {
                const sb = g0 + tt * tile + L * nb + j; // stream byte in the source
                const lb = tt * nb + j; // stream byte in the lane's run
                const w = lb / 4; // the lane's word
                const db = g0 + 4 * (((w / 4) * 32 + L) * 4 + w % 4) + (lb % 4); // stream byte in the destination
                dst[4 * (db / 4) + 3 - db % 4] = src[4 * (sb / 4) + 3 - sb % 4];
            };
        }
    }
    return out;
}

/// One expert of a projection: its trellis's device address and K2, its scales as stored (fp16).
pub const Expert = struct { trellis: u64, k2: i32, suh: []const u8, svh: []const u8 };

/// A projection's tables: int64 addresses, int32 K2s, suh [E, K] and svh [E, N] fp16.
pub const Tables = struct {
    ptrs: []i64,
    k2s: []i32,
    suh: []align(16) u8,
    svh: []align(16) u8,
    k2_lo: i32,
    k2_hi: i32,

    pub fn deinit(t: *Tables, gpa: std.mem.Allocator) void {
        gpa.free(t.ptrs);
        gpa.free(t.k2s);
        gpa.free(t.suh);
        gpa.free(t.svh);
        t.* = undefined;
    }
};

/// The tables of `experts` in order, the shared expert last; every suh / svh must have the projection's K and N.
pub fn tables(gpa: std.mem.Allocator, experts: []const Expert) !Tables {
    if (experts.len == 0) return error.NoExperts;
    const kb = experts[0].suh.len;
    const nb = experts[0].svh.len;
    const ptrs = try gpa.alloc(i64, experts.len);
    errdefer gpa.free(ptrs);
    const k2s = try gpa.alloc(i32, experts.len);
    errdefer gpa.free(k2s);
    const suh = try gpa.alignedAlloc(u8, .@"16", try std.math.mul(usize, experts.len, kb));
    errdefer gpa.free(suh);
    const svh = try gpa.alignedAlloc(u8, .@"16", try std.math.mul(usize, experts.len, nb));
    var t: Tables = .{ .ptrs = ptrs, .k2s = k2s, .suh = suh, .svh = svh, .k2_lo = std.math.maxInt(i32), .k2_hi = 0 };
    errdefer gpa.free(svh);
    for (experts, 0..) |e, i| {
        if (e.suh.len != kb or e.svh.len != nb) return error.ExpertShapesDiffer;
        t.ptrs[i] = @bitCast(e.trellis);
        t.k2s[i] = e.k2;
        @memcpy(t.suh[i * kb ..][0..kb], e.suh);
        @memcpy(t.svh[i * nb ..][0..nb], e.svh);
        t.k2_lo = @min(t.k2_lo, e.k2);
        t.k2_hi = @max(t.k2_hi, e.k2);
    }
    return t;
}

const testing = std.testing;

test "strips: each 128-column block's tiles in K order" {
    const a = testing.allocator;
    const kt = 3;
    const nt = 16;
    const words = 2; // one int32 a tile, to keep the check readable
    var src: [kt * nt * words * 2]u8 = undefined;
    // tile (k, n) holds the byte 16 k + n
    for (0..kt) |k| for (0..nt) |n| @memset(src[(k * nt + n) * words * 2 ..][0 .. words * 2], @intCast(16 * k + n));
    const out = try strips(a, &src, kt, nt, words);
    defer a.free(out);
    // out[nb][k][j] = tile (k, 8 nb + j)
    for (0..2) |nb| for (0..kt) |k| for (0..8) |j| {
        try testing.expectEqual(@as(u8, @intCast(16 * k + 8 * nb + j)), out[((nb * kt + k) * 8 + j) * words * 2]);
    };
    try testing.expectError(error.BadTrellis, strips(a, src[0..8], kt, nt, words));
}

test "lanes: the numpy reference's layout (to_lanes) at every dense3 width" {
    // fixtures/lanes-ref.bin: for K2 4, 6, 8, 10, 12, 16: u32 K2, u32 NB, u32 KT, strips words, lanes words
    const a = testing.allocator;
    var rest: []const u8 = @embedFile("fixtures/lanes-ref.bin");
    var cases: usize = 0;
    while (rest.len > 0) : (cases += 1) {
        const k2 = std.mem.readInt(u32, rest[0..4], .little);
        const nb = std.mem.readInt(u32, rest[4..8], .little);
        const kt = std.mem.readInt(u32, rest[8..12], .little);
        const bytes = 4 * nb * kt * 32 * k2;
        const src = rest[12..][0..bytes];
        const want = rest[12 + bytes ..][0..bytes];
        const got = try lanes(a, src, kt, k2);
        defer a.free(got);
        try testing.expectEqualSlices(u8, want, got);
        rest = rest[12 + 2 * bytes ..];
    }
    try testing.expectEqual(@as(usize, 6), cases);
}

test "expert tables: addresses, K2s and stacked scales in order, the width range" {
    const a = testing.allocator;
    const s1 = [_]u8{ 1, 2, 3, 4 };
    const s2 = [_]u8{ 5, 6, 7, 8 };
    const v1 = [_]u8{ 9, 9 };
    const v2 = [_]u8{ 7, 7 };
    var t = try tables(a, &.{ .{ .trellis = 0x7f00_0000_1000, .k2 = 4, .suh = &s1, .svh = &v1 }, .{ .trellis = 0x7f00_0002_0000, .k2 = 10, .suh = &s2, .svh = &v2 } });
    defer t.deinit(a);
    try testing.expectEqualSlices(i64, &.{ 0x7f00_0000_1000, 0x7f00_0002_0000 }, t.ptrs);
    try testing.expectEqualSlices(i32, &.{ 4, 10 }, t.k2s);
    try testing.expectEqualSlices(u8, &.{ 1, 2, 3, 4, 5, 6, 7, 8 }, t.suh);
    try testing.expectEqualSlices(u8, &.{ 9, 9, 7, 7 }, t.svh);
    try testing.expect(t.k2_lo == 4 and t.k2_hi == 10);
    try testing.expectError(error.ExpertShapesDiffer, tables(a, &.{ .{ .trellis = 1, .k2 = 4, .suh = &s1, .svh = &v1 }, .{ .trellis = 2, .k2 = 4, .suh = s2[0..2], .svh = &v2 } }));
}

test "real expert table constructor balances every allocation failure" {
    const C = struct {
        fn run(a: std.mem.Allocator) !void {
            var t = try tables(a, &.{.{ .trellis = 7, .k2 = 4, .suh = &.{ 1, 2 }, .svh = &.{ 3, 4 } }});
            defer t.deinit(a);
        }
    };
    try testing.checkAllAllocationFailures(testing.allocator, C.run, .{});
}
