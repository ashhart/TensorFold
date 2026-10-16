//! A group read as smaller ones dequantizes to the same float64 bits at every width, and the codes keep their bytes.
const std = @import("std");
const st = @import("../../core/safetensors.zig");
const mlx = @import("../../core/quant/mlx.zig");
const affine = @import("affine.zig");
const lane = @import("pack_lane.zig");
const widen = @import("pack_widen.zig");

/// A [rows, k] matrix at `bits`/`group` with seeded codes and finite bf16 scales and biases, owned by `a`.
pub fn synthetic(a: std.mem.Allocator, seed: u64, rows: usize, k: usize, bits: u8, group: usize) !lane.Q {
    var prng = std.Random.DefaultPrng.init(seed);
    const rand = prng.random();
    const kw = k * bits / 32;
    const w = try a.alloc(u8, rows * kw * 4);
    rand.bytes(w);
    const tables = [2][]u8{ try a.alloc(u8, rows * (k / group) * 2), try a.alloc(u8, rows * (k / group) * 2) };
    for (tables) |t| for (std.mem.bytesAsSlice(u16, t)) |*v| {
        const f: f32 = (rand.float(f32) - 0.5) * 4;
        v.* = @intCast(@as(u32, @bitCast(f)) >> 16);
    };
    var shape_w: [st.max_rank]usize = @splat(0);
    shape_w[0] = rows;
    shape_w[1] = kw;
    var shape_s = shape_w;
    shape_s[1] = k / group;
    return .{
        .w = .{ .dtype = .u32, .rank = 2, .shape = shape_w, .bytes = w },
        .s = .{ .dtype = .bf16, .rank = 2, .shape = shape_s, .bytes = tables[0] },
        .b = .{ .dtype = .bf16, .rank = 2, .shape = shape_s, .bytes = tables[1] },
        .rows = rows,
        .k = k,
        .kw = kw,
        .spec = try affine.Spec.init(bits, group),
    };
}

fn reference(a: std.mem.Allocator, q: lane.Q) ![]f64 {
    const out = try a.alloc(f64, q.rows * q.k);
    try mlx.reference(.{ .words = q.w, .scales = q.s, .biases = q.b, .bits = q.spec.bits, .group = @intCast(q.spec.group) }, out);
    return out;
}

test "a group read as smaller ones is an exact identity in float64 at every width" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_][2]usize{ .{ 128, 64 }, .{ 128, 32 }, .{ 64, 32 } }) |pair| for ([_]u8{ 2, 3, 4, 5, 6, 8 }, 0..) |bits, i| {
        const q = try synthetic(a, 0x5eed + i + pair[1], 4, 512, bits, pair[0]);
        const h = try widen.regroup(a, q, pair[1]);
        try std.testing.expectEqual(pair[1], h.spec.group);
        try std.testing.expectEqual(bits, h.spec.bits);
        try std.testing.expectEqual(512 / pair[1], h.s.dim(1));
        try std.testing.expectEqualSlices(u8, q.w.bytes, h.w.bytes);
        const want = try reference(a, q);
        const got = try reference(a, h);
        try std.testing.expectEqualSlices(u64, std.mem.bytesAsSlice(u64, std.mem.sliceAsBytes(want)), std.mem.bytesAsSlice(u64, std.mem.sliceAsBytes(got)));
    };
}

test "a group is never read as a larger one, and its own group passes through untouched" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const q = try synthetic(a, 7, 2, 128, 6, 32);
    try std.testing.expectEqual(q.s.bytes.ptr, (try widen.regroup(a, q, 32)).s.bytes.ptr);
    try std.testing.expectError(error.InexactRegroup, widen.regroup(a, q, 64));
}
