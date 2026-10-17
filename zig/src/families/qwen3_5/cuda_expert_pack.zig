//! The loader's bounded affine-3 staging: one shared scratch pair, including a final partial expert chunk.
const std = @import("std");
const cuda = @import("cuda");
const A3 = cuda.experts3.Affine3Experts;
const aligned = @import("cuda_engine.zig").aligned;
pub const limit = 32;
pub fn chunks(ctx: anytype, comptime pack: anytype, stage: u64, capacity: usize, raw: [3]u64, raw_len: usize, out: u64, count: usize, n: usize, k: usize, m: usize, mi: usize) !void {
    const at = aligned(raw_len);
    const size = A3.scratchBytes(limit, n, k);
    if (at + aligned(size[0]) + size[1] > capacity) return error.StageTooSmall;
    const scratch: [2]u64 = .{ stage + at, stage + at + aligned(size[0]) };
    var c: usize = 0;
    while (c < count) : (c += limit) {
        const sb = c * n * (k / 64) * 2;
        try pack(ctx, raw[0] + c * n * k * 3 / 8, raw[1] + sb, raw[2] + sb, scratch, out + c * m * A3.bytes(1, n, k), @min(limit, count - c), n, k, m, mi);
    }
}
const Mock = struct {
    calls: usize = 0,
    counts: [2]usize = undefined,
    raw: [2][3]u64 = undefined,
    out: [2]u64 = undefined,
    scratch: [2]u64 = undefined,
    fn pack(x: *@This(), w: u64, s: u64, b: u64, tmp: [2]u64, out: u64, count: usize, n: usize, k: usize, m: usize, mi: usize) !void {
        try std.testing.expectEqual(@as(usize, 64), n);
        try std.testing.expectEqual(@as(usize, 128), k);
        try std.testing.expectEqual(@as(usize, 2), m);
        try std.testing.expectEqual(@as(usize, 1), mi);
        x.counts[x.calls] = count;
        x.raw[x.calls] = .{ w, s, b };
        x.out[x.calls] = out;
        x.scratch = tmp;
        x.calls += 1;
    }
};
test "the loader's chunks bound the scratch and pack 32 experts then 3" {
    const n = 64;
    const k = 128;
    const count = 35;
    const raw_len = count * (n * k * 3 / 8 + 2 * n * (k / 64) * 2);
    const size = A3.scratchBytes(limit, n, k);
    const capacity = aligned(raw_len) + aligned(size[0]) + size[1];
    var mock: Mock = .{};
    try std.testing.expectError(error.StageTooSmall, chunks(&mock, Mock.pack, 0x100000, capacity - 1, .{ 1000, 2000, 3000 }, raw_len, 6000, count, n, k, 2, 1));
    try std.testing.expectEqual(@as(usize, 0), mock.calls);
    try chunks(&mock, Mock.pack, 0x100000, capacity, .{ 1000, 2000, 3000 }, raw_len, 6000, count, n, k, 2, 1);
    try std.testing.expectEqual([2]usize{ 32, 3 }, mock.counts);
    try std.testing.expectEqual([3]u64{ 1000 + 32 * n * k * 3 / 8, 2000 + 32 * n * (k / 64) * 2, 3000 + 32 * n * (k / 64) * 2 }, mock.raw[1]);
    try std.testing.expectEqual(6000 + 32 * 2 * A3.bytes(1, n, k), mock.out[1]);
    try std.testing.expectEqual(0x100000 + aligned(raw_len), mock.scratch[0]);
    try std.testing.expectEqual(0x100000 + capacity, mock.scratch[1] + size[1]);
}
