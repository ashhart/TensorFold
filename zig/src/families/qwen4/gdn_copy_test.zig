//! Production CUDA convolution-tail copies are checked against raw prefix slices with guard gaps.
const std = @import("std");
const gdn = @import("cuda_gdn.zig");
const a = std.testing.allocator;
const View = struct { buffer: []u8, offset: usize = 0, len: usize };
const Backend = struct {
    calls: usize = 0,
    fail_at: ?usize = null,
    pub fn copy(b: *@This(), dst: []u8, d: usize, src: []u8, s: usize, n: usize) !void {
        b.calls += 1;
        if (b.calls == b.fail_at) return error.PlantedCopy;
        @memcpy(dst[d..][0..n], src[s..][0..n]);
    }
};
test "Qwen4 GDN: actual CUDA owner copies every raw prefix tail with strided guards" {
    for ([_]usize{ 1, 2, 3, 4, 16 }) |rows| {
        const p: gdn.Params = .{ .rows = rows, .keys = 1, .values = 2, .eps = 1e-6, .row_stride = 131088 };
        const z = try gdn.sizes(p);
        const x = try a.alloc(u8, z.projection + 32);
        defer a.free(x);
        const prior = try a.alloc(u8, z.conv_state + 32);
        defer a.free(prior);
        const dest = try a.alloc(u8, try gdn.extent(p, z.conv_state) + 32);
        defer a.free(dest);
        var random = std.Random.DefaultPrng.init(0x367);
        random.random().bytes(x);
        random.random().bytes(prior);
        @memset(dest, 0xa5);
        var b: Backend = .{};
        try gdn.saveConv(&b, p, View{ .buffer = dest, .offset = 16, .len = dest.len - 32 }, View{ .buffer = x, .offset = 16, .len = z.projection }, View{ .buffer = prior, .offset = 16, .len = z.conv_state });
        try std.testing.expectEqual(rows * 3, b.calls);
        const w = z.conv_state / 3;
        for (0..rows) |r| for (0..3) |t| {
            const at = r + t + 1;
            const want = if (at < 3) prior[16 + at * w ..][0..w] else x[16 + (at - 3) * (z.projection / rows) ..][0..w];
            try std.testing.expectEqualSlices(u8, want, dest[16 + r * p.row_stride + t * w ..][0..w]);
        };
        for (dest, 0..) |byte, i| {
            const payload = i >= 16 and i < dest.len - 16 and (i - 16) % p.row_stride < z.conv_state;
            if (!payload) try std.testing.expectEqual(@as(u8, 0xa5), byte);
        }
        const calls = b.calls;
        try std.testing.expectError(error.Qwen4GdnBuffer, gdn.saveConv(&b, p, View{ .buffer = dest, .len = 0 }, View{ .buffer = x, .len = z.projection }, View{ .buffer = prior, .len = z.conv_state }));
        try std.testing.expectEqual(calls, b.calls);
        b.fail_at = calls + 2;
        try std.testing.expectError(error.PlantedCopy, gdn.saveConv(&b, p, View{ .buffer = dest, .len = dest.len - 32 }, View{ .buffer = x, .len = z.projection }, View{ .buffer = prior, .len = z.conv_state }));
    }
}
