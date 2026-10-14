const std = @import("std");
const mtl = @import("metal");
const View = @import("view.zig").View;

pub fn bf(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @intCast((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}
pub fn fp(x: u16) f32 {
    return @bitCast(@as(u32, x) << 16);
}
pub fn view(b: mtl.Buffer) View {
    return .{ .buffer = b };
}
pub const Buffers = struct {
    device: mtl.Device,
    items: [32]mtl.Buffer = undefined,
    count: usize = 0,

    pub fn get(b: *Buffers, comptime T: type, n: usize) !mtl.Buffer {
        if (b.count == b.items.len) return error.TooManyTestBuffers;
        const x = try b.device.buffer(try std.math.mul(usize, @sizeOf(T), n), mtl.ResourceOptions.shared);
        b.items[b.count] = x;
        b.count += 1;
        @memset(x.slice(T, n), 0);
        return x;
    }
    pub fn deinit(b: *Buffers) void {
        for (b.items[0..b.count]) |x| x.deinit();
    }
};
pub fn done(cb: mtl.CommandBuffer, e: mtl.ComputeEncoder) !void {
    e.end();
    cb.commit();
    cb.wait();
    try std.testing.expect(cb.failure() == null);
}
pub fn close(expected: f64, actual: f32, tolerance: f64) !void {
    try std.testing.expect(@abs(expected - @as(f64, actual)) <= tolerance);
}
