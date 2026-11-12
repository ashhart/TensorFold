//! The DSpark drafter's kernels on the GPU against host references: the target's taps.
const std = @import("std");
const mtl = @import("metal");
const c = @import("config.zig");
const kernels = @import("kernels.zig");
const bf16_round = @import("bf16_round");

const shared = mtl.ResourceOptions.shared;

const Rig = struct {
    device: mtl.Device,
    queue: mtl.Queue,
    k: kernels.Kernels,

    fn init() !Rig {
        const device = mtl.Device.init() catch return error.SkipZigTest;
        errdefer device.deinit();
        const queue = try device.queue();
        errdefer queue.deinit();
        return .{ .device = device, .queue = queue, .k = try kernels.load(device) };
    }

    fn deinit(r: *Rig) void {
        r.k.deinit();
        r.queue.deinit();
        r.device.deinit();
    }

    fn buffer(r: *Rig, comptime T: type, n: usize) !mtl.Buffer {
        const b = try r.device.buffer(n * @sizeOf(T), shared);
        @memset(b.slice(u8, n * @sizeOf(T)), 0);
        return b;
    }

    fn run(r: *Rig, comptime key: []const u8, bufs: []const mtl.Buffer, value: anytype, at: usize, grid: [2]usize) !void {
        const cb = r.queue.commandBuffer();
        const e = cb.compute(.serial);
        e.setPipeline(r.k.get(key));
        for (bufs, 0..) |b, i| e.setBuffer(b, 0, if (i >= at) i + 1 else i);
        e.setValue(value, at);
        e.dispatchThreads(mtl.Size.of(grid[0], grid[1], 1), mtl.Size.of(@min(grid[0], 256), 1, 1));
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |_| return error.GpuFailed;
    }
};

/// Round to nearest even, as MLX casts.
const toBf16 = bf16_round.bits;

fn noise(seed: u64, out: []u16, scale: f32) void {
    var prng = std.Random.DefaultPrng.init(seed);
    for (out) |*v| v.* = toBf16((prng.random().float(f32) * 2 - 1) * scale);
}

test "lfm2 dspark: the tap kernel copies each row's hidden values into its tap's columns, the rest untouched" {
    var r = try Rig.init();
    defer r.deinit();
    const rows = 5;
    const taps = 3;
    const h = try r.buffer(u16, rows * c.hidden);
    defer h.deinit();
    const out = try r.buffer(u16, rows * taps * c.hidden);
    defer out.deinit();
    noise(7, h.slice(u16, rows * c.hidden), 4);
    try r.run("lfm2_tap", &.{ h, out }, [2]u32{ taps * c.hidden, c.hidden }, 2, .{ c.hidden / 8, rows });
    const want = h.slice(u16, rows * c.hidden);
    const got = out.slice(u16, rows * taps * c.hidden);
    for (0..rows) |row| {
        try std.testing.expectEqualSlices(u16, want[row * c.hidden ..][0..c.hidden], got[(row * taps + 1) * c.hidden ..][0..c.hidden]);
        for (got[row * taps * c.hidden ..][0..c.hidden]) |v| try std.testing.expectEqual(@as(u16, 0), v);
        for (got[(row * taps + 2) * c.hidden ..][0..c.hidden]) |v| try std.testing.expectEqual(@as(u16, 0), v);
    }
}

test "lfm2 dspark: toBf16 is core's bits(), 0xffffffff gives a NaN, other words the plain + copy's bits" {
    try std.testing.expect(std.math.isNan(@as(f32, @bitCast(@as(u32, toBf16(@bitCast(@as(u32, 0xffff_ffff)))) << 16))));
    var prng = std.Random.DefaultPrng.init(1863);
    for (0..1 << 19) |i| { // every bf16 value's tie and carry edges, then random words
        const lo = [_]u32{ 0x7fff, 0x8000, 0x8001, 0xffff };
        const w: u32 = if (i < 1 << 18) @as(u32, @intCast(i >> 2)) << 16 | lo[i & 3] else prng.random().int(u32);
        const v: f32 = @bitCast(w);
        if (std.math.isNan(v)) {
            try std.testing.expect(std.math.isNan(@as(f32, @bitCast(@as(u32, toBf16(v)) << 16))));
        } else try std.testing.expectEqual(@as(u16, @intCast((w + 0x7fff + ((w >> 16) & 1)) >> 16)), toBf16(v));
    }
}
