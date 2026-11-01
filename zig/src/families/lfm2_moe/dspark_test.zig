//! The DSpark drafter's kernels on the GPU against host references: the target's taps.
const std = @import("std");
const mtl = @import("metal");
const c = @import("config.zig");
const kernels = @import("kernels.zig");

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
fn toBf16(f: f32) u16 {
    const b: u32 = @bitCast(f);
    return @truncate((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}

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
