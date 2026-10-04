//! Shared setup for the GPU test programs: device, queue and the embedded metallib.
const std = @import("std");
const mtl = @import("metal");
const metallib = @import("metallib");

pub const Gpu = struct {
    device: mtl.Device,
    queue: mtl.Queue,
    lib: mtl.Library,

    pub fn init() !Gpu {
        const device = try mtl.Device.init();
        const queue = try device.queue();
        const lib = try mtl.Library.fromBytes(device, metallib.bytes);
        return .{ .device = device, .queue = queue, .lib = lib };
    }

    pub fn pipeline(self: Gpu, name: []const u8, indirect: bool) !mtl.Pipeline {
        return mtl.Pipeline.init(self.device, self.lib, name, indirect);
    }

    /// A shared, untracked buffer: our encoders order access themselves.
    pub fn buffer(self: Gpu, comptime T: type, len: usize) !mtl.Buffer {
        return self.device.buffer(len * @sizeOf(T), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    }

    pub fn deinit(self: Gpu) void {
        self.lib.deinit();
        self.queue.deinit();
        self.device.deinit();
    }
};

/// Commit, wait, and fail on a GPU error.
pub fn run(cb: mtl.CommandBuffer) !void {
    cb.commit();
    cb.wait();
    if (cb.failure()) |text| {
        std.debug.print("command buffer failed: {s}\n", .{text});
        return error.GpuFailed;
    }
}

pub fn median(values: []f64) f64 {
    std.mem.sort(f64, values, {}, std.sort.asc(f64));
    return values[values.len / 2];
}

/// tf_round's map v -> 1664525 v + 1013904223 applied `times` times, as one (a, c) pair (wrapping u32).
pub fn lcgPower(times: u32) [2]u32 {
    var result = [2]u32{ 1, 0 };
    var step = [2]u32{ 1664525, 1013904223 };
    var k = times;
    while (k > 0) : (k >>= 1) {
        // temporaries first: an array literal assigned to itself writes element 0 before reading element 1's inputs
        if (k & 1 == 1) {
            const a = step[0] *% result[0];
            const c = step[0] *% result[1] +% step[1];
            result = .{ a, c };
        }
        const a = step[0] *% step[0];
        const c = step[0] *% step[1] +% step[1];
        step = .{ a, c };
    }
    return result;
}

/// out[i] = in[i] after `iters` tf_round steps, for `n` words at the given byte offsets.
pub fn roundKernel(enc: mtl.ComputeEncoder, p: mtl.Pipeline, in: mtl.Buffer, in_off: usize, out: mtl.Buffer, out_off: usize, n: u32, iters: u32) void {
    enc.setPipeline(p);
    enc.setBuffer(in, in_off, 0);
    enc.setBuffer(out, out_off, 1);
    enc.setValue(n, 2);
    enc.setValue(iters, 3);
    enc.dispatchThreads(mtl.Size.of(n, 1, 1), mtl.Size.of(256, 1, 1));
}

test "lcgPower matches stepping" {
    var v: u32 = 2654435761;
    for (0..4000) |_| v = v *% 1664525 +% 1013904223;
    const map = lcgPower(4000);
    try std.testing.expectEqual(v, map[0] *% 2654435761 +% map[1]);
}
