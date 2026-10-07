//! A model-free cancellation fixture shows the FP32 bias-sum behavior dense Qwen projections require.
const std = @import("std");
const metal = @import("metal");
const row = @import("tensorfold").row_projection;

fn bf(x: f32) u16 {
    const bits: u32 = @bitCast(x);
    return @truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
}

fn values(buffer: metal.Buffer, comptime T: type) []T {
    return @as([*]T, @ptrCast(@alignCast(buffer.contents())))[0 .. buffer.length() / @sizeOf(T)];
}

pub fn main(init: std.process.Init) !void {
    _ = init;
    const pool = metal.objc.Pool.push();
    defer pool.pop();
    const device = try metal.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const n = 64;
    const k = 128;
    const opts = metal.ResourceOptions.shared | metal.ResourceOptions.untracked;
    const w = try device.buffer(n * k / 2, opts);
    defer w.deinit();
    const scales = try device.buffer(n * k / 64 * 2, opts);
    defer scales.deinit();
    const biases = try device.buffer(n * k / 64 * 2, opts);
    defer biases.deinit();
    const x = try device.buffer(32 * k * 2, opts);
    defer x.deinit();
    const y = try device.buffer(32 * n * 2, opts);
    defer y.deinit();
    @memset(values(w, u32), 0);
    @memset(values(scales, u16), 0);
    @memset(values(biases, u16), bf(1));
    for (values(x, u16), 0..) |*v, i| v.* = bf(switch (i % 4) {
        0 => @as(f32, 1),
        2 => @as(f32, -1),
        else => @as(f32, 1.0 / 1024.0),
    });
    var pipes = try row.Pipelines.load(device);
    defer pipes.deinit();
    const weights = row.Weights{ .w = w, .scales = scales, .biases = biases, .n = n, .k = k };
    var failures: usize = 0;
    for ([_]usize{ 1, 2, 4, 8, 16, 32 }) |rows| {
        const call = try row.call(&pipes, weights, rows, false);
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        e.setPipeline(call.pipeline);
        e.setBuffer(x, 0, 0);
        e.setBuffer(w, 0, 1);
        e.setBuffer(scales, 0, 2);
        e.setBuffer(biases, 0, 3);
        e.setValue(call.dims, 4);
        e.setBuffer(y, 0, 5);
        e.dispatchGroups(.{ .width = call.groups }, .{ .width = call.threads });
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |message| {
            std.debug.print("row seam GPU failure: {s}\n", .{message});
            return error.GpuFailed;
        }
        var unequal: usize = 0;
        for (values(y, u16)[0 .. rows * n]) |actual| if (actual != bf(1.0 / 16.0)) {
            unequal += 1;
        };
        if (unequal != 0) failures += 1;
        std.debug.print("rows {d}: FP32 reference 0.0625, first actual bf16 bits 0x{x}, unequal {d}/{d}\n", .{ rows, values(y, u16)[0], unequal, rows * n });
    }
    if (failures != 0) return error.Fp32BiasSumRequired;
}
