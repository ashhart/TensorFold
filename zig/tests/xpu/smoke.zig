//! End-to-end check of the Level Zero path: SPIR-V module, device buffers, a launch, a copy back.

const std = @import("std");
const xpu = @import("xpu");
const rt = @import("rt.zig");

pub fn run() !void {
    var driver = try xpu.Driver.open();
    defer driver.close();
    var ctx = try xpu.Context.init(&driver, rt.ordinal);
    defer ctx.deinit();
    var stream = try xpu.Stream.init(&ctx);
    defer stream.deinit();
    var module = try xpu.Module.load(&ctx, xpu.kernels.vadd);
    defer module.unload();
    var kernel = try module.kernel("vadd", .{ 64, 1, 1 });
    defer kernel.deinit();

    const count: usize = 1 << 20;
    const gpa = std.heap.page_allocator;
    const a = try gpa.alloc(f32, count);
    const b = try gpa.alloc(f32, count);
    const c = try gpa.alloc(f32, count);
    for (a, b, 0..) |*x, *y, i| {
        x.* = @floatFromInt(i);
        y.* = 2.0 * @as(f32, @floatFromInt(i)) + 1.0;
    }
    var da = try xpu.DeviceBuffer.fromHost(&ctx, stream, std.mem.sliceAsBytes(a));
    defer da.free();
    var db = try xpu.DeviceBuffer.fromHost(&ctx, stream, std.mem.sliceAsBytes(b));
    defer db.free();
    var dc = try xpu.DeviceBuffer.alloc(&ctx, count * 4);
    defer dc.free();
    try xpu.launch.dispatch(kernel, stream, .{ @intCast(count / 64), 1, 1 }, .{ da, db, dc });
    try dc.download(stream, 0, std.mem.sliceAsBytes(c));

    var bad: usize = 0;
    for (c, 0..) |v, i| {
        if (v != 3.0 * @as(f32, @floatFromInt(i)) + 1.0) bad += 1;
    }
    std.debug.print("vadd {d} elements: {d} mismatches\n", .{ count, bad });
    if (bad != 0) return error.Mismatch;
}
