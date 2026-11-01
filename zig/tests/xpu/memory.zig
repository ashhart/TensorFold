//! Free device memory as the lane host asks it (Gpu.memory, less this process's allocations): Sysman or the core total.

const std = @import("std");
const xpu = @import("xpu");

pub const panic = std.debug.FullPanic(@import("rig.zig").panicDrain);

fn free(ctx: *xpu.Context) !u64 {
    const mem = try ctx.gpu.memory();
    return @min(mem.free, mem.size -| xpu.rt.alloc_now);
}

pub fn main() !void {
    xpu.stop.install();
    var drv = try xpu.Driver.open();
    var ctx = try xpu.Context.init(&drv, null);
    const stream = try xpu.Stream.init(&ctx);
    var r = try xpu.rt.Runtime.init(&ctx, stream);
    const sysman = std.c.access("/dev/dri", 0) == 0 or std.c.access("/dev/dxg", 0) != 0;
    const mem = try ctx.gpu.memory();
    std.debug.print("{s}: {s}, size {d:.2} GiB, free {d:.2} GiB\n", .{ ctx.name(), if (sysman) "Sysman" else "core total (no Sysman: WDDM)", @as(f64, @floatFromInt(mem.size)) / (1 << 30), @as(f64, @floatFromInt(mem.free)) / (1 << 30) });
    const before = try free(&ctx);
    const b = try r.alloc(512 << 20);
    _ = b;
    const after = try free(&ctx);
    std.debug.print("free as the lane host sees it: {d:.2} GiB, after a 512 MiB allocation {d:.2} GiB\n", .{ @as(f64, @floatFromInt(before)) / (1 << 30), @as(f64, @floatFromInt(after)) / (1 << 30) });
    if (before == 0 or after >= before) return error.NoFreeMemory;
    std.debug.print("memory: ok\n", .{});
}
