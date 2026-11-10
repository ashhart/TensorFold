//! The CUDA test runner's device and embedded-kernel information query.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");

pub fn info(gpu: check.Gpu) !void {
    var name_buf: [256]u8 = undefined;
    const name = try gpu.ctx.name(&name_buf);
    const mem = try gpu.ctx.memInfo();
    std.debug.print("RESULT driver CUDA {d}, device {s}, sm_{d}, {d} SMs, {d} MiB total, {d} MiB free, kernels embedded {}\n", .{
        try gpu.d.version(),      name,
        try gpu.ctx.capability(), try gpu.ctx.attribute(.multiprocessor_count),
        mem.total >> 20,          mem.free >> 20,
        cuda.kernels.available,
    });
}
